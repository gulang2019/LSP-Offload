# We want to parallel the compute with communication
'''
Conceptually, the main thread of torch initiates the communication, we launch a cpu thread to modify the result, and send the result back to the gpu; some operations on gpu need to wait for this result to be modified.
'''
import torch
import torch.nn as nn
from typing import Union, Tuple, List, Dict, Callable
import transformers
import numpy as np
import time
from gradient_compressor import IntCompressorDispatcher
from state import State

from utils import NVTXContext
import kernels.fused_adam as fused_adam

PROFILE = False

class Context:
    h2dStream: torch.cuda.streams.Stream = None
    d2hStream: torch.cuda.streams.Stream = None
    @staticmethod
    def init():
        Context.h2dStream = torch.cuda.Stream()
        Context.d2hStream = torch.cuda.Stream()
    def __init__(self):
        self.stack = []
        self.compute_time = []
        self.upload_wait = []
        self.offload_wait = []
        self.upload_comm = []
        self.offload_comm = []
    def push(self, obj: Tuple[torch.cuda.Event, torch.Tensor]):
        self.stack.append(obj)
    def pop(self) -> Tuple[torch.cuda.Event, torch.Tensor]:
        item = self.stack[-1]
        self.stack.pop()
        return item

def _process_split(ctx, optim, dtype, pinned_buffer_grad, state: State):
    def _impl():
        with torch.no_grad():
            t = time.perf_counter()
            event, pinned_buffer_B = ctx.pop()
            _, pinned_buffer_A = ctx.pop()
            event.synchronize()
            if state == State.PROFILE: ctx.offload_wait.append(time.perf_counter() - t)
            t = time.perf_counter()
            grad = pinned_buffer_A.to(dtype) @ pinned_buffer_B.to(dtype)
            optim.step(grad.contiguous())
            pinned_buffer_grad.copy_(grad)
            if state == State.PROFILE: ctx.compute_time.append(time.perf_counter() - t)
    return _impl

def _process(ctx, optim, dtype, state: State):
    def _impl():
        # print(f'_process {ctx.idx} called')
        with NVTXContext(f'process_{ctx.idx}', PROFILE):
            with torch.no_grad():
                t = time.perf_counter()
                event, pinned_buffer_grad = ctx.pop()
                event.synchronize()
                if state == State.PROFILE: ctx.offload_wait.append(time.perf_counter() - t)
                # grad = pinned_buffer_grad.to(dtype)
                t = time.perf_counter()
                # grad = grad.contiguous()
                # if not torch.isnan(pinned_buffer_grad).sum().item() == 0:
                #     print(f'nan found in pinned_buffer_grad of {ctx.idx}')
                # assert torch.isnan(pinned_buffer_grad).sum().item() == 0
                optim.step(pinned_buffer_grad)
                # pinned_buffer_grad.copy_(grad.to(pinned_buffer_grad.dtype))
                if state == State.PROFILE: ctx.compute_time.append(time.perf_counter() - t)
    return _impl
gpu_buffer = None
def _h2d_comm(ctx, pinned_buffer, device, state: State):
    # global gpu_buffer
    # if gpu_buffer is None:
    #     gpu_buffer = torch.empty_like(pinned_buffer, device = device, dtype = dtype)
    # assert pinned_buffer.size() == gpu_buffer.size()
    def _impl():
        # print(f'h2d_comm {ctx.idx} called')
        with NVTXContext(f'h2dcomm_{ctx.idx}', PROFILE):
            with torch.cuda.stream(ctx.h2dStream):
                t = time.perf_counter()
                new_buffer = pinned_buffer.to(device, non_blocking= state != State.PROFILE_COMM)
                # weight.add_(decompressor(new_buffer.to(weight.dtype)))
                # gpu_buffer.copy_(pinned_buffer, non_blocking= state != State.PROFILE_COMM)
                # weight.add_(decompressor(gpu_buffer.to(weight.dtype)))
                if state == State.PROFILE_COMM:
                    ctx.upload_comm.append((time.perf_counter() - t, pinned_buffer.numel() * pinned_buffer.element_size()))
                event = torch.cuda.Event()
                event.record(ctx.h2dStream)
                ctx.push((event, new_buffer))
            
    return _impl

def _d2h_comm(ctx, device_buffer, pinned_buffer, state: State):
    _event = torch.cuda.Event()
    _event.record(torch.cuda.current_stream())
    def _impl():
        # print(f'd2h_comm {ctx.idx} called')
        with NVTXContext(f'd2h_comm_{ctx.idx}', PROFILE):
            with torch.no_grad():
                nonlocal _event
                ctx.d2hStream.wait_event(_event)
                # torch.cuda.synchronize()
                with torch.cuda.stream(ctx.d2hStream):
                    t = time.perf_counter()
                    assert torch.isnan(device_buffer).sum().item() == 0
                    pinned_buffer.copy_(device_buffer, non_blocking=state != State.PROFILE_COMM)
                    
                    if state == State.PROFILE_COMM:
                        # print('d2h_comm', device_buffer.numel() * device_buffer.element_size(), time.perf_counter() - t)
                        ctx.offload_comm.append((time.perf_counter() - t, device_buffer.numel() * device_buffer.element_size()))
                    event = torch.cuda.Event()
                    event.record(ctx.d2hStream)
                    ctx.push((event, pinned_buffer))
    return _impl

def _update(ctx, weight, decompressor: Callable[[torch.Tensor], torch.Tensor], state: State):
    def _impl():
        # print(f'update {ctx.idx} called')
        # print(f'max memory usage bwfore update {ctx.idx}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
        with NVTXContext(f'update_{ctx.idx}', PROFILE):
            with torch.no_grad():
                t = time.perf_counter()
                event, delta = ctx.pop()
                assert torch.isnan(delta).sum().item() == 0
                torch.cuda.current_stream().wait_event(event)
                if state == State.PROFILE: ctx.upload_wait.append(time.perf_counter() - t)
                weight.add_(decompressor(delta.to(weight.dtype)))
                del delta
        # print(f'max memory usage after update {ctx.idx}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
    return _impl

class LoRaOffloadMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, optim):
        output = input @ weight
        if optim is not None:
            ctx.optim = optim
            ctx.optim.forward(input)
            ctx.save_for_backward(weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert hasattr(ctx, 'optim')
        weight, = ctx.saved_tensors # input: B x SL x HS, weight: HS x HS, S: SL x rank_approx
        
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            # For the gradient w.r.t. the input, if needed
            grad_input = grad_output @ weight.t()

        if ctx.needs_input_grad[1]:
            # grad_output: B x SL x HS, input: B x SL x HS, Weight HS x HS
            ctx.optim.backward(grad_output.reshape(-1, grad_output.size(-1)))
            grad_weight = None
                
        return grad_input, grad_weight, None

class LoRaOffloadLinear(nn.Module):
    def __init__(self, 
                 module: Union[nn.Linear, transformers.pytorch_utils.Conv1D], 
                 lr: float = 1e-3,
                 offloading_device = 'cpu',
                 optim_dtype: torch.dtype = torch.float32,
                 int_compress_tag: str = None,
                 intermidiate_size: torch.Tensor = None,
                 use_grad_ckpt: bool = False,
                 global_state: Dict = None):
        super(LoRaOffloadLinear, self).__init__()
        if isinstance(module, nn.Linear):
            self.in_features = module.in_features
            self.out_features = module.out_features
            self.weight = torch.nn.Parameter(module.weight.data.t())
        elif isinstance(module, transformers.pytorch_utils.Conv1D):
            self.in_features = module.weight.size(0)
            self.out_features = module.weight.size(1)
            self.weight = module.weight
        self.weight.requires_grad = True
        self.bias = module.bias
        self.intermidiate_size = intermidiate_size
        self.offloading_device = offloading_device

        self.optim = ParallelOffloadAdamMMOptim(self.weight,
                                        lr = lr,
                                        offloading_device=offloading_device,
                                        optim_dtype = optim_dtype,
                                        int_compress_tag=int_compress_tag,
                                        intermidiate_size=intermidiate_size,
                                        use_grad_ckpt = use_grad_ckpt,
                                        global_state = global_state)

    def forward(self, input, *args, **kwargs):
        ret = LoRaOffloadMatmulFunction.apply(input, self.weight, self.optim if self.training else None)
        if self.bias is not None:
            ret += self.bias
        return ret

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, int_size = {self.intermidiate_size},"
                f" offloading_device={self.offloading_device}, ")

class ParallelOffloadAdamMMOptim:
    def __init__(self, 
                 weight: torch.Tensor, 
                 lr: float, 
                 offloading_device = 'cpu',
                 optim_dtype: torch.dtype = torch.float32,
                 int_compress_tag: str = None,
                 intermidiate_size: int = None,
                 use_grad_ckpt: bool = False,
                 global_state: Dict = None):
        self.weight = weight
        self.compute_device = weight.device
        self.optim_dtype = optim_dtype
        self.optim = Adam(lr, device = offloading_device, dtype = optim_dtype)
        self.offloading_device = offloading_device
        self.int_compress_tag = int_compress_tag
        self.intermidiate_size = intermidiate_size
        assert int_compress_tag in IntCompressorDispatcher
        self.intermidiate_compressor = IntCompressorDispatcher[int_compress_tag](intermidiate_size)
        self.split_comm = self.int_compress_tag is not None
        self.global_state = global_state
        self.inited = False
        if self.split_comm:
            self.pinned_buffer_A = None
            self.pinned_buffer_B = None
        self.pinned_buffer_grad = None
        self._grad = None
        
        self.current_device = self.offloading_device
        
        self.use_grad_ckpt = use_grad_ckpt
        # A hack impl to determine whether this is the first forward pass
        if self.use_grad_ckpt:
            self.first_FWD = False
        
    def _new_pinned_buffer_if_needed(self, size, old_buffer):
        if old_buffer is None or size != old_buffer.size():
            return torch.empty(size, device = self.offloading_device, dtype = self.optim_dtype, pin_memory=True)
        return old_buffer
    
    def init(self, left_compress_size, right_compress_size):
        self.inited = True
        self.optim.init((left_compress_size, right_compress_size))
        if self.offloading_device.__str__() == self.compute_device.__str__():
            return
        self.left_compress_size = left_compress_size
        if self.split_comm:
            self.pinned_buffer_A = self._new_pinned_buffer_if_needed((left_compress_size, self.intermidiate_size), self.pinned_buffer_A)
            self.pinned_buffer_B = self._new_pinned_buffer_if_needed((self.intermidiate_size, right_compress_size), self.pinned_buffer_B)
        self.pinned_buffer_grad = self._new_pinned_buffer_if_needed((left_compress_size, right_compress_size), self.pinned_buffer_grad)
        

    def to(self, device):
        if not self.inited: return
        self.current_device = device
        if device == 'cpu':                
            if not hasattr(self, "m_buf"):
                self.m_buf = torch.empty_like(self.optim.m, pin_memory=True, device = 'cpu')
                self.v_buf = torch.empty_like(self.optim.v, pin_memory=True, device = 'cpu')
            self.m_buf.copy_(self.optim.m)
            self.v_buf.copy_(self.optim.v)
            del self.optim.m 
            del self.optim.v
        else: 
            assert hasattr(self, "m_buf")
            self.optim.m = self.m_buf.to(device)
            self.optim.v = self.v_buf.to(device)
            
    # this function is only called during training
    def forward(self, input): # A: B x SL x HS .view(-1, input.size(-1)).t(), 
        # print('fwd called!', self.use_grad_ckpt, self.first_FWD, self.idx)
        # print(f'max memory usage before fwd layer {self.idx} {self.first_FWD}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
        if self.use_grad_ckpt:
            self.first_FWD = not self.first_FWD
        # print('forward of ', self.idx, self.first_FWD)
        with torch.no_grad():
            if self.global_state['state'] == State.RUNNING_PROFILE:
                if not self.use_grad_ckpt or not self.first_FWD: # we keep the second 
                    self.A = input.view(-1, input.size(-1)).t()
                
                return
            assert self.inited
            
            if self.offloading_device.__str__() == self.compute_device.__str__():
                self.A = self.gradient_compressor.left_compress(self, input.view(-1, input.size(-1)).t())
                
                return
                        
            if (not self.use_grad_ckpt) or self.first_FWD:
                if hasattr(self, 'fwd_hooks'):
                    # print('fwd_hookss of ', self.ctx.idx, 'len: ', len(self.fwd_hooks), self.use_grad_ckpt, self.first_FWD)
                    for hook in self.fwd_hooks:
                        hook()
                    self.fwd_hooks.clear()

            if self.split_comm and ((not self.use_grad_ckpt) or not self.first_FWD):
                A = self.gradient_compressor.left_compress(self, input.view(-1, input.size(-1)).t())
                func = _d2h_comm(self.ctx, self.intermidiate_compressor.left_compress(A).to(self.optim_dtype), self.pinned_buffer_A, self.global_state['state'])
                self.scheduler.schedule_fwd(self, func)
            elif (not self.split_comm) and ((not self.use_grad_ckpt) or not self.first_FWD): 
                self.A = self.gradient_compressor.left_compress(self, input.view(-1, input.size(-1)).t())
            
            if (not self.use_grad_ckpt) or self.first_FWD:
                if hasattr(self, 'fwd_end_hooks'):
                    # print('fwd_end_hooks of ', self.ctx.idx, 'len: ', len(self.fwd_end_hooks))
                    for hook in self.fwd_end_hooks:
                        hook()
                    self.fwd_end_hooks.clear()
        # print(f'max memory usage after fwd layer {self.idx} {self.first_FWD}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
            
    def backward(self, B):
        # print(f'max memory usage before bwd layer {self.idx}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
        # print('bwd called',  self.idx, self.global_state['state'])
        # print('peak_mem_usage', torch.cuda.memory_stats('cuda:0')['allocated_bytes.all.peak'] / 1024 / 1024)
        with torch.no_grad():
            state = self.global_state['state']
            if state == State.RUNNING_PROFILE:
                grad = self.A @ B
                self.A = None
                if state == State.RUNNING_PROFILE:
                    if self._grad is not None:
                        self._grad.add_(grad.to(self._grad.dtype).to(self._grad.device))
                    else: 
                        self._grad = grad.detach().float().cpu()
                del grad
                return
            
            assert self.inited
            
            if self.offloading_device.__str__() == self.compute_device.__str__():
                assert not self.split_comm
                grad = (self.A @ self.gradient_compressor.right_compress(self, B)).to(self.optim_dtype)
                self.optim.step(grad)
                self.weight.add_(self.gradient_compressor.decompress(self, grad.to(self.weight.dtype)))
                self.A = None 
                return
            
            if hasattr(self, 'bwd_hooks'):
                # print('bwd_hooks of ', self.ctx.idx, 'len: ', len(self.bwd_hooks))
                for hook in self.bwd_hooks:
                    hook()
                self.bwd_hooks.clear()
            B = self.gradient_compressor.right_compress(self, B)
            if state == State.PROFILE_COMPUTE:
                # print('self.A:', self.A.size(), 'B:', B.size())
                self.gradient_compressor.decompressor(self)((self.A @ B).to(self.optim_dtype).to(self.weight.dtype))
                return
            # self.A = None
            if self.split_comm:
                d2h_comm_func = _d2h_comm(self.ctx, self.intermidiate_compressor.right_compress(B), self.pinned_buffer_B, state, self.optim_dtype)
                process_func = _process_split(self.ctx, self.optim, self.optim_dtype, self.pinned_buffer_grad, state, )
            else:
                d2h_comm_func = _d2h_comm(self.ctx, (self.A@B).to(self.optim_dtype), self.pinned_buffer_grad, state)
                self.A = None
                process_func = _process(self.ctx, self.optim, self.optim_dtype, state)
            h2d_comm_func = _h2d_comm(self.ctx, self.pinned_buffer_grad, self.weight.device, state)
            update_func = _update(self.ctx, self.weight, self.gradient_compressor.decompressor(self), state)
            self.scheduler.schedule_bwd(self, d2h_comm_func, process_func, h2d_comm_func, update_func)
            
            if hasattr(self, 'bwd_end_hooks'):
                # print('bwd_end_hooks of ', self.ctx.idx, 'len: ', len(self.bwd_end_hooks))
                for hook in self.bwd_end_hooks:
                    hook()
                self.bwd_end_hooks.clear()
        # print(f'max memory usage after bwd layer {self.idx}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
        
class Scheduler:
    def __init__(self, 
                 modules: List[ParallelOffloadAdamMMOptim],
                 fcfs_point: int = 0,
                fcfs_process_delay: int = 0,
                fcfs_h2d_delay: int = 0,
                fcfs_upd_delay: int = 0,
                lcfs_h2d_delay: int = 0,
                lcfs_process_delay: int = 0,
                lcfs_d2h_delay: int = 0,
                 zero: bool = False,
                 verbose = False):
        self.modules = modules
        Context.init()

        for i, module in enumerate(modules):
            module.scheduler = self
            module.fwd_hooks = []
            module.fwd_end_hooks = []
            module.idx = i
            module.ctx = Context()
            module.ctx.idx = i
            module.bwd_hooks = []
            module.bwd_end_hooks = []
        self.fcfs_point = fcfs_point
        self.fcfs_process_delay = fcfs_process_delay
        self.fcfs_h2d_delay = fcfs_h2d_delay
        self.fcfs_upd_delay = fcfs_upd_delay
        self.lcfs_h2d_delay = lcfs_h2d_delay
        self.lcfs_process_delay = lcfs_process_delay
        self.lcfs_d2h_delay = lcfs_d2h_delay
        self.zero = zero
        self.verbose = verbose
        assert self.fcfs_process_delay <= self.fcfs_h2d_delay
        assert self.lcfs_h2d_delay <= self.lcfs_process_delay and self.lcfs_h2d_delay <= self.lcfs_d2h_delay
        
    def __str__(self):
        print(f'Scheduler: modules: {len(self.modules)}', end = ' ')
        if self.zero:
            print('zero')
        return f'fcfs_point: {self.fcfs_point}, fcfs_process_delay: {self.fcfs_process_delay}, fcfs_h2d_delay: {self.fcfs_h2d_delay}, lcfs_h2d_delay: {self.lcfs_h2d_delay}, lcfs_process_delay: {self.lcfs_process_delay}, lcfs_d2h_delay: {self.lcfs_d2h_delay}'

    def schedule_fwd(self, module: ParallelOffloadAdamMMOptim, func):
        module.fwd_end_hooks.append(func)
    
    def clear(self):
        for module in self.modules:
            module.fwd_hooks.clear()
            module.fwd_end_hooks.clear()
            module.bwd_hooks.clear()
            module.bwd_end_hooks.clear()
    
    def schedule_bwd(self, module: ParallelOffloadAdamMMOptim, d2h_comm_func, process_func, h2d_comm_func, update_func):
        if self.zero:
            # all schedules happen at the end of the backward pass
            module.bwd_end_hooks.append(d2h_comm_func)
            # self.modules[0].bwd_end_hooks.append(process_func)
            # self.modules[0].bwd_end_hooks.append(h2d_comm_func)
            # self.modules[0].bwd_end_hooks.append(update_func)
            self.modules[0].fwd_hooks.insert(0, update_func)
            self.modules[0].fwd_hooks.insert(0, h2d_comm_func)
            self.modules[0].fwd_hooks.insert(0, process_func)
            return

        def _schedule_fcfs(delay, funcs, use_end_hook: bool = True):
            func_names = '_'.join([f.__qualname__ for f in funcs]) if isinstance(funcs, list) else funcs.__qualname__
            if delay <= module.idx:
                module_for_sch = self.modules[module.idx - delay]
                hooks = module_for_sch.bwd_end_hooks if use_end_hook else module_for_sch.bwd_hooks
                if self.verbose:
                    print(f'FCFS: {func_names} of {module.idx} to bwd_end_hooks of {module.idx - delay}')
            else: 
                module_for_sch = self.modules[min(delay - module.idx, module.idx)]
                hooks = module_for_sch.fwd_hooks
                if self.verbose:
                    print(f'FCFS: {func_names} of {module.idx} to fwd_hooks of {min(delay - module.idx, module.idx)}')
            if isinstance(funcs, list):
                hooks.extend(funcs)
            else: hooks.append(funcs)
            return
        def _schedule_lcfs(delay, funcs):
            func_names = '_'.join([f.__qualname__ for f in funcs]) if isinstance(funcs, list) else funcs.__qualname__
            if delay <= module.idx:
                hooks = self.modules[module.idx - delay].fwd_hooks
                if self.verbose:
                    print(f'LCFS: {func_names} of {module.idx} to fwd_hooks of {module.idx - delay}')
            else: 
                hooks = self.modules[min(module.idx, delay - module.idx)].bwd_end_hooks
                if self.verbose:
                    print(f'LCFS: {func_names} of {module.idx} to bwd_end_hooks of {min(module.idx, delay - module.idx)}')
            if isinstance(funcs, list):
                for func in funcs[::-1]:
                    hooks.insert(0, func)
            else: hooks.insert(0, funcs)
            return
            
        if module.idx >= self.fcfs_point:
            # use first come first serve
            module.bwd_end_hooks.append(d2h_comm_func)
            _schedule_fcfs(self.fcfs_process_delay, process_func, True)
            _schedule_fcfs(self.fcfs_h2d_delay, h2d_comm_func, True)
            _schedule_fcfs(self.fcfs_upd_delay, update_func, True)
        else: 
            # use last come first serve
            _schedule_lcfs(self.lcfs_h2d_delay, [h2d_comm_func, update_func])
            _schedule_lcfs(self.lcfs_process_delay, process_func)
            _schedule_lcfs(self.lcfs_d2h_delay, [d2h_comm_func])
        
        # if module.idx >= self.process_delay:
        #     # print(f'schedule process_func {module.idx} to bwd_end_hooks of {module.idx - self.process_delay}')
        #     self.modules[module.idx - self.process_delay].bwd_end_hooks.append(process_func)
        # else: 
        #     idx = min(self.process_delay - module.idx, module.idx)
        #     # print(f'schedule process_func {module.idx} to fwd_hooks of {idx}')
        #     self.modules[idx].fwd_hooks.append(process_func)
        # if not self.comm_reverse:
        #     if module.idx >= self.h2d_comm_delay:
        #         # print(f'schedule h2d_comm_func {module.idx} to bwd_end_hooks of {module.idx - self.h2d_comm_delay}')
        #         self.modules[module.idx - self.h2d_comm_delay].bwd_end_hooks.append(h2d_comm_func)
        #     else: 
        #         idx = min(self.h2d_comm_delay - module.idx, module.idx)
        #         # print(f'schedule h2d_comm_func {module.idx} to fwd_hooks of {idx}')
        #         self.modules[idx].fwd_hooks.append(h2d_comm_func)
        #         self.modules[idx].fwd_hooks.append(update_func)
        # else:
        #     idx = min(max(module.idx - self.h2d_comm_delay, self.process_delay - module.idx), module.idx)
        #     self.modules[idx].fwd_hooks.append(h2d_comm_func)
        #     # print(f'COMM REVERSE schedule h2d_comm_func {module.idx} to fwd_hooks of {idx}')
        #     self.modules[idx].fwd_hooks.append(update_func)
    
    def report(self):
        # for module in self.modules:
        #     print(f'module {module.ctx.idx}:')
        #     print('upload_comm:', module.ctx.upload_comm)
        #     print('offload_comm:', module.ctx.offload_comm)
        #     print('upload_wait:', module.ctx.upload_wait)
        #     print('offload_wait:', module.ctx.offload_wait)
            
        upload_amount = np.sum([np.mean([comm[1] for comm in module.ctx.upload_comm]) for module in self.modules]) / 1024 / 1024
        upload_time = np.sum([np.mean([comm[0] for comm in module.ctx.upload_comm]) for module in self.modules])
        offload_amount = np.sum([np.mean([comm[1] for comm in module.ctx.offload_comm]) for module in self.modules]) / 1024 / 1024
        offload_time = np.sum([np.mean([comm[0] for comm in module.ctx.offload_comm]) for module in self.modules])
        upload_wait = np.sum([np.mean(module.ctx.upload_wait) for module in self.modules], axis = 0)
        offload_wait = np.sum([np.mean(module.ctx.offload_wait) for module in self.modules], axis = 0)
        compute_time = np.sum([np.mean(module.ctx.compute_time) for module in self.modules], axis = 0)
        print(f'upload_amount: {upload_amount} MB')
        print(f'upload_time: {upload_time:.4f}')
        print(f'offload_amount: {offload_amount} MB')
        print(f'offload_time: {offload_time:.4f}')
        print(f'upload_wait: {np.mean(upload_wait):.4f} +- {np.std(upload_wait):.4f}')
        print(f'offload_wait: {np.mean(offload_wait):.4f} +- {np.std(offload_wait):.4f}')
        print(f'compute_time: {np.mean(compute_time):.4f} +- {np.std(compute_time):.4f}')
        
        return {'upload_amount': upload_amount, 'upload_time': upload_time, 'offload_amount': offload_amount, 'offload_time': offload_time, 'upload_wait': np.mean(upload_wait), 'offload_wait': np.mean(offload_wait), 'compute_time': np.mean(compute_time)}

class ScheduleEnv:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __enter__(self):
        return self.scheduler
    
    def __exit__(self, *args): 
        self.scheduler.clear()

class Adam:
    def __init__(self,
                lr = 1e-3,
                beta = 0.9,
                gamma = 0.999,
                eps = 1e-4,
                device = 'cuda',
                dtype: torch.dtype = torch.float32):
        # self.v = torch.zeros_like(param, device = device, dtype = dtype).contiguous()
        # self.m = torch.zeros_like(param, device = device, dtype = dtype).contiguous()
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.t = 0
        self.dtype = dtype
        self.device = device

    def init(self, shape):
        self.m = torch.zeros(shape, device = self.device, dtype = self.dtype).contiguous()
        self.v = torch.zeros(shape, device = self.device, dtype = self.dtype).contiguous()
    
    def step(self, grad):
        self.t += 1
        assert grad.dtype == self.dtype
        with torch.no_grad():
            # copied = grad.clone()
            if torch.any(torch.isnan(grad)):
                print('nan detected')
                print('grad:', grad.min().item(), grad.max().item(), self.beta, self.gamma, self.eps, self.t)
            fused_adam.fused_adam(grad, self.m, self.v, self.beta, self.gamma, self.lr, self.eps, self.t)
            
            # if torch.any(torch.isnan(grad)):
            #     print('nan detected after update')
            #     for t, name in zip([copied, self.m, self.v], ['copied', 'm', 'v']):
            #         print(name, t.min().item(), t.max().item())
            #     print('beta, gamma, eps, t', self.beta, self.gamma, self.eps, self.t)
            #     exit(0)
            # del copied
        return grad

from transformers.models.bert.modeling_bert import BertSelfOutput, BertOutput, BertIntermediate
from transformers.models.roberta.modeling_roberta import RobertaSelfOutput, RobertaOutput, RobertaIntermediate
from lora_modules import LoRALayer

def replace_linear_layer_with_parallel_linear(model, 
                        modules,
                        use_lora, 
                        lora_rank,
                        int_compress_method = None,
                        int_compress_rank = None,
                        offload = None,
                        lr = None,
                        optim_dtype = torch.float32,
                        use_grad_ckpt = False,
                        global_state: Dict = None
    ):
    new_modules = []
    def replace(module, tag = ''):
        for name, child in module.named_children():
            if (isinstance(child, nn.Linear) or isinstance(child, transformers.pytorch_utils.Conv1D)) and\
                name != 'lm_head':
                if (modules is None or (('attn.qkv' in modules and name in ('q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value', 'c_attn')) or\
                    ('attn.o' in modules and name in ('o_proj')) or \
                    ('attn.o' in modules and name == 'dense' and isinstance(module, BertSelfOutput)) or\
                    ('attn.o' in modules and name == 'dense' and isinstance(module, RobertaSelfOutput)) or\
                    ('mlp.gate' in modules and name in ('gate_proj')) or\
                    ('mlp.up' in modules and name in ('up_proj', 'c_fc')) or\
                    ('mlp.up' in modules and name == 'dense' and isinstance(module, BertIntermediate)) or\
                    ('mlp.up' in modules and name == 'dense' and isinstance(module, RobertaIntermediate)) or\
                    ('mlp.down' in modules and name in ('down_proj', 'c_proj')) or\
                    ('mlp.down' in modules and name == 'dense' and isinstance(module, BertOutput))or\
                    ('mlp.down' in modules and name == 'dense' and isinstance(module, RobertaOutput)))):
                    if use_lora: 
                        new_module = LoRALayer(child, lora_rank)
                    else:
                        new_module = LoRaOffloadLinear(child,
                        lr = lr,
                        offloading_device = offload,
                        optim_dtype=optim_dtype,
                        int_compress_tag = int_compress_method,
                        intermidiate_size = int_compress_rank,
                        use_grad_ckpt=use_grad_ckpt,
                        global_state = global_state)
                        new_modules.append(new_module.optim)
                    setattr(module, name, new_module)
            else:
                replace(child, tag + name + '.')
    replace(model)
    return new_modules

if __name__ == '__main__':
    import argparse 
    import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--sch-fcfs-point', type=int, default=0)
    parser.add_argument('--sch-fcfs-process-delay', type=int, default=0)
    parser.add_argument('--sch-fcfs-h2d-delay', type=int, default=0)
    parser.add_argument('--sch-fcfs-upd-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-h2d-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-process-delay', type=int, default=0)
    parser.add_argument('--sch-lcfs-d2h-delay', type=int, default=0)
    parser.add_argument('--sch-verbose', action = 'store_true')
    parser.add_argument('--modules', choices = ['attn.qkv', 'attn.o', 'mlp.gate', 'mlp.up', 'mlp.down'], default = None, nargs = '+')
    parser.add_argument('--compress', action = 'store_true')
    parser.add_argument('--compress-size', type=int, default=None) 
    parser.add_argument('--nnz', type=int, default=None) 
    parser.add_argument('--int-compress', choices = list(IntCompressorDispatcher.keys()), type=str, default = None)
    parser.add_argument('--int-size', type=int, default=None)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--n_repeat', type=int, default=10)
    parser.add_argument('--gradient-checkpointing', action = 'store_true')
    parser.add_argument('--offload', type = str, choices = ['cpu', 'cuda:0', 'cuda:1'], default = None)
    parser.add_argument('--truncate', type = int, default = None)
    parser.add_argument('--zero', action = 'store_true')
    parser.add_argument('--optim-dtype', type = str, choices = ['float32', 'float16', 'bfloat16'], default = 'float32')
    args = parser.parse_args()
    if args.dtype == 'float32':
        args.dtype = torch.float32
    elif args.dtype == 'float16':
        args.dtype = torch.float16
    elif args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16
    
    if args.optim_dtype == 'float32':
        args.optim_dtype = torch.float32
    elif args.optim_dtype == 'float16':
        args.optim_dtype = torch.float16
    elif args.optim_dtype == 'bfloat16':
        args.optim_dtype = torch.bfloat16
        
    torch.manual_seed(1)
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype = args.dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.truncate is not None:
        if isinstance(model, transformers.GPT2LMHeadModel):
            model.transformer.h.__delitem__(slice(args.truncate, None))
        elif isinstance(model, transformers.LlamaForCausalLM):
            model.model.layers.__delitem__(slice(args.truncate, None))
        else: raise NotImplementedError
            
    # t = time.perf_counter()
    # optim = torch.optim.SGD(model.parameters(), lr = 1e-3)
    # for i in tqdm.tqdm(range(args.n_repeat)):
    #     x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
    #     y = model(x, labels = x)
    #     err = y.loss
    #     optim.zero_grad()
    #     err.backward()
    #     optim.step()
    # torch.cuda.synchronize()
    # print('average time: ', (time.perf_counter() - t) / args.n_repeat, 'error: ', err.item())
    
    n_params = sum([p.numel() for p in model.parameters()])
    print('number of parameters: ', n_params)
    print('model', model)
    
    global_state = {'state': State.RUNNING}
    modules = replace_linear_layer_with_parallel_linear(model,
        modules = args.modules,
        use_lora = False,
        lora_rank= None,
        offload = args.offload,
        int_compress_method=args.int_compress,
        int_compress_rank=args.int_size,
        use_grad_ckpt=args.gradient_checkpointing,
        global_state = global_state,
        lr = args.lr,
        optim_dtype = args.optim_dtype)
    print(model)
    model.train()
    from gradient_compressor import LearnedCountSketchCompressor, GradientCompressor
    if args.compress:
        gradient_compressor = LearnedCountSketchCompressor(sketch_size= args.compress_size, 
                                                           n_unempty = args.nnz)
    else: gradient_compressor = GradientCompressor()
    gradient_compressor.fit(modules)
    for module in modules: 
        module.gradient_compressor = gradient_compressor
    scheduler = Scheduler(modules,
        fcfs_point = args.sch_fcfs_point,
        fcfs_process_delay = args.sch_fcfs_process_delay,
        fcfs_h2d_delay = args.sch_fcfs_h2d_delay,
        fcfs_upd_delay= args.sch_fcfs_upd_delay,
        lcfs_h2d_delay = args.sch_lcfs_h2d_delay,
        lcfs_process_delay = args.sch_lcfs_process_delay,
        lcfs_d2h_delay = args.sch_lcfs_d2h_delay,
        zero = args.zero,
        verbose = args.sch_verbose)
    print(scheduler)
    
    global_state['state'] = State.RUNNING
    for i in tqdm.tqdm(range(5), desc = 'warmup'):
        x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
        y = model(x, labels = x)
        err = y.loss
        err.backward()
    struct = tqdm.tqdm(range(args.n_repeat), desc = 'struct')
    acc_loss = None
    start = time.perf_counter()
    for i in struct:
        x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
        y = model(x, labels = x)
        err = y.loss
        acc_loss = err.item() if acc_loss is None else 0.9 * acc_loss + 0.1 * err.item()
        struct.set_description(f'loss: {acc_loss}')
        err.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print('average running time: ', elapsed / args.n_repeat)
    
    global_state['state'] = State.PROFILE_COMPUTE
    start = time.perf_counter()
    acc_loss = None
    struct = tqdm.tqdm(range(args.n_repeat), desc = 'struct')
    for i in struct:
        x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
        y = model(x, labels = x)
        # print(f'max memory usage after fwd {i}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
        err = y.loss
        acc_loss = err.item() if acc_loss is None else 0.9 * acc_loss + 0.1 * err.item()
        struct.set_description(f'loss: {acc_loss}')
        err.backward()
        # print(f'max memory usage after bwd {i}: ', torch.cuda.max_memory_allocated() / 1024 / 1024)
    torch.cuda.synchronize()
    print('average compute time: ', (time.perf_counter() - start) / args.n_repeat)
    
    
    global_state['state'] = State.PROFILE
    acc_loss = None
    from utils import NVTXContext
    warmup_iter = 2
    end_iter = 3
    for i in range(5):
        if i == warmup_iter:
            torch.cuda.cudart().cudaProfilerStart()
            PROFILE = True
        x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
        with NVTXContext(f'FWD {i}', i >= warmup_iter and i <= end_iter):
            y = model(x, labels = x)
        with NVTXContext(f'BWD {i}', i >= warmup_iter and i <= end_iter):
            err = y.loss
            err.backward()
        if i == end_iter:
            PROFILE = False
            torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    
    print('max memory usage: ', torch.cuda.max_memory_allocated() / 1024 / 1024)

    # end2end_times = {}
    # # for state, tag in zip([State.PROFILE, State.PROFILE_COMPUTE, State.PROFILE_COMM, ], ['Tot', 'GPU_Compute', 'Comm']):
    # for state, tag in zip([State.PROFILE], ['Tot']):
    #     global_state['state'] = state
    #     if state == State.PROFILE_COMM: gradient_compressor.init_profile()
    #     t = time.perf_counter()
    #     for i in tqdm.tqdm(range(args.n_repeat), desc = tag):
    #         x = torch.randint(0, tokenizer.vocab_size, (args.bs, args.seq_len)).cuda()
    #         y = model(x, labels = x)
    #         err = y.loss
    #         err.backward()
    #     torch.cuda.synchronize()
    #     end2end_times[tag] = (time.perf_counter() - t) / args.n_repeat
        
    # print('compress:', args.compress, 'bs', args.bs)
    # scheduler.report()
    # gradient_compressor.report_and_end_profile(args.n_repeat)
    # for k, v in end2end_times.items():
    #     print(k, v)
    