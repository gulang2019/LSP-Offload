import torch
from compress import *
import torch.nn as nn
from typing import Union
import transformers
import time
import torch.optim.adam
import random 
import math
import kernels.fused_adam as fused_adam
random.seed(0)

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
                 compress_method: str,
                 rank: int, 
                 lr: float = 1e-3,
                 offloading_device = 'cpu',
                 delta_compress_method: str = None,
                 delta_compress_rank: int = None, 
                 global_state: dict = dict(),
                 tag: str = None,
                 optim_dtype: torch.dtype = torch.float32):
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
        self.compress_method = compress_method
        self.rank = rank
        self.offloading_device = offloading_device
        self.delta_compress_method = delta_compress_method
        self.delta_compress_rank = delta_compress_rank
        self.tag = tag
        self.global_state = global_state
        self.optim = OffloadAdamMMOptim(self.weight,
                                        compress_method,
                                        rank,
                                        lr,
                                        offloading_device,
                                        delta_compress_method,
                                        delta_compress_rank,
                                        global_state,
                                        tag,
                                        optim_dtype,
                                        gradient_checkpointing=self.global_state.get('gradient_checkpointing', False))

    def forward(self, input, *args, **kwargs):
        ret = LoRaOffloadMatmulFunction.apply(input, self.weight, self.optim if self.training else None)
        if self.bias is not None:
            ret += self.bias
        return ret

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, "
                f"compress_method={self.compress_method}, offloading_device={self.offloading_device}, "
                f"delta_compress_method={self.delta_compress_method})")

class Adam:
    def __init__(self,
                param,
                lr = 1e-3,
                beta = 0.9,
                gamma = 0.999,
                eps = 1e-4,
                device = 'cuda',
                dtype: torch.dtype = torch.float32):
        self.v = torch.zeros_like(param, device = device, dtype = dtype).contiguous()
        self.m = torch.zeros_like(param, device = device, dtype = dtype).contiguous()
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.t = 0
        self.dtype = dtype
    def step(self, grad):
        self.t += 1
        assert grad.dtype == self.dtype
        with torch.no_grad():
            fused_adam.fused_adam(grad, self.m, self.v, self.beta, self.gamma, self.lr, self.eps, self.t)
            # self.m = self.beta * self.m + (1 - self.beta) * grad
            # self.v = self.gamma * self.v + (1 - self.gamma) * grad * grad
            # m_hat = self.m / (1 - self.beta**self.t)
            # v_hat = self.v / (1 - self.gamma**self.t)
            # grad = - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return grad

from compress import Compressor, LoraPower, ImpSampling, TopK, LoraFirst, CountSketch

class OffloadAdamMMOptim:
    def __init__(self, 
                 weight: torch.Tensor, 
                 compress_method: str, 
                 rank: int, 
                 lr: float, 
                 offloading_device = 'cpu',
                 delta_compress_method: str = None,
                 delta_compress_rank: int = None, 
                 global_state = dict(),
                 tag: str = None, 
                 optim_dtype: torch.dtype = torch.float32,
                 gradient_checkpointing: bool = False):
        self.weight = weight
        if compress_method == 'power':
            self.compressor = LoraPower(rank, offloading_device, offloading_dtype = optim_dtype)
        elif compress_method == 'imp_sampling':
            self.compressor = ImpSampling(rank, offloading_device, offloading_dtype = optim_dtype)
        elif compress_method == 'CountSketch':
            self.compressor = CountSketch(rank, offloading_device, offloading_dtype = optim_dtype)
        elif compress_method == 'topk':
            self.compressor = TopK(rank, offloading_device, offloading_dtype = optim_dtype)
        elif compress_method == 'lora_first':
            self.compressor = LoraFirst(rank, offloading_device, offloading_dtype = optim_dtype)
        elif compress_method == 'drop':
            self.compressor = Compressor(offloading_device, offloading_dtype = optim_dtype, split_comm = True)
        elif compress_method is None:
            self.compressor = Compressor(offloading_device, offloading_dtype = optim_dtype, split_comm = False)
        else: raise NotImplementedError(f'compress_method: {compress_method} is not implemented')
        self.optim_dtype = optim_dtype
        self.rank = rank
        self.optim = Adam(self.weight, lr, device = offloading_device, dtype = optim_dtype)
        self.offloading_device = offloading_device
        self.delta_compress_method = delta_compress_method
        if delta_compress_method == 'power':
            self.delta_compress_impl = LoraPower.compress_and_comm
        elif delta_compress_method == 'topk':
            self.delta_compress_impl = TopK.compress_and_comm
        elif delta_compress_method is None: # no approximation
            self.delta_compress_impl = lambda x, _, device, dtype: (x.to(device).to(dtype), x.numel() * x.element_size())
        else: raise NotImplementedError(f'delta_compress_method: {delta_compress_method} is not implemented')
        self.delta_compress_rank = 32 if delta_compress_rank is None else delta_compress_rank
        self.global_state = global_state
        self.global_state['offload_modules'].append(self)
        self.tag = tag
        self.gradient_checkpointing = gradient_checkpointing
        
        self.A = None
        
        # for selecting subspace
        self._grad = None
            
    # this function is only called during training
    def forward(self, input): # A: B x SL x HS .view(-1, input.size(-1)).t(), 
        '''
        following behaviors may happen:
        1. send the compressed input to the offloading device;
        2. store the compressed input for update;
        3. do nothing; 
        '''
        skip_compression = self.global_state.get('state', None) == 'profile' or\
                        self.global_state.get('ProfileState', False) == 'Compute'
        do_profile = self.global_state.get('ProfileState', False) == 'Offload'
        with torch.no_grad():
            # for split communication with gradient checkpointing, we only call the forward function once
            comm_amount = 0
            if (self.gradient_checkpointing and not input.requires_grad and (self.compressor.split_comm and not skip_compression)) or \
                (self.gradient_checkpointing and input.requires_grad and (not self.compressor.split_comm or skip_compression)) or\
                (not self.gradient_checkpointing):
                if do_profile:
                    torch.cuda.synchronize(device = self.weight.device)
                    time_start = time.time()
                A = input.view(-1, input.size(-1)).t()
                if not skip_compression and not self.compressor.split_comm:
                    comm_amount = self.compressor.forward(self.gradient_compressor.left_compress(self, A))
                if not self.compressor.split_comm or skip_compression: # store the compressed input for update
                    self.A = A
                if self.global_state.get('ProfileState', False) == 'Compute':
                    torch.cuda.synchronize()
                    self.global_state['FWDTimeStamp'].append((self.tag, time.time()))
                if do_profile:
                    torch.cuda.synchronize()
                    elapsed = time.time() - time_start
                    self.global_state['OffloadTimeStamp'][self.tag] = elapsed
                    self.global_state['OffloadTime'] += elapsed
                    self.global_state['OffloadComm'] += comm_amount

    def backward(self, B):
        '''
        following behaviors may happen:
        1. send the compressed grad_output to the offloading device; 
        2. do low rank approximation on the gradient and send them to the offloading device; (input, grad_output)
        '''
        if self.global_state.get('state', None) == 'profile':
            grad = self.A @ B
            self.A = None
            if self._grad is not None:
                self._grad.add_(grad.to(self._grad.device).to(self._grad.dtype))
            else: self._grad = grad.detach().cpu().float()
            return
        profile_state = self.global_state.get('ProfileState', False)
        if profile_state == 'Compute':
            grad = self.A @ B
            self.A = None
            torch.cuda.synchronize()
            self.global_state['BWDTimeStamp'].append((self.tag, time.time()))
            return 
        elif profile_state == 'Offload':
            torch.cuda.synchronize(device = self.weight.device)
            time_start = time.time()
        with torch.no_grad():
            grad, comm_amount = self.compressor.backward(self.gradient_compressor.left_compress(self, self.A) if self.A is not None else None,\
                self.gradient_compressor.right_compress(self, B))
            self.A = None
            if profile_state == 'Offload':
                self.global_state['OffloadComm'] += comm_amount
                torch.cuda.synchronize()
                elapsed = time.time() - time_start
                self.global_state['OffloadTime'] += elapsed
                self.global_state['OffloadTimeStamp'][self.tag] += elapsed
                time_start = time.time()
            if 'cuda' in self.offloading_device:
                torch.cuda.set_device(self.offloading_device)
            grad = self.optim.step(grad)
            if profile_state == 'Offload':
                self.global_state['NumParams'] += grad.numel()
                torch.cuda.synchronize()
                elapsed = time.time() - time_start
                self.global_state['OptimTime'] += elapsed
                self.global_state['OptimTimeStamp'][self.tag] = elapsed
                time_start = time.time()
            
            assert str(grad.device) == self.offloading_device
            if self.weight.device != grad.device:
                grad, comm_amount = self.delta_compress_impl(grad, self.delta_compress_rank, self.weight.device, self.weight.dtype)
            self.weight += self.gradient_compressor.decompress(self, grad)
            if profile_state == 'Offload':
                torch.cuda.synchronize(device = self.weight.device)
                elapsed = time.time() - time_start
                self.global_state['UploadTime'] += elapsed 
                self.global_state['UploadComm'] += comm_amount
                self.global_state['UploadTimeStamp'][self.tag] = elapsed
      
class TestModule(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 hdim = 768, 
                 compress_method = 'power',
                 rank = 2,
                 lr = 1e-3,
                 offloading_device = 'cpu',
                 delta_compress_method = 'power'):
        super(TestModule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hdim = hdim
        self.fc1 = LoRaOffloadLinear(nn.Linear(in_features, hdim), compress_method, rank, lr, offloading_device, delta_compress_method)
        self.relu = nn.ReLU()
        self.fc2 = LoRaOffloadLinear(nn.Linear(hdim, out_features), compress_method, rank, lr, offloading_device, delta_compress_method) 
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
def test():
    torch.random.manual_seed(0)
    test_x = torch.randn(32, 768).to('cuda')
    w = torch.randn(768, 1).to('cuda')
    for compress_method in [None, 'power', 'imp_sampling', 'topk', 'lora_first']:
        for delta_compress_method in [None, 'power', 'topk']:
            t = time.time()
            torch.manual_seed(0)
            module = TestModule(768, 1).cuda()
            module.to('cuda')
            for i in range(1000):
                x = torch.randn(32, 768).to('cuda')
                with torch.no_grad():
                    label = x @ w
                y = module(x)
                l = nn.MSELoss()(y, label)
                l.backward()
            with torch.no_grad():
                test_loss = nn.MSELoss()(module(test_x), (test_x @ w))
            print(f'compress_method: {compress_method}, delta_compress_method: {delta_compress_method}, test_loss: {test_loss}, time: {time.time() - t}')

if __name__ == '__main__':
    test()
    example_output = '''
compress_method: None, delta_compress_method: None, test_loss: 0.38337019085884094, time: 4.22867226600647
compress_method: None, delta_compress_method: power, test_loss: 0.38337019085884094, time: 3.1178665161132812
compress_method: None, delta_compress_method: topk, test_loss: 0.38337019085884094, time: 4.328294992446899
compress_method: power, delta_compress_method: None, test_loss: 0.38337019085884094, time: 4.133617639541626
compress_method: power, delta_compress_method: power, test_loss: 0.38337019085884094, time: 3.767718553543091
compress_method: power, delta_compress_method: topk, test_loss: 0.38337019085884094, time: 3.7440412044525146
compress_method: imp_sampling, delta_compress_method: None, test_loss: 0.38337019085884094, time: 3.5613739490509033
compress_method: imp_sampling, delta_compress_method: power, test_loss: 0.38337019085884094, time: 4.273885250091553
compress_method: imp_sampling, delta_compress_method: topk, test_loss: 0.38337019085884094, time: 4.3453528881073
compress_method: topk, delta_compress_method: None, test_loss: 0.38337019085884094, time: 4.5648353099823
compress_method: topk, delta_compress_method: power, test_loss: 0.38337019085884094, time: 4.268109560012817
compress_method: topk, delta_compress_method: topk, test_loss: 0.38337019085884094, time: 3.6876425743103027
'''