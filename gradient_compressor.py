from typing import List
import time
import torch
import tqdm
import copy
import math

class IntermidiateCompressor:
    def __init__(self, intermidiate_size):
        self.intermidiate_size = intermidiate_size
    
    def left_compress(self, tensor):
        return tensor

    def right_compress(self, tensor):
        return tensor

class ImpSamplingCompressor(IntermidiateCompressor):
    def __init__(self, intermidiate_size):
        super().__init__(intermidiate_size)
        
    def left_compress(self, tensor):
        d, n = tensor.size()
        leverage_scores = torch.norm(tensor, dim = 0) + 1e-6
        leverage_scores = leverage_scores / torch.sum(leverage_scores)
        indices = torch.multinomial(leverage_scores.float(), self.intermidiate_size, replacement = True)
        scales = 1 / torch.sqrt(leverage_scores[indices] * self.intermidiate_size) # (self.intermidiate_size,)
        self.sketch_matrix = torch.zeros((n, self.intermidiate_size), dtype = tensor.dtype, device = tensor.device)
        self.sketch_matrix[indices, torch.arange(self.intermidiate_size)] = scales
        return tensor @ self.sketch_matrix
    
    def right_compress(self, tensor):
        ret = self.sketch_matrix.t() @ tensor
        self.sketch_matrix = None
        return ret
    
class CountSketchCompressor(IntermidiateCompressor):
    def __init__(self, intermidiate_size):
        super().__init__(intermidiate_size)

    def left_compress(self, tensor):
        d, n = tensor.size()
        indices = torch.randint(0, self.intermidiate_size, (n,), dtype=torch.long, device = tensor.device)
        signs = 2 * torch.randint(0, 2, (n,), dtype=tensor.dtype, device = tensor.device) - 1
        self.sketch_matrix = torch.zeros((n, self.intermidiate_size), dtype = tensor.dtype, device = tensor.device)
        self.sketch_matrix[torch.arange(n), indices] = signs
        return tensor @ self.sketch_matrix

    def right_compress(self, tensor):
        ret = self.sketch_matrix.t() @ tensor
        self.sketch_matrix = None
        return ret
    
class LoraFirstCompressor(IntermidiateCompressor):
    def __init__(self, intermidiate_size):
        super().__init__(intermidiate_size)
        
    def left_compress(self, tensor):
        m, n = tensor.size()
        U = torch.randn(int(self.intermidiate_size), m, device = tensor.device, dtype = tensor.dtype) / math.sqrt(int(self.intermidiate_size))
        y = U @ tensor
        ortho = torch.qr(y.t().float())[0].to(tensor.dtype) # n x \hat{k}
        Aortho = tensor@ortho
        self.ortho = ortho.t()
        return Aortho
    
    def right_compress(self, tensor):
        ret = tensor @ self.ortho
        self.ortho = None
        return ret

IntCompressorDispatcher = {
    'ImpSampling': ImpSamplingCompressor,
    'CountSketch': CountSketchCompressor,
    'LoraFirst': LoraFirstCompressor,
    None: IntermidiateCompressor
}

class GradientCompressor:
    def __init__(self) -> None:
        self.doing_profile = False
        self.n_profile_data = 1
        
    def set_num_profile_data(self, n_profile_data):
        self.n_profile_data = n_profile_data

    def fit(self, modules):
        for module in modules:
            assert hasattr(module, 'weight')
            module.init(module.weight.size(0), module.weight.size(1))
        return
    
    def init_profile(self):
        self.left_compress_times = []
        self.right_compress_times = []
        self.decompress_times = []
        self.doing_profile = True
        
    def report_and_end_profile(self, n_repeat):
        left_compress_time = sum(self.left_compress_times) / n_repeat
        right_compress_time = sum(self.right_compress_times) / n_repeat
        decompress_time = sum(self.decompress_times) / n_repeat
        print(f'Left compress: {left_compress_time:.4f} s')
        print(f'Right compress: {right_compress_time:.4f} s')
        print(f'decompress_times: {decompress_time:.4f} s')
        self.doing_profile = False
    
    def need_profile(self):
        return False
        
    def left_compress(self, module, tensor):
        if self.doing_profile:
            torch.cuda.synchronize()
            t = time.perf_counter()
        ret = self.left_compress_impl(module, tensor)
        if self.doing_profile:
            torch.cuda.synchronize()
            self.left_compress_times.append(time.perf_counter() - t)
        return ret
    
    def right_compress(self, module, tensor):
        if self.doing_profile:
            torch.cuda.synchronize()
            t = time.perf_counter()
        ret = self.right_compress_impl(module, tensor)
        if self.doing_profile:
            torch.cuda.synchronize()
            self.right_compress_times.append(time.perf_counter() - t)
        return ret 
    
    def decompress(self, module, tensor):
        if self.doing_profile:
            torch.cuda.synchronize()
            t = time.perf_counter()
        ret = self.decompress_impl(module, tensor)
        if self.doing_profile:
            torch.cuda.synchronize()
            self.decompress_times.append(time.perf_counter() - t)
        return ret
    
    def left_compress_impl(self, module, tensor):
        return tensor
    
    def right_compress_impl(self, module, tensor):
        return tensor
    
    def decompress_impl(self, module, tensor):
        return tensor
    
    def decompressor(self, module):
        return lambda tensor: self.decompress(module, tensor)
    
class GaussianCompressor(GradientCompressor):
    def __init__(self, rank, update_freq = 100):
        super().__init__()
        self.rank = rank
        self.update_freq = update_freq
        self.counter = 0
        self.sketch_matrices = {}
    
    def need_profile(self):
        return False
    
    def fit(self, modules):
        for module in modules:
            module.init(self.rank, self.rank)
        return 
    
    def _update(self, module):
        m, n = module.weight.size()
        U = torch.randn((self.rank, m), device = module.weight.device, dtype = module.weight.dtype) / math.sqrt(self.rank)
        V = torch.randn((self.rank, n), device = module.weight.device, dtype = module.weight.dtype) / math.sqrt(self.rank)
        self.sketch_matrices[module] = (U, V)
    
    def left_compress_impl(self, module, tensor):
        if module not in self.sketch_matrices or\
            self.counter % self.update_freq == 0:
            self._update(module)
            self.counter += 1
        return self.sketch_matrices[module][0] @ tensor
     
    def right_compress_impl(self, module, tensor):
        return tensor @ self.sketch_matrices[module][1].t()
    
    def decompress_impl(self, module, tensor):
        U, V = self.sketch_matrices[module]
        return U.t() @ tensor @ V
        
class SVDGradientCompressor(GradientCompressor):
    def __init__(self, 
                 minimum_rank,
                 max_rank, 
                 energy_threshold: float = 0.99,
                 approx_svd: bool = False):
        super().__init__()
        self.minimum_rank = minimum_rank
        self.max_rank = max_rank
        self.energy_threshold = energy_threshold
        self.approx_svd = approx_svd

        self._Us = {}
        self._Vs = {}
    
    def need_profile(self):
        return True
        
    @staticmethod
    def _approx_svd(A: torch.Tensor, k):
        m, n = A.size()
        U = torch.randn(int(2*k), m, device = A.device, dtype = A.dtype) / math.sqrt(int(2*k))
        y = U @ A
        ortho = torch.linalg.qr(y.t().float())[0].to(A.dtype) # n x \hat{k}
        Aortho = A@ortho
        U, S, V = torch.svd(Aortho.float())
        return U, S, ortho.float() @ V

    def fit(self, modules):
        start = time.perf_counter()
        losses = []
        for module in tqdm.tqdm(modules, desc = 'SVD Compression'):
            grad = module._grad.cuda().to(module.weight.dtype)
            module._grad = None
            if self.approx_svd:
                U, S, V = self._approx_svd(grad, self.max_rank)
            else:
                U, S, V = torch.svd(grad.float())
            torch.cumsum(S, 0, out = S)
            S.div_(S[-1].item())
            rank = torch.argmax((S >= self.energy_threshold).to(torch.int32)).item() + 1
            rank = math.ceil(rank / 32) * 32
            rank = min(max(rank, self.minimum_rank), self.max_rank)
            _V = V[:, :rank].to(module.weight.dtype)
            _U = U[:, :rank].t().to(module.weight.dtype)
            
            U_old = self._Us.get(module, None)
            V_old = self._Vs.get(module, None)
            
            U_proj = ((_U @ U_old.t()) if U_old is not None else _U).to(module.offloading_device).to(module.optim_dtype)
            V_proj = ((V_old.t() @ _V) if V_old is not None else _V).to(module.offloading_device).to(module.optim_dtype)
            module.optim.m = U_proj @ module.optim.m @ V_proj
            module.optim.v = (U_proj ** 2) @ module.optim.v @ (V_proj ** 2)
            
            self._Us[module] = _U
            self._Vs[module] = _V
            
            module.init(rank, rank)
            
            losses.append(((_U.t() @ _U @ grad @ _V @ _V.t() - grad).norm() / grad.norm()).item())
            # print(f'rank: {rank}, weight: {module.weight.size()}, Compression ratio: {(1 - rank**2 / module.weight.numel()) * 100:.2f}%, Energy: {S[rank-1].item()*100:.2f}%')
        
        # report total memory 
        tot_elem = sum([u.numel() + v.numel() for u, v in zip(self._Us.values(), self._Vs.values())])
        tot_mem = sum([u.numel() * u.element_size() + v.numel() * v.element_size() for u, v in zip(self._Us.values(), self._Vs.values())])
        print(f'Total memory: {tot_mem / 1024 / 1024:.2f} MB, Tot elem: {tot_elem}, Average loss: {sum(losses) / len(losses):.4f}, Time: {time.perf_counter() - start:.2f}s')
        
    def left_compress_impl(self, module, tensor):
        return self._Us[module] @ tensor if module in self._Us else tensor
    
    def right_compress_impl(self, module, tensor):
        return tensor @ self._Vs[module] if module in self._Vs else tensor
    
    def decompress_impl(self, module, tensor):
        return self._Us[module].t() @ tensor @ self._Vs[module].t() if module in self._Us else tensor

class CountSketch:
    def __init__(self, sketch_size, compress_size, nnz_col, device, dtype, cs_init):
        self.indices = torch.randint(0, sketch_size, (nnz_col, compress_size), device=device)
        if cs_init == 'binary':
            self.values = torch.nn.Parameter((torch.randint(0, 1, (nnz_col, compress_size), dtype = dtype, device = device) * 2 - 1) / math.sqrt(nnz_col))
        elif cs_init == 'gaussian':
            self.values = torch.nn.Parameter(torch.randn((nnz_col, compress_size), dtype = dtype, device = device) / math.sqrt(nnz_col)) 
        else: 
            raise ValueError(f'Unknown initialization method {cs_init}')
        self.shape = (sketch_size, compress_size)
    
    def get_sketch(self, train = False):
        return torch.zeros(self.shape, dtype=self.values.dtype, device=self.values.device).scatter_(0, self.indices, self.values if train else self.values.data)    

    def n_storage(self):
        return self.indices.numel() * self.indices.element_size() + self.values.numel() * self.values.element_size()

class LearnedCountSketchCompressor(GradientCompressor):
    def __init__(self,
                 sketch_size=1024, 
                 n_unempty=8,
                 n_iter = 1000,
                 n_ft_iter = 100,
                 lr = 1e-1,
                 ft_lr = 1e-1,
                 reuse_sketch = False,
                 cs_init: str = 'binary',
                 thresh_hold = 0.2):
        super().__init__()
        self.sketch_size = sketch_size
        self.n_unempty = n_unempty
        self.sketches_map = {}
        self.common_fit_iter = n_iter
        self.finetune_iter = n_ft_iter
        self.lr = lr
        self.ft_lr = ft_lr
        self.reuse_sketch = reuse_sketch
        self.cs_init = cs_init
        assert self.cs_init in ('gaussian', 'binary')
        self.thresh_hold = thresh_hold
    
    def __str__(self):
        return f'LearnedCountSketchCompressor(sketch_size={self.sketch_size}, n_unempty={self.n_unempty}, reuse_sketch={self.reuse_sketch}, cs_init={self.cs_init}, thresh_hold={self.thresh_hold})'
    
    def need_profile(self):
        return True
    
    def train(self, ts:List[torch.Tensor], 
                    sketch_l: CountSketch,
                    sketch_r: CountSketch,
                    optim: torch.optim.Optimizer, iterations=100):
        for _ in range(iterations):
            for t in ts:
                optim.zero_grad()
                sketch_mtx_l = sketch_l.get_sketch(train = True)
                sketch_mtx_r = sketch_r.get_sketch(train = True)
                if sketch_mtx_l.size(1) > sketch_mtx_r.size(1):
                    approx = sketch_mtx_l.t() @ ((sketch_mtx_l @ (t @ sketch_mtx_r.t())) @ sketch_mtx_r)
                else:
                    approx = (sketch_mtx_l.t() @ ((sketch_mtx_l @ t) @ sketch_mtx_r.t())) @ sketch_mtx_r
                loss = torch.norm(approx - t) / torch.norm(t)
                loss.backward()
                optim.step()
        with torch.no_grad():
            losses = []
            sketch_mtx_l = sketch_l.get_sketch(train = False)
            sketch_mtx_r = sketch_r.get_sketch(train = False)
            for t in ts:
                if sketch_mtx_l.size(1) > sketch_mtx_r.size(1):
                    approx = sketch_mtx_l.t() @ ((sketch_mtx_l @ (t @ sketch_mtx_r.t())) @ sketch_mtx_r)
                else:
                    approx = (sketch_mtx_l.t() @ ((sketch_mtx_l @ t) @ sketch_mtx_r.t())) @ sketch_mtx_r
                loss = torch.norm(approx - t) / torch.norm(t)
                losses.append(loss.item())
            loss = sum(losses) / len(losses) if len(losses) else -1
        return loss
        
    def _fit(self, shape: torch.Size, dtype: torch.dtype, tensors: List[torch.Tensor], sketch_l = None, sketch_r = None, n_iter = 100, lr = 1e-1):
        m, n = shape
        if sketch_l is None:
            sketch_l = CountSketch(self.sketch_size, m, self.n_unempty, 'cuda', dtype, self.cs_init) 
        if sketch_r is None:
            sketch_r = CountSketch(self.sketch_size, n, self.n_unempty, 'cuda', dtype, self.cs_init)
        optim = torch.optim.Adam([sketch_l.values, sketch_r.values], lr=lr)
        loss = self.train(tensors, sketch_l, sketch_r, optim, n_iter)
        del optim
        del tensors
        return sketch_l, sketch_r, loss
    
    def eval(self, modules):
        losses = []
        for module in modules:
            sketch_l, sketch_r = self.sketches_map[module]
            with torch.no_grad():
                grad = module._grad.cuda().to(module.weight.dtype)
                sketch_mtx_l = sketch_l.get_sketch()
                sketch_mtx_r = sketch_r.get_sketch()
                if sketch_mtx_l.size(1) > sketch_mtx_r.size(1):
                    approx = sketch_mtx_l.t() @ ((sketch_mtx_l @ (grad @ sketch_mtx_r.t())) @ sketch_mtx_r)
                else:
                    approx = (sketch_mtx_l.t() @ ((sketch_mtx_l @ grad) @ sketch_mtx_r.t())) @ sketch_mtx_r
                loss = torch.norm(approx - grad) / torch.norm(grad)
                losses.append(loss.item())
        loss = sum(losses) / len(losses)
        print(f'Eval loss: {loss:.4f}')
        return loss
    
    def fit(self, modules):
        first_fit = len(self.sketches_map) == 0
        if not first_fit and (self.eval(modules) < self.thresh_hold):
            for module in modules:
                module._grad = None
                if module.current_device == 'cpu':
                    module.to('cuda')
            print('Skip learn sketching because the error is below the threshold')
            return 
        start_time = time.perf_counter()
        
        # we first classify the modules by their shapes
        modules_map = {}
        for module in modules:
            shape = module.weight.size()
            if shape not in modules_map:
                modules_map[shape] = []
            modules_map[shape].append(module)
        dtype = modules[0].weight.dtype
        
        # then we compress the gradients of the same shape
        losses = []
        
        for shape, modules in modules_map.items():
            if first_fit or not self.reuse_sketch:
                if first_fit: 
                    sketch_l_common, sketch_r_common, _ = self._fit(shape, dtype, [], n_iter = 0, lr = self.lr)
                else:
                    sketch_l_common, sketch_r_common, _ = self._fit(
                        shape, dtype,
                        [module._grad.cuda().to(module.weight.dtype) for module in modules], 
                        n_iter = self.common_fit_iter, lr = self.lr)

            for module in tqdm.tqdm(modules, desc = f'Learn sketching for shape {shape}'):
                if first_fit or not self.reuse_sketch:
                    sketch_l = copy.deepcopy(sketch_l_common)
                    sketch_r = copy.deepcopy(sketch_r_common)
                else: 
                    l, r = self.sketches_map[module]
                    sketch_l = copy.deepcopy(l)
                    sketch_r = copy.deepcopy(r)
                if first_fit:
                    _, _, loss = self._fit(shape, dtype, [], sketch_l, sketch_r, n_iter = 0, lr = self.ft_lr)
                else:
                    _, _, loss = self._fit(shape, dtype, 
                                           [module._grad.cuda().to(module.weight.dtype)], 
                                           sketch_l, sketch_r, n_iter = self.finetune_iter, lr = self.ft_lr)

                losses.append(loss)
                with torch.no_grad():
                    sketch_l_mtx = sketch_l.get_sketch()
                    sketch_r_mtx = sketch_r.get_sketch()
                    old_sketch_l, old_sketch_r = self.sketches_map.get(module, (None, None))
                    sketch_l_mtx_old = old_sketch_l.get_sketch() if old_sketch_l is not None else None
                    sketch_r_mtx_old = old_sketch_r.get_sketch() if old_sketch_r is not None else None
                    l_proj = (sketch_l_mtx @ sketch_l_mtx_old.t() if sketch_l_mtx_old is not None else sketch_l_mtx).to('cuda').to(module.optim_dtype)
                    r_proj = (sketch_r_mtx @ sketch_r_mtx_old.t() if sketch_r_mtx_old is not None else sketch_r_mtx).to('cuda').to(module.optim_dtype)
                    self.sketches_map[module] = (sketch_l, sketch_r)
                    if module.inited:
                        if module.current_device == 'cpu':
                            module.to('cuda')
                        module.optim.m = l_proj @ module.optim.m @ r_proj.t()
                        module.optim.v = (l_proj ** 2) @ module.optim.v @ (r_proj.t() ** 2)
                        module.optim.m.nan_to_num_(nan=0.0, posinf = 1.0, neginf = -1.0)
                        module.optim.v.nan_to_num_(nan=0.0, posinf = 1.0)
                module._grad = None 
                module.init(self.sketch_size, self.sketch_size)
            
        print('Finish learn sketching')
        tot_mem = sum([u.n_storage() + v.n_storage() for u, v in self.sketches_map.values()])
        print(f'Total memory: {tot_mem / 1024 / 1024:.2f} MB, Average loss: {sum(losses) / len(losses):.4f}, Time: {time.perf_counter() - start_time:.2f}s')
        
    def left_compress_impl(self, module, tensor):
        if module not in self.sketches_map: return tensor
        sketch_l, _ = self.sketches_map[module]
        sketch_l_mtx = sketch_l.get_sketch()
        return sketch_l_mtx @ tensor
    
    def right_compress_impl(self, module, tensor):
        if module not in self.sketches_map: return tensor
        _, sketch_r = self.sketches_map[module]
        sketch_r_mtx = sketch_r.get_sketch()
        return tensor @ sketch_r_mtx.t()
    
    def decompress_impl(self, module, tensor):
        if module not in self.sketches_map: return tensor
        sketch_l, sketch_r = self.sketches_map[module]
        sketch_l_mtx = sketch_l.get_sketch()
        sketch_r_mtx = sketch_r.get_sketch()
        if sketch_l_mtx.size(1) > sketch_r_mtx.size(1):
            return sketch_l_mtx.t() @ (tensor @ sketch_r_mtx)
        return sketch_l_mtx.t() @ tensor @ sketch_r_mtx
    
if __name__ == '__main__':
    sketch_size, compress_size, other_dim = 1024, 1600, 4800
    n_unempty = 16
    n_repeat = 100
    dtype = torch.float16
    sketch = CountSketch(sketch_size, compress_size, n_unempty, torch.device('cuda'), dtype)

    tensor = torch.randn(compress_size, other_dim, dtype=dtype, device=torch.device('cuda'))
    t = time.perf_counter()
    for _ in range(n_repeat):
        sketch_mtx = sketch.get_sketch()
        approx = sketch_mtx.t() @ (sketch_mtx @ tensor)
    torch.cuda.synchronize()
    print(f'Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    sketch_mtx = sketch.get_sketch()
    sketch_t_CSR = sketch_mtx.t().to_sparse_csr()
    sketch_CSR = sketch_mtx.to_sparse_csr()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = sketch_CSR.t() @ (sketch_CSR @ tensor)
    torch.cuda.synchronize()
    print(f'CSR Left compressed Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    sketch_mtx = sketch.get_sketch()
    sketch_t_CSC = sketch_mtx.t().to_sparse_csc()
    sketch_CSC = sketch_mtx.to_sparse_csc()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = sketch_t_CSC @ (sketch_CSC @ tensor)
    torch.cuda.synchronize()
    print(f'CSC Left compressed Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    dense = torch.randn(n_unempty, compress_size, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = dense.t() @ (dense @ tensor)
    torch.cuda.synchronize()
    print(f'Reference time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    tensor = torch.randn(other_dim, compress_size, dtype=dtype, device=torch.device('cuda'))
    t = time.perf_counter()
    for _ in range(n_repeat):
        sketch_mtx = sketch.get_sketch()
        approx = (tensor @ sketch_mtx.t()) @ sketch_mtx
    torch.cuda.synchronize()
    print(f'Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    sketch_mtx = sketch.get_sketch()
    sketch_t_csr = sketch_mtx.t().to_sparse_csr()
    sketch_csr = sketch_mtx.to_sparse_csr()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = (sketch_t_csr @ (sketch_csr @ tensor.t())).t()
    torch.cuda.synchronize()
    print(f'CSR Right compressed Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    sketch_mtx = sketch.get_sketch()
    sketch_t_csc = sketch_mtx.t().to_sparse_csc()
    sketch_csc = sketch_mtx.to_sparse_csc()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = (tensor @ sketch_t_csc) @ sketch_csc
    torch.cuda.synchronize()
    print(f'CSC Right compressed Average time: {(time.perf_counter() - t) / n_repeat:.4f} s')
    
    
    dense = torch.randn(n_unempty, compress_size, dtype=dtype, device=torch.device('cuda'))
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n_repeat):
        approx = (tensor @ dense.t()) @ dense
    torch.cuda.synchronize()
    print(f'Reference time: {(time.perf_counter() - t) / n_repeat:.4f} s')