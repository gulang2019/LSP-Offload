import torch
from lora_modules import lora_approx
import numpy as np
import math 

class Compressor:
    def __init__(self, device, offloading_dtype, split_comm: bool):
        self.offloading_device = device
        self.offloading_dtype = offloading_dtype
        self.split_comm = split_comm
    
    def _offload(self, tensor):
        return tensor.to(self.offloading_device).to(self.offloading_dtype), tensor.numel() * tensor.element_size()
    
    def forward(self, A):
        if self.split_comm:
            self.A, comm = self._offload(A)
            return comm
        return 0
    
    def backward(self, A, B):
        if self.split_comm:
            B, comm = self._offload(B)
            grad = self.A @ B
            self.A = None
            return grad, comm
        grad = A @ B
        if grad.device == self.offloading_device:
            return grad, 0
        return self._offload(grad)

class ImpSampling(Compressor):
    def __init__(self, rank = 32, offloading_device = 'cpu', offloading_dtype: torch.dtype = torch.float32):
        super().__init__(offloading_device, offloading_dtype, True)
        self.rank = rank
        
    def forward(self, A):
        d, n = A.size()
        leverage_scores = torch.norm(A, dim = 0) + 1e-6
        leverage_scores = leverage_scores / torch.sum(leverage_scores)
        indices = torch.multinomial(leverage_scores.float(), self.rank, replacement = True)
        scales = 1 / torch.sqrt(leverage_scores[indices] * self.rank) # (self.rank,)
        self.sketch_matrix = torch.zeros((n, self.rank), dtype = A.dtype, device = A.device)
        self.sketch_matrix[indices, torch.arange(self.rank)] = scales
        self.A, amt = self._offload(A @ self.sketch_matrix)
        return amt
        
    def backward(self, _, B):
        b, amt = self._offload(self.sketch_matrix.t() @ B)
        grad = self.A @ b
        self.sketch_matrix = None
        self.A = None
        return grad, amt

class CountSketch(Compressor):
    def __init__(self, rank = 32, offloading_device = 'cpu', update_step = 10, offloading_dtype: torch.dtype = torch.float32):
        super().__init__(offloading_device, offloading_dtype, True)
        self.rank = rank
        self.sketch_matrix = None
        self.update_step = update_step
        self.counter = 0
        self.n = 0
        
    def forward(self, A):
        d, n = A.size()
        # A: M x K; B: K x N
        if self.sketch_matrix is None or n > self.n:
            hash_indices = torch.randint(0, self.rank, [n], dtype = torch.long, device = A.device)
            signs = torch.randint(low = 0, high = 2, size=[n], dtype = A.dtype, device = A.device) * 2 - 1
            self.sketch_matrix = torch.zeros((n, self.rank), dtype = A.dtype, device = A.device)
            self.sketch_matrix[torch.arange(n), hash_indices] = signs
            self.counter = 0
            self.n = n
        self.counter += 1
        self.A, amt = self._offload(A @ self.sketch_matrix[:n, :])
        return amt
        
    def backward(self, _, B):
        b, amt = self._offload(self.sketch_matrix[:B.size(0), :].t() @ B)
        grad = self.A @ b
        if self.counter % self.update_step == 0:
            self.sketch_matrix = None
        self.A = None
        return grad, amt

class LoraFirst(Compressor):
    def __init__(self, rank, offloading_device, offloading_dtype: torch.dtype = torch.float32):
        super().__init__(offloading_device, offloading_dtype, True)
        self.rank = rank
    
    def forward(self, A):
        m, n = A.size()
        U = torch.randn(int(self.rank), m, device = A.device, dtype = A.dtype) / np.sqrt(int(self.rank))
        y = U @ A
        ortho = torch.qr(y.t().float())[0].to(A.dtype) # n x \hat{k}
        Aortho = A@ortho
        self.ortho = ortho.t()
        self.Aortho, amt = self._offload(Aortho)
        return amt
    
    def backward(self, _, B):
        B = self.ortho @ B 
        self.ortho = None
        B, amt = self._offload(B)
        ret = self.Aortho @ B
        self.Aortho = None
        return ret, amt

class LoraPower(Compressor):
    @staticmethod
    def compress_and_comm(A: torch.Tensor, rank: int, device: torch.device, dtype: torch.dtype):
        m, n = A.size()
        U = torch.randn(rank, m, device = A.device, dtype = A.dtype)
        y = U @ A
        ortho = torch.qr(y.t().float())[0].to(A.dtype)
        Aortho = A @ ortho
        amt1 = Aortho.numel() * Aortho.element_size()
        Aortho = Aortho.to(device).to(dtype)
        amt2 = ortho.numel() * ortho.element_size()
        ortho = ortho.to(device).to(dtype)
        return Aortho @ ortho, amt1 + amt2
    
    def __init__(self, rank = 32, offloading_device = 'cpu', offloading_dtype: torch.dtype = torch.float32):
        super().__init__(offloading_device, offloading_dtype, False)
        self.rank = rank

    def forward(self, A):
        return 0
    
    def backward(self, A, B):
        grad = A @ B
        return self.compress_and_comm(grad, self.rank, self.offloading_device, self.offloading_dtype)
            
class TopK(Compressor):
    @staticmethod
    def compress_and_comm(A: torch.Tensor, rank: int, device: torch.device, dtype: torch.dtype):
        n_elem = int(rank * (A.size(0) + A.size(1))) if isinstance(rank, int) else int(rank * A.numel())
        n_elem = min(n_elem, A.numel())
        values, indices = torch.topk(A.abs().reshape(-1), n_elem)
        comm_amount = values.numel() * values.element_size() + indices.numel() * indices.element_size()
        values = values.to(device).to(A.dtype)
        indices = indices.to(device)
        A = torch.zeros_like(A, device = device, dtype = dtype)
        A[indices // A.size(1), indices % A.size(1)] = values
        return A, comm_amount
    
    def __init__(self, rank = 32, offloading_device = 'cpu', offloading_dtype: torch.dtype = torch.float32):
        super().__init__(offloading_device, offloading_dtype, False)
        self.rank = rank
    
    def forward(self, _):
        return 0
                
    def backward(self, A, B):
        grad = A @ B
        return self.compress_and_comm(grad, self.rank, self.offloading_device, self.offloading_dtype)
    
class CountSketch:
    def __init__(self, k: int,
                 n: int,
                 normalized: bool = False,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.k = k
        self.n = n
        self.normalized = normalized
        self.device = device
        self.dtype = dtype
    def update(self):
        self.values = torch.randint(0, 2, (self.n,), device = self.device, dtype = self.dtype) * 2 - 1
        self.indices = torch.randint(0, self.k, (self.n,), device = self.device, dtype = torch.long)
        if self.normalized:
            sketch = torch.zeros((self.k, self.n), device = self.device, dtype = self.dtype)
            sketch[self.indices, torch.arange(self.n)] = 1
            sum = torch.sum(sketch, dim = -1)
            scale = math.sqrt(self.n / self.k) / torch.sqrt(sum)    
            print(scale.size(), sketch.size())
            sketch = scale.unsqueeze(-1) * sketch
            self.values.mul_(sketch.sum(dim=0))
    def compress(self, A):
        sketch = torch.zeros((self.k, self.n), device = self.device, dtype = self.dtype)
        sketch[self.indices, torch.arange(self.n)] = self.values
        return sketch @ A

    def decompress(self, B):
        sketch = torch.zeros((self.k, self.n), device = self.device, dtype = self.dtype)
        sketch[self.indices, torch.arange(self.n)] = self.values
        return sketch.t() @ B

class RandomSelect:
    def __init__(self, k: int,
                 n: int,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.k = k
        self.n = n
        self.device = device
        self.dtype = dtype
    def update(self):
        self.indices = torch.randint(0, self.n, (self.k,), device = self.device, dtype = torch.long)
    
    def compress(self, A):
        return A[self.indices]
    
    def decompress(self, B):
        ret = torch.zeros((self.n, self.k), device = self.device, dtype = self.dtype)
        ret[self.indices] = B
        return ret

class SVD:
    def __init__(self, k: int,
                 n: int,
                 device: torch.device = 'cpu',
                 dtype: torch.dtype = torch.float32):
        self.k = k
        self.n = n
        self.device = device
        self.dtype = dtype
    def update(self):
        pass    
    def compress(self, A):
        U, S, V = torch.svd(A)
        self.U_k = U[:, :self.k]
        return self.U_k.t() @ A
    
    def decompress(self, B):
        return self.U_k @ B

if __name__ == '__main__':
    A = torch.randn(1024, 32)
    for method in [CountSketch, RandomSelect, SVD]:
        sketch = method(32, 1024)
        sketch.update()
        B = sketch.compress(A)
        A_ = sketch.decompress(B)
        print(method.__class__.__name__, A, A_, torch.max(torch.abs(A - A_)))