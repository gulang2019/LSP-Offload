import torch 
import numpy as np
import time 

def lora_mm_CountSketch(A: torch.Tensor, B: torch.Tensor, k=32):
    d, n = A.size()
    _n, m = B.size()
    assert n == _n
    # A: M x K; B: K x N
    hash_indices = torch.randint(0, k, [n], dtype = torch.long, device = A.device)
    signs = torch.randint(low = 0, high = 2, size=[n], dtype = A.dtype, device = A.device) * 2 - 1

    sketch_matrix = torch.zeros((n, k), dtype = A.dtype, device = A.device)
    sketch_matrix[torch.arange(n), hash_indices] = signs
    # print('sketch_matrix:', sketch_matrix.size(), sketch_matrix.dtype)
    return (A @ sketch_matrix) @ (sketch_matrix.t() @ B)

def lora_mm_gaussian(A: torch.Tensor, B: torch.Tensor, k=32):
    d, n = A.size()
    _n, m = B.size()
    assert n == _n
    sketch_matrix = torch.randn((n, k), dtype = A.dtype, device = A.device) / np.sqrt(k)
    # print('sketch_matrix:', sketch_matrix.size(), sketch_matrix.dtype)
    return (A @ sketch_matrix) @ (sketch_matrix.t() @ B)

def lora_mm_topk(A: torch.Tensor, B: torch.Tensor, k):
    rate = (k / A.size(1)) if isinstance(k, int) else k
    return topk(A, rate) @ topk(B, rate)

def lora_mm_imp_sampling(A: torch.Tensor, B: torch.Tensor, k=32):
    d, n = A.size()
    _n, m = B.size()
    assert n == _n
    leverage_scores = torch.norm(A, dim = 0) * torch.norm(B, dim = 1)
    leverage_scores = leverage_scores / torch.sum(leverage_scores)
    indices = torch.multinomial(leverage_scores, k, replacement = True).to(A.device)
    scales = 1 / torch.sqrt(leverage_scores[indices] * k) # (k,)
    sketch_matrix = torch.zeros((n, k), dtype = A.dtype, device = A.device)
    sketch_matrix[indices, torch.arange(k)] = scales
    return (A @ sketch_matrix) @ (sketch_matrix.t() @ B)

def lora_mm_imp_sampling_aonly(A: torch.Tensor, B: torch.Tensor, k=32):
    d, n = A.size()
    _n, m = B.size()
    assert n == _n
    leverage_scores = torch.norm(A, dim = 0) 
    leverage_scores = leverage_scores / torch.sum(leverage_scores)
    indices = torch.multinomial(leverage_scores, k, replacement = True).to(A.device)
    scales = 1 / torch.sqrt(leverage_scores[indices] * k) # (k,)
    sketch_matrix = torch.zeros((n, k), dtype = A.dtype, device = A.device)
    sketch_matrix[indices, torch.arange(k)] = scales
    return (A @ sketch_matrix) @ (sketch_matrix.t() @ B)
    
def lora_mm_topk_sampling(A: torch.Tensor, B: torch.Tensor, k=32):
    d, n = A.size()
    _n, m = B.size()
    assert n == _n
    k = min(k, n)
    leverage_scores = torch.norm(A, dim = 0) * torch.norm(B, dim = 1)
    leverage_scores = leverage_scores / torch.sum(leverage_scores)
    # print('A:', A.size(), 'B:', B.size(), 'leverage_scores:', leverage_scores.size(), 'k:', k)
    _, indices = torch.topk(leverage_scores, k)
    sketch_matrix = torch.zeros((n, k), dtype = A.dtype, device = A.device)
    sketch_matrix[indices, torch.arange(k)] = 1
    return (A @ sketch_matrix) @ (sketch_matrix.t() @ B)

def lora_svd(A: torch.Tensor, k):
    U, S, V = torch.svd(A.float())
    return (U[:, :k] @ torch.diag(S[:k]) @ V[:, :k].t()).to(A.dtype)

def lora_power(A: torch.Tensor, k):
    m, n = A.size()
    U = torch.randn(k, m, device = A.device, dtype = A.dtype) / np.sqrt(k)
    y = U @ A # k x n
    ortho = torch.qr(y.t().float())[0].to(A.dtype) # n x \hat{k}
    return A @ ortho @ ortho.t()

def lora_approx(A: torch.Tensor, k):
    m, n = A.size()
    U = torch.randn(int(1.5*k), m, device = A.device, dtype = A.dtype) / np.sqrt(int(1.5*k))
    y = U @ A
    ortho = torch.qr(y.t().float())[0].to(A.dtype) # n x \hat{k}
    Aortho = A@ortho
    return lora_svd(Aortho, k) @ ortho.t()

def lora_gaussian(A: torch.Tensor, k):
    U = torch.randn(A.size(1), k, device = A.device) / np.sqrt(k)
    return A @ U @ U.t()

def lora_first(x,y,rank):
    return lora_svd(x, rank) @ y

def lora_approx_first(x,y,rank):
    return lora_approx(x, rank) @ y

def lora_approx_second(x,y,rank):
    return x @ lora_approx(y.t(), rank).t()

def lora_svd_second(x,y,rank):
    return x @ lora_svd(y, rank)

def leverage_score_sampling(A: torch.Tensor, k = 32):
    scores = torch.qr(A.float())[0].norm(dim=-1) ** 2
    probs = scores / scores.sum()
    indices = torch.multinomial(probs, k, replacement = False).to(A.device)
    mask = torch.zeros_like(probs, device=A.device, dtype=A.dtype)
    mask[indices] = 1 /  (k * probs[indices])
    return A * mask.unsqueeze(-1)

def mm_leverage_score_sampling(A: torch.Tensor, B: torch.Tensor, k = 32):
    return leverage_score_sampling(A, k) @ B

def norm_sampling(A: torch.Tensor, k = 32):
    scores = A.norm(dim=-1)
    probs = scores / scores.sum()
    indices = torch.multinomial(probs, k, replacement = False).to(A.device)
    mask = torch.zeros_like(probs, device=A.device)
    mask[indices] = 1 /  (k * probs[indices])
    return A * mask.unsqueeze(-1)

def norm_sampling_t(A: torch.Tensor, k = 32):
    return norm_sampling(A.t(), k).t()

def leverage_score_sampling_t(A: torch.Tensor, k = 32):
    return leverage_score_sampling(A.t(), k).t()

def topk(A: torch.Tensor, rate):
    k = int(rate * (A.size(0) + A.size(1))) if isinstance(rate, int) else int(rate * A.numel())
    _, indices = torch.topk(A.abs().reshape(-1), k)
    mask = torch.zeros_like(A.reshape(-1), device = A.device, dtype = A.dtype)
    mask[indices] = 1
    return A * mask.view(A.size())

def lora_topk(A: torch.Tensor, k):
    A_topk = topk(A, 0.05)
    residual = A - A_topk
    approx_residual = lora_power(residual, k)
    return A_topk + approx_residual

APPROX_MAP = {
# approx matmul  
    'CountSketch': lora_mm_CountSketch,
    'gaussian': lora_mm_gaussian,
    'imp_sampling': lora_mm_imp_sampling,
    'imp_sampling_aonly': lora_mm_imp_sampling_aonly,
    'topk_sampling': lora_mm_topk_sampling,
    'mm_leverage_score_sampling': mm_leverage_score_sampling,
    'leverage_score_sampling': leverage_score_sampling,
    'lora_first': lora_first,
    'lora_approx_first': lora_approx_first,
# low rank approx
    'gaussian_lora': lora_gaussian,
    'power': lora_power,
    'approx': lora_approx,
    'mm_topk': lora_mm_topk,
    'svd': lora_svd,
    'norm_sampling': norm_sampling,
    'norm_sampling_t': norm_sampling_t,
    'leverage_score_sampling_t': leverage_score_sampling_t,
    'topk': topk,
}

def test_lora_mm(type):
    A = torch.randn(768, 512).cuda()
    B = torch.randn(512, 768).cuda()
    if type == 'svd':
        func = lora_svd
    elif type == 'CountSketch':
        func = lora_mm_CountSketch
    elif type == 'gaussian':
        func = lora_mm_gaussian
    elif type == 'imp_sampling':
        func = lora_mm_imp_sampling
    elif type == 'topk_sampling':
        func = lora_mm_topk_sampling
    elif type == 'mm_topk':
        func = lora_mm_topk
    elif type == 'imp_sampling_aonly':
        func = lora_mm_imp_sampling_aonly
    elif type == 'power':
        func = lora_power
    elif type == 'approx':
        func = lora_approx
    elif type == 'gaussian_lora':
        func = lora_gaussian
    elif type == 'topk':
        func = topk
    print(f'--{type}--')
    start = time.time()
    for i in range(100):
        ground_truth = A @ B
    ground_truth_elapsed = round((time.time() - start) * 1000, 1)
    ground_truth_norm = torch.norm(ground_truth).item()
    norm_a = torch.norm(A).item()
    norm_b = torch.norm(B).item()
    
    results = dict()
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        start = time.time()
        if type in ('svd', 'power', 'approx', 'gaussian_lora', 'topk'):
            fnorm = torch.norm(func(ground_truth, k) - ground_truth).item()
        else:
            fnorms = []
            for i in range(100):
                norm = torch.norm(func(A, B, k) - ground_truth).item()
                fnorms.append(norm)
            fnorm = np.mean(fnorms)
        end = time.time()
        elapsed = round((end - start) * 1000, 1)
        fnorm_to_ground_truth = round(fnorm / ground_truth_norm, 3)
        fnorm_to_prod_ab = round(fnorm / (norm_a * norm_b), 3)
        if type in ('svd', 'power', 'approx', 'gaussian_lora'):
            elapsed += ground_truth_elapsed
        results[k] = {'fnorm_to_ground_truth': fnorm_to_ground_truth, 'fnorm_to_ab': fnorm_to_prod_ab, 'runtime': elapsed}
        print('k:', k, 'fnorm_to_ground_truth:', fnorm_to_ground_truth, 'fnorm_to_prod_ab:', fnorm_to_prod_ab)
    return results

def test_lora():
    results = {}
    to_ground_truth = {}
    to_ab_norm = {}
    runtime = {}
    for type in ['svd', 'CountSketch', 'gaussian', 'imp_sampling', 'topk_sampling', 'power', 'approx', 'gaussian_lora']:
        results[type] = test_lora_mm(type)
        to_ground_truth[type] = {k: results[type][k]['fnorm_to_ground_truth'] for k in results[type]}
        to_ab_norm[type] = {k: results[type][k]['fnorm_to_ab'] for k in results[type]}
        runtime[type] = {k: results[type][k]['runtime'] for k in results[type]}
    import pandas as pd
    df = pd.DataFrame(to_ground_truth)
    print(df)
    df = pd.DataFrame(to_ab_norm)
    print(df)
    df = pd.DataFrame(runtime)
    print(df)

if __name__ == '__main__':
    # test_lora_mm('CountSketch')
    # test_lora()
    A = torch.randn(10, 10).cuda()
    A_topk = topk(A, 2)
    print(A)
    print(A_topk)
    