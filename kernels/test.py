import torch
import fused_adam

def adam_gt(
    grad, m, v, beta, gamma, lr, t, eps
): 
    with torch.no_grad():
        m = beta * m + (1 - beta) * grad
        v = gamma * v + (1 - gamma) * grad * grad
        m_hat = m / (1 - beta**t)
        v_hat = v / (1 - gamma**t)
        grad = - lr * m_hat / (torch.sqrt(v_hat) + eps)
        
def test_correctness(shape, dtype, device):
    # param = torch.randn(shape, dtype=dtype, device='cuda', requires_grad=True)
    grad = torch.randn(shape, dtype=dtype, device=device)
    m = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device).abs() + 1e-6
    grad_gt = grad.clone()
    m_gt = m.clone()
    v_gt = v.clone()
    beta = 0.9
    gamma = 0.999
    lr = 0.001
    eps = 1e-4
    t = 10 
    with torch.no_grad():
        m_gt = beta * m_gt + (1 - beta) * grad_gt
        v_gt = gamma * v_gt + (1 - gamma) * grad_gt * grad_gt
        m_hat = m_gt / (1 - beta**t)
        v_hat = v_gt / (1 - gamma**t)
        grad_gt = - lr * m_hat / (torch.sqrt(v_hat) + eps)
    fused_adam.fused_adam(grad, m, v, beta, gamma, lr, eps, t)

    max_err  = torch.max(torch.abs(grad - grad_gt))
    max_m_err = torch.max(torch.abs(m - m_gt))
    max_v_err = torch.max(torch.abs(v - v_gt))
    print(f'max_err={max_err:.3e}, max_m_err={max_m_err:.3e}, max_v_err={max_v_err:.3e}')
    
    assert torch.allclose(grad, grad_gt, atol = 1e-2)
    assert torch.allclose(m, m_gt, atol=1e-1)
    assert torch.allclose(v, v_gt, atol = 1e-1)

import time 

def benchmark(shape, dtype, device, n_repeat = 100):
    grad = torch.randn(shape, dtype=dtype, device=device)
    m = torch.randn(shape, dtype=dtype, device=device)
    v = torch.randn(shape, dtype=dtype, device=device).abs() + 1e-6
    grad_gt = grad.clone()
    m_gt = m.clone()
    v_gt = v.clone()
    beta = 0.9
    gamma = 0.999
    lr = 0.001
    eps = 1e-4
    t = 10
    
    
    start = time.perf_counter()
    for i in range(n_repeat): 
        fused_adam.fused_adam(grad, m, v, beta, gamma, lr, eps, t)
    fused_time = time.perf_counter() - start    
    
    start = time.perf_counter()
    for i in range(n_repeat):
        with torch.no_grad():
            m_gt = beta * m_gt + (1 - beta) * grad_gt
            v_gt = gamma * v_gt + (1 - gamma) * grad_gt * grad_gt
            m_hat = m_gt / (1 - beta**t)
            v_hat = v_gt / (1 - gamma**t)
            grad_gt = - lr * m_hat / (torch.sqrt(v_hat) + eps)
    vanilla_time = time.perf_counter() - start
    
    return fused_time / n_repeat, vanilla_time / n_repeat

# test_correctness((1024), torch.float32, 'cpu')
# test_correctness((768, 2304), torch.float32, 'cuda')
# test_correctness((768, 2304), torch.bfloat16, 'cuda')
# test_correctness((1024), torch.float32, 'cuda:1')
    

for device in ['cpu', 'cuda']:
    for dtype in [torch.float32, torch.bfloat16]:
        if device == 'cpu' and dtype == torch.bfloat16:
            continue
        tot_vanilla, tot_fused = 0, 0
        for shape, cnt in [((4096, 4096), 4), ((4096, 11008), 3)]:
            fused_time, vanilla_time = benchmark(shape, dtype, device)
            tot_fused += fused_time * cnt
            tot_vanilla += vanilla_time * cnt
        print(f'device={device}, dtype={dtype}, fused_time={tot_fused:.3e}s, vanilla_time={tot_vanilla:.3e}s')
        print(f'speedup={tot_vanilla / tot_fused:.1f}x')
            