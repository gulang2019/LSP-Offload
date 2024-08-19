import os
import torch 
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_matrix(matrix, fig, axes):
    S = torch.linalg.svdvals(matrix).cpu().numpy()
    cax = axes[0].imshow(matrix.cpu().numpy())
    cbar = fig.colorbar(cax, ax=axes[0])
    hist, bins = np.histogram(matrix.flatten().abs().cpu().numpy())
    axes[1].plot(bins[:-1], np.cumsum(hist) / hist.sum())
    axes[2].plot(np.arange(1, 1 + len(S)), np.cumsum(S) / S.sum())

def matrix_analyzer(matrices, save_dir = 'matrix', tag: str = 'matrix'):
    # sparsity
    # rank
    print('Analyze', tag)
    fig, axes = plt.subplots(len(matrices), 3, tight_layout = True, figsize = (9, len(matrices) * 3))
    for i, (name, matrix) in enumerate(matrices):
        if len(matrix.size()) == 4: # (B, H, SL, SL/HS)
            head = random.randint(0, matrix.size(1) - 1)
            name = f'{name} H{head}'
            visualize_matrix(matrix[:, head, :, :].squeeze(0), fig, axes[i, :])
        elif len(matrix.size()) == 3: # (B, SL, HS)
            visualize_matrix(matrix.squeeze(0), fig, axes[i, :])
        elif len(matrix.size()) ==2:
            visualize_matrix(matrix, fig, axes[i, :])
        else: 
            raise ValueError(f'Unsupported matrix size {matrix.size()} {tag} {name}')
        axes[i, 0].set_ylabel(f'{name} {matrix.size()}')
    fig.suptitle(tag)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f'{save_dir}/{tag}.png')
    plt.close(fig)
    torch.save(matrices, f'{save_dir}/{tag}.pt')
    

class NVTXContext:
    def __init__(self, name, enabled = True):
        self.name = name 
        self.enabled = enabled

    def __enter__(self):
        # self.allocated = torch.cuda.memory_allocated() / 1024 / 1024
        if self.enabled:
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, *args):
        # new_allocated = torch.cuda.memory_allocated() / 1024 / 1024 - self.allocated
        # if new_allocated > 0:
        #     print(f'NVTX {self.name} allocated {new_allocated:.1f} MB')
        if self.enabled:
            torch.cuda.nvtx.range_pop()


def profile_comm(offload_device, n_repeat):
    import time 
    
    def comm(m, n, dtype, from_device,  to_device):
        a = torch.randn(m, n).to(dtype).to(from_device)
        if 'cuda' in from_device: 
            torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(n_repeat):
            b = a.to(to_device)
        if 'cuda' in to_device: 
            torch.cuda.synchronize()
        end = time.perf_counter()
        return (end - start) / n_repeat
    from sklearn import linear_model
    def linear_regression(x,y):
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        return regr.coef_, regr.intercept_, regr.score(x, y)
    
    print('comm:')
    for from_device, to_device in zip(('cuda:0', offload_device), (offload_device, 'cuda:0')):
        comms = []
        ranks = [1, 2, 4, 8, 16, 64, 256, 4096]
        for r in ranks:
            t = comm(4096, r, torch.float16, from_device, to_device)
            comms.append(t)
        k, b, r2 = linear_regression(np.array(ranks).reshape(-1,1), np.array(comms))
        k = k[0] / 4096 / 2
        print(f'from_device: {from_device}, to_device: {to_device}, {1/k/1e9:.2f} GB/s, overhead {b*1e3:.2f} s, r2: {r2:.2f}')