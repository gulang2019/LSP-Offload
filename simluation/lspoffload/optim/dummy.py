# copy dependencies from transformers/optimization.py
from typing import Any, Callable, Dict, Iterable

import torch
from torch import nn

from .lsp_optimizer import LspOptimizer


class Dummy(LspOptimizer):
    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lsp_args: Dict[str, Any],
    ):
        super().__init__(params, {"lsp_args": lsp_args})

    @torch.no_grad()
    def step(self, closure: Callable = None, save_grad_dir=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        losses = []
        if closure is not None:
            loss = closure()

        if "step" not in self.defaults:
            self.defaults["step"] = 0

        if self.defaults["compressor_mgr"] is None:
            self.init_compressors()

        for gidx, group in enumerate(self.param_groups):
            for pidx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "dim" not in group:
                    group["dim"] = 2

                # LspOffload Projection
                if "compressor" in state:
                    uncompressed_grad = grad
                    grad = state["compressor"].compress(grad)
                    decompressed_grad = state["compressor"].decompress(grad)

                    loss = torch.norm(uncompressed_grad - decompressed_grad) / torch.norm(uncompressed_grad)

                    if save_grad_dir is not None:
                        torch.save(
                            decompressed_grad, f"{save_grad_dir}/decompressed_grad_{gidx}_{pidx}_{state['step']}.pt"
                        )
                        torch.save(
                            uncompressed_grad, f"{save_grad_dir}/uncompressed_grad_{gidx}_{pidx}_{state['step']}.pt"
                        )

                    losses.append(loss)

        self.defaults["step"] += 1

        return torch.stack(losses).mean()
