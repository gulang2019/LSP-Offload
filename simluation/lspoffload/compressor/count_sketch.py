import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import torch
from torch.nn.parameter import Parameter as Parameter
from tqdm.auto import tqdm

from .compressor import Compressor, CompressorArgs, CompressorManager


@dataclass
class CountSketchArgs(CompressorArgs):
    """
    Args:
        output_dim: The output dimension of the count sketch
        nonzero_dim: The number of non-zero dimension in the count sketch
        init_method: The initialization method for the count
    """

    _compressor_name: str = "count_sketch"
    output_dim: int | Dict[int, int] = 1024
    nonzero_dim: int = 8
    init_method: Literal["binary", "gaussian"] = "binary"


class CountSketch:
    def __init__(
        self,
        input_dim: int,
        args: CountSketchArgs,
        device: torch.device,
        dtype: torch.dtype,
    ):
        # self.indices = torch.randperm(args.output_dim, device=device)[: args.nonzero_dim].repeat(input_rank, 1).T
        if isinstance(args.output_dim, int):
            self.output_dim = args.output_dim
        elif isinstance(args.output_dim, dict):
            self.output_dim = args.output_dim.get(input_dim, -1)
            if self.output_dim == -1:
                raise ValueError(f"output_dim is not defined for input_dim {input_dim}")
        else:
            raise ValueError(f"Unknown output_dim type {type(args.output_dim)}")

        self.indices = CountSketch._rand_indices(self.output_dim, args.nonzero_dim, input_dim, device)
        if args.init_method == "binary":
            values = torch.randint(0, 2, (args.nonzero_dim, input_dim), dtype=dtype, device=device)  # {0, 1}
            values = values * 2 - 1  # Convert to {-1, 1}
            values /= math.sqrt(args.nonzero_dim)

            self.values = torch.nn.Parameter(values)
        elif args.init_method == "gaussian":
            values = torch.randn((args.nonzero_dim, input_dim), dtype=dtype, device=device)
            values /= math.sqrt(args.nonzero_dim)

            self.values = torch.nn.Parameter(values)
        else:
            raise ValueError(f"Unknown initialization method {args.init_method}")
        self.shape = (self.output_dim, input_dim)

    def get_sketch(self, train: bool = False) -> torch.Tensor:
        return torch.zeros(self.shape, dtype=self.values.dtype, device=self.values.device).scatter_(
            0, self.indices, self.values if train else self.values.data
        )

    @staticmethod
    def _rand_indices(output_dim: int, nonzero_dim: int, input_dim: int, device: torch.device) -> torch.Tensor:
        # return torch.randint(0, output_dim, (nonzero_dim, input_rank), device=device)
        weights = torch.ones(output_dim, device=device).expand(input_dim, -1)
        return torch.multinomial(weights, nonzero_dim, replacement=False).T


class CountSketchCompressor(Compressor):
    def __init__(
        self,
        left_rank: int,
        right_rank: int,
        count_sketech_args: CountSketchArgs,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.left_rank = left_rank
        self.right_rank = right_rank
        self.count_sketch_args = count_sketech_args
        self.device = device
        self.dtype = dtype

        self.init_count_sketch()

    def init_count_sketch(self):
        self.left_sketch = CountSketch(
            self.left_rank,
            self.count_sketch_args,
            self.device,
            self.dtype,
        )

        self.right_sketch = CountSketch(
            self.right_rank,
            self.count_sketch_args,
            self.device,
            self.dtype,
        )

    def compress(self, full_rank_grad: torch.Tensor, train: bool = False):
        return torch.linalg.multi_dot(
            [self.left_sketch.get_sketch(train), full_rank_grad, self.right_sketch.get_sketch(train).T]
        )

    def decompress(self, low_rank_grad: torch.Tensor, train: bool = False):
        return torch.linalg.multi_dot(
            [self.left_sketch.get_sketch(train).T, low_rank_grad, self.right_sketch.get_sketch(train)]
        )

    def parameters(self) -> List[Parameter]:
        return [self.left_sketch.values, self.right_sketch.values]


class CountSketchManager(CompressorManager):
    def __init__(
        self,
        skip_optimizing_threshold: float = 0.2,
        reuse_sketches: bool = False,
        optimize_lr: float = 0.1,
        cs_optimizer_type: Literal["adamw", "adam", "sgd"] = "adamw",
        cs_lr_scheduler_type: Literal["step", "cosine", "none"] = "none",
        common_optimize_iter: int = 100,
        specific_optimize_iter: int = 100,
        **kwargs,
    ) -> None:
        super().__init__()

        self.compressor_args = CountSketchArgs(**kwargs)

        self.skip_optimizing_threshold = skip_optimizing_threshold
        self.reuse_sketches = reuse_sketches
        self.optimize_lr = optimize_lr
        self.common_optimize_iter = common_optimize_iter
        self.specific_optimize_iter = specific_optimize_iter

        self.cs_optimizer_tyep = cs_optimizer_type
        self.cs_lr_scheduler_type = cs_lr_scheduler_type

        self.compressors: Dict[Tuple[int, int, int], CountSketchCompressor] = {}

        # Temporarily disable because it is not necessary
        self.first_fit = False

    def init_compressors(self, lsp_param_groups: List[Dict[str, Any]], state: Dict[torch.Tensor, Any]):
        for gidx, group in enumerate(lsp_param_groups):
            assert group.get("use_lsp", False), "CountSketchCompressor only supports LSP parameters"
            for param in group["params"]:
                grad = param.grad
                left_rank = grad.size(-2)
                right_rank = grad.size(-1)
                # Use the same compressor for the same rank for memory and performance reasons.
                # And some experiments prove that it is not much worse then using different compressors.
                if (left_rank, right_rank, gidx) not in self.compressors:
                    self.compressors[(left_rank, right_rank, gidx)] = CountSketchCompressor(
                        left_rank,
                        right_rank,
                        self.compressor_args,
                        param.device,
                        param.dtype,
                    )
                state[param]["compressor"] = self.compressors[(left_rank, right_rank, gidx)]

    def optimize_compressors(self, lsp_param_groups: List[Dict[str, Any]], state: Dict[torch.Tensor, Any]):
        full_rank_grads = []
        for gidx, groups in enumerate(lsp_param_groups):
            for p in groups["params"]:
                if p.grad is None:
                    continue
                full_rank_grads.append((p.grad, state[p], gidx))

        self._fit(full_rank_grads)

    def should_update(self, steps: int):
        return steps % self.update_interval == 0

    @torch.no_grad()
    def _eval(self, full_rank_grads: List[torch.Tensor]) -> float:
        losses = []

        for full_rank_grad, _, gidx in full_rank_grads:
            left_rank = full_rank_grad.shape[0]
            right_rank = full_rank_grad.shape[1]

            compressor = self.compressors[(left_rank, right_rank, gidx)]
            decompressed_grad = compressor.decompress(compressor.compress(full_rank_grad))

            loss = torch.norm(full_rank_grad - decompressed_grad) / torch.norm(full_rank_grad)
            losses.append(loss.item())
        loss = sum(losses) / len(losses)
        print("LSPoffload CountSketch eval loss:", loss)
        return loss

    def _fit(self, full_rank_grads: List[torch.Tensor]) -> float:
        # Skip if error is small enough
        if not self.first_fit and self._eval(full_rank_grads) < self.skip_optimizing_threshold:
            return

        grads_map: Dict[Tuple[int, int], List[torch.Tensor]] = {}
        state_dict_map: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for full_rank_grad, state_dict, gidx in full_rank_grads:
            left_rank = full_rank_grad.size(-2)
            right_rank = full_rank_grad.size(-1)
            if (left_rank, right_rank, gidx) not in grads_map:
                grads_map[(left_rank, right_rank, gidx)] = []
                state_dict_map[(left_rank, right_rank, gidx)] = []
            grads_map[(left_rank, right_rank, gidx)].append(full_rank_grad)
            state_dict_map[(left_rank, right_rank, gidx)].append(state_dict)

        losses = []

        for shape, compressor in self.compressors.items():
            # First, init and optimize each compressor for all the gradients with
            # the same shape at once
            old_compressor = deepcopy(compressor)
            if self.first_fit or not self.reuse_sketches:
                compressor.init_count_sketch()
                self._optimize_compressor(
                    compressor,
                    grads_map.get(shape, []),
                    n_iter=self.common_optimize_iter if not self.first_fit else 0,
                    lr=self.optimize_lr,
                    use_tqdm=True,
                )

            # Second, optimize each compressor for the gradients with the same shape
            # one by one
            for full_rank_grad in grads_map.get(shape, []):
                # if self.first_fit or not self.reuse_sketches:
                #     old_compressor = deepcopy(compressor)
                # else:
                #     old_compressor = None
                loss = self._optimize_compressor(
                    compressor,
                    [full_rank_grad],
                    n_iter=self.specific_optimize_iter,
                    lr=self.optimize_lr,
                )

                losses.append(loss)

            # Reproject the optimizer subspace
            for state_dict in state_dict_map.get(shape, []):
                if "exp_avg" in state_dict:
                    state_dict["exp_avg"][:] = compressor.compress(old_compressor.decompress(state_dict["exp_avg"]))
                if "exp_avg_sq" in state_dict:
                    l_proj = compressor.left_sketch.get_sketch() @ old_compressor.left_sketch.get_sketch().T
                    r_proj = old_compressor.right_sketch.get_sketch() @ compressor.right_sketch.get_sketch().T

                    state_dict["exp_avg_sq"][:] = torch.linalg.multi_dot(
                        [l_proj**2, state_dict["exp_avg_sq"], r_proj**2]
                    )

        return sum(losses) / len(losses)

    def _optimize_compressor(
        self,
        compressor: CountSketchCompressor,
        full_rank_grads: List[torch.Tensor],
        n_iter: int,
        lr: float,
        use_tqdm: bool = False,
    ) -> float:
        # optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr)
        if self.cs_optimizer_tyep == "adamw":
            optimizer = torch.optim.AdamW(compressor.parameters(), lr=lr)
        elif self.cs_optimizer_tyep == "adam":
            optimizer = torch.optim.Adam(compressor.parameters(), lr=lr)
        elif self.cs_optimizer_tyep == "sgd":
            optimizer = torch.optim.SGD(compressor.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type {self.cs_optimizer_tyep}")

        if self.cs_lr_scheduler_type == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_iter, gamma=0.5)
        elif self.cs_lr_scheduler_type == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)
        elif self.cs_lr_scheduler_type == "none":
            lr_scheduler = None

        if use_tqdm:
            pbar = tqdm(total=n_iter, desc="Optimizing compressor")
        for _ in range(n_iter):
            fit_loss = torch.tensor(0.0, device=full_rank_grads[0].device)
            for full_rank_grad in full_rank_grads:
                optimizer.zero_grad()
                compressed_grad = compressor.compress(full_rank_grad, train=True)
                decompressed_grad = compressor.decompress(compressed_grad, train=True)
                loss = torch.norm(full_rank_grad - decompressed_grad) / torch.norm(full_rank_grad)
                loss.backward()

                optimizer.step()
                fit_loss += loss

            if lr_scheduler is not None:
                lr_scheduler.step()

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(loss=(fit_loss / len(full_rank_grads)).item())

        losses = []
        with torch.no_grad():
            for full_rank_grad in full_rank_grads:
                compressed_grad = compressor.compress(full_rank_grad)
                decompressed_grad = compressor.decompress(compressed_grad)
                loss = torch.norm(full_rank_grad - decompressed_grad) / torch.norm(full_rank_grad)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        if use_tqdm:
            pbar.set_postfix(loss=avg_loss)
            pbar.close()

        return avg_loss
