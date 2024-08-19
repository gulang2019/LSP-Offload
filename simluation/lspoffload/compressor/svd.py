from dataclasses import dataclass
from typing import Dict, List, Literal

import torch
from torch.nn.parameter import Parameter as Parameter
from tqdm.auto import tqdm

from .compressor import Compressor, CompressorArgs, CompressorManager


@dataclass
class SvdArgs(CompressorArgs):
    """
    Args:
        output_dim: The output dimension of the count sketch
        nonzero_dim: The number of non-zero dimension in the count sketch
        init_method: The initialization method for the count
    """

    _compressor_name: str = "svd"
    rank: int = 128
    side: Literal["left", "both"] = "both"


class SvdCompressor(Compressor):
    def __init__(
        self,
        rank: int,
        side: Literal["left", "both"] = "both",
    ):
        self.rank = rank
        self.side = side
        self.lproj, self.rproj = None, None

    def compress(self, full_rank_grad: torch.Tensor):
        if self.lproj is None or self.rproj is None:
            raise ValueError("The compressor is not initialized")

        if self.side == "left":
            return torch.linalg.multi_dot([self.lproj.T, full_rank_grad])
        elif self.side == "both":
            return torch.linalg.multi_dot([self.lproj.T, full_rank_grad, self.rproj.T])
        else:
            raise ValueError(f"Invalid side {self.side}")

    def decompress(self, low_rank_grad: torch.Tensor):
        if self.lproj is None or self.rproj is None:
            raise ValueError("The compressor is not initialized")

        if self.side == "left":
            return torch.linalg.multi_dot([self.lproj, low_rank_grad])
        elif self.side == "both":
            return torch.linalg.multi_dot([self.lproj, low_rank_grad, self.rproj])
        else:
            raise ValueError(f"Invalid side {self.side}")

    def optimize(self, gradient: torch.Tensor):
        self.lproj, self.rproj = self._get_orthogonal_matrix(gradient, self.rank)

    def _get_orthogonal_matrix(self, gradient: torch.Tensor, rank: int) -> torch.Tensor:
        if gradient.data.dtype != torch.float:
            float_data = False
            original_dtype = gradient.data.dtype
            matrix = gradient.data.float()
        else:
            float_data = True
            matrix = gradient.data

        U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)

        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_dtype)
            B = B.to(original_dtype)

        return A, B

    def parameters(self) -> List[Parameter]:
        return []


class SvdManager(CompressorManager):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()

        self.compressor_args = SvdArgs(**kwargs)

    def init_compressors(self, lsp_param_groups: List[Dict[str, torch.Any]], state: Dict[torch.Tensor, torch.Any]):
        for group in lsp_param_groups:
            for param in group["params"]:
                state[param]["compressor"] = SvdCompressor(self.compressor_args.rank, self.compressor_args.side)

    def optimize_compressors(self, lsp_param_groups: List[Dict[str, torch.Any]], state: Dict[torch.Tensor, torch.Any]):
        for group in lsp_param_groups:
            for param in tqdm(group["params"], desc="Optimizing compressors"):
                state[param]["compressor"].optimize(param.grad)

    def should_update(self, steps: int):
        return steps % self.update_interval == 0
