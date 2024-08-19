from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


class Compressor(ABC):
    @abstractmethod
    def compress(self, full_rank_grad: torch.Tensor):
        pass

    @abstractmethod
    def decompress(self, low_rank_grad: torch.Tensor):
        pass

    @abstractmethod
    def parameters(self) -> List[torch.nn.Parameter]:
        return []


@dataclass
class CompressorArgs(ABC):
    _compressor_name: str
    # left_rank: int
    # right_rank: int
    pass


class CompressorManager(ABC):
    @abstractmethod
    def init_compressors(self, lsp_param_groups: List[Dict[str, Any]], state: Dict[torch.Tensor, Any]):
        pass

    @abstractmethod
    def optimize_compressors(self, lsp_param_groups: List[Dict[str, Any]], state: Dict[torch.Tensor, Any]):
        pass

    @abstractmethod
    def should_update(self, steps: int):
        pass
