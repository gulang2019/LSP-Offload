from .compressor import Compressor, CompressorArgs, CompressorManager
from .count_sketch import (
    CountSketchArgs,
    CountSketchCompressor,
    CountSketchManager,
)
from .svd import SvdArgs, SvdCompressor, SvdManager


def get_manager_cls_by_name(name: str) -> CompressorManager:
    if name == "count_sketch":
        return CountSketchManager
    elif name == "svd":
        return SvdManager
    else:
        raise NotImplementedError


__all__ = [
    "CompressorManager",
    "Compressor",
    "CompressorArgs",
    "CountSketchManager",
    "CountSketchArgs",
    "CountSketchCompressor",
    "SvdManager",
    "SvdArgs",
    "SvdCompressor",
    "get_manager_cls_from_args",
]
