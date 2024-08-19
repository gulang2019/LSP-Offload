from abc import ABC
from collections.abc import Iterable
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer

from lspoffload.compressor import CompressorManager, get_manager_cls_by_name


class LspOptimizer(Optimizer, ABC):
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[Dict[str, Any]], defaults: Dict[str, Any]) -> None:
        if "lsp_args" not in defaults or len(defaults["lsp_args"]) == 0:
            defaults["lsp_args"] = {
                "_compressor_name": "count_sketch",
            }

        defaults["compressor_mgr"] = None

        super().__init__(params, defaults)

    @staticmethod
    def _is_lsp_param_group(param_group: Dict[str, Any]) -> bool:
        return "use_lsp" in param_group and param_group["use_lsp"]

    def init_compressors(self):
        if self.defaults["compressor_mgr"] is None:
            lsp_mgr: CompressorManager = get_manager_cls_by_name(self.defaults["lsp_args"].pop("_compressor_name"))(
                **self.defaults["lsp_args"]
            )
            self.defaults["compressor_mgr"] = lsp_mgr
        else:
            lsp_mgr = self.defaults["compressor_mgr"]

        # Filter out the params that are not compressed
        lsp_param_groups = [g for g in self.param_groups if self._is_lsp_param_group(g)]

        lsp_mgr.init_compressors(lsp_param_groups, self.state)

    def update_compressors(self):
        lsp_mgr: Optional[CompressorManager] = self.defaults.get("compressor_mgr", None)

        if lsp_mgr is None:
            self.init_compressors()
            lsp_mgr = self.defaults["compressor_mgr"]

        # Filter out the params that are not compressed
        lsp_param_groups = [g for g in self.param_groups if self._is_lsp_param_group(g)]

        lsp_mgr.optimize_compressors(lsp_param_groups, self.state)

    def should_update(self) -> bool:
        lsp_mgr: Optional[CompressorManager] = self.defaults.get("compressor_mgr", None)

        if lsp_mgr is None:
            return False

        return lsp_mgr.should_update(self.defaults["steps"])
