"""Batch 仿射归一化：与 ``cfg.data.normalization`` 扁平结构一致（各键含 ``mean`` / ``std`` 列表）。"""

from __future__ import annotations

from copy import copy
from typing import Any, Mapping, Union

import torch
from omegaconf import DictConfig, OmegaConf


def _cfg_dict(cfg: Union[DictConfig, Mapping[str, Any]]) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else dict(cfg)


def _padding_mask(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    c = mean.numel()
    if x.ndim >= 2 and x.shape[-1] == c:
        return x.abs().sum(dim=-1) == 0
    if c == 1:
        return x == 0
    return torch.zeros(x.shape[:-1] if x.ndim > 1 else x.shape, dtype=torch.bool, device=x.device)


def _expand_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if mask.shape == ref.shape:
        return mask
    if mask.ndim < ref.ndim:
        m = mask
        while m.ndim < ref.ndim:
            m = m.unsqueeze(-1)
        return m.expand_as(ref)
    return mask


def _affine(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, inverse: bool) -> torch.Tensor:
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    mask = _padding_mask(x, mean)
    y = x * std + mean if inverse else (x - mean) / std
    return y.masked_fill(_expand_mask(mask, y), 0)


class BatchNormalizer:
    """读取 ``data/normalization`` 下每个 batch 字段的 ``mean`` / ``std``，做 ``(x-mean)/std`` 及反变换。"""

    def __init__(self, stats: dict[str, dict[str, torch.Tensor]]) -> None:
        self._stats = stats

    @classmethod
    def from_config(cls, cfg: Union[DictConfig, Mapping[str, Any]]) -> BatchNormalizer:
        stats: dict[str, dict[str, torch.Tensor]] = {}
        for name, block in _cfg_dict(cfg).items():
            if isinstance(block, Mapping) and "mean" in block and "std" in block:
                stats[name] = {
                    "mean": torch.tensor(block["mean"], dtype=torch.float32),
                    "std": torch.tensor(block["std"], dtype=torch.float32),
                }
        return cls(stats)

    def normalize(self, batch: dict) -> dict:
        out = copy(batch)
        for key, par in self._stats.items():
            if key in out:
                out[key] = _affine(out[key], par["mean"], par["std"], inverse=False)
        return out

    def inverse(self, batch: dict) -> dict:
        out = copy(batch)
        for key, par in self._stats.items():
            if key in out:
                out[key] = _affine(out[key], par["mean"], par["std"], inverse=True)
        return out

    def inverse_tensor(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if key not in self._stats:
            return x
        par = self._stats[key]
        return _affine(x, par["mean"], par["std"], inverse=True)

    def inverse_future(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_tensor("future", x)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {k: {kk: vv.detach().cpu().tolist() for kk, vv in v.items()} for k, v in self._stats.items()}
