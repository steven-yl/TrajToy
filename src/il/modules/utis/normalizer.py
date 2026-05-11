"""仿射归一化

``BatchNormalizer``：整 batch 字典一次性变换。
``Normalizer``：按字段对张量做 ``(x-mean)/std``，末维为特征维 ``C``，支持单帧 ``(C,)``、序列 ``(T, C)``、
或任意 ``(..., C)``，便于在 ``Dataset.__getitem__`` 中逐样本使用。
"""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import torch

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

    def __init__(self, parameters: dict[str, dict[str, Any]]) -> None:
        self._parameters = {
            k: {"mean": torch.as_tensor((par["mean"]), dtype=torch.float32), "std": torch.as_tensor((par["std"]), dtype=torch.float32)}
            for k, par in parameters.items()
        }

    def normalize(self, batch: dict) -> dict:
        out = copy(batch)
        for key, par in self._parameters.items():
            if key in out:
                out[key] = _affine(out[key], par["mean"], par["std"], inverse=False)
        return out

    def inverse(self, batch: dict) -> dict:
        out = copy(batch)
        for key, par in self._parameters.items():
            if key in out:
                out[key] = _affine(out[key], par["mean"], par["std"], inverse=True)
        return out

    def inverse_tensor(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if key not in self._parameters:
            return x
        par = self._parameters[key]
        return _affine(x, par["mean"], par["std"], inverse=True)

    def inverse_future(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_tensor("future", x)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {k: {kk: vv.detach().cpu().tolist() for kk, vv in v.items()} for k, v in self._parameters.items()}

def _as_torch(x: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, bool]:
    if isinstance(x, torch.Tensor):
        return x, False
    return torch.as_tensor(np.asarray(x), dtype=torch.float32), True


def _restore_type(y: torch.Tensor, *, was_numpy: bool, like: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    if was_numpy:
        return y.detach().cpu().numpy().astype(np.asarray(like).dtype, copy=False)
    return y


class Normalizer:
    """单样本、按字段的仿射归一化；``x`` 形状为 ``(..., C)``，``C`` 与 ``stats[key]['mean']`` 长度一致。"""

    def __init__(self, parameters: dict[str, dict[str, Any]]) -> None:
        self._parameters = {
            k: {"mean": torch.as_tensor((par["mean"]), dtype=torch.float32), "std": torch.as_tensor((par["std"]), dtype=torch.float32)}
            for k, par in parameters.items()
        }

    def normalize(self, key: str, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if key not in self._parameters:
            return x
        t, was_numpy = _as_torch(x)
        par = self._parameters[key]
        y = _affine(t, par["mean"], par["std"], inverse=False)
        return _restore_type(y, was_numpy=was_numpy, like=x)

    def inverse(self, key: str, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if key not in self._parameters:
            return x
        t, was_numpy = _as_torch(x)
        par = self._parameters[key]
        y = _affine(t, par["mean"], par["std"], inverse=True)
        return _restore_type(y, was_numpy=was_numpy, like=x)

    def apply(self, sample: dict[str, Any], *, inverse: bool = False) -> dict[str, Any]:
        """对 ``sample`` 中同时出现在 ``stats`` 里的键做归一化或反归一化；其它键原样拷贝。"""
        out = copy(sample)
        fn = self.inverse if inverse else self.normalize
        for key in self._parameters:
            if key in out and out[key] is not None:
                out[key] = fn(key, out[key])
        return out

    def inverse_tensor(self, key: str, x: torch.Tensor) -> torch.Tensor:
        out = self.inverse(key, x)
        if not isinstance(out, torch.Tensor):
            return torch.as_tensor(out, dtype=torch.float32)
        return out

    def inverse_future(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_tensor("future", x)