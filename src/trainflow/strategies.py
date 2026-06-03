from __future__ import annotations

import warnings
from abc import ABC
from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class Strategy(ABC):
    def setup(self, trainer: Any) -> None:
        return None

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        return metrics

    def reduce_bool_any(self, value: bool) -> bool:
        """Return True if ``value`` is True on any rank (no-op single-device)."""
        return value

    def no_sync_context(self, model: torch.nn.Module):
        return nullcontext()

    @property
    def is_global_zero(self) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0


class SingleDeviceStrategy(Strategy):
    pass


class DDPStrategy(Strategy):
    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not dist.is_available() or not dist.is_initialized():
            return model
        device_ids = None
        if torch.cuda.is_available():
            device_ids = [torch.cuda.current_device()]
        return DistributedDataParallel(model, device_ids=device_ids, find_unused_parameters=False)

    def reduce_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        if not dist.is_available() or not dist.is_initialized():
            return metrics
        world_size = dist.get_world_size()
        reduced: dict[str, float] = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device="cuda" if torch.cuda.is_available() else "cpu")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = float(tensor.item() / world_size)
        return reduced

    def reduce_bool_any(self, value: bool) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor(
            1 if value else 0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())

    def no_sync_context(self, model: torch.nn.Module):
        if isinstance(model, DistributedDataParallel):
            return model.no_sync()
        return nullcontext()


def build_strategy(name: str | None) -> Strategy:
    if name is None or name == "none":
        return SingleDeviceStrategy()
    normalized = name.lower()
    if normalized == "ddp":
        return DDPStrategy()
    if normalized in {"fsdp", "deepspeed"}:
        warnings.warn(
            f"Strategy '{name}' is not implemented yet; falling back to SingleDeviceStrategy.",
            UserWarning,
            stacklevel=2,
        )
        return SingleDeviceStrategy()
    raise ValueError(f"Unsupported strategy: {name}")
