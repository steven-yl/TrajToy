"""Declarative metric aggregation.

Models call ``self.log(name, value, on_step=..., on_epoch=...)`` inside their step methods; the
Trainer owns a :class:`MetricCollector` per stage that accumulates values and produces:

- step metrics: emitted immediately (subject to ``log_every_n_steps``).
- epoch metrics: aggregated with ``reduce_fx`` (default mean) over the epoch, then reduced across
  ranks once at epoch end.

This replaces ad-hoc mutation of ``trainer.current_metrics`` and gives step/epoch scopes a single,
well-defined meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class _Entry:
    values: list[float] = field(default_factory=list)
    reduce_fx: str = "mean"
    on_step: bool = False
    on_epoch: bool = True
    prog_bar: bool = False
    last: float = 0.0


def _reduce(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    if how == "mean":
        return float(np.mean(values))
    if how == "sum":
        return float(np.sum(values))
    if how == "max":
        return float(np.max(values))
    if how == "min":
        return float(np.min(values))
    if how == "last":
        return float(values[-1])
    raise ValueError(f"Unknown reduce_fx {how!r}; expected mean/sum/max/min/last.")


class MetricCollector:
    """Accumulates ``self.log(...)`` calls for one stage (train/val/test)."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._entries: dict[str, _Entry] = {}

    def log(
        self,
        name: str,
        value: float,
        *,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: str = "mean",
        prog_bar: bool = False,
    ) -> None:
        entry = self._entries.get(name)
        if entry is None:
            entry = _Entry(reduce_fx=reduce_fx, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)
            self._entries[name] = entry
        entry.values.append(float(value))
        entry.last = float(value)
        entry.on_step = on_step
        entry.on_epoch = on_epoch
        entry.reduce_fx = reduce_fx
        entry.prog_bar = prog_bar

    def step_metrics(self) -> dict[str, float]:
        """Most-recent value for entries flagged ``on_step``, namespaced with the stage prefix."""
        out: dict[str, float] = {}
        for name, entry in self._entries.items():
            if entry.on_step:
                out[f"{self.prefix}/{name}"] = entry.last
        return out

    def epoch_metrics(self) -> dict[str, float]:
        """Reduced value over the epoch for entries flagged ``on_epoch``."""
        out: dict[str, float] = {}
        for name, entry in self._entries.items():
            if entry.on_epoch and entry.values:
                out[f"{self.prefix}/{name}"] = _reduce(entry.values, entry.reduce_fx)
        return out

    def prog_bar_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, entry in self._entries.items():
            if entry.prog_bar:
                out[name] = entry.last
        return out

    def reset(self) -> None:
        self._entries.clear()

    def __bool__(self) -> bool:
        return bool(self._entries)
