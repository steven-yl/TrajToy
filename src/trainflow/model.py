from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from .trainer import Trainer


class TrainableModel(ABC, nn.Module):
    """Base model contract for TrainFlow."""

    # Set by the Trainer during fit/validate/test so ``self.log`` can route to the active collector.
    _trainer: "Trainer | None" = None

    @abstractmethod
    def forward(self, batch: Any) -> Any:
        """Return either loss tensor or a dict including `loss`."""

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """Return optimizer or structured optimizer/scheduler config."""

    def log(
        self,
        name: str,
        value: Any,
        *,
        on_step: bool = False,
        on_epoch: bool = True,
        reduce_fx: str = "mean",
        prog_bar: bool = False,
    ) -> None:
        """Declaratively record a metric from inside a step method.

        The Trainer aggregates ``on_epoch`` metrics over the epoch and reduces them across ranks
        once at epoch end; ``on_step`` metrics are emitted immediately (subject to
        ``log_every_n_steps``). No-op if called outside a Trainer-driven loop.
        """
        trainer = self._trainer
        if trainer is None:
            return
        import torch

        if isinstance(value, torch.Tensor):
            value = float(value.detach().item())
        trainer._collect_metric(
            name,
            float(value),
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            prog_bar=prog_bar,
        )

    def training_step(self, batch: Any) -> Any:
        return self.forward(batch)

    def validation_step(self, batch: Any) -> Any:
        return self.forward(batch)

    def test_step(self, batch: Any) -> Any:
        return self.forward(batch)

    def predict_step(self, batch: Any) -> Any:
        return self.forward(batch)

    def on_before_backward(self, loss: Any) -> None:
        return None

    def on_after_backward(self) -> None:
        return None
