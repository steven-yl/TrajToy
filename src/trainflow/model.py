from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class TrainableModel(ABC, nn.Module):
    """Base model contract for TrainFlow."""

    @abstractmethod
    def forward(self, batch: Any) -> Any:
        """Return either loss tensor or a dict including `loss`."""

    @abstractmethod
    def configure_optimizers(self) -> Any:
        """Return optimizer or structured optimizer/scheduler config."""

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
