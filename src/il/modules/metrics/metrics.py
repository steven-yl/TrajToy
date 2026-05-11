"""轨迹评估指标：TorchMetrics 风格的 Metrics 基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


# ── TorchMetrics 风格基类 ────────────────────────────────────────────


class Metrics(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        """
        Args:
            *args: 
            **kwargs: 
        Returns:
            dict[str, torch.Tensor]: 
        """
        pass