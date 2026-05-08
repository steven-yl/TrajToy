# 实现loss的基类，基于ABC
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Args:
            *args: 
            **kwargs: 
        Returns:
            torch.Tensor: 
        """
        pass
