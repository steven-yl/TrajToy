"""轨迹预测损失：ADE + FDE。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from il.modules.loss.loss import Loss
    
class DiffusionMSELoss(Loss):
    """
    Diffusion MSE Loss
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            pred: (B, F, 4) [x, y, heading, v]
            target: (B, F, 4) [x, y, heading, v]
        Returns:
            (total_loss, {"mse_loss"})
        """        
        mask = mask.to(dtype=pred.dtype).unsqueeze(-1)
        mse_loss = (F.mse_loss(pred, target, reduction="none") * mask).sum() / mask.sum().clamp(min=1)

        return {
            "loss": mse_loss,
        }
