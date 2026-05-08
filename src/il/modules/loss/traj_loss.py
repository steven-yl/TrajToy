"""轨迹预测损失：ADE + FDE。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from il.modules.loss.loss import Loss

def _last_valid_idx(mask: torch.Tensor) -> torch.Tensor:
    rev = mask.flip(dims=[1]).to(dtype=torch.int64)
    idx = mask.size(1) - 1 - rev.argmax(dim=1)
    idx = idx.masked_fill(mask.to(dtype=torch.int64).sum(dim=1) == 0, 0)
    return idx
    
class TrajLoss(Loss):
    """ADE + FDE 损失，支持 mask 和 huber / mse 两种回归损失。"""

    def __init__(self, cfg_ade_weight: float, cfg_fde_weight: float) -> None:
        super().__init__()
        self._cfg_ade_weight = float(cfg_ade_weight)
        self._cfg_fde_weight = float(cfg_fde_weight)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            pred / target: (B, F, 4) [x, y, heading, v]
            mask: (B, F)
        Returns:
            (total_loss, {"xy_ade", "xy_fde", "heading_ade", "heading_fde", "speed_ade", "speed_fde", ...})
        """
        # 兼容 bool/float mask：统一转为浮点用于加权计算。
        mask_f = mask.to(dtype=pred.dtype)

        # ADE / FDE 仅基于 xy 位移
        pred_xy = pred[..., :2]
        target_xy = target[..., :2]

        disp = torch.norm(pred_xy - target_xy, dim=-1)  # (B, F)
        valid = mask_f.sum(dim=-1).clamp(min=1)

        xy_ade = ((disp * mask_f).sum(dim=-1) / valid).mean()

        last_idx = _last_valid_idx(mask)
        xy_fde = disp.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        _loss = lambda p, t: F.mse_loss(p, t, reduction="none")

        pw_xy = _loss(pred[..., :2], target[..., :2]).sum(-1)       # (B, F)
        pw_h = _loss(pred[..., 2:3], target[..., 2:3]).squeeze(-1)  # (B, F)
        pw_v = _loss(pred[..., 3:4], target[..., 3:4]).squeeze(-1)  # (B, F)
        pw = pw_xy + pw_h + pw_v

        reg = ((pw * mask_f).sum(-1) / valid).mean()
        loss = self._cfg_ade_weight * reg + self._cfg_fde_weight * xy_fde

        return {
            "loss": loss,
            "xy_ade": xy_ade.detach(),
            "xy_fde": xy_fde.detach(),
            "reg_loss": reg.detach(),
        }
