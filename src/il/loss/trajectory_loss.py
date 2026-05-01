"""轨迹预测损失：ADE + FDE。"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig

import torch.nn.functional as F


class TrajectoryLoss(nn.Module):
    """ADE + FDE 损失，支持 mask 和 huber / mse 两种回归损失。"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        lc = cfg.loss
        self._ade_w = lc.ade_weight
        self._fde_w = lc.fde_weight
        self._loss_type = lc.loss_type
        self._huber_delta = lc.huber_delta

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

        last_idx = self._last_valid_idx(mask)
        xy_fde = disp.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        # 回归损失分头计算
        if self._loss_type == "huber":
            _loss = lambda p, t: F.huber_loss(p, t, reduction="none", delta=self._huber_delta)
        else:
            _loss = lambda p, t: F.mse_loss(p, t, reduction="none")

        pw_xy = _loss(pred[..., :2], target[..., :2]).sum(-1)       # (B, F)
        pw_h = _loss(pred[..., 2:3], target[..., 2:3]).squeeze(-1)  # (B, F)
        pw_v = _loss(pred[..., 3:4], target[..., 3:4]).squeeze(-1)  # (B, F)
        pw = pw_xy + pw_h + pw_v

        reg = ((pw * mask_f).sum(-1) / valid).mean()
        total = self._ade_w * reg + self._fde_w * xy_fde

        # 航向误差按角度环绕计算到 [-pi, pi] 后再取绝对值。
        heading_abs = ((pred[..., 2] - target[..., 2] + torch.pi) % (2 * torch.pi) - torch.pi).abs()  # (B, F)
        speed_abs = (pred[..., 3] - target[..., 3]).abs()    # (B, F)
        heading_ade = ((heading_abs * mask_f).sum(dim=-1) / valid).mean()
        speed_ade = ((speed_abs * mask_f).sum(dim=-1) / valid).mean()
        heading_fde = heading_abs.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()
        speed_fde = speed_abs.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        return total, {
            # backward-compatible aliases
            "ade": xy_ade.detach(),
            "fde": xy_fde.detach(),
            "heading_error": heading_ade.detach(),
            "speed_error": speed_ade.detach(),
            # detailed metrics
            "xy_ade": xy_ade.detach(),
            "xy_fde": xy_fde.detach(),
            "heading_ade": heading_ade.detach(),
            "heading_fde": heading_fde.detach(),
            "speed_ade": speed_ade.detach(),
            "speed_fde": speed_fde.detach(),
            "regression": reg.detach(), "total": total.detach(),
        }

    @staticmethod
    def _last_valid_idx(mask: torch.Tensor) -> torch.Tensor:
        rev = mask.flip(dims=[1]).to(dtype=torch.int64)
        idx = mask.size(1) - 1 - rev.argmax(dim=1)
        idx = idx.masked_fill(mask.to(dtype=torch.int64).sum(dim=1) == 0, 0)
        return idx
