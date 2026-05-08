"""模型评估：ADE / FDE 等指标在训练过程中的在线累积。"""

from __future__ import annotations

import torch

from il.modules.metrics.metrics import Metric

def _last_valid_idx(mask: torch.Tensor) -> torch.Tensor:
    rev = mask.flip(dims=[1]).to(dtype=torch.int64)
    idx = mask.size(1) - 1 - rev.argmax(dim=1)
    idx = idx.masked_fill(mask.to(dtype=torch.int64).sum(dim=1) == 0, 0)
    return idx

class TrajMetrics(Metric):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """pred / target: (B, F, 4) [x, y, heading, v]；mask: (B, F)。"""
        mask_f = mask.to(dtype=pred.dtype)
        valid = mask_f.sum(dim=-1).clamp(min=1)
        pred_xy = pred[..., :2]
        target_xy = target[..., :2]
        disp = torch.norm(pred_xy - target_xy, dim=-1)
        xy_ade = ((disp * mask_f).sum(dim=-1) / valid).mean()
        last_idx = _last_valid_idx(mask)
        xy_fde = disp.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        h_diff = pred[..., 2] - target[..., 2]
        h_err = torch.atan2(torch.sin(h_diff), torch.cos(h_diff)).abs()
        heading_ade = ((h_err * mask_f).sum(dim=-1) / valid).mean()
        heading_fde = h_err.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        spd_err = (pred[..., 3] - target[..., 3]).abs()
        speed_ade = ((spd_err * mask_f).sum(dim=-1) / valid).mean()
        speed_fde = spd_err.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        return {
            "xy_ade": xy_ade.detach(),
            "xy_fde": xy_fde.detach(),
            "heading_ade": heading_ade.detach(),
            "heading_fde": heading_fde.detach(),
            "speed_ade": speed_ade.detach(),
            "speed_fde": speed_fde.detach(),
        }