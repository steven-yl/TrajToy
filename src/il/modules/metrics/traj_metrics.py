"""模型评估：ADE / FDE 等指标在训练过程中的在线累积。"""

from __future__ import annotations

import torch

from il.modules.metrics.metrics import Metrics


def _last_valid_idx(mask: torch.Tensor) -> torch.Tensor:
    rev = mask.flip(dims=[1]).to(dtype=torch.int64)
    idx = mask.size(1) - 1 - rev.argmax(dim=1)
    idx = idx.masked_fill(mask.to(dtype=torch.int64).sum(dim=1) == 0, 0)
    return idx


def _cum_longitudinal_arc_m(target_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """沿 GT 折线从原点到各 future 点的累积弧长（米），(B, F)。无效段不计入增量。"""
    mask_b = mask.bool()
    dt = target_xy.dtype
    b = target_xy[:, 0].norm(dim=-1) * mask_b[:, 0].to(dt)
    if target_xy.size(1) <= 1:
        return b.unsqueeze(1)
    seg = (target_xy[:, 1:] - target_xy[:, :-1]).norm(dim=-1)
    valid_seg = mask_b[:, 1:] & mask_b[:, :-1]
    inc = torch.cat([b.unsqueeze(1), seg * valid_seg.to(dt)], dim=1)
    return torch.cumsum(inc, dim=1)


def _fde_xy_at_arc_threshold(
    disp: torch.Tensor,
    cum_arc_m: torch.Tensor,
    mask: torch.Tensor,
    dist_m: float,
) -> torch.Tensor:
    """GT 累积弧长首次达到 ``dist_m`` 的时刻的 xy 位移误差；达不到的样本不参与平均。"""
    mask_b = mask.bool()
    reach = (cum_arc_m >= float(dist_m)) & mask_b
    eligible = reach.any(dim=1)
    if not bool(eligible.any().item()):
        return disp.new_tensor(0.0)
    idx = reach.to(disp.dtype).argmax(dim=1).unsqueeze(1).long()
    gathered = disp.gather(1, idx).squeeze(1)
    return gathered[eligible].mean()


def _lon_fde_metric_key(dist_m: float) -> str:
    if float(dist_m).is_integer():
        return f"xy_fde_lon_{int(dist_m)}m"
    s = str(dist_m).replace(".", "p")
    return f"xy_fde_lon_{s}m"


class TrajMetrics(Metrics):
    def __init__(self, lon_fde_thresholds_m: tuple[float, ...] | list[float] = (5.0, 15.0)) -> None:
        super().__init__()
        self._lon_fde_thresholds_m = tuple(float(x) for x in lon_fde_thresholds_m)

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

        cum_arc = _cum_longitudinal_arc_m(target_xy, mask)

        h_diff = pred[..., 2] - target[..., 2]
        h_err = torch.atan2(torch.sin(h_diff), torch.cos(h_diff)).abs()
        heading_ade = ((h_err * mask_f).sum(dim=-1) / valid).mean()
        heading_fde = h_err.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        spd_err = (pred[..., 3] - target[..., 3]).abs()
        speed_ade = ((spd_err * mask_f).sum(dim=-1) / valid).mean()
        speed_fde = spd_err.gather(1, last_idx.unsqueeze(1)).squeeze(1).mean()

        out: dict[str, torch.Tensor] = {
            "xy_ade": xy_ade.detach(),
            "xy_fde": xy_fde.detach(),
            "heading_ade": heading_ade.detach(),
            "heading_fde": heading_fde.detach(),
            "speed_ade": speed_ade.detach(),
            "speed_fde": speed_fde.detach(),
        }
        for d in self._lon_fde_thresholds_m:
            out[_lon_fde_metric_key(d)] = _fde_xy_at_arc_threshold(disp, cum_arc, mask, d).detach()
        return out
