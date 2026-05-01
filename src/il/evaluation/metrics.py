"""模型评估：ADE / FDE 指标计算 & 轨迹预测。"""

from __future__ import annotations

import numpy as np
import torch

from omegaconf import DictConfig

from il.model import TrajectoryPredictor
from il.loss import TrajectoryLoss


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    """将 batch 中道路 + 历史 + 未来相关 key 搬到 device。"""
    keys = [
        "history", "history_mask",
        "centerline", "centerline_mask",
        "left_boundary", "left_boundary_mask",
        "right_boundary", "right_boundary_mask",
        "lane_dividers", "lane_dividers_mask",
        "max_v", "max_v_mask",
        "future", "future_mask",
    ]
    return {k: batch[k].to(device) for k in keys if k in batch}


def _model_forward(model: TrajectoryPredictor, b: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        b["history"], b["history_mask"],
        b["centerline"], b["centerline_mask"],
        b["left_boundary"], b["left_boundary_mask"],
        b["right_boundary"], b["right_boundary_mask"],
        b["lane_dividers"], b["lane_dividers_mask"],
        b["max_v"], b["max_v_mask"],
    )


@torch.no_grad()
def compute_metrics(
    model: TrajectoryPredictor,
    dataloader: torch.utils.data.DataLoader,
    cfg: DictConfig,
) -> dict[str, float]:
    """在数据集上计算 xy/heading/speed 的 ADE/FDE 指标。"""
    device = torch.device(cfg.device)
    model.eval()
    loss_fn = TrajectoryLoss(cfg)

    total_xy_ade = total_xy_fde = 0.0
    total_heading_ade = total_heading_fde = 0.0
    total_speed_ade = total_speed_fde = 0.0
    count = 0

    for batch in dataloader:
        b = _batch_to_device(batch, device)
        pred = _model_forward(model, b)
        _, comp = loss_fn(pred, b["future"], b["future_mask"])
        bs = b["history"].size(0)
        total_xy_ade += comp["xy_ade"].item() * bs
        total_xy_fde += comp["xy_fde"].item() * bs
        total_heading_ade += comp["heading_ade"].item() * bs
        total_heading_fde += comp["heading_fde"].item() * bs
        total_speed_ade += comp["speed_ade"].item() * bs
        total_speed_fde += comp["speed_fde"].item() * bs
        count += bs

    if count == 0:
        return {
            "ade": 0.0, "fde": 0.0,  # backward-compatible aliases
            "xy_ade": 0.0, "xy_fde": 0.0,
            "heading_ade": 0.0, "heading_fde": 0.0,
            "speed_ade": 0.0, "speed_fde": 0.0,
            "count": 0,
        }
    xy_ade = total_xy_ade / count
    xy_fde = total_xy_fde / count
    return {
        "ade": xy_ade, "fde": xy_fde,  # backward-compatible aliases
        "xy_ade": xy_ade, "xy_fde": xy_fde,
        "heading_ade": total_heading_ade / count,
        "heading_fde": total_heading_fde / count,
        "speed_ade": total_speed_ade / count,
        "speed_fde": total_speed_fde / count,
        "count": count,
    }


@torch.no_grad()
def predict_trajectory(
    model: TrajectoryPredictor,
    batch: dict[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """对单个 batch 预测轨迹，返回 (B, F, 4) numpy [x, y, heading, v]。"""
    device = torch.device(device)
    model.eval()
    b = _batch_to_device(batch, device)
    return _model_forward(model, b).cpu().numpy()
