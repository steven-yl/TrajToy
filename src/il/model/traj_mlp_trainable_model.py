"""Trajectory MLP model adapted to TrainFlow contracts."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from trainflow.model import TrainableModel
from il.modules.loss.traj_loss import Loss
from il.modules.metrics.traj_metrics import Metrics

def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    lr_scheduler: str,
    cosine_t_max: int,
    cosine_eta_min: float,
    warmup_steps: int,
    warmup_start_factor: float,
    lr_step_size: int,
    lr_gamma: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """按名称构造学习率调度器；步数均指 TrainFlow 每次 ``optimizer.step()`` 对应的 scheduler step。"""
    kind = lr_scheduler.strip().lower()
    if kind in ("none", "null", ""):
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
        )
    if kind == "cosine_warmup":
        ws = max(0, int(warmup_steps))
        if ws == 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
            )
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(warmup_start_factor),
            end_factor=1.0,
            total_iters=ws,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=cosine_eta_min
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warm, cosine], milestones=[ws]
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(lr_step_size), gamma=float(lr_gamma)
        )
    raise ValueError(
        f"Unknown lr_scheduler {lr_scheduler!r}; "
        "expected none, cosine, cosine_warmup, step."
    )


class TrajMlpTrainableModel(TrainableModel):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Loss,
        metrics_fn: Metrics,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",
        cosine_t_max: int = 10000,
        cosine_eta_min: float = 0.0,
        warmup_steps: int = 0,
        warmup_start_factor: float = 1e-8,
        lr_step_size: int = 30,
        lr_gamma: float = 0.5,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.lr_scheduler_name = str(lr_scheduler)
        self.cosine_t_max = int(cosine_t_max)
        self.cosine_eta_min = float(cosine_eta_min)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_factor = float(warmup_start_factor)
        self.lr_step_size = int(lr_step_size)
        self.lr_gamma = float(lr_gamma)
        self.traj_mlp = model
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = _build_lr_scheduler(
            optimizer,
            lr_scheduler=self.lr_scheduler_name,
            cosine_t_max=self.cosine_t_max,
            cosine_eta_min=self.cosine_eta_min,
            warmup_steps=self.warmup_steps,
            warmup_start_factor=self.warmup_start_factor,
            lr_step_size=self.lr_step_size,
            lr_gamma=self.lr_gamma,
        )
        out: dict[str, Any] = {"optimizer": optimizer}
        if scheduler is not None:
            out["lr_scheduler"] = scheduler
        return out


    def _predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.traj_mlp(
            history=batch["history"],
            history_mask=batch["history_mask"],
            centerline=batch["centerline"],
            centerline_mask=batch["centerline_mask"],
            left_boundary=batch["left_boundary"],
            left_boundary_mask=batch["left_boundary_mask"],
            right_boundary=batch["right_boundary"],
            right_boundary_mask=batch["right_boundary_mask"],
            lane_dividers=batch["lane_dividers"],
            lane_dividers_mask=batch["lane_dividers_mask"],
            max_v=batch["max_v"],
            max_v_mask=batch["max_v_mask"],
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = self._predict(batch)
        return self.loss_fn(pred, batch["future"], batch["future_mask"])

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = self._predict(batch)
        out = self.loss_fn(pred, batch["future"], batch["future_mask"])
        out.update(self.metrics_fn(pred, batch["future"], batch["future_mask"]))
        return out

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = self._predict(batch)
        out = self.loss_fn(pred, batch["future"], batch["future_mask"])
        out.update(self.metrics_fn(pred, batch["future"], batch["future_mask"]))
        return out

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"pred_future": self._predict(batch), "batch": batch}