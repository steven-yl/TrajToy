"""Trajectory MLP model adapted to TrainFlow contracts."""

from __future__ import annotations

import torch

from trainflow.model import TrainableModel
from il.modules.loss.traj_loss import TrajLoss
from il.modules.metrics.traj_metrics import TrajMetrics
from il.modules.model.mlp.mlp_trajectory_predictor import MLPTrajectoryPredictor


class TrajMLP(TrainableModel):
    def __init__(
        self,
        traj_predictor: MLPTrajectoryPredictor,
        loss_fn: TrajLoss,
        metrics_fn: TrajMetrics,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.traj_mlp = traj_predictor
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

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
        return {"pred": self._predict(batch), "batch": batch}