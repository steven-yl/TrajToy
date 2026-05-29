"""Trajectory diffusion model adapted to TrainFlow contracts."""

from __future__ import annotations

from typing import Any

import torch

from trainflow.model import TrainableModel
from il.modules.loss.loss import Loss
from il.modules.metrics.metrics import Metrics
from il.modules.utis.normalizer import Normalizer
from il.modules.utis.lr_scheduler import build_lr_scheduler
from il.modules.utis.diffusion_core import DiffusionPipeline
import logging


class TrajDiffusionTrainableModel(TrainableModel):
    def __init__(self,
            predictor: DiffusionPipeline,
            loss_fn: Loss,
            metrics_fn: Metrics,
            normalizer: Normalizer,
            lr: float = 1.0e-3,
            weight_decay: float = 1.0e-4,
            lr_scheduler: str = "cosine_warmup",
            cosine_t_max: int = 40000,
            cosine_eta_min: float = 1.0e-5,
            warmup_steps: int = 500,
            warmup_start_factor: float = 1.0e-8,
            lr_step_size: int = 30,
            lr_gamma: float = 0.5,
        ) -> None:
        super().__init__()
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.normalizer = normalizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.cosine_t_max = cosine_t_max
        self.cosine_eta_min = cosine_eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_factor = warmup_start_factor
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            lr_scheduler=self.lr_scheduler,
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

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        pred_noise, target_noise = self.predictor(norm_batch["future"], norm_batch)
        loss = self.loss_fn(pred_noise, target_noise,  batch["future_mask"])
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        # pred_future = self.predictor.sample(norm_batch, norm_batch["future"].shape)
        # pred_future = self.normalizer.inverse_future(pred_future)
        # metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        # out = {"pred_future": pred_future}
        # out.update(metrics)
        
        pred_future, x_samples = self.predictor.grid_sample(norm_batch, norm_batch["future"].shape)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples})

        out.update({"loss": metrics["xy_ade"]})
        return out

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        # pred_future = self.predictor.sample(norm_batch, norm_batch["future"].shape)
        # pred_future = self.normalizer.inverse_future(pred_future)
        # metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        # out = {"pred_future": pred_future}
        # out.update(metrics)
        
        pred_future, x_samples = self.predictor.grid_sample(norm_batch, norm_batch["future"].shape)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]
        
        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples})

        out.update({"loss": metrics["xy_ade"]})
        return out

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        # pred_future = self.predictor.sample(norm_batch, norm_batch["future"].shape)
        # pred_future = self.normalizer.inverse_future(pred_future)
        # out = {"pred_future": pred_future}
        
        pred_future, x_samples = self.predictor.grid_sample(norm_batch, norm_batch["future"].shape)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]
        out = {"pred_future": pred_future}
        out.update({"x_samples": x_samples, "batch": batch})
        return out
       
       