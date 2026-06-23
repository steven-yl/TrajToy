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


class StatesProcess(object):
    def __init__(self):
        pass
    # ------------------------------------------------------------------
    # heading 角度量在欧氏扩散空间不连续（±π 跳变），用 sin/cos 参与扩散；
    # 速度 v 与 xy 强相关，由后处理从 xy 位移推出，避免冗余建模：
    #   future 4 维 [x, y, heading, v] -> 扩散 4 维 [x, y, sin(h), cos(h)]
    # ------------------------------------------------------------------
    FUTURE_STEP_DT = 0.2  # history_future_dt(0.1s) × future_interval(2)
    @staticmethod
    def _velocity_from_xy(xy: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        """由 xy 路径相邻位移推标量速度 (m/s)，``xy``: [..., F, 2] -> [..., F]。

        与 ``_heading_from_xy`` 同样在前补原点，首点速度为「原点 -> 首点」位移 / dt。
        """
        if dt is None:
            dt = StatesProcess.FUTURE_STEP_DT
        origin = torch.zeros_like(xy[..., :1, :])
        pts = torch.cat([origin, xy], dim=-2)          # [..., F+1, 2]
        diff = pts[..., 1:, :] - pts[..., :-1, :]      # [..., F, 2]
        return torch.linalg.vector_norm(diff, ord=2, dim=-1) / dt

    @staticmethod
    def postprocessXYHV(future3: torch.Tensor) -> torch.Tensor:
        """[..., 4]=[x,y,sin(h),cos(h)] -> [..., 4]=[x,y,heading,v]（v 由 xy 位移推出）。"""
        x = future3[..., 0]
        y = future3[..., 1]
        h = torch.atan2(future3[..., 2], future3[..., 3])
        v = StatesProcess._velocity_from_xy(future3[..., :2])
        return torch.stack([x, y, h, v], dim=-1)

    @staticmethod
    def preprocess(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """把 future 编码为 4 维 [x,y,sin(h),cos(h)] 后再做归一化；其余条件字段照常归一化。

        归一化的 future 字段使用 4 维 mean/std（见 normalizer 配置），
        因此必须在 ``normalizer.apply`` 之前完成 4->4 编码。
        """
        enc = dict(batch)
        future = batch["future"]
        x = future[..., 0]
        y = future[..., 1]
        h = future[..., 2]
        h_sin = torch.sin(h)
        h_cos = torch.cos(h)

        enc["future"] = torch.stack([x, y, h_sin, h_cos], dim=-1)
        return enc # self.normalizer.apply(enc)

    @staticmethod
    def postprocess(sample3: torch.Tensor) -> torch.Tensor:
        """采样输出（归一化 4 维）-> 反归一化 -> 解码为 4 维 [x,y,heading,v]。"""
        return StatesProcess.postprocessXYHV(sample3) # self.normalizer.inverse_future(sample3))

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
        norm_batch = StatesProcess.preprocess(norm_batch)
        pred, target = self.predictor(norm_batch["future"], norm_batch)
        loss = self.loss_fn(pred, target, norm_batch["future_mask"])
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        norm_batch = StatesProcess.preprocess(norm_batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self.normalizer.inverse_future(StatesProcess.postprocess(pred_future))
        x_samples = [self.normalizer.inverse_future(StatesProcess.postprocess(x)) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps})

        out.update({"loss": torch.tensor(0.0, device=pred_future.device)})
        return out

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        norm_batch = StatesProcess.preprocess(norm_batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self.normalizer.inverse_future(StatesProcess.postprocess(pred_future))
        x_samples = [self.normalizer.inverse_future(StatesProcess.postprocess(x)) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps})

        out.update({"loss": torch.tensor(0.0, device=pred_future.device)})
        return out

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self.normalizer.apply(batch)
        norm_batch = StatesProcess.preprocess(norm_batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self.normalizer.inverse_future(StatesProcess.postprocess(pred_future)) # self._postprocess(pred_future)
        x_samples = [self.normalizer.inverse_future(StatesProcess.postprocess(x)) for x in x_samples] # self._postprocess(x) for x in x_samples]
        out = {"pred_future": pred_future}
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps, "batch": batch})
        return out
