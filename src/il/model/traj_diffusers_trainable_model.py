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

    # ------------------------------------------------------------------
    # heading 角度量在欧氏扩散空间不连续（±π 跳变），干脆不让扩散建模 heading：
    #   future 4 维 [x, y, heading, v] -> 扩散 3 维 [x, y, v]
    # heading 在后处理中由预测的 xy 路径切线方向推出（局部车体系下 ego 位于原点、
    # 朝向为 0，故首点 heading 用「原点 -> 首点」方向锚定）。
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_future(future: torch.Tensor) -> torch.Tensor:
        """[..., 4]=[x,y,heading,v] -> [..., 3]=[x,y,v]（丢弃 heading，由后处理还原）。"""
        x = future[..., 0]
        y = future[..., 1]
        v = future[..., 3]
        return torch.stack([x, y, v], dim=-1)

    @staticmethod
    def _heading_from_xy(xy: torch.Tensor) -> torch.Tensor:
        """由 xy 路径的有限差分切线方向推 heading，``xy``: [..., F, 2] -> [..., F]。

        局部车体系下 ego 当前位于原点 (0,0)，朝向为 0，因此在序列前补一个原点，
        使首个 future 点的 heading 用「原点 -> 首点」的位移方向锚定；其余点用相邻
        前向差分。位移近零（停车）时 atan2(0,0)=0，与 ego 朝向一致，可接受。
        """
        origin = torch.zeros_like(xy[..., :1, :])
        pts = torch.cat([origin, xy], dim=-2)          # [..., F+1, 2]
        diff = pts[..., 1:, :] - pts[..., :-1, :]      # [..., F, 2]
        return torch.atan2(diff[..., 1], diff[..., 0])  # [..., F]

    def _decode_future(self, future3: torch.Tensor) -> torch.Tensor:
        """[..., 3]=[x,y,v] -> [..., 4]=[x,y,heading,v]（heading 由 xy 切线推出）。"""
        x = future3[..., 0]
        y = future3[..., 1]
        v = future3[..., 2]
        heading = self._heading_from_xy(future3[..., :2])
        return torch.stack([x, y, heading, v], dim=-1)

    def _prepare(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """把 future 编码为 3 维 [x,y,v] 后再做归一化；其余条件字段照常归一化。

        归一化的 future 字段使用 3 维 mean/std（见 normalizer 配置），
        因此必须在 ``normalizer.apply`` 之前完成 4->3 编码。
        """
        enc = dict(batch)
        enc["future"] = self._encode_future(batch["future"])
        return self.normalizer.apply(enc)

    def _postprocess(self, sample3: torch.Tensor) -> torch.Tensor:
        """采样输出（归一化 3 维）-> 反归一化 -> 解码为 4 维 [x,y,heading,v]。"""
        return self._decode_future(self.normalizer.inverse_future(sample3))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self._prepare(batch)
        pred_noise, target_noise = self.predictor(norm_batch["future"], norm_batch)
        loss = self.loss_fn(pred_noise, target_noise, batch["future_mask"])
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self._prepare(batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self._postprocess(pred_future)
        x_samples = [self._postprocess(x) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps})

        out.update({"loss": torch.tensor(0.0, device=pred_future.device)})
        return out

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self._prepare(batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self._postprocess(pred_future)
        x_samples = [self._postprocess(x) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"pred_future": pred_future}
        out.update(metrics)
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps})

        out.update({"loss": torch.tensor(0.0, device=pred_future.device)})
        return out

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        norm_batch = self._prepare(batch)
        pred_future, x_samples, x_samples_timesteps = self.predictor.grid_sample(
            norm_batch, norm_batch["future"].shape,
        )
        pred_future = self._postprocess(pred_future)
        x_samples = [self._postprocess(x) for x in x_samples]
        out = {"pred_future": pred_future}
        out.update({"x_samples": x_samples, "x_samples_timesteps": x_samples_timesteps, "batch": batch})
        return out
