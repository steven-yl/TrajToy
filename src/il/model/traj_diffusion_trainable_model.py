"""Trajectory diffusion model adapted to TrainFlow contracts."""

from __future__ import annotations

from typing import Any

import torch

from trainflow.model import TrainableModel
from il.modules.loss.loss import Loss
from il.modules.metrics.metrics import Metrics

from il.modules.utis.lr_scheduler import build_lr_scheduler

class TrajDiffusionModelWrapper(torch.nn.Module):
    """DDPM 包装：训练时随机 t 预测噪声；推理时完整反向马尔可夫链采样。"""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Loss,
        metrics_fn: Metrics,
        diffusion_schedule: str = "linear",
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.diffusion_schedule = str(diffusion_schedule)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)

        betas = self._build_betas(
            self.diffusion_schedule,
            self.diffusion_steps,
            self.beta_start,
            self.beta_end,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([betas.new_tensor([1.0]), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).clamp(
            min=1e-12
        )
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt((1.0 - alphas_cumprod).clamp(min=0.0)),
            persistent=False,
        )

    @staticmethod
    def _build_betas(
        schedule: str,
        num_steps: int,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError(f"`diffusion_steps` must be positive, got {num_steps}.")
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        if schedule == "cosine":
            s = 0.008
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0.0001, 0.9999)
        raise ValueError(f"Unsupported diffusion schedule: {schedule!r}.")

    @staticmethod
    def _as_loss_dict(loss_out: Any) -> dict[str, torch.Tensor]:
        if isinstance(loss_out, dict):
            return loss_out
        return {"loss": loss_out}

    def _prepare_condition(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cond = dict(batch)
        cond.pop("future", None)
        cond.pop("future_mask", None)
        return cond

    def _q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[timesteps]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[timesteps]
        while sqrt_acp.ndim < x0.ndim:
            sqrt_acp = sqrt_acp.unsqueeze(-1)
            sqrt_om = sqrt_om.unsqueeze(-1)
        x_t = sqrt_acp * x0 + sqrt_om * eps
        return x_t, eps

    def _predict_eps(
        self,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # Keep compatibility with different model signatures.
        try:
            return self.model(x_noisy, timesteps, cond)
        except TypeError:
            return self.model(
                x_noisy=x_noisy,
                timestep=timesteps,
                cond=cond,
            )

    def _estimate_x0(
        self,
        x_noisy: torch.Tensor,
        pred_eps: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_acp = self.sqrt_alphas_cumprod[timesteps]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[timesteps]
        while sqrt_acp.ndim < x_noisy.ndim:
            sqrt_acp = sqrt_acp.unsqueeze(-1)
            sqrt_om = sqrt_om.unsqueeze(-1)
        return (x_noisy - sqrt_om * pred_eps) / sqrt_acp.clamp(min=1e-12)

    def _diffusion_forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = batch["future"]
        timesteps = torch.randint(
            0,
            self.diffusion_steps,
            (x0.shape[0],),
            device=x0.device,
        )
        x_noisy, eps = self._q_sample(x0, timesteps)
        pred_eps = self._predict_eps(x_noisy, timesteps, self._prepare_condition(batch))
        return pred_eps, eps, timesteps, x_noisy

    def _p_sample_step(
        self,
        x: torch.Tensor,
        t: int,
        cond: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """DDPM 单步：p(x_{t-1} | x_t)，与训练时的 ε 参数化一致。"""
        b = x.shape[0]
        t_vec = torch.full((b,), t, device=x.device, dtype=torch.long)
        eps = self._predict_eps(x, t_vec, cond)

        beta_t = self.betas[t]
        alpha_t = 1.0 - beta_t
        acp_t = self.alphas_cumprod[t]
        acp_prev = self.alphas_cumprod_prev[t]
        denom = (1.0 - acp_t).clamp(min=1e-12)

        pred_x0 = self._estimate_x0(x, eps, t_vec)
        mean = (acp_prev.sqrt() * beta_t / denom) * pred_x0 + (alpha_t.sqrt() * (1.0 - acp_prev) / denom) * x

        if t == 0:
            return mean
        var = self.posterior_variance[t].clamp(min=1e-20)
        return mean + var.sqrt() * torch.randn_like(x)

    def sample_trajectory(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """从纯噪声开始完整反向采样得到 future 轨迹 (B, L, C)。"""
        cond = self._prepare_condition(batch)
        ref = batch["future"]
        x = torch.randn(ref.shape, device=ref.device, dtype=ref.dtype)
        for t in range(self.diffusion_steps - 1, -1, -1):
            x = self._p_sample_step(x, t, cond)
        return x

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """推理：等价于 ``sample_trajectory``。"""
        return self.sample_trajectory(batch)

    def _compute_noise_loss(
        self,
        pred_eps: torch.Tensor,
        eps: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self._as_loss_dict(self.loss_fn(pred_eps, eps, future_mask))

    def compute_step_output(
        self,
        batch: dict[str, torch.Tensor],
        with_metrics: bool,
    ) -> dict[str, torch.Tensor]:
        pred_eps, eps, timesteps, x_noisy = self._diffusion_forward(batch)
        out = self._compute_noise_loss(pred_eps, eps, batch["future_mask"])
        if with_metrics:
            pred_x0 = self._estimate_x0(x_noisy, pred_eps, timesteps)
            out.update(self.metrics_fn(pred_x0, batch["future"], batch["future_mask"]))
            out.update({"pred_future": pred_x0})
        return out


class TrajDiffusionTrainableModel(TrainableModel):
    def __init__(self,
            predictor: TrajDiffusionModelWrapper,
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
        return self.predictor.compute_step_output(batch, with_metrics=True)

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.predictor.compute_step_output(batch, with_metrics=True)

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.predictor.compute_step_output(batch, with_metrics=True)

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_future = self.predictor(batch)
        return {"pred_future": pred_future, "batch": batch}