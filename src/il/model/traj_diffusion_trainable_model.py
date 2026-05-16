"""Trajectory diffusion model adapted to TrainFlow contracts."""

from __future__ import annotations

from typing import Any

import torch

from trainflow.model import TrainableModel
from il.modules.loss.loss import Loss
from il.modules.metrics.metrics import Metrics
from il.modules.utis.normalizer import Normalizer
from il.modules.utis.lr_scheduler import build_lr_scheduler
import logging

class TrajDiffusionModelWrapper(torch.nn.Module):
    """DDPM 包装：训练时随机 t 预测噪声；推理时完整反向马尔可夫链采样。"""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Loss,
        diffusion_schedule: str = "linear",
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        df_type: str = "ddpm_eps",
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.diffusion_schedule = str(diffusion_schedule)
        self.diffusion_steps = int(diffusion_steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.df_type = df_type
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
        eta: float = 0.5,
        df_type: str = "ddpm_eps",  # "ddpm_eps", "ddpm_x0", "ddim", "ddim_rand"
        t_next: int | None = None,
    ) -> torch.Tensor:
        """单步反向：从调度下标 ``t`` 去噪到更干净的下标 ``t_next``（``t_next < t``，DDPM 连续步传 ``t-1``）。"""
        b = x.shape[0]
        t_vec = torch.full((b,), t, device=x.device, dtype=torch.long)
        eps = self._predict_eps(x, t_vec, cond)

        beta_t = self.betas[t]
        alpha_t = 1.0 - beta_t
        acp_t = self.alphas_cumprod[t]
        if t_next is None:
            t_next = t - 1
        if t_next < 0:
            acp_prev = self.alphas_cumprod_prev[0]
        else:
            acp_prev = self.alphas_cumprod[t_next]
        denom = (1.0 - acp_t).clamp(min=1e-12)

        pred_x0 = self._estimate_x0(x, eps, t_vec)

        mean = None
        if df_type == "ddpm_eps" or df_type == "ddpm_x0":
            if df_type == "ddpm_x0":
                # DDPM 公式 predict by x0
                mean = (acp_prev.sqrt() * beta_t / denom) * pred_x0 + (alpha_t.sqrt() * (1.0 - acp_prev) / denom) * x
            elif df_type == "ddpm_eps":
                # DDPM 公式 predict by eps
                mean = 1 / alpha_t.sqrt() * (x - beta_t / denom.sqrt() * eps)
            else:
                raise ValueError(f"Unsupported df_type: {df_type!r}.")
            if t == 0:
                return mean
            var = self.posterior_variance[t].clamp(min=1e-20)
            return mean + var.sqrt() * torch.randn_like(x)
        if df_type == "ddim" or df_type == "ddim_rand":
            if df_type == "ddim":
                # DDIM 确定性采样 (η=0): x_{t-1} = √ᾱ_{t-1} * x̂₀ + √(1-ᾱ_{t-1}) * ε_θ
                x = acp_prev.sqrt() * pred_x0 + (1.0 - acp_prev).clamp(min=0.0).sqrt() * eps
            elif df_type == "ddim_rand":
                # DDIM 随机 (η>0): σ_t = η * √((1-ᾱ_prev)/(1-ᾱ_t)) * √(1 - ᾱ_t/ᾱ_prev)
                acp_prev_safe = acp_prev.clamp(min=1e-12)
                sigma_t = (
                    eta
                    * ((1.0 - acp_prev) / denom).clamp(min=0.0).sqrt()
                    * (1.0 - (acp_t / acp_prev_safe).clamp(max=1.0)).clamp(min=0.0).sqrt()
                )
                direction_coeff = (1.0 - acp_prev - sigma_t ** 2).clamp(min=0.0).sqrt()
                x = acp_prev.sqrt() * pred_x0 + direction_coeff * eps + sigma_t * torch.randn_like(x)
            else:
                raise ValueError(f"Unsupported df_type: {df_type!r}.")
            return x
        raise ValueError(f"Unsupported df_type: {df_type!r}.")

    def sample_trajectory(self, batch: dict[str, torch.Tensor], sample_num: int = 0) -> torch.Tensor:
        """从纯噪声开始完整反向采样得到 future 轨迹 (B, L, C)。"""
        cond = self._prepare_condition(batch)
        ref = batch["future"]
        x = torch.randn(ref.shape, device=ref.device, dtype=ref.dtype)
        sample_steps = self.diffusion_steps // sample_num if sample_num > 0 else None
        x_samples = []
        if self.df_type == "ddim_rand" or self.df_type == "ddim":
            stride = max(1, self.diffusion_steps // 10)
            t = self.diffusion_steps - 1
            while t >= 0:
                t_next = max(0, t - stride)
                x = self._p_sample_step(x, t, cond, df_type=self.df_type, t_next=t_next)
                if sample_steps is not None and t % sample_steps == 0:
                    x_samples.append(x.clone())
                if t_next == 0:
                    break
                t = t_next
        else:
            for t in range(self.diffusion_steps - 1, -1, -1):
                t_next = t - 1
                x = self._p_sample_step(x, t, cond, df_type=self.df_type, t_next=t_next)
                if sample_steps is not None and t % sample_steps == 0:
                    x_samples.append(x.clone())
        return x, x_samples

    def forward(self, batch: dict[str, torch.Tensor], sample_num: int = 3) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """推理：等价于 ``sample_trajectory``。"""
        return self.sample_trajectory(batch, sample_num=sample_num)

    def compute_noise_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_eps, eps, timesteps, x_noisy = self._diffusion_forward(batch)
        loss = self._as_loss_dict(self.loss_fn(pred_eps, eps, batch["future_mask"]))
        return loss, x_noisy, pred_eps, timesteps

    def compute_x0(
        self,
        x_noisy: torch.Tensor,
        pred_eps: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self._estimate_x0(x_noisy, pred_eps, timesteps)
    # 顺序采样过程
    
    def order_diffusion_forward(
        self,
        future: torch.Tensor,
        sample_num: int = 1,
    ) -> list[torch.Tensor]:
        x0 = future.clone()
        step = self.diffusion_steps // sample_num if sample_num > 0 else 1
        timesteps = torch.arange(0, self.diffusion_steps, step=step, device=x0.device)
        timesteps = timesteps.unsqueeze(0).repeat(x0.shape[0], 1)  # (B, T)

        # 对于每个 timestep（列），针对所有样本（行）采样
        x_noisy_list = []
        eps_list = []
        for i in range(timesteps.shape[1]):
            t = timesteps[:, i]
            x_noisy, eps = self._q_sample(x0, t)
            x_noisy_list.append(x_noisy)
            eps_list.append(eps)
        # 堆叠输出: [(B, ...)] -> (T, B, ...)
        x_noisy = torch.stack(x_noisy_list, dim=1)  # (T, B, ...)
        eps = torch.stack(eps_list, dim=1)          # (T, B, ...)
        # x_noisy = x_noisy.transpose(0, 1)           # (B, T, ...)
        # eps = eps.transpose(0, 1)                   # (B, T, ...)
        return x_noisy, eps, timesteps

class TrajDiffusionTrainableModel(TrainableModel):
    def __init__(self,
            predictor: TrajDiffusionModelWrapper,
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
        loss, x_noisy, pred_eps, timesteps = self.predictor.compute_noise_loss(self.normalizer.apply(batch))
        out = loss

        pred_future = self.predictor.compute_x0(x_noisy, pred_eps, timesteps)
        pred_future = self.normalizer.inverse_future(pred_future)
        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out.update({"pred_future": pred_future})
        out.update(metrics)
        return out

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # loss, x_noisy, pred_eps, timesteps = self.predictor.compute_noise_loss(self.normalizer.apply(batch))
        # pred_future = self.predictor.compute_x0(x_noisy, pred_eps, timesteps)
        # pred_future = self.normalizer.inverse_future(pred_future)
        # metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        # loss.update({"pred_future": pred_future})
        # loss.update(metrics)
        # return loss

        pred_future, x_samples = self.predictor(self.normalizer.apply(batch), sample_num=3)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"loss": torch.tensor(0.0, device=pred_future.device)}
        out.update(metrics)
        out.update({"pred_future": pred_future, "x_samples": x_samples})
        return out

    def test_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # loss, x_noisy, pred_eps, timesteps = self.predictor.compute_noise_loss(self.normalizer.apply(batch))
        # pred_future = self.predictor.compute_x0(x_noisy, pred_eps, timesteps)
        # pred_future = self.normalizer.inverse_future(pred_future)
        # metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        # loss.update({"pred_future": pred_future})
        # loss.update(metrics)
        # return loss

        pred_future, x_samples = self.predictor(self.normalizer.apply(batch), sample_num=3)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]

        metrics = self.metrics_fn(pred_future, batch["future"], batch["future_mask"])
        out = {"loss": torch.tensor(0.0, device=pred_future.device)}
        out.update(metrics)
        out.update({"pred_future": pred_future, "x_samples": x_samples})
        return out

    def predict_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred_future, x_samples = self.predictor(self.normalizer.apply(batch), sample_num=3)
        pred_future = self.normalizer.inverse_future(pred_future)
        x_samples = [self.normalizer.inverse_future(x) for x in x_samples]
        return {"pred_future": pred_future, "x_samples": x_samples, "batch": batch}