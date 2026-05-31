from __future__ import annotations

import torch
from torch import nn
from diffusers import SchedulerMixin


class DiffusionProcess:
    """正向扩散:对 ``x0`` 加噪并由外部传入的 ``model`` / ``noise_scheduler`` 预测噪声。

    无实例状态;``model`` 与 ``noise_scheduler`` 均在调用时注入。
    """

    @staticmethod
    def q_sample(
        noise_scheduler: SchedulerMixin,
        gt_sample: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向扩散 ``q(x_t | x_0)``:返回 ``(noisy_x0, target_noise)``。"""
        target_noise = torch.randn_like(gt_sample)
        noisy_x0 = noise_scheduler.add_noise(gt_sample, target_noise, timesteps)
        return noisy_x0, target_noise

    @staticmethod
    def target_for_loss(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        gt_sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (gt_sample.shape[0],),
            device=gt_sample.device,
        )
        noisy_x0, target_noise = DiffusionProcess.q_sample(
            noise_scheduler, gt_sample, timesteps,
        )
        pred_noise = model(noisy_x0, timesteps, batch)
        return pred_noise, target_noise


class DiffusionSampler:
    """反向扩散:从纯噪声出发逐步去噪到 ``x0``。

    ``model`` 与 ``noise_scheduler`` 均在调用时注入;采样循环内须复用同一 scheduler 实例。
    """

    @staticmethod
    def step(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 训练时 model 拿到的是 (B,) 形状的 timesteps;采样循环里 ``timestep``
        # 是 0-d 标量,这里扩到 (B,) 与训练分布对齐。
        # diffusers 的 ``scheduler.step`` 仍要求标量 / 0-d,保留原 ``timestep``。
        t_vec = timestep.expand(sample.shape[0])
        model_output = model(sample, t_vec, batch)
        # ``scheduler.step`` 返回 ``SchedulerOutput``,需要取 ``prev_sample``。
        return noise_scheduler.step(model_output, timestep, sample).prev_sample

    @staticmethod
    @torch.no_grad()
    def sample(
        model: nn.Module,
        noise_scheduler: SchedulerMixin,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
        num_inference_steps: int | None = None,
        intermediates_num: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if device is None:
            device = next(model.parameters()).device

        # DDPM 必须走满 num_train_timesteps,DDIM/DPM-Solver 等可显式传更小值加速。
        n_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else noise_scheduler.config.num_train_timesteps
        )
        noise_scheduler.set_timesteps(n_steps, device=device)
        timesteps = noise_scheduler.timesteps

        sample_result = torch.randn(shape, device=device)
        intermediates_sample: list[torch.Tensor] = []
        snap_stride = (
            max(1, len(timesteps) // intermediates_num)
            if intermediates_num > 0
            else 0
        )

        for i, t in enumerate(timesteps):
            sample_result = DiffusionSampler.step(
                model, noise_scheduler, sample_result, t, batch,
            )
            if snap_stride and i % snap_stride == 0:
                intermediates_sample.append(sample_result)

        if intermediates_num > 0:
            return sample_result, intermediates_sample
        return sample_result


class DiffusionPipeline(nn.Module):
    """组合训练用的 :class:`DiffusionProcess` 与采样用的 :class:`DiffusionSampler`。

    ``model`` 与 ``noise_scheduler`` 由本类持有(Hydra 注入);process / sampler 无状态,
    调用时将二者作为参数传入。

    注:diffusers 的 ``SchedulerMixin`` 不是 ``nn.Module``,不会进入 ``state_dict``,
    加载权重后需仍由配置重建 scheduler(与 diffusers Pipeline 惯例一致)。
    """

    def __init__(self, noise_scheduler: SchedulerMixin, model: nn.Module, num_inference_steps: int | None) -> None:
        super().__init__()
        self.noise_scheduler = noise_scheduler
        self.model = model
        # 反向采样步数;``None`` 表示走 ``num_train_timesteps``,
        # DDPM 应保持默认,DDIM/DPM-Solver 等可显式传更小值加速。
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 正向扩散训练:返回 ``(pred_noise, target_noise)``,外部接 loss。
        return DiffusionProcess.target_for_loss(
            self.model, self.noise_scheduler, sample, batch,
        )

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return DiffusionSampler.sample(
            self.model,
            self.noise_scheduler,
            batch,
            shape,
            device=device,
            num_inference_steps=self.num_inference_steps,
        )

    def grid_sample(
        self,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
        intermediates_num: int = 5,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return DiffusionSampler.sample(
            self.model,
            self.noise_scheduler,
            batch,
            shape,
            device=device,
            num_inference_steps=self.num_inference_steps,
            intermediates_num=intermediates_num,
        )
