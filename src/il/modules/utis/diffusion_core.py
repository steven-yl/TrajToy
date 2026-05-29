from __future__ import annotations

import torch
from torch import nn
from diffusers import SchedulerMixin


class DiffusionProcess:
    """正向扩散:对 ``x0`` 加噪并由外部传入的 ``model`` 预测噪声。

    本类不持有 ``model``,使其只负责扩散数学与调度,可被复用在多个网络上。
    """

    def __init__(self, noise_scheduler: SchedulerMixin) -> None:
        self.noise_scheduler = noise_scheduler

    def q_sample(
        self,
        gt_sample: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向扩散 ``q(x_t | x_0)``:返回 ``(noisy_x0, target_noise)``。"""
        target_noise = torch.randn_like(gt_sample)
        noisy_x0 = self.noise_scheduler.add_noise(gt_sample, target_noise, timesteps)
        return noisy_x0, target_noise

    def target_for_loss(
        self,
        model: nn.Module,
        gt_sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (gt_sample.shape[0],),
            device=gt_sample.device,
        )
        noisy_x0, target_noise = self.q_sample(gt_sample, timesteps)
        pred_noise = model(noisy_x0, timesteps, batch)
        return pred_noise, target_noise


class DiffusionSampler:
    """反向扩散:从纯噪声出发逐步去噪到 ``x0``。``model`` 在调用时注入。"""

    def __init__(self, noise_scheduler: SchedulerMixin) -> None:
        self.noise_scheduler = noise_scheduler

    def step(
        self,
        model: nn.Module,
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
        return self.noise_scheduler.step(model_output, timestep, sample).prev_sample

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
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
            else self.noise_scheduler.config.num_train_timesteps
        )
        self.noise_scheduler.set_timesteps(n_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        sample_result = torch.randn(shape, device=device)
        intermediates_sample: list[torch.Tensor] = []
        snap_stride = (
            max(1, len(timesteps) // intermediates_num)
            if intermediates_num > 0
            else 0
        )

        for i, t in enumerate(timesteps):
            sample_result = self.step(model, sample_result, t, batch)
            if snap_stride and i % snap_stride == 0:
                intermediates_sample.append(sample_result)

        if intermediates_num > 0:
            return sample_result, intermediates_sample
        return sample_result


class DiffusionPipeline(nn.Module):
    """组合训练用的 :class:`DiffusionProcess` 与采样用的 :class:`DiffusionSampler`。

    ``model`` 由本类唯一持有,通过 ``nn.Module`` 的属性赋值机制完成注册;
    process 与 sampler 在调用时以参数形式接收 ``model``,自身不再保留引用。
    """

    def __init__(self, noise_scheduler: SchedulerMixin, model: nn.Module, num_inference_steps: int | None) -> None:
        super().__init__()
        self.model = model
        self.diffusion_process = DiffusionProcess(noise_scheduler)
        self.diffusion_sampler = DiffusionSampler(noise_scheduler)
        # 反向采样步数;``None`` 表示走 ``num_train_timesteps``,
        # DDPM 应保持默认,DDIM/DPM-Solver 等可显式传更小值加速。
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 正向扩散训练:返回 ``(pred_noise, target_noise)``,外部接 loss。
        return self.diffusion_process.target_for_loss(self.model, sample, batch)

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return self.diffusion_sampler.sample(
            self.model,
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
        return self.diffusion_sampler.sample(
            self.model,
            batch,
            shape,
            device=device,
            num_inference_steps=self.num_inference_steps,
            intermediates_num=intermediates_num,
        )
