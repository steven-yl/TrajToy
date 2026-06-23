from __future__ import annotations

import torch
from torch import nn
from diffusers import SchedulerMixin


class FlowMatchingProcess:
    """Flow Matching 训练: 对 ``x0`` 与噪声做线性插值, 由 model 预测 velocity ``v = x1 - x0``。

    无实例状态; ``model`` 与 ``flow_scheduler`` 均在调用时注入。
    训练插值按 scheduler 配置公式计算, 不调用 ``scale_noise`` —— 后者依赖
    ``scheduler.timesteps`` 可变状态, 会在 ``set_timesteps`` (验证/采样) 后被污染。
    """

    @staticmethod
    def _build_train_schedule(
        flow_scheduler: SchedulerMixin,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由 config 重建固定训练时间表 ``(sigmas, timesteps)``, 与 scheduler 初始化一致。"""
        n = flow_scheduler.config.num_train_timesteps
        # 与 FlowMatchEulerDiscreteScheduler.__init__ 对齐: [T, T-1, ..., 1]
        sigmas = torch.arange(n, 0, -1, dtype=torch.float32) / n
        if not flow_scheduler.config.use_dynamic_shifting:
            shift = float(flow_scheduler.config.shift)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * n
        if device is not None:
            sigmas = sigmas.to(device)
            timesteps = timesteps.to(device)
        return sigmas, timesteps

    @staticmethod
    def q_sample(
        gt_sample: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向插值 ``x_t = (1 - σ) x_0 + σ x_1``: 返回 ``(x_t, noise)``。"""
        noise = torch.randn_like(gt_sample)
        sigma = sigmas.to(dtype=gt_sample.dtype).view(-1, *([1] * (gt_sample.ndim - 1)))
        x_t = (1.0 - sigma) * gt_sample + sigma * noise
        return x_t, noise

    @staticmethod
    def _sample_train_batch(
        flow_scheduler: SchedulerMixin,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从固定训练时间表均匀采样 ``(timesteps, sigmas)``。"""
        train_sigmas, train_timesteps = FlowMatchingProcess._build_train_schedule(
            flow_scheduler, device=device,
        )
        indices = torch.randint(0, train_sigmas.shape[0], (batch_size,), device=device)
        return train_timesteps[indices], train_sigmas[indices]

    @staticmethod
    def target_for_loss(
        model: nn.Module,
        flow_scheduler: SchedulerMixin,
        gt_sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps, sigmas = FlowMatchingProcess._sample_train_batch(
            flow_scheduler, gt_sample.shape[0], gt_sample.device,
        )
        x_t, noise = FlowMatchingProcess.q_sample(gt_sample, sigmas)
        pred = model(x_t, timesteps, batch)
        # rectified flow 目标: v = x_1 - x_0
        target = noise - gt_sample
        return pred, target


class FlowMatchingSampler:
    """Flow Matching 采样: 从纯噪声出发, 用 Euler 积分沿 velocity field 生成 ``x0``。

    ``model`` 与 ``flow_scheduler`` 均在调用时注入; 采样循环内须复用同一 scheduler 实例。
    """

    @staticmethod
    def step(
        model: nn.Module,
        flow_scheduler: SchedulerMixin,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # 训练时 model 拿到的是 (B,) 形状的 timesteps; 采样循环里 ``timestep``
        # 是 0-d 标量, 这里扩到 (B,) 与训练分布对齐。
        # diffusers 的 ``scheduler.step`` 仍要求标量 / 0-d, 保留原 ``timestep``。
        t_vec = timestep.expand(sample.shape[0])
        model_output = model(sample, t_vec, batch)
        # ``scheduler.step`` 执行 Euler 步: x_{t+dt} = x_t + dt * v_θ(x_t, t)
        return flow_scheduler.step(model_output, timestep, sample).prev_sample

    @staticmethod
    @torch.no_grad()
    def sample(
        model: nn.Module,
        flow_scheduler: SchedulerMixin,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
        num_inference_steps: int | None = None,
        intermediates_num: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[int | float]]:
        if device is None:
            device = next(model.parameters()).device

        n_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else flow_scheduler.config.num_train_timesteps
        )
        flow_scheduler.set_timesteps(n_steps, device=device)
        timesteps = flow_scheduler.timesteps

        # σ=1 处为纯噪声, 对应 flow matching 路径起点 x_1
        sample_result = torch.randn(shape, device=device)
        x_samples: list[torch.Tensor] = []
        x_samples_timesteps: list[int | float] = []
        snap_stride = (
            max(1, len(timesteps) // intermediates_num)
            if intermediates_num > 0
            else 0
        )

        for i, t in enumerate(timesteps):
            sample_result = FlowMatchingSampler.step(
                model, flow_scheduler, sample_result, t, batch,
            )
            if snap_stride and (i == 0 or i == len(timesteps) - 1 or i % snap_stride == 0):
                x_samples.append(sample_result)
                x_samples_timesteps.append(t.item() if isinstance(t, torch.Tensor) else t)
        if intermediates_num > 0:
            return sample_result, x_samples, x_samples_timesteps
        return sample_result


class FlowMatchingPipeline(nn.Module):
    """组合训练用的 :class:`FlowMatchingProcess` 与采样用的 :class:`FlowMatchingSampler`。

    ``model`` 与 ``flow_scheduler`` 由本类持有 (Hydra 注入); process / sampler 无状态,
    调用时将二者作为参数传入。

    推荐使用 ``diffusers.FlowMatchEulerDiscreteScheduler`` 作为 ``flow_scheduler``。
    注: diffusers 的 ``SchedulerMixin`` 不是 ``nn.Module``, 不会进入 ``state_dict``,
    加载权重后需仍由配置重建 scheduler (与 diffusers Pipeline 惯例一致)。
    """

    def __init__(
        self,
        flow_scheduler: SchedulerMixin,
        model: nn.Module,
        num_inference_steps: int | None,
    ) -> None:
        super().__init__()
        self.flow_scheduler = flow_scheduler
        self.model = model
        # Euler 积分步数; ``None`` 表示走 ``num_train_timesteps``。
        self.num_inference_steps = num_inference_steps

    def forward(
        self,
        sample: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Flow Matching 训练: 返回 ``(pred_velocity, target_velocity)``, 外部接 loss。
        return FlowMatchingProcess.target_for_loss(
            self.model, self.flow_scheduler, sample, batch,
        )

    def sample(
        self,
        batch: dict[str, torch.Tensor],
        shape: torch.Size | tuple[int, ...],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        return FlowMatchingSampler.sample(
            self.model,
            self.flow_scheduler,
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
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[int | float]]:
        return FlowMatchingSampler.sample(
            self.model,
            self.flow_scheduler,
            batch,
            shape,
            device=device,
            num_inference_steps=self.num_inference_steps,
            intermediates_num=intermediates_num,
        )
