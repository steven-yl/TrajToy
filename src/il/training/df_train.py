"""Diffusion 训练入口（基于 TrainerBase）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from il.loss import TrajectoryLoss
from il.utils.normalizer import BatchNormalizer
OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

from il.dataset import create_dataloaders
from il.model import Conditional1DUNet
from torch_ema import ExponentialMovingAverage as EMA
from .trainer import TrainerBase


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a dataloader batch dict onto ``device``."""
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

def prepare_diffusion_schedule(cfg: DictConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """Pre-compute DDPM schedule constants (linear or cosine)."""
    schedule_cfg = cfg.trainer.schedule_cfg
    schedule_name = schedule_cfg.type
    num_steps = schedule_cfg.num_steps
    beta_start = schedule_cfg.beta_start
    beta_end = schedule_cfg.beta_end

    if schedule_name == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    elif schedule_name == "cosine":
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Schedule '{schedule_name}' not implemented.")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        "posterior_variance": posterior_variance,
    }


def add_noise(
    x: torch.Tensor, schedule: dict[str, torch.Tensor], timesteps: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDPM forward: x_t = sqrt(alphas_cumprod_t) * x_0 + sqrt(1 - alphas_cumprod_t) * eps."""
    t = timesteps
    sqrt_acp = schedule["sqrt_alphas_cumprod"][t]
    sqrt_om = schedule["sqrt_one_minus_alphas_cumprod"][t]
    while sqrt_acp.ndim < x.ndim:
        sqrt_acp = sqrt_acp.unsqueeze(-1)
        sqrt_om = sqrt_om.unsqueeze(-1)
    eps = torch.randn_like(x)
    return sqrt_acp * x + sqrt_om * eps, eps, t


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid future steps (mask shape (B, L) or (B, L, 1))."""
    if mask.ndim == 2:
        m = mask.float().unsqueeze(-1)
    else:
        m = mask.float()
    m = m.expand_as(pred)
    se = (pred - target) ** 2 * m
    denom = m.sum().clamp(min=1.0)
    return se.sum() / denom


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    cond: dict,
    schedule: dict[str, torch.Tensor],
    shape: tuple[int, int, int],
    *,
    log_timing: bool = False,
) -> torch.Tensor:
    """Reverse DDPM: Gaussian noise -> trajectory, same layout as ``future`` (B, L, C).

    Args:
        log_timing: 若为 True，在 CUDA 同步后打印本轮反向扩散总耗时与平均每步毫秒数。
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    betas = schedule["betas"].to(device=device, dtype=dtype)
    alphas = 1.0 - betas
    alphas_cumprod = schedule["alphas_cumprod"].to(device=device, dtype=dtype)
    posterior_variance = schedule["posterior_variance"].to(device=device, dtype=dtype)
    T = betas.shape[0]
    b = shape[0]
    if log_timing and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = torch.randn(shape, device=device, dtype=dtype)
    for i in range(T - 1, -1, -1):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        eps = model(x, t, cond)
        alpha_t = alphas[i]
        alpha_bar_t = alphas_cumprod[i]
        beta_t = betas[i]
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps)
        if i > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(posterior_variance[i].clamp(min=1e-20))
            x = mean + sigma * noise
        else:
            x = mean
    if log_timing and device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if log_timing:
        ms_per = (elapsed / max(T, 1)) * 1000.0
        print(
            f"[sample_ddpm] T={T} batch={b} shape={shape} "
            f"elapsed={elapsed:.3f}s ({ms_per:.2f} ms/step)"
        )
    return x


def _ddim_timestep_sequence(T: int, num_inference_steps: int | None) -> list[int]:
    """在固定训练日程长度 ``T`` 上构造严格递减、含 ``T-1`` 与 ``0`` 的 DDIM 时间索引。"""
    if num_inference_steps is None or int(num_inference_steps) >= T:
        return list(range(T - 1, -1, -1))
    # 子序列至少两个锚点（最噪与最净），且不超过训练步数 T
    n = max(2, min(int(num_inference_steps), T))
    raw: list[int] = []
    for k in range(n):
        v = (T - 1) * (1.0 - k / max(n - 1, 1))
        raw.append(int(round(v)))
    ids = sorted({max(0, min(T - 1, x)) for x in raw}, reverse=True)
    if ids[0] != T - 1:
        ids.insert(0, T - 1)
    if ids[-1] != 0:
        ids.append(0)
    out = [ids[0]]
    for j in ids[1:]:
        if j < out[-1]:
            out.append(j)
    return out


@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    cond: dict,
    schedule: dict[str, torch.Tensor],
    shape: tuple[int, int, int],
    *,
    eta: float = 0.0,
    num_inference_steps: int | None = None,
    log_timing: bool = False,
) -> torch.Tensor:
    """DDIM 反向采样，输出布局同 ``future`` (B, L, C)。

    **与 DDPM 训练兼容**：当前管线里模型学习的是前向公式
    ``x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε`` 中的 ε；DDIM 用同一 ε 重构
    ``x̂_0 = (x_t - √(1-ᾱ_t) ε) / √(ᾱ_t)``，因此 **DDPM 训练的 checkpoint 可直接用于 DDIM**，
    前提是 ``schedule`` 与训练时一致（同一 ``betas`` / ``alphas_cumprod``）。

    **减少采样步数**：设 ``num_inference_steps`` 为小于训练步数 ``T`` 的整数（≥2），
    只在原日程上取子序列做 DDIM 跳跃，**不必改训练时的 ``num_steps``**。
    ``None`` 或 ``>= T`` 时等价于完整 ``T`` 步（与早期实现一致）。

    ``eta=0`` 为确定性 DDIM（除初始 ``x_T`` 外无额外噪声）；``eta∈(0,1]`` 为随机 DDIM。

    Args:
        eta: 随机强度，通常 ∈ [0, 1]；传入值会被钳制到该区间。
        num_inference_steps: DDIM 子序列长度（网格点个数，含两端）；模型前向次数约为该值。
        log_timing: 若为 True，在 CUDA 同步后打印本轮总耗时与平均每步毫秒数。
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    betas = schedule["betas"].to(device=device, dtype=dtype)
    alphas_cumprod = schedule["alphas_cumprod"].to(device=device, dtype=dtype)
    T = betas.shape[0]
    b = shape[0]
    eta = min(1.0, max(0.0, float(eta)))

    ts_seq = _ddim_timestep_sequence(T, num_inference_steps)
    n_infer = len(ts_seq)

    if log_timing and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x = torch.randn(shape, device=device, dtype=dtype)
    for k in range(len(ts_seq) - 1):
        t_curr = ts_seq[k]
        t_next = ts_seq[k + 1]
        t = torch.full((b,), t_curr, device=device, dtype=torch.long)
        eps = model(x, t, cond)
        alpha_bar = alphas_cumprod[t_curr]
        alpha_prev = alphas_cumprod[t_next]

        sqrt_ab = torch.sqrt(alpha_bar.clamp(min=1e-20))
        pred_x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / sqrt_ab

        # σ_t = η * sqrt( (1-ᾱ_{next})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{next}) )
        oom = (1.0 - alpha_prev) / (1.0 - alpha_bar).clamp(min=1e-20)
        oom = oom * (1.0 - alpha_bar / alpha_prev.clamp(min=1e-20))
        sigma = eta * torch.sqrt(torch.clamp(oom, min=0.0))
        dir_coef = torch.sqrt(torch.clamp(1.0 - alpha_prev - sigma * sigma, min=0.0))
        x = torch.sqrt(alpha_prev) * pred_x0 + dir_coef * eps
        if eta > 0:
            x = x + sigma * torch.randn_like(x)

    # eps = model(x, torch.zeros((b,), device=device, dtype=torch.long), cond)
    # alpha_bar = alphas_cumprod[0]
    # sqrt_ab = torch.sqrt(alpha_bar.clamp(min=1e-20))
    # x = (x - torch.sqrt(1.0 - alpha_bar) * eps) / sqrt_ab

    if log_timing and device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    if log_timing:
        denom = max(n_infer, 1)
        ms_per = (elapsed / denom) * 1000.0
        print(
            f"[sample_ddim] train_T={T} infer_grid={n_infer} batch={b} shape={shape} eta={eta} "
            f"elapsed={elapsed:.3f}s ({ms_per:.2f} ms/model-call)"
        )
    return x


def _build_future_traj_figure(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_samples: int = 4,
    title: str = "DDPM rollout vs GT (ego-local xy)",
):
    """构造 ``future`` xy 对比 matplotlib Figure（需在物理空间已反归一化）。_headless 使用 Agg。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_np = pred.detach().float().cpu().numpy()
    gt_np = gt.detach().float().cpu().numpy()
    m = mask.detach().cpu().numpy().astype(bool)
    B = min(pred_np.shape[0], max_samples)
    if B == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "empty batch", ha="center", va="center")
        fig.suptitle(title)
        return fig
    ncols = min(B, 4)
    nrows = (B + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    for i in range(B):
        r, c = divmod(i, ncols)
        ax = axs[r][c]
        mi = m[i]
        if not mi.any():
            ax.set_title(f"sample {i} (empty mask)")
            ax.axis("off")
            continue
        ax.plot(gt_np[i, mi, 0], gt_np[i, mi, 1], "o-", color="tab:green", ms=3, lw=1.5, label="gt")
        ax.plot(pred_np[i, mi, 0], pred_np[i, mi, 1], "s-", color="tab:red", ms=3, lw=1.5, label="pred")
        ax.scatter(gt_np[i, mi, 0][0], gt_np[i, mi, 1][0], c="black", s=40, zorder=5, marker="x")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(f"#{i}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    for j in range(B, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def _log_future_traj(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    tb_writer: Any | None,
    tb_step: int | None,
    tb_tag: str,
    traj_figure_path: Path | None,
    max_samples: int,
    title: str,
) -> None:
    """写入 TensorBoard ``add_figure``；可选保存 PNG。"""
    import matplotlib.pyplot as plt

    want_tb = tb_writer is not None and tb_step is not None
    want_file = traj_figure_path is not None
    if not want_tb and not want_file:
        return
    fig = _build_future_traj_figure(pred, gt, mask, max_samples=max_samples, title=title)
    try:
        if want_tb:
            tb_writer.add_figure(tb_tag, fig, global_step=int(tb_step), close=False)
        if want_file:
            traj_figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(traj_figure_path, dpi=140)
    finally:
        plt.close(fig)


@torch.no_grad()
def compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    schedule: dict[str, torch.Tensor],
    cfg: DictConfig,
    *,
    device: torch.device | None = None,
    rollout: bool = True,
    batch_norm: BatchNormalizer | None = None,
    traj_figure_path: Path | None = None,
    traj_figure_max_samples: int = 4,
    tb_writer: Any | None = None,
    tb_step: int | None = None,
    tb_tag: str = "val/future_xy",
) -> dict[str, Any]:
    """在 ``loader`` 上评估：噪声预测 masked MSE；可选完整 DDPM 采样后算轨迹 ADE/FDE。

    Args:
        model: ``Conditional1DUNet`` 等，输入 noisy (B,L,C)、timestep (B,)、cond batch dict。
        loader: 数据加载器。
        schedule: :func:`prepare_diffusion_schedule` 的返回值。
        cfg: 完整 Hydra 配置（供 :class:`TrajectoryLoss`）。
        device: 默认 ``cfg.device``。
        rollout: 若为 False，仅返回 ``loss`` / ``noise_mse`` 与 ``count``，不做反向扩散（省时间）。
        batch_norm: 若给定，与训练时一致地对 batch 做归一化后再算指标。
        traj_figure_path: 可选；若给定且 ``rollout=True``，第一个 batch 另存 PNG。
        traj_figure_max_samples: 图中最多展示的 batch 内样本数。
        tb_writer: 可选 ``SummaryWriter``；与 ``tb_step`` 同时给出时，第一个 batch 调用 ``add_figure``。
        tb_step: TensorBoard 的 ``global_step``（验证常用 ``epoch``）。
        tb_tag: TensorBoard 图像 tag，例如 ``val/future_xy``。
    """
    if device is None:
        device = torch.device(cfg.device)
    model.eval()
    loss_fn = TrajectoryLoss(cfg)
    num_steps = len(schedule["betas"])

    total_noise = 0.0
    total_xy_ade = total_xy_fde = 0.0
    total_heading_ade = total_heading_fde = 0.0
    total_speed_ade = total_speed_fde = 0.0
    count = 0
    traj_figure_saved = False

    for batch in tqdm(loader, desc="Computing metrics", unit="batch", leave=False):
        batch = _batch_to_device(batch, device)
        if batch_norm is not None:
            normalize_batch = batch_norm.normalize(batch)
        else:
            normalize_batch = batch
        x = normalize_batch["future"]
        future_mask = normalize_batch["future_mask"]
        cond = normalize_batch
        bsz = x.shape[0]

        timestep = torch.randint(0, num_steps, (bsz,), device=device)
        x_noisy, eps, t_for_model = add_noise(x, schedule, timestep)
        pred_eps = model(x_noisy, t_for_model, cond)
        noise_mse = _masked_mse(pred_eps, eps, future_mask)
        total_noise += noise_mse.item() * bsz

        if rollout:
            sched_eval = cfg.trainer.schedule_cfg
            ni = sched_eval.get("num_inference_steps", None)
            if ni is not None and int(ni) < num_steps:
                pred_traj = sample_ddim(
                    model,
                    cond,
                    schedule,
                    x.shape,
                    num_inference_steps=int(ni),
                )
            else:
                pred_traj = sample_ddpm(model, cond, schedule, x.shape)
            if batch_norm is not None:
                pred_vis = batch_norm.inverse_future(pred_traj)
                gt_vis = batch_norm.inverse_future(x)
            else:
                pred_vis = pred_traj
                gt_vis = x
            _, comp = loss_fn(pred_vis, gt_vis, future_mask)
            total_xy_ade += comp["xy_ade"].item() * bsz
            total_xy_fde += comp["xy_fde"].item() * bsz
            total_heading_ade += comp["heading_ade"].item() * bsz
            total_heading_fde += comp["heading_fde"].item() * bsz
            total_speed_ade += comp["speed_ade"].item() * bsz
            total_speed_fde += comp["speed_fde"].item() * bsz
            want_vis = not traj_figure_saved and (
                traj_figure_path is not None or (tb_writer is not None and tb_step is not None)
            )
            if want_vis:
                _log_future_traj(
                    pred_vis,
                    gt_vis,
                    future_mask,
                    tb_writer=tb_writer,
                    tb_step=tb_step,
                    tb_tag=tb_tag,
                    traj_figure_path=traj_figure_path,
                    max_samples=traj_figure_max_samples,
                    title=f"{tb_tag} (step {tb_step})" if tb_step is not None else tb_tag,
                )
                traj_figure_saved = True
        count += bsz

    if count == 0:
        out: dict[str, Any] = {
            "loss": 0.0,
            "noise_mse": 0.0,
            "xy_ade": 0.0,
            "xy_fde": 0.0,
            "heading_ade": 0.0,
            "heading_fde": 0.0,
            "speed_ade": 0.0,
            "speed_fde": 0.0,
            "ade": 0.0,
            "fde": 0.0,
            "count": 0,
        }
        return out

    noise_avg = total_noise / count
    out: dict[str, Any] = {
        "loss": noise_avg,
        "noise_mse": noise_avg,
        "count": count,
    }
    if rollout:
        out.update(
            {
                "ade": total_xy_ade / count,
                "fde": total_xy_fde / count,
                "xy_ade": total_xy_ade / count,
                "xy_fde": total_xy_fde / count,
                "heading_ade": total_heading_ade / count,
                "heading_fde": total_heading_fde / count,
                "speed_ade": total_speed_ade / count,
                "speed_fde": total_speed_fde / count,
            }
        )
    return out


class DFTrainer(TrainerBase):
    """Diffusion 训练器。"""

    def run(self) -> None:
        cfg = self.cfg
        trainer_cfg = cfg.trainer
        log_cfg = cfg.log

        device = torch.device(cfg.device)
        save_dir = Path(log_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.yaml"
        OmegaConf.save(cfg, config_path)
        print(f"配置已保存至: {config_path}")

        print("加载数据...")
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        print(
            f"训练集: {len(train_loader.dataset)}, "
            f"验证集: {len(val_loader.dataset)}, "
            f"测试集: {len(test_loader.dataset)}"
        )

        schedule = prepare_diffusion_schedule(cfg, device)

        model = Conditional1DUNet(cfg).to(device)
        ckpt_path_raw = trainer_cfg.get("checkpoint_path", None)
        start_epoch = 1
        global_step = 0
        ckpt_state: dict | None = None
        if ckpt_path_raw:
            ckpt_path = Path(ckpt_path_raw)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到检查点: {ckpt_path}")
            try:
                loaded = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                loaded = torch.load(ckpt_path, map_location=device)
            if not isinstance(loaded, dict):
                ckpt_state = {"model_state_dict": loaded}
            else:
                ckpt_state = loaded
            state_dict = ckpt_state.get("model_state_dict", ckpt_state)
            model.load_state_dict(state_dict, strict=False)
            print(f"已从检查点加载模型权重: {ckpt_path}")
            global_step = int(ckpt_state.get("global_step", 0))
            start_epoch = int(ckpt_state.get("epoch", 0)) + 1

        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainer_cfg.lr,
            weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
        )
        steps_per_epoch = len(train_loader)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, trainer_cfg.num_epochs * steps_per_epoch),
            eta_min=float(trainer_cfg.get("lr_min", 1e-6)),
        )

        if ckpt_state is not None:
            if "optimizer_state_dict" in ckpt_state:
                optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            if "lr_scheduler_state_dict" in ckpt_state:
                lr_scheduler.load_state_dict(ckpt_state["lr_scheduler_state_dict"])

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
        except ImportError:
            pass

        ema = EMA(model.parameters(), decay=float(trainer_cfg.get("ema_decay", 0.9999)))
        if ckpt_state is not None and ckpt_state.get("ema_state_dict"):
            ema.load_state_dict(ckpt_state["ema_state_dict"])

        max_grad_norm = float(trainer_cfg.get("max_grad_norm", 0.0))

        val_rollout = bool(trainer_cfg.get("val_rollout", False))
        val_interval = int(trainer_cfg.get("val_interval", 1))
        best_val_score = float("inf")
        save_interval = int(getattr(log_cfg, "save_interval", 10) or 0)

        batch_norm = BatchNormalizer.from_config(cfg.data.normalization)
        try:
            epoch_pbar = tqdm(
                range(start_epoch, trainer_cfg.num_epochs + 1),
                desc="Epochs",
                unit="epoch",
            )
            for epoch in epoch_pbar:
                model.train()
                t0 = time.monotonic()
                epoch_t0 = t0
                losses: list[float] = []
                batch_pbar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch}/{trainer_cfg.num_epochs}",
                    unit="batch",
                    leave=False,
                )
                for _, batch in batch_pbar:
                    batch = _batch_to_device(batch, device)
                    batch = batch_norm.normalize(batch)
                    x = batch["future"]
                    future_mask = batch["future_mask"]
                    cond = batch
                    timestep = torch.randint(
                        0, len(schedule["betas"]), (x.shape[0],), device=device
                    )
                    x_noisy, eps, t_for_model = add_noise(x, schedule, timestep)
                    pred = model(x_noisy, t_for_model, cond)
                    loss = _masked_mse(pred, eps, future_mask)

                    optimizer.zero_grad()
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    ema.update()

                    losses.append(loss.item())
                    global_step += 1

                    dt = time.monotonic() - t0
                    batch_pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        time=f"{dt:.2f}s",
                    )
                    t0 = time.monotonic()

                    if writer:
                        writer.add_scalar("train/loss", loss.item(), global_step)
                        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)

                mean_loss = sum(losses) / max(len(losses), 1)
                if writer:
                    writer.add_scalar("epoch_train/loss", mean_loss, epoch)

                dt_epoch = time.monotonic() - epoch_t0
                print(
                    f"Epoch {epoch}/{trainer_cfg.num_epochs} "
                    f"mean_loss={mean_loss:.4f} ({dt_epoch:.1f}s)"
                )

                if val_interval > 0 and epoch % val_interval == 0:
                    val_metrics = compute_metrics(
                        model,
                        val_loader,
                        schedule,
                        cfg,
                        device=device,
                        rollout=val_rollout,
                        batch_norm=batch_norm,
                        tb_writer=writer,
                        tb_step=epoch if val_rollout else None,
                        tb_tag="val/future_xy",
                    )
                    print(f"Epoch {epoch}/{trainer_cfg.num_epochs} 验证集: {val_metrics}")
                    postfix: dict[str, str] = {
                        "mean_train_loss": f"{mean_loss:.4f}",
                        "val_loss": f"{val_metrics['loss']:.4f}",
                    }
                    if val_rollout and "xy_ade" in val_metrics:
                        postfix["val_xy_ade"] = f"{val_metrics['xy_ade']:.4f}"
                        postfix["val_xy_fde"] = f"{val_metrics['xy_fde']:.4f}"
                    epoch_pbar.set_postfix(**postfix)

                    if writer:
                        writer.add_scalar("epoch_val/loss", val_metrics["loss"], epoch)
                        if val_rollout:
                            for k in (
                                "xy_ade",
                                "xy_fde",
                                "heading_ade",
                                "heading_fde",
                                "speed_ade",
                                "speed_fde",
                            ):
                                if k in val_metrics:
                                    writer.add_scalar(f"epoch_val/{k}", val_metrics[k], epoch)

                    score = val_metrics["xy_ade"] if val_rollout else val_metrics["loss"]
                    if score < best_val_score:
                        best_val_score = score
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                                "ema_state_dict": ema.state_dict(),
                                "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
                            },
                            save_dir / "best_model.pt",
                        )
                        tag = (
                            f"val_xy_ade={val_metrics['xy_ade']:.4f}"
                            if val_rollout
                            else f"val_loss={val_metrics['loss']:.4f}"
                        )
                        print(f"  ★ 保存最优模型 ({tag})")
                else:
                    epoch_pbar.set_postfix(mean_train_loss=f"{mean_loss:.4f}")

                if save_interval > 0 and epoch % save_interval == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                            "ema_state_dict": ema.state_dict(),
                        },
                        save_dir / f"epoch_{epoch}.pt",
                    )
                    print(f"  已保存检查点 epoch_{epoch}.pt")

            torch.save(
                {
                    "epoch": trainer_cfg.num_epochs,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                },
                save_dir / "final_model.pt",
            )
            print(f"最终模型已保存: {save_dir / 'final_model.pt'}")

            if val_interval > 0:
                test_metrics = compute_metrics(
                    model,
                    test_loader,
                    schedule,
                    cfg,
                    device=device,
                    rollout=val_rollout,
                    batch_norm=batch_norm,
                    tb_writer=writer,
                    tb_step=global_step if val_rollout else None,
                    tb_tag="test/future_xy",
                )
                print(f"测试集: {test_metrics}")
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            raise
        finally:
            if writer:
                writer.close()
