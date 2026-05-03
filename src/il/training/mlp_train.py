"""Hydra 训练入口。

用法:
    python -m il.train                          # 使用默认配置
    python -m il.train data.batch_size=32       # 覆盖单个参数
    python -m il.train --cfg job                # 查看最终配置
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from tqdm.auto import tqdm

from il.dataset import create_dataloaders
from il.model import TrajectoryPredictor
from il.loss import TrajectoryLoss
from il.evaluation import compute_metrics
from .trainer import TrainerBase


OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

# ── 工具函数 ─────────────────────────────────────────────────────────


def _build_scheduler(optimizer, trainer_cfg: DictConfig, steps_per_epoch: int):
    if trainer_cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer_cfg.num_epochs * steps_per_epoch,
        )
    if trainer_cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=trainer_cfg.lr_step_size, gamma=trainer_cfg.lr_gamma,
        )
    return None


# ── 训练器 ───────────────────────────────────────────────────────────


class MLPTrainer(TrainerBase):
    """轨迹 MLP 监督学习训练。"""

    def run(self) -> None:
        cfg = self.cfg
        device = torch.device(cfg.device)
        save_dir = Path(cfg.log.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.yaml"
        OmegaConf.save(OmegaConf.structured(cfg), config_path)
        print(f"配置已保存至: {config_path}")
        print("加载数据...")
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        print(
            f"训练集: {len(train_loader.dataset)}, "
            f"验证集: {len(val_loader.dataset)}, "
            f"测试集: {len(test_loader.dataset)}"
        )

        model = TrajectoryPredictor(cfg).to(device)
        ckpt_path = cfg.trainer.get("checkpoint_path", None)
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到预训练模型: {ckpt_path}")
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict, strict=False)
            print(f"已加载预训练模型: {ckpt_path}")
        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.trainer.lr,
            weight_decay=cfg.trainer.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, cfg.trainer, len(train_loader))
        loss_fn = TrajectoryLoss(cfg)

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
        except ImportError:
            pass

        best_val_ade = float("inf")
        global_step = 0
        trainer_cfg = cfg.trainer
        log_cfg = cfg.log

        epoch_pbar = tqdm(range(1, trainer_cfg.num_epochs + 1), desc="Epochs", unit="epoch")
        for epoch in epoch_pbar:
            model.train()
            losses, xy_ades, xy_fdes = [], [], []
            heading_ades, heading_fdes = [], []
            speed_ades, speed_fdes = [], []
            t0 = time.monotonic()

            batch_pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{trainer_cfg.num_epochs}",
                unit="batch",
                leave=False,
            )
            for batch_idx, batch in batch_pbar:
                b = {
                    k: v.to(device)
                    for k, v in batch.items()
                    if k not in ("vehicle_params",)
                }

                pred = model(
                    history=b["history"],
                    history_mask=b["history_mask"],
                    centerline=b["centerline"],
                    centerline_mask=b["centerline_mask"],
                    left_boundary=b["left_boundary"],
                    left_boundary_mask=b["left_boundary_mask"],
                    right_boundary=b["right_boundary"],
                    right_boundary_mask=b["right_boundary_mask"],
                    lane_dividers=b["lane_dividers"],
                    lane_dividers_mask=b["lane_dividers_mask"],
                    max_v=b["max_v"],
                    max_v_mask=b["max_v_mask"],
                )
                loss, comp = loss_fn(pred, b["future"], b["future_mask"])

                optimizer.zero_grad()
                loss.backward()
                if trainer_cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), trainer_cfg.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1
                losses.append(comp["total"].item())
                xy_ades.append(comp["xy_ade"].item())
                xy_fdes.append(comp["xy_fde"].item())
                heading_ades.append(comp["heading_ade"].item())
                heading_fdes.append(comp["heading_fde"].item())
                speed_ades.append(comp["speed_ade"].item())
                speed_fdes.append(comp["speed_fde"].item())

                if writer:
                    writer.add_scalar("train/loss", comp["total"].item(), global_step)
                    writer.add_scalar("train/xy_ade", comp["xy_ade"].item(), global_step)
                    writer.add_scalar("train/xy_fde", comp["xy_fde"].item(), global_step)
                    writer.add_scalar("train/heading_ade", comp["heading_ade"].item(), global_step)
                    writer.add_scalar("train/heading_fde", comp["heading_fde"].item(), global_step)
                    writer.add_scalar("train/speed_ade", comp["speed_ade"].item(), global_step)
                    writer.add_scalar("train/speed_fde", comp["speed_fde"].item(), global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

                batch_pbar.set_postfix(
                    loss=f"{comp['total'].item():.4f}",
                    xy_ade=f"{comp['xy_ade'].item():.4f}",
                    xy_fde=f"{comp['xy_fde'].item():.4f}",
                    heading_ade=f"{comp['heading_ade'].item():.4f}",
                    heading_fde=f"{comp['heading_fde'].item():.4f}",
                    speed_ade=f"{comp['speed_ade'].item():.4f}",
                    speed_fde=f"{comp['speed_fde'].item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

                if (batch_idx + 1) % log_cfg.log_interval == 0:
                    print(
                        f"  [{epoch}/{trainer_cfg.num_epochs}] batch {batch_idx + 1}/{len(train_loader)} "
                        f"loss={comp['total'].item():.4f} "
                        f"xy_ade={comp['xy_ade'].item():.4f} xy_fde={comp['xy_fde'].item():.4f} "
                        f"heading_ade={comp['heading_ade'].item():.4f} heading_fde={comp['heading_fde'].item():.4f} "
                        f"speed_ade={comp['speed_ade'].item():.4f} speed_fde={comp['speed_fde'].item():.4f}"
                    )

            dt = time.monotonic() - t0
            ml = float(np.mean(losses))
            mxya, mxyf = float(np.mean(xy_ades)), float(np.mean(xy_fdes))
            mha, mhf = float(np.mean(heading_ades)), float(np.mean(heading_fdes))
            msa, msf = float(np.mean(speed_ades)), float(np.mean(speed_fdes))

            val = compute_metrics(model, val_loader, cfg)
            print(
                f"Epoch {epoch}/{trainer_cfg.num_epochs} ({dt:.1f}s) "
                f"train_loss={ml:.4f} "
                f"xy_ade={mxya:.4f} xy_fde={mxyf:.4f} "
                f"heading_ade={mha:.4f} heading_fde={mhf:.4f} "
                f"speed_ade={msa:.4f} speed_fde={msf:.4f} | "
                f"val_xy_ade={val['xy_ade']:.4f} val_xy_fde={val['xy_fde']:.4f} "
                f"val_heading_ade={val['heading_ade']:.4f} val_heading_fde={val['heading_fde']:.4f} "
                f"val_speed_ade={val['speed_ade']:.4f} val_speed_fde={val['speed_fde']:.4f}"
            )
            epoch_pbar.set_postfix(
                train_loss=f"{ml:.4f}",
                val_xy_ade=f"{val['xy_ade']:.4f}",
                val_xy_fde=f"{val['xy_fde']:.4f}",
            )

            if writer:
                for tag, v in [
                    ("epoch_train/train_loss", ml),
                    ("epoch_train/xy_ade", mxya),
                    ("epoch_train/xy_fde", mxyf),
                    ("epoch_train/heading_ade", mha),
                    ("epoch_train/heading_fde", mhf),
                    ("epoch_train/speed_ade", msa),
                    ("epoch_train/speed_fde", msf),
                    ("epoch_val/xy_ade", val["xy_ade"]),
                    ("epoch_val/xy_fde", val["xy_fde"]),
                    ("epoch_val/heading_ade", val["heading_ade"]),
                    ("epoch_val/heading_fde", val["heading_fde"]),
                    ("epoch_val/speed_ade", val["speed_ade"]),
                    ("epoch_val/speed_fde", val["speed_fde"]),
                ]:
                    writer.add_scalar(tag, v, epoch)

            if val["xy_ade"] < best_val_ade:
                best_val_ade = val["xy_ade"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_xy_ade": val["xy_ade"],
                        "val_xy_fde": val["xy_fde"],
                        "val_heading_ade": val["heading_ade"],
                        "val_heading_fde": val["heading_fde"],
                        "val_speed_ade": val["speed_ade"],
                        "val_speed_fde": val["speed_fde"],
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  ★ 保存最优模型 val_xy_ade={val['xy_ade']:.4f}")

            if epoch % log_cfg.save_interval == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_dir / f"epoch_{epoch}.pt",
                )

        print("\n" + "=" * 60)
        test = compute_metrics(model, test_loader, cfg)
        print(
            f"测试集: "
            f"xy_ade={test['xy_ade']:.4f} xy_fde={test['xy_fde']:.4f} "
            f"heading_ade={test['heading_ade']:.4f} heading_fde={test['heading_fde']:.4f} "
            f"speed_ade={test['speed_ade']:.4f} speed_fde={test['speed_fde']:.4f} "
            f"({test['count']} 样本)"
        )

        torch.save(
            {"epoch": trainer_cfg.num_epochs, "model_state_dict": model.state_dict()},
            save_dir / "final_model.pt",
        )
        if writer:
            writer.close()
