"""Hydra 训练入口。

用法:
    python -m il.train                          # 使用默认配置
    python -m il.train data.batch_size=32       # 覆盖单个参数
    python -m il.train --cfg job                # 查看最终配置
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from tqdm.auto import tqdm

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

from il.dataset import create_dataloaders
from il.model import TrajectoryPredictor
from il.loss import TrajectoryLoss
from il.evaluation import compute_metrics


# ── 工具函数 ─────────────────────────────────────────────────────────


def _build_scheduler(optimizer, cfg: DictConfig, steps_per_epoch: int):
    tc = cfg.train
    if tc.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.num_epochs * steps_per_epoch,
        )
    if tc.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=tc.lr_step_size, gamma=tc.lr_gamma,
        )
    return None


# ── 训练主函数 ────────────────────────────────────────────────────────


def train(cfg: DictConfig) -> None:
    """执行完整训练流程。"""
    device = torch.device(cfg.device)

    save_dir = Path(cfg.log.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置 (OmegaConf YAML)
    config_path = save_dir / "config.yaml"
    OmegaConf.save(OmegaConf.structured(cfg), config_path)
    print(f"配置已保存至: {config_path}")
    print(OmegaConf.to_yaml(cfg))

    # 数据
    print("加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    print(f"训练集: {len(train_loader.dataset)}, "
          f"验证集: {len(val_loader.dataset)}, "
          f"测试集: {len(test_loader.dataset)}")

    # 模型
    model = TrajectoryPredictor(cfg).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg, len(train_loader))
    loss_fn = TrajectoryLoss(cfg)

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
    except ImportError:
        pass

    best_val_ade = float("inf")
    global_step = 0
    tc = cfg.train
    lc = cfg.log

    epoch_pbar = tqdm(range(1, tc.num_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        losses, ades, fdes = [], [], []
        t0 = time.monotonic()

        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{tc.num_epochs}",
            unit="batch",
            leave=False,
        )
        for batch_idx, batch in batch_pbar:
            b = {k: v.to(device) for k, v in batch.items()
                 if k not in ("vehicle_params",)}

            pred = model(
                b["history"], b["history_mask"],
                b["centerline"], b["centerline_mask"],
                b["left_boundary"], b["left_boundary_mask"],
                b["right_boundary"], b["right_boundary_mask"],
                b["lane_dividers"], b["lane_dividers_mask"],
                b["max_v"], b["max_v_mask"],
            )
            loss, comp = loss_fn(pred, b["future"], b["future_mask"])

            optimizer.zero_grad()
            loss.backward()
            if tc.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            losses.append(comp["total"].item())
            ades.append(comp["ade"].item())
            fdes.append(comp["fde"].item())

            if writer:
                writer.add_scalar("train/loss", comp["total"].item(), global_step)
                writer.add_scalar("train/ade", comp["ade"].item(), global_step)
                writer.add_scalar("train/fde", comp["fde"].item(), global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            batch_pbar.set_postfix(
                loss=f"{comp['total'].item():.4f}",
                ade=f"{comp['ade'].item():.4f}",
                fde=f"{comp['fde'].item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if (batch_idx + 1) % lc.log_interval == 0:
                print(f"  [{epoch}/{tc.num_epochs}] batch {batch_idx+1}/{len(train_loader)} "
                      f"loss={comp['total'].item():.4f} ade={comp['ade'].item():.4f} fde={comp['fde'].item():.4f}")

        dt = time.monotonic() - t0
        ml, ma, mf = float(np.mean(losses)), float(np.mean(ades)), float(np.mean(fdes))

        val = compute_metrics(model, val_loader, cfg)
        print(f"Epoch {epoch}/{tc.num_epochs} ({dt:.1f}s) "
              f"train_loss={ml:.4f} ade={ma:.4f} fde={mf:.4f} | "
              f"val_ade={val['ade']:.4f} val_fde={val['fde']:.4f}")
        epoch_pbar.set_postfix(
            train_loss=f"{ml:.4f}",
            val_ade=f"{val['ade']:.4f}",
            val_fde=f"{val['fde']:.4f}",
        )

        if writer:
            for tag, v in [("epoch/train_loss", ml), ("epoch/train_ade", ma),
                           ("epoch/train_fde", mf), ("epoch/val_ade", val["ade"]),
                           ("epoch/val_fde", val["fde"])]:
                writer.add_scalar(tag, v, epoch)

        if val["ade"] < best_val_ade:
            best_val_ade = val["ade"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_ade": val["ade"], "val_fde": val["fde"]},
                       save_dir / "best_model.pt")
            print(f"  ★ 保存最优模型 val_ade={val['ade']:.4f}")

        if epoch % lc.save_interval == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       save_dir / f"epoch_{epoch}.pt")

    # 测试
    print("\n" + "=" * 60)
    test = compute_metrics(model, test_loader, cfg)
    print(f"测试集: ADE={test['ade']:.4f} FDE={test['fde']:.4f} ({test['count']} 样本)")

    torch.save({"epoch": tc.num_epochs, "model_state_dict": model.state_dict()},
               save_dir / "final_model.pt")
    if writer:
        writer.close()


# ── Hydra 入口 ───────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra 入口。"""
    train(cfg)


if __name__ == "__main__":
    main()
