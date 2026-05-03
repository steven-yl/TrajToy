"""Diffusion 训练入口（基于 TrainerBase）。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from datetime import datetime

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

from il.dataset import create_dataloaders
from il.model import Conditional1DUNet
from il.utils.diffusion import ScheduleDDPM, training_loop, ScheduleLogLinear, samples
from torch_ema import ExponentialMovingAverage as EMA
from .trainer import TrainerBase


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
        OmegaConf.save(OmegaConf.structured(cfg), config_path)
        print(f"配置已保存至: {config_path}")
        print("加载数据...")
        train_loader, val_loader, test_loader = create_dataloaders(cfg)
        print(
            f"训练集: {len(train_loader.dataset)}, "
            f"验证集: {len(val_loader.dataset)}, "
            f"测试集: {len(test_loader.dataset)}"
        )

        epochs = trainer_cfg.num_epochs
        sample_batch_size = trainer_cfg.sample_batch_size
        lr = trainer_cfg.lr

        schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.002, N=1000)
        model = Conditional1DUNet(cfg).to(device)

        ckpt_path = trainer_cfg.get("checkpoint_path", None)
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"未找到检查点: {ckpt_path}")
            try:
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt_path, map_location=device)
            state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict, strict=False)
            print(f"已从检查点加载模型权重: {ckpt_path}")

        print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
        except ImportError:
            pass

        ema = EMA(model.parameters(), decay=0.9999)
        global_step = 0
        prev_epoch = None
        epoch_losses: list[float] = []
        t0 = time.monotonic()

        def flush_epoch(
            ep: int,
            losses: list[float],
            dt_s: float,
            *,
            optimizer: torch.optim.Optimizer,
            step_marker: int,
        ) -> None:
            ml = float(np.mean(losses))
            print(f"Epoch {ep}/{epochs} ({dt_s:.1f}s) train_loss={ml:.6f}")
            if writer:
                writer.add_scalar("epoch_train/train_loss", ml, ep)
            if log_cfg.save_interval and ep % log_cfg.save_interval == 0:
                torch.save(
                    {
                        "epoch": ep,
                        "global_step": step_marker,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    save_dir / f"epoch_{ep}.pt",
                )
                print(f"  已保存检查点: {save_dir / f'epoch_{ep}.pt'}")

        try:
            for ns in training_loop(train_loader, model, schedule, epochs=epochs, lr=lr):
                global_step += 1
                loss_val = ns.loss.item()
                ns.pbar.set_description(f"Loss={loss_val:.5}")

                if prev_epoch is not None and ns.epoch != prev_epoch:
                    flush_epoch(
                        prev_epoch + 1,
                        epoch_losses,
                        time.monotonic() - t0,
                        optimizer=ns.optimizer,
                        step_marker=global_step - 1,
                    )
                    epoch_losses = []
                    t0 = time.monotonic()

                epoch_losses.append(loss_val)

                if writer:
                    writer.add_scalar("train/loss", loss_val, global_step)
                    writer.add_scalar(
                        "train/lr",
                        ns.optimizer.param_groups[0]["lr"],
                        global_step,
                    )

                ema.update()

                if global_step % log_cfg.log_interval == 0:
                    print(
                        f"  step {global_step} epoch={ns.epoch + 1}/{epochs} "
                        f"loss={loss_val:.6f}"
                    )

                prev_epoch = ns.epoch

            if epoch_losses and prev_epoch is not None:
                flush_epoch(
                    prev_epoch + 1,
                    epoch_losses,
                    time.monotonic() - t0,
                    optimizer=ns.optimizer,
                    step_marker=global_step,
                )

            sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)
            sigmas = sample_schedule.sample_sigmas(10)
            print(
                f"采样: sigma_steps={len(sigmas) - 1}, gam=2.1, batchsize={sample_batch_size}"
            )
            with ema.average_parameters():
                last_xt = None
                for xt in samples(
                    model,
                    sigmas,
                    gam=2.1,
                    batchsize=sample_batch_size,
                ):
                    last_xt = xt
                if writer and last_xt is not None:
                    writer.add_scalar(
                        "sample/x_abs_mean",
                        last_xt.detach().abs().mean().item(),
                        global_step,
                    )

            torch.save(
                {
                    "epoch": epochs,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                },
                save_dir / "final_model.pt",
            )
            print(f"最终模型已保存: {save_dir / 'final_model.pt'}")
        finally:
            if writer:
                writer.close()
