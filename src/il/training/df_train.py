"""Diffusion 训练入口（基于 TrainerBase）。"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

from il.dataset import create_dataloaders
from il.model import Conditional1DUNet
from il.utils.diffusion import ScheduleDDPM, training_loop, ScheduleLogLinear, samples
from accelerate import Accelerator
from il.model import Scaled
from torch_ema import ExponentialMovingAverage as EMA
from .trainer import TrainerBase


class DFTrainer(TrainerBase):
    """Diffusion 训练器。"""

    def run(self) -> None:

        trainer_cfg = self.cfg.trainer

        epochs = int(trainer_cfg.get("num_epochs", 1000))
        sample_batch_size = int(trainer_cfg.get("sample_batch_size", 64))
        lr = float(trainer_cfg.get("learning_rate", 2e-4))

        device = torch.device(trainer_cfg.get("device", trainer_cfg.get("device", "cpu")))
        train_loader, _, _ = create_dataloaders(self.cfg)
        schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.002, N=1000)
        model = Conditional1DUNet(self.cfg).to(device)

        ema = EMA(model.parameters(), decay=0.9999)
        for ns in training_loop(train_loader, model, schedule, epochs=epochs, lr=lr):
            ns.pbar.set_description(f"Loss={ns.loss.item():.5}")

        sample_schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=35, N=1000)
        with ema.average_parameters():
            samples(
                model,
                sample_schedule.sample_sigmas(10),
                gam=2.1,
                batchsize=sample_batch_size,
            )

            ckpt_path = trainer_cfg.get("checkpoint_path", "checkpoint.pth")
            torch.save(model.state_dict(), ckpt_path)

