from trainflow.callbacks import Callback
from omegaconf import DictConfig
from typing import Any

class TrajMLPCallback(Callback):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_start(trainer, batch, batch_idx)

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        return super().on_train_batch_end(trainer, outputs, batch, batch_idx)

    def on_validation_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        return super().on_validation_batch_start(trainer, batch, batch_idx)

    def on_validation_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        return super().on_validation_batch_end(trainer, outputs, batch, batch_idx)