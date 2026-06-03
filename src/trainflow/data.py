from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch.utils.data import DataLoader


class DataModule(ABC):
    """Unified data lifecycle for train/val/test/predict stages."""

    def prepare_data(self) -> None:
        """Download/cache/tokenize in a single-process-safe way."""

    def setup(self, stage: Optional[str] = None) -> None:
        """Build datasets for a specific stage on every process."""

    def set_epoch(self, epoch: int) -> None:
        """Inform the data pipeline of the current epoch (e.g. for ``DistributedSampler``).

        No-op by default; subclasses backed by a ``DistributedSampler`` override this to call
        ``sampler.set_epoch(epoch)`` so shuffling varies correctly across epochs under DDP.
        """

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader:
        raise NotImplementedError("val_dataloader() is required for validation.")

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("test_dataloader() is required for testing.")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("predict_dataloader() is required for prediction.")
