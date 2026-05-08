from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from trainflow.data import DataModule
from trainflow.model import TrainableModel


class ToyModel(TrainableModel):
    """Tiny classifier on synthetic Gaussian blobs (no extra deps)."""

    def __init__(self, dim_in: int = 10, num_classes: int = 2) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.net = torch.nn.Linear(dim_in, num_classes)

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            raise TypeError("ToyModel.forward expects a dict batch or (x, y) tuple/list.")
        logits = self.net(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_step(self, batch: Any) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if isinstance(x, dict):
            x = x["x"]
        return self.net(x)


class ToyDataModule(DataModule):
    def __init__(
        self,
        n_train: int = 256,
        n_val: int = 64,
        batch_size: int = 32,
        dim_in: int = 10,
        num_classes: int = 2,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size
        self.dim_in = dim_in
        self.num_classes = num_classes
        self.seed = seed
        self._g = torch.Generator().manual_seed(seed)

    def setup(self, stage: str | None = None) -> None:
        def make_split(n: int) -> TensorDataset:
            x = torch.randn(n, self.dim_in, generator=self._g)
            y = torch.randint(0, self.num_classes, (n,), generator=self._g)
            return TensorDataset(x, y)

        if stage in (None, "fit", "validate"):
            self.train_ds = make_split(self.n_train)
            self.val_ds = make_split(self.n_val)
        if stage in (None, "test"):
            if not hasattr(self, "test_ds"):
                self.test_ds = make_split(self.n_val)
        if stage in (None, "predict"):
            if not hasattr(self, "predict_ds"):
                self.predict_ds = TensorDataset(
                    torch.randn(self.n_val, self.dim_in, generator=self._g),
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False)
