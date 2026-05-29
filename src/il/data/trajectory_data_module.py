"""轨迹数据：PyTorch ``Dataset`` + TrainFlow ``DataModule`` 封装。"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from trainflow.data import DataModule
import logging

class TrajectoryDataModule(DataModule):
    """划分 train/val/test 并构造 DataLoader。

    构造时传入已配置好的 ``torch.utils.data.Dataset`` 实例（``data_set``），本模块仅负责
    ``dm_cfg_*``：划分比例与 DataLoader 参数。
    """

    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        data_set: Dataset | None = None,
        train_data_set: Dataset | None = None,
        val_data_set: Dataset | None = None,
        test_data_set: Dataset | None = None,
    ) -> None:
        super().__init__()
        self._data_set = data_set
        self._train_data_set = train_data_set
        self._val_data_set = val_data_set
        self._test_data_set = test_data_set
        self._train_ratio = float(train_ratio)
        self._val_ratio = float(val_ratio)
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._pin_memory = bool(pin_memory)
        self._full_ds: Dataset | None = None
        self._train_ds: Dataset | None = None
        self._val_ds: Dataset | None = None
        self._test_ds: Dataset | None = None

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._full_ds is not None:
            return
        if self._data_set is not None:
            ds = self._data_set
            total = len(ds)
            if total == 0:
                raise ValueError("数据集为空")

            t_end = int(total * self._train_ratio)
            v_end = t_end + int(total * self._val_ratio)

            self._full_ds = ds
            self._train_ds = Subset(ds, range(0, t_end))
            self._val_ds = Subset(ds, range(t_end, v_end))
            self._test_ds = Subset(ds, range(v_end, total))
        elif self._train_data_set is not None and self._val_data_set is not None and self._test_data_set is not None:
            self._train_ds = self._train_data_set
            self._val_ds = self._val_data_set
            self._test_ds = self._test_data_set
            self._full_ds = ConcatDataset(
                [self._train_ds, self._val_ds, self._test_ds]
            )
            if len(self._full_ds) == 0:
                raise ValueError("full数据集为空")
            if len(self._val_ds) == 0:
                raise ValueError("val数据集为空")
        logging.info(f" full_ds: {len(self._full_ds)}, train_ds: {len(self._train_ds)}, val_ds: {len(self._val_ds)}, test_ds: {len(self._test_ds)}")

    def _loader_kw(self) -> dict:
        return {"num_workers": self._num_workers, "pin_memory": self._pin_memory}

    def train_dataloader(self) -> DataLoader:
        self.setup(stage="fit")
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True,
            **self._loader_kw(),
        )

    def val_dataloader(self) -> DataLoader:
        self.setup(stage="validate")
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self._batch_size,
            shuffle=False,
            **self._loader_kw(),
        )

    def test_dataloader(self) -> DataLoader:
        self.setup(stage="test")
        assert self._test_ds is not None
        return DataLoader(
            self._test_ds,
            batch_size=self._batch_size,
            shuffle=False,
            **self._loader_kw(),
        )

    def predict_dataloader(self) -> DataLoader:
        """与 test 相同划分，用于无标签推理遍历。"""
        return self.test_dataloader()
