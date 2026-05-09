"""轨迹数据：``TrajectoryDataset`` + TrainFlow ``DataModule`` 封装。"""

from __future__ import annotations

from typing import Optional

from torch.utils.data import DataLoader, Subset

from trainflow.data import DataModule

from il.data.dataset.trajectory_dataset import TrajectoryDataset


class TrajectoryDataModule(DataModule):
    """划分 train/val/test 并构造 DataLoader。

    构造时传入已配置好的 ``TrajectoryDataset`` 实例（``data_set``），本模块仅负责
    ``dm_cfg_*``：划分比例与 DataLoader 参数。
    """

    def __init__(
        self,
        train_ratio: float,
        val_ratio: float,
        batch_size: int,
        num_workers: int,
        data_set: TrajectoryDataset,
    ) -> None:
        super().__init__()
        self._data_set = data_set
        self._train_ratio = float(train_ratio)
        self._val_ratio = float(val_ratio)
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._full_ds: TrajectoryDataset | None = None
        self._train_ds: Subset | None = None
        self._val_ds: Subset | None = None
        self._test_ds: Subset | None = None

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        if self._full_ds is not None:
            return
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

        print(f" full_ds: {len(self._full_ds)}, train_ds: {len(self._train_ds)}, val_ds: {len(self._val_ds)}, test_ds: {len(self._test_ds)}")

    def _loader_kw(self) -> dict:
        return {"num_workers": self._num_workers, "pin_memory": True}

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
