from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


class Callback:
    def on_fit_start(self, trainer: Any) -> None: ...
    def on_fit_end(self, trainer: Any) -> None: ...
    def on_train_epoch_start(self, trainer: Any) -> None: ...
    def on_train_epoch_end(self, trainer: Any) -> None: ...
    def on_validation_epoch_start(self, trainer: Any) -> None: ...
    def on_validation_epoch_end(self, trainer: Any) -> None: ...
    def on_test_epoch_start(self, trainer: Any) -> None: ...
    def on_test_epoch_end(self, trainer: Any) -> None: ...
    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None: ...
    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_validation_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None: ...
    def on_validation_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_test_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None: ...
    def on_test_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_before_backward(self, trainer: Any, loss: torch.Tensor) -> None: ...
    def on_after_backward(self, trainer: Any) -> None: ...
    def on_before_optimizer_step(self, trainer: Any) -> None: ...
    def on_after_optimizer_step(self, trainer: Any) -> None: ...
    def state_dict(self) -> dict[str, Any]:
        return {}
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return None


@dataclass
class CallbackState:
    monitor: str
    best: float | None = None
    wait: int = 0


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath: str = "checkpoint",
        monitor: str = "val/loss",
        mode: str = "min",
        save_top_k: int = 1,
        filename: str = "{epoch:03d}-{step:08d}.ckpt",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.best: float | None = None
        self._saved: list[tuple[float, Path]] = []

    def _is_better(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_validation_epoch_end(self, trainer: Any) -> None:
        if self.monitor not in trainer.current_metrics:
            return
        score = float(trainer.current_metrics[self.monitor])
        path = self.dirpath / self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)
        if self.save_top_k <= 0:
            trainer.save_checkpoint(path)
            return
        if self._is_better(score) or len(self._saved) < self.save_top_k:
            trainer.save_checkpoint(path)
            self._saved.append((score, path))
            self._saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
            self.best = self._saved[0][0]
            while len(self._saved) > self.save_top_k:
                _, p = self._saved.pop(-1)
                if p.exists():
                    p.unlink()

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "saved": [(s, str(p)) for s, p in self._saved]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best = state_dict.get("best")
        self._saved = [(s, Path(p)) for s, p in state_dict.get("saved", [])]


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", mode: str = "min", patience: int = 5, min_delta: float = 0.0) -> None:
        self.state = CallbackState(monitor=monitor)
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta

    def _improved(self, current: float) -> bool:
        if self.state.best is None:
            return True
        if self.mode == "min":
            return current < self.state.best - self.min_delta
        return current > self.state.best + self.min_delta

    def on_validation_epoch_end(self, trainer: Any) -> None:
        if self.state.monitor not in trainer.current_metrics:
            return
        current = float(trainer.current_metrics[self.state.monitor])
        if self._improved(current):
            self.state.best = current
            self.state.wait = 0
            return
        self.state.wait += 1
        if self.state.wait >= self.patience:
            trainer.should_stop = True

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.state.best, "wait": self.state.wait}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state.best = state_dict.get("best")
        self.state.wait = state_dict.get("wait", 0)


class LearningRateMonitor(Callback):
    def __init__(self, name: str = "lr") -> None:
        self.name = name

    def on_after_optimizer_step(self, trainer: Any) -> None:
        if not trainer.optimizers:
            return
        lrs = [group["lr"] for group in trainer.optimizers[0].param_groups]
        trainer.log({self.name: float(sum(lrs) / len(lrs))})


class TrainProgressBar(Callback):
    """Terminal progress bar for training epochs (uses ``tqdm`` if installed)."""

    def __init__(self, leave: bool = True) -> None:
        self.leave = leave
        self._pbar: Any = None

    def on_train_epoch_start(self, trainer: Any) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.train_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(total=total, desc=f"train epoch {trainer.current_epoch}", leave=self.leave)

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._pbar is not None:
            self._pbar.update(1)

    def on_train_epoch_end(self, trainer: Any) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None



class ValidationProgressBar(Callback):
    """Terminal progress bar for validation epochs (uses ``tqdm`` if installed)."""

    def __init__(self, leave: bool = True) -> None:
        self.leave = leave
        self._pbar: Any = None

    def on_validation_epoch_start(self, trainer: Any) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.val_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(total=total, desc=f"val epoch {trainer.current_epoch}", leave=self.leave)

    def on_validation_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._pbar is None:
            return
        self._pbar.update(1)

    def on_validation_epoch_end(self, trainer: Any) -> None:
        if self._pbar is None:
            return
        self._pbar.close()
        self._pbar = None


class TestProgressBar(Callback):
    """Terminal progress bar for test epochs (uses ``tqdm`` if installed)."""

    def __init__(self, leave: bool = True) -> None:
        self.leave = leave
        self._pbar: Any = None

    def on_test_epoch_start(self, trainer: Any) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.test_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(total=total, desc=f"test epoch {trainer.current_epoch}", leave=self.leave)


    def on_test_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._pbar is None:
            return
        self._pbar.update(1)

    def on_test_epoch_end(self, trainer: Any) -> None:
        if self._pbar is None:
            return
        self._pbar.close()
        self._pbar = None



class PredictProgressBar(Callback):
    """Terminal progress bar for predict epochs (uses ``tqdm`` if installed)."""

    def __init__(self, leave: bool = True) -> None:
        self.leave = leave
        self._pbar: Any = None

    def on_predict_epoch_start(self, trainer: Any) -> None:
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.predict_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(total=total, desc=f"predict epoch {trainer.current_epoch}", leave=self.leave)


    def on_predict_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._pbar is None:
            return
        self._pbar.update(1)

    def on_predict_epoch_end(self, trainer: Any) -> None:
        if self._pbar is None:
            return
        self._pbar.close()
        self._pbar = None

class TimeMonitor(Callback):
    """Track wall-clock time for fit / train epoch / val epoch / test epoch.

    - Uses ``time.perf_counter`` (monotonic, high-precision) for elapsed time.
    - Logs durations via ``trainer.log`` so they flow into CSV / TensorBoard.
    - Optionally prints a one-line summary per epoch.
    """

    def __init__(self, log_metrics: bool = True, verbose: bool = True) -> None:
        self.log_metrics = log_metrics
        self.verbose = verbose
        self._fit_start: float | None = None
        self._train_epoch_start: float | None = None
        self._val_epoch_start: float | None = None
        self._test_epoch_start: float | None = None

    @staticmethod
    def _elapsed(start: float | None) -> float | None:
        if start is None:
            return None
        return max(0.0, time.perf_counter() - start)

    def on_fit_start(self, trainer: Any) -> None:
        self._fit_start = time.perf_counter()

    def on_fit_end(self, trainer: Any) -> None:
        elapsed = self._elapsed(self._fit_start)
        if elapsed is None:
            return
        if self.log_metrics:
            trainer.log({"time/fit_sec": float(elapsed)})
        if self.verbose:
            print(f"[TimeMonitor] fit finished in {elapsed:.2f}s")

    def on_train_epoch_start(self, trainer: Any) -> None:
        self._train_epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: Any) -> None:
        elapsed = self._elapsed(self._train_epoch_start)
        if elapsed is None:
            return
        if self.log_metrics:
            trainer.log({"time/train_epoch_sec": float(elapsed)})
        if self.verbose:
            print(
                f"[TimeMonitor] train epoch {trainer.current_epoch} took {elapsed:.2f}s"
            )

    def on_validation_epoch_start(self, trainer: Any) -> None:
        self._val_epoch_start = time.perf_counter()

    def on_validation_epoch_end(self, trainer: Any) -> None:
        elapsed = self._elapsed(self._val_epoch_start)
        if elapsed is None:
            return
        if self.log_metrics:
            trainer.log({"time/val_epoch_sec": float(elapsed)})
        if self.verbose:
            print(
                f"[TimeMonitor] val epoch {trainer.current_epoch} took {elapsed:.2f}s"
            )

    def on_test_epoch_start(self, trainer: Any) -> None:
        self._test_epoch_start = time.perf_counter()

    def on_test_epoch_end(self, trainer: Any) -> None:
        elapsed = self._elapsed(self._test_epoch_start)
        if elapsed is None:
            return
        if self.log_metrics:
            trainer.log({"time/test_epoch_sec": float(elapsed)})
        if self.verbose:
            print(
                f"[TimeMonitor] test epoch {trainer.current_epoch} took {elapsed:.2f}s"
            )