from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from trainflow.loggers import TqdmLogger
from trainflow._rank_zero import rank_zero_warn, rank_zero_info


tqdm_logger = logging.getLogger('tqdm_progress')
_TQDM_OUT = TqdmLogger(tqdm_logger)


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
    def on_predict_epoch_start(self, trainer: Any) -> None: ...
    def on_predict_epoch_end(self, trainer: Any) -> None: ...
    def on_predict_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None: ...
    def on_predict_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None: ...
    def on_before_backward(self, trainer: Any, loss: torch.Tensor) -> None: ...
    def on_after_backward(self, trainer: Any) -> None: ...
    def on_before_optimizer_step(self, trainer: Any) -> None: ...
    def on_after_optimizer_step(self, trainer: Any) -> None: ...
    def on_exception(self, trainer: Any, exception: BaseException) -> None: ...
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
        every_n_epochs: int = 1,
        save_on_exception: bool = True,
        save_last: bool = True,
    ) -> None:
        """Checkpoint 回调。

        Args:
            mode: ``min`` / ``max`` 按 ``monitor`` 指标保留最优；``epoch`` 每隔固定 epoch 保存（忽略 ``monitor``）。
            every_n_epochs: 仅 ``mode=="epoch"`` 时生效，每多少个 epoch 在验证结束后存盘一次（≥1）。
            save_top_k: 保留的快照数量。``min``/``max`` 模式保留最优的 ``save_top_k`` 个；
                ``epoch`` 模式保留最近 ``save_top_k`` 个定期快照；≤0 时不删旧文件（全部保留）。
            save_on_exception: 训练异常中断时在 global-zero rank 写一份 ``interrupted.ckpt``，便于恢复。
            save_last: 每个验证 epoch 结束后额外写一份 ``final.ckpt``（最新状态，便于断点续训）。
                大模型下逐 epoch 全量落盘开销较大时可设为 ``False`` 关闭。
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        if mode not in ("min", "max", "epoch"):
            raise ValueError(f"ModelCheckpoint.mode 应为 min、max 或 epoch，收到 {mode!r}")
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.save_last = bool(save_last)
        self.best: float | None = None
        self._saved: list[tuple[float, Path]] = []
        self.save_on_exception = bool(save_on_exception)
        self._seen_monitor = False

    def _is_better(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_validation_epoch_end(self, trainer: Any) -> None:
        # Only the global-zero rank writes checkpoints under DDP (single rank otherwise), and only
        # during ``fit`` so standalone ``validate()``/``test()`` do not write or prune checkpoints.
        if not trainer.strategy.is_global_zero:
            return
        if getattr(trainer, "stage", "fit") != "fit":
            return
        # "Latest" snapshot for resume; optional so large models can skip per-epoch full dumps.
        if self.save_last:
            trainer.save_checkpoint(self.dirpath / "final.ckpt")
        if self.mode == "epoch":
            self._save_periodic(trainer)
        else:
            self._save_monitored(trainer)

    def _save_periodic(self, trainer: Any) -> None:
        """``mode='epoch'``: snapshot every ``every_n_epochs`` and keep the most recent ``save_top_k``."""
        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return
        path = self.dirpath / self.filename.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )
        trainer.save_checkpoint(path)
        if self.save_top_k <= 0:
            return
        # Order is recency (append order); drop the oldest beyond save_top_k.
        self._saved.append((float(trainer.global_step), path))
        while len(self._saved) > self.save_top_k:
            _, p = self._saved.pop(0)
            if p.exists():
                p.unlink()

    def _save_monitored(self, trainer: Any) -> None:
        """``mode='min'/'max'``: keep the best ``save_top_k`` snapshots by ``monitor``."""
        if self.monitor not in trainer.current_metrics:
            return
        self._seen_monitor = True
        path = self.dirpath / f"{self.mode}_{self.filename.format(epoch=trainer.current_epoch, step=trainer.global_step)}"
        score = float(trainer.current_metrics[self.monitor])
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

    def on_fit_end(self, trainer: Any) -> None:
        # Surface a likely metric-name typo for monitored modes instead of silently saving nothing.
        if self.mode in ("min", "max") and not self._seen_monitor:
            rank_zero_warn(
                f"ModelCheckpoint monitor {self.monitor!r} was never found in metrics; "
                "no monitored checkpoint was saved. Check the metric name "
                "(available keys are namespaced like 'val/loss').",
                UserWarning,
                stacklevel=2,
            )

    def on_exception(self, trainer: Any, exception: BaseException) -> None:
        # Best-effort emergency checkpoint so a crash mid-training is recoverable.
        if not self.save_on_exception:
            return
        if not trainer.strategy.is_global_zero:
            return
        try:
            trainer.save_checkpoint(self.dirpath / "interrupted.ckpt")
        except Exception:  # pragma: no cover — never mask the original exception
            pass

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.best, "saved": [(s, str(p)) for s, p in self._saved]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best = state_dict.get("best")
        self._saved = [(s, Path(p)) for s, p in state_dict.get("saved", [])]


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val/loss", mode: str = "min", patience: int = 5, min_delta: float = 0.0) -> None:
        self.state = CallbackState(monitor=monitor)
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self._seen_monitor = False

    def _improved(self, current: float) -> bool:
        if self.state.best is None:
            return True
        if self.mode == "min":
            return current < self.state.best - self.min_delta
        return current > self.state.best + self.min_delta

    def on_validation_epoch_end(self, trainer: Any) -> None:
        # Runs on ALL ranks during fit (never rank-guarded) so the stop decision is consistent and
        # collective ops do not hang. Skipped outside fit so standalone validate() has no side effect.
        if getattr(trainer, "stage", "fit") != "fit":
            return
        if self.state.monitor not in trainer.current_metrics:
            return
        self._seen_monitor = True
        current = float(trainer.current_metrics[self.state.monitor])
        if self._improved(current):
            self.state.best = current
            self.state.wait = 0
            return
        self.state.wait += 1
        if self.state.wait >= self.patience:
            trainer.should_stop = True

    def on_fit_end(self, trainer: Any) -> None:
        # Surface a likely metric-name typo instead of silently never stopping.
        if not self._seen_monitor:
            rank_zero_warn(
                f"EarlyStopping monitor {self.state.monitor!r} was never found in metrics; "
                "early stopping had no effect. Check the metric name "
                f"(available keys are namespaced like 'val/loss').",
                UserWarning,
                stacklevel=2,
            )

    def state_dict(self) -> dict[str, Any]:
        return {"best": self.state.best, "wait": self.state.wait}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.state.best = state_dict.get("best")
        self.state.wait = state_dict.get("wait", 0)


class LearningRateMonitor(Callback):
    """Log average LR of the first optimizer, throttled to the Trainer's logging cadence.

    ``trainer.log`` triggers a cross-rank all-reduce under DDP, so emitting every optimizer step
    would add a synchronisation per step. We throttle to ``trainer.log_every_n_steps`` to match the
    Trainer's own metric logging and keep the DDP overhead negligible.
    """

    def __init__(self, name: str = "lr") -> None:
        self.name = name

    def on_after_optimizer_step(self, trainer: Any) -> None:
        if not trainer.optimizers:
            return
        every = max(1, int(getattr(trainer, "log_every_n_steps", 1)))
        if trainer.global_step % every != 0:
            return
        lrs = [group["lr"] for group in trainer.optimizers[0].param_groups]
        trainer.log({self.name: float(sum(lrs) / len(lrs))})


class TrainProgressBar(Callback):
    """Terminal progress bar for training epochs (uses ``tqdm`` if installed)."""

    def __init__(self, leave: bool = True) -> None:
        self.leave = leave
        self._pbar: Any = None

    def on_train_epoch_start(self, trainer: Any) -> None:
        if not trainer.strategy.is_global_zero:
            self._pbar = None
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.train_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(
            total=total,
            desc=f"train epoch {trainer.current_epoch}",
            leave=self.leave,
            file=_TQDM_OUT,
        )

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
        if not trainer.strategy.is_global_zero:
            self._pbar = None
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.val_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(
            total=total,
            desc=f"val epoch {trainer.current_epoch}",
            leave=self.leave,
            file=_TQDM_OUT,
        )

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
        if not trainer.strategy.is_global_zero:
            self._pbar = None
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.test_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(
            total=total,
            desc=f"test epoch {trainer.current_epoch}",
            leave=self.leave,
            file=_TQDM_OUT,
        )


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
        if not trainer.strategy.is_global_zero:
            self._pbar = None
            return
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            self._pbar = None
            return
        dl = trainer.datamodule.predict_dataloader()
        total = len(dl) if hasattr(dl, "__len__") else None
        self._pbar = tqdm(
            total=total,
            desc=f"predict epoch {trainer.current_epoch}",
            leave=self.leave,
            file=_TQDM_OUT,
        )


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
            rank_zero_info(f"[TimeMonitor] fit finished in {elapsed:.2f}s")

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