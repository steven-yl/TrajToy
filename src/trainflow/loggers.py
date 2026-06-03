from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter

import logging
import sys

from ._rank_zero import rank_zero_only

class TqdmLogger:
    """将 tqdm 进度条的输出重定向到指定的日志记录器。"""
    def __init__(self, logger: logging.Logger, stderr: bool = True):
        self.logger = logger
        self.stderr = stderr

    def write(self, msg: str) -> None:
        # 关键步骤：移除每行开头可能导致重复的 '\r' 回车符
        if msg and msg.strip():
            self.logger.info(msg.lstrip('\r'))
        if msg and self.stderr:
            try:
                sys.stderr.write(msg)
            except Exception:
                self.logger.debug("TqdmLogger failed to write to stderr", exc_info=True)

    def flush(self) -> None:
        if self.stderr:
            try:
                sys.stderr.flush()
            except Exception:
                self.logger.debug("TqdmLogger failed to flush stderr", exc_info=True)

class Logger(ABC):
    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        raise NotImplementedError

    def finalize(self) -> None:
        return None


class NoOpLogger(Logger):
    """Drop-in logger when metrics should not be written anywhere."""

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        return None


class CSVLogger(Logger):
    """Append-only CSV metric logger that tolerates evolving metric keys.

    Different stages emit different key sets (train step rows, val epoch rows, ``time/*`` rows …).
    A plain ``DictWriter`` with a fixed header would raise ``ValueError`` as soon as a row contains
    a key absent from the header. This logger tracks the union of all keys seen so far and rewrites
    the file with an expanded header whenever a new column appears, so every row is preserved and
    aligned under a consistent schema.
    """

    def __init__(self, save_dir: str = "logs", filename: str = "metrics.csv") -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._file_path = path / filename
        self._fieldnames: list[str] = []
        self._rows: list[dict[str, float]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Resume appending to a non-empty CSV by reading back its rows/columns."""
        if not (self._file_path.exists() and self._file_path.stat().st_size > 0):
            return
        with self._file_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._fieldnames = list(reader.fieldnames or [])
            for raw in reader:
                self._rows.append({k: v for k, v in raw.items() if v not in (None, "")})

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        new_keys = [k for k in row if k not in self._fieldnames]
        # "step" should stay first; other keys keep first-seen order.
        if not self._fieldnames:
            self._fieldnames = ["step"] + [k for k in row if k != "step"]
            rewrite = True
        elif new_keys:
            self._fieldnames.extend(new_keys)
            rewrite = True
        else:
            rewrite = False
        self._rows.append(row)
        if rewrite:
            self._rewrite_all()
        else:
            self._append_row(row)

    def _append_row(self, row: dict[str, float]) -> None:
        with self._file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)

    def _rewrite_all(self) -> None:
        """Rewrite header + all rows so older rows align under any newly added columns."""
        with self._file_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow(row)
    # File writes only happen on global rank 0 (no-op single-device change).
    log_metrics = rank_zero_only(log_metrics)


class TensorBoardLogger(Logger):
    def __init__(self, save_dir: str = "logs/tensorboard") -> None:
        self.writer = SummaryWriter(log_dir=save_dir)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, float(value), global_step=step)
    log_metrics = rank_zero_only(log_metrics)

    def finalize(self) -> None:
        self.writer.flush()
        self.writer.close()


class LoggerCollection(Logger):
    def __init__(self, loggers: list[Logger] | None = None) -> None:
        self.loggers = list(loggers or [])

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    def finalize(self) -> None:
        for logger in self.loggers:
            logger.finalize()
