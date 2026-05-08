from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


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
    def __init__(self, save_dir: str = "logs", filename: str = "metrics.csv") -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._file_path = path / filename
        self._header_written = self._file_path.exists() and self._file_path.stat().st_size > 0

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        write_header = not self._header_written
        with self._file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


class TensorBoardLogger(Logger):
    def __init__(self, save_dir: str = "logs/tensorboard") -> None:
        self.writer = SummaryWriter(log_dir=save_dir)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        for key, value in metrics.items():
            self.writer.add_scalar(key, float(value), global_step=step)

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
