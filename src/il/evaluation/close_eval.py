"""闭环验证入口：基类与 Hydra 实例化。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hydra.utils import instantiate
from omegaconf import DictConfig


class CloseEvalBase(ABC):
    @abstractmethod
    def evaluate_closed_loop(self, cfg: DictConfig) -> None:
        """在仿真环境中执行闭环评估。"""


def instantiate_close_eval(cfg: DictConfig) -> CloseEvalBase:
    return instantiate(cfg.close_eval)
