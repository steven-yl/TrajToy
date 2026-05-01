"""Trainer 抽象基类：与 Hydra ``instantiate`` + ``train.py`` 调用约定对齐。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


class TrainerBase(ABC):
    """训练 / 评估流程的抽象基类。

    ``train.py`` 约定::

        trainer = instantiate(cfg.training.trainer)
        trainer(cfg)

    子类实现 :meth:`run`。若 YAML 节点上除 ``_target_`` / ``_convert_`` / ``_partial_``
    外还有其它字段，Hydra 会作为 ``__init__`` 关键字传入；未使用的关键字应被
    ``**kwargs`` 吸收或显式列出，避免实例化报错。
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __call__(self, cfg: DictConfig) -> None:
        """Hydra 入口：绑定本次运行的根配置并执行 :meth:`run`。"""
        self.cfg = cfg
        self.run()

    @abstractmethod
    def run(self) -> None:
        """执行完整训练或评估循环。"""
        ...

    @property
    def train_cfg(self) -> Any:
        """便捷访问 ``cfg.train``（若存在）。"""
        return self.cfg.get("train") if hasattr(self, "cfg") else None
