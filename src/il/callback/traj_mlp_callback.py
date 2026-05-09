"""轨迹可视化 Callback：在验证阶段将预测轨迹与 GT 可视化到 TensorBoard。"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from trainflow.callbacks import Callback
from trainflow.loggers import TensorBoardLogger, LoggerCollection
from il.data.visualization import TrajectoryDatasetVisualizer


class TrajVisualizationCallback(Callback):
    """训练过程中可视化轨迹预测结果。

    在每个 validation epoch 结束时，对前 N 个 batch 的第一个样本进行可视化，
    同时绘制 GT future 和模型预测 future，写入 TensorBoard。

    Parameters
    ----------
    log_every_n_epochs : int
        每隔多少个 epoch 可视化一次，默认 1。
    num_samples : int
        每次可视化的样本数量，默认 4。
    tag_prefix : str
        TensorBoard tag 前缀。
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 4,
        tag_prefix: str = "val_vis",
    ) -> None:
        self._log_every_n_epochs = max(1, int(log_every_n_epochs))
        self._num_samples = max(1, int(num_samples))
        self._tag_prefix = tag_prefix
        self._collected_samples: list[dict[str, Any]] = []

    def on_validation_epoch_start(self, trainer: Any) -> None:
        self._collected_samples = []

    def on_validation_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        if len(self._collected_samples) >= self._num_samples:
            return

        # 需要模型预测结果；从 model 重新推理获取 pred tensor
        model = trainer.model
        model.eval()
        with torch.no_grad():
            pred = model._predict(batch)  # (B, F, 2) or (B, F, 4)

        batch_size = pred.shape[0]
        remaining = self._num_samples - len(self._collected_samples)
        num_to_collect = min(batch_size, remaining)

        for i in range(num_to_collect):
            sample_data = {k: v[i] for k, v in batch.items()}
            sample_data["pred_future"] = pred[i]
            self._collected_samples.append(sample_data)

    def on_validation_epoch_end(self, trainer: Any) -> None:
        if trainer.current_epoch % self._log_every_n_epochs != 0:
            return

        writer = self._get_tb_writer(trainer)
        if writer is None:
            return

        for idx, sample in enumerate(self._collected_samples):
            tag = f"{self._tag_prefix}/sample_{idx}"
            TrajVisualizationVisualizer.log_to_tensorboard(
                writer,
                sample,
                tag=tag,
                global_step=trainer.global_step,
                title=f"Epoch {trainer.current_epoch} Sample {idx}",
            )

    @staticmethod
    def _get_tb_writer(trainer: Any):
        """从 trainer.logger 中提取 SummaryWriter。"""
        logger = trainer.logger
        if isinstance(logger, TensorBoardLogger):
            return logger.writer
        if isinstance(logger, LoggerCollection):
            for sub_logger in logger.loggers:
                if isinstance(sub_logger, TensorBoardLogger):
                    return sub_logger.writer
        return None


class TrajVisualizationVisualizer(TrajectoryDatasetVisualizer):
    """扩展 TrajectoryDatasetVisualizer，额外支持绘制模型预测轨迹。

    data dict 中额外的 key:
        - pred_future: (F, 2) 或 (F, 4) 模型预测的未来轨迹
    """

    COLORS = {
        **TrajectoryDatasetVisualizer.COLORS,
        "prediction": "#e74c3c",  # 鲜红色，区分于 GT future
    }

    @classmethod
    def _draw(cls, ax, data: dict[str, Any], **kwargs) -> None:
        """先绘制基础数据，再叠加预测轨迹。"""
        import numpy as np

        # 调用父类绘制 GT 部分
        super()._draw(ax, data, **kwargs)

        # 绘制模型预测轨迹
        pred_future = data.get("pred_future")
        if pred_future is None:
            return

        pred_np = cls._to_numpy(pred_future)  # (F, 2) or (F, 4)
        pred_xy = pred_np[:, :2]

        # 使用 future_mask（如果有）来确定有效长度
        future_mask = data.get("future_mask")
        mask = cls._to_numpy(future_mask) if future_mask is not None else None

        cls._plot_polyline(
            ax, pred_xy, mask,
            color=cls.COLORS["prediction"], linewidth=2.5,
            linestyle="--", label="Prediction", marker="D", markersize=3.0,
            alpha=0.9,
        )

        # 预测轨迹方向箭头（如果有 theta 信息）
        show_arrows = kwargs.get("show_arrows", True)
        arrow_interval = kwargs.get("arrow_interval", 3)
        if show_arrows and pred_np.shape[1] >= 3:
            valid_pred = pred_np
            if mask is not None:
                valid_pred = pred_np[np.asarray(mask, dtype=bool)]
            for i in range(0, len(valid_pred), arrow_interval):
                cls._plot_arrow(
                    ax, valid_pred[i, 0], valid_pred[i, 1], valid_pred[i, 2],
                    length=0.5, color=cls.COLORS["prediction"], linewidth=1.0,
                )