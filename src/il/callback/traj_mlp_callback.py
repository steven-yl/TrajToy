"""轨迹可视化 Callback：在验证阶段将预测轨迹与 GT 可视化到 TensorBoard。"""

from __future__ import annotations

from typing import Any
from trainflow.callbacks import Callback
from trainflow.loggers import TensorBoardLogger, LoggerCollection
from il.data.visualization import TrajectoryDatasetVisualizer, DiffusionProcessVisualizer


class TrajVisualizationCallback(Callback):
    """训练过程中可视化轨迹预测结果。

    在每个 validation epoch 结束时，收集前 N 个样本，
    同时绘制 GT future 和模型预测 future，以网格形式写入 TensorBoard。

    直接使用 TrajectoryDatasetVisualizer（已原生支持 pred_future key）。

    Parameters
    ----------
    log_every_n_epochs : int
        每隔多少个 epoch 可视化一次，默认 1。
    num_samples : int
        每次可视化的样本数量，默认 4。
    tag : str
        TensorBoard tag 名称。
    ncols : int
        多子图网格每行列数，默认 2。
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 4,
        tag: str = "val_vis/trajectory",
        ncols: int = 2,
    ) -> None:
        self._log_every_n_epochs = max(1, int(log_every_n_epochs))
        self._num_samples = max(1, int(num_samples))
        self._tag = tag
        self._ncols = ncols
        self._collected_samples: list[dict[str, Any]] = []

    def on_validation_epoch_start(self, trainer: Any) -> None:
        self._collected_samples = []

    def on_validation_batch_end(
        self, trainer: Any, outputs: Any, batch: Any, batch_idx: int,
    ) -> None:
        if len(self._collected_samples) >= self._num_samples:
            return

        batch_size = outputs["pred_future"].shape[0]
        remaining = self._num_samples - len(self._collected_samples)
        num_to_collect = min(batch_size, remaining)

        for i in range(num_to_collect):
            sample_data = {k: v[i].detach().cpu() for k, v in batch.items()}
            sample_data["pred_future"] = outputs["pred_future"][i].detach().cpu()
            if "x_samples" in outputs:
                sample_data["x_samples"] = [x[i].detach().cpu() for x in outputs["x_samples"]]
            self._collected_samples.append(sample_data)

    def on_validation_epoch_end(self, trainer: Any) -> None:
        if not self._collected_samples:
            return
        if trainer.current_epoch % self._log_every_n_epochs != 0:
            return

        writer = self._get_tb_writer(trainer)
        if writer is None:
            return

        titles = [f"Sample {i}" for i in range(len(self._collected_samples))]

        # 轨迹预测可视化（GT + Prediction）
        TrajectoryDatasetVisualizer.log_to_tensorboard(
            writer,
            self._collected_samples,
            tag=self._tag,
            global_step=trainer.global_step,
            title=titles,
            ncols=self._ncols,
        )

        # 扩散去噪过程可视化（如果存在 x_samples）
        for idx, sample in enumerate(self._collected_samples):
            if "x_samples" not in sample or not sample["x_samples"]:
                continue
            step_dicts = DiffusionProcessVisualizer.build_step_dicts(sample)
            num_steps = len(step_dicts)
            step_titles = [
                f"t={num_steps - 1 - i}" if i < num_steps - 1 else "t=0 (final)"
                for i in range(num_steps)
            ]
            DiffusionProcessVisualizer.log_to_tensorboard(
                writer,
                step_dicts,
                tag=f"{self._tag}/diffusion_sample_{idx}",
                global_step=trainer.global_step,
                title=step_titles,
                ncols=min(4, num_steps),
            )

    @staticmethod
    def _get_tb_writer(trainer: Any):
        """从 trainer.logger 中提取 TensorBoard SummaryWriter。"""
        logger = trainer.logger
        if isinstance(logger, TensorBoardLogger):
            return logger.writer
        if isinstance(logger, LoggerCollection):
            for sub_logger in logger.loggers:
                if isinstance(sub_logger, TensorBoardLogger):
                    return sub_logger.writer
        return None
