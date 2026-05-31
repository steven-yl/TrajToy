"""轨迹可视化 Callback：在验证阶段将预测轨迹与 GT 可视化到 TensorBoard。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from trainflow.callbacks import Callback
from trainflow.loggers import TensorBoardLogger, LoggerCollection
from il.data.visualization import TrajectoryDatasetVisualizer, DiffusionProcessVisualizer


def get_tensorboard_writer(trainer: Any):
    """从 trainer.logger 中提取 TensorBoard SummaryWriter。"""
    logger = trainer.logger
    if isinstance(logger, TensorBoardLogger):
        return logger.writer
    if isinstance(logger, LoggerCollection):
        for sub_logger in logger.loggers:
            if isinstance(sub_logger, TensorBoardLogger):
                return sub_logger.writer
    return None


def _tb_scalar(value: Any) -> int | float | str | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    return str(value)


class TrajVisualizationCallback(Callback):
    """训练过程中可视化轨迹预测结果。

    在每个 validation epoch 结束时，收集前 N 个样本，
    同时绘制 GT future 和模型预测 future，以单列多行形式写入 TensorBoard。

    直接使用 TrajectoryDatasetVisualizer（已原生支持 pred_future key）。

    Parameters
    ----------
    log_every_n_epochs : int
        每隔多少个 epoch 可视化一次，默认 1。
    num_samples : int
        每次可视化的样本数量，默认 4。
    tag : str
        TensorBoard tag 名称。
    image_save_dir : str | None
        图片保存目录；为 ``null`` 时不保存。Hydra 下可设为 ``${hydra:runtime.output_dir}/tb_images``。
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 4,
        tag: str = "val_vis/trajectory",
        tensorboard_show: bool = False,
        image_save_dir: str | None = None,
    ) -> None:
        self._log_every_n_epochs = max(1, int(log_every_n_epochs))
        self._num_samples = max(1, int(num_samples))
        self._tag = tag
        self._tensorboard_show = tensorboard_show
        self._image_save_dir = Path(image_save_dir) if image_save_dir else None
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

        writer = get_tensorboard_writer(trainer)
        if writer is None:
            return

        titles = [f"Sample {i}" for i in range(len(self._collected_samples))]
        e, s = trainer.current_epoch, trainer.global_step
        if self._tensorboard_show:
            TrajectoryDatasetVisualizer.log_to_tensorboard(
                writer,
                self._collected_samples,
                tag=self._tag,
                global_step=s,
                title=titles,
            )

        if self._image_save_dir:
            TrajectoryDatasetVisualizer.log_to_image(
                self._collected_samples,
                save_path=self._image_save_dir / f"trajectory_e{e}_s{s}.png"
            )

        for idx, sample in enumerate(self._collected_samples):
            if "x_samples" not in sample or not sample["x_samples"]:
                continue
            step_dicts = DiffusionProcessVisualizer.build_step_dicts(sample)
            num_steps = len(step_dicts)
            step_titles = [
                f"t={num_steps - 1 - i}" if i < num_steps - 1 else "t=0 (final)"
                for i in range(num_steps)
            ]
            if self._tensorboard_show:
                DiffusionProcessVisualizer.log_to_tensorboard(
                    writer,
                    step_dicts,
                    tag=f"{self._tag}/diffusion_sample_{idx}",
                    global_step=s,
                    title=step_titles,
                )
            if self._image_save_dir:
                DiffusionProcessVisualizer.log_to_image(
                    step_dicts,
                    save_path=self._image_save_dir / f"diffusion_{idx}_e{e}_s{s}.png",
                )


class HParamsCallback(Callback):
    """将超参数写入 TensorBoard HPARAMS 面板。

    在训练结束（``on_fit_end``）时调用 ``SummaryWriter.add_hparams`` 一次，
    并附带 ``trainer.current_metrics`` 中的指标用于对比。

    Parameters
    ----------
    hparams : dict[str, Any]
        超参数字典；非 int/float/str/bool 的值会转为字符串。
    metric_keys : list[str] | None
        写入 TB 的指标键；``None`` 时使用当前 ``trainer.current_metrics`` 全部项。
    """

    def __init__(
        self,
        hparams: dict[str, Any],
        metric_keys: list[str] | None = None,
    ) -> None:
        self._hparams = hparams
        self._metric_keys = metric_keys
        self._logged = False

    def on_fit_end(self, trainer: Any) -> None:
        if self._logged:
            return

        writer = get_tensorboard_writer(trainer)
        if writer is None:
            return

        metrics = trainer.current_metrics
        if self._metric_keys is not None:
            metrics = {k: metrics[k] for k in self._metric_keys if k in metrics}

        writer.add_hparams(
            {str(k): _tb_scalar(v) for k, v in self._hparams.items()},
            {str(k): float(v) for k, v in metrics.items()},
        )
        self._logged = True

# add_text
class TextCallback(Callback):
    def __init__(self, tag: str, text: str) -> None:
        self._tag = tag
        self._text = str(text)

    def on_fit_start(self, trainer: Any) -> None:
        writer = get_tensorboard_writer(trainer)
        if writer is None:
            return
        writer.add_text(self._tag, self._text)

# add_histogram
# GRAPHS