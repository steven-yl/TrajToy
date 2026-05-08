from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

try:
    from hydra.utils import instantiate
except ImportError as exc:  # pragma: no cover
    instantiate = None  # type: ignore[assignment]
    _HYDRA_IMPORT_ERROR = exc
else:
    _HYDRA_IMPORT_ERROR = None


def _require_hydra() -> None:
    if instantiate is None:
        raise ImportError(
            "Hydra is required for trainflow.hydra_build. Install with: pip install hydra-core"
        ) from _HYDRA_IMPORT_ERROR


def instantiate_trainflow(cfg: DictConfig) -> tuple[Any, Any, Any]:
    """Build ``(trainer, model, datamodule)`` from ``cfg.trainflow`` using Hydra ``instantiate``.

    Expected layout::

        trainflow:
          trainer:
            _target_: trainflow.trainer.Trainer
            ...
          model:
            _target_: ...
          data:
            _target_: ...

    Nested nodes with ``_target_`` (e.g. ``callbacks``, ``logger`` under ``trainer``) are
    recursively instantiated when ``_recursive_=True``.
    """
    _require_hydra()
    if not OmegaConf.is_config(cfg):
        raise TypeError("cfg must be an OmegaConf DictConfig")

    tf = cfg.get("trainflow")
    if tf is None:
        raise ValueError(
            "Missing config key `trainflow`. Add `trainflow.trainer`, `trainflow.model`, "
            "`trainflow.data` (each with `_target_`)."
        )

    trainer_cfg = tf.get("trainer")
    model_cfg = tf.get("model")
    data_cfg = tf.get("data")

    if trainer_cfg is None or model_cfg is None or data_cfg is None:
        raise ValueError(
            "`trainflow` must define `trainer`, `model`, and `data` subconfigs."
        )

    trainer = instantiate(trainer_cfg, _recursive_=True)
    model = instantiate(model_cfg, _recursive_=True)
    datamodule = instantiate(data_cfg, _recursive_=True)
    return trainer, model, datamodule


def run_fit(cfg: DictConfig) -> None:
    """Instantiate components from Hydra config and run ``trainer.fit(model, datamodule)``."""
    trainer, model, datamodule = instantiate_trainflow(cfg)
    trainer.fit(model, datamodule)
