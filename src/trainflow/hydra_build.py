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


def _resolve_strict_weights_only(cfg: DictConfig) -> tuple[bool, bool]:
    """Match IL YAML keys ``strict`` / ``weights_only`` or ``checkpoint_*`` variants."""
    strict = OmegaConf.select(cfg, "checkpoint_strict")
    if strict is None:
        strict = OmegaConf.select(cfg, "strict", default=True)
    weights_only = OmegaConf.select(cfg, "checkpoint_weights_only")
    if weights_only is None:
        weights_only = OmegaConf.select(cfg, "weights_only", default=False)
    return bool(strict), bool(weights_only)


def _fit_resume_checkpoint_path(cfg: DictConfig) -> Any | None:
    if bool(cfg.get("auto_load_checkpoint", False)):
        return OmegaConf.select(cfg, "resume_checkpoint")
    return OmegaConf.select(cfg, "resume_checkpoint")


def instantiate_model_and_datamodule(cfg: DictConfig) -> tuple[Any, Any]:
    _require_hydra()
    if not OmegaConf.is_config(cfg):
        raise TypeError("cfg must be an OmegaConf DictConfig")

    tf = cfg.get("trainflow")
    if tf is None:
        raise ValueError(
            "Missing config key `trainflow`. Add `trainflow.model`, "
            "`trainflow.data` (each with `_target_`)."
        )

    model_cfg = tf.get("model")
    data_cfg = tf.get("data")
    if model_cfg is None or data_cfg is None:
        raise ValueError(
            "`trainflow` must define `model` and `data` subconfigs."
        )

    model = instantiate(model_cfg, _recursive_=True)
    datamodule = instantiate(data_cfg, _recursive_=True)
    return model, datamodule


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
    """Instantiate trainflow and run ``fit``. Resume when ``resume_checkpoint`` is set or
    ``auto_load_checkpoint`` is true with ``checkpoint_path``.
    """
    trainer, model, datamodule = instantiate_trainflow(cfg)
    ckpt = _fit_resume_checkpoint_path(cfg)
    if ckpt is None:
        trainer.fit(model, datamodule)
        return
    strict, weights_only = _resolve_strict_weights_only(cfg)
    trainer.fit(
        model,
        datamodule,
        ckpt_path=ckpt,
        ckpt_strict=strict,
        ckpt_weights_only=weights_only,
    )


def run_validate(cfg: DictConfig) -> None:
    print("start validate...")
    trainer, model, datamodule = instantiate_trainflow(cfg)
    trainer.model = model
    trainer.datamodule = datamodule
    ckpt = _fit_resume_checkpoint_path(cfg)
    if ckpt is None:
        raise ValueError("Missing config key `resume_checkpoint` (required for validate).")
    strict, weights_only = _resolve_strict_weights_only(cfg)
    trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
    metrics = trainer.validate()
    print("validate done!")
    return metrics


def run_test(cfg: DictConfig) -> None:
    print("start test...")
    trainer, model, datamodule = instantiate_trainflow(cfg)
    trainer.model = model
    trainer.datamodule = datamodule
    ckpt = _fit_resume_checkpoint_path(cfg)
    if ckpt is None:
        raise ValueError("Missing config key `resume_checkpoint` (required for test).")
    strict, weights_only = _resolve_strict_weights_only(cfg)
    trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
    metrics = trainer.test()
    print("test done!")
    return metrics

def run_predict(cfg: DictConfig) -> list[Any]:
    print("start predict...")
    trainer, model, datamodule = instantiate_trainflow(cfg)
    trainer.model = model
    trainer.datamodule = datamodule
    ckpt = _fit_resume_checkpoint_path(cfg)
    if ckpt is None:
        raise ValueError("Missing config key `resume_checkpoint` (required for predict).")
    strict, weights_only = _resolve_strict_weights_only(cfg)
    trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
    outputs = trainer.predict()
    print("predict done!")
    return outputs