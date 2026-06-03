from __future__ import annotations

import logging
import os
import platform
import sys
from typing import Any

import torch
import torch.nn as nn
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


def resolve_strict_weights_only(cfg: DictConfig) -> tuple[bool, bool]:
    """Match IL YAML keys ``strict`` / ``weights_only`` or ``checkpoint_*`` variants."""
    strict = OmegaConf.select(cfg, "checkpoint_strict")
    if strict is None:
        strict = OmegaConf.select(cfg, "strict", default=True)
    weights_only = OmegaConf.select(cfg, "checkpoint_weights_only")
    if weights_only is None:
        weights_only = OmegaConf.select(cfg, "weights_only", default=False)
    return bool(strict), bool(weights_only)


def _select_resume_checkpoint(cfg: DictConfig) -> Any | None:
    """Read the configured ``resume_checkpoint`` path (no gating). ``None`` when unset."""
    return OmegaConf.select(cfg, "resume_checkpoint")


def fit_resume_checkpoint_path(cfg: DictConfig) -> Any | None:
    """Resume checkpoint path for ``fit``, or ``None`` when not resuming.

    ``auto_load_checkpoint`` gates resumption: when false (default), ``fit`` starts fresh even if a
    ``resume_checkpoint`` value is present; when true, the configured ``resume_checkpoint`` is used.
    Eval/predict entrypoints bypass this gate via :func:`_select_resume_checkpoint` since a
    checkpoint is mandatory there.
    """
    if not bool(cfg.get("auto_load_checkpoint", False)):
        return None
    return _select_resume_checkpoint(cfg)


def instantiate_trainer_and_model(cfg: DictConfig) -> tuple[Any, Any]:
    _require_hydra()
    if not OmegaConf.is_config(cfg):
        raise TypeError("cfg must be an OmegaConf DictConfig")

    tf = cfg.get("trainflow")
    if tf is None:
        raise ValueError(
            "Missing config key `trainflow`. Add `trainflow.trainer`, `trainflow.model`, "
            "`trainflow.data` (each with `_target_`)."
        )

    model_cfg = tf.get("model")
    trainer_cfg = tf.get("trainer")
    if model_cfg is None or trainer_cfg is None:
        raise ValueError(
            "`trainflow` must define `model` and `trainer` subconfigs."
        )

    model = instantiate(model_cfg, _recursive_=True)
    trainer = instantiate(trainer_cfg, _recursive_=True)
    return trainer, model


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


def _dist_world_size() -> int:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _print_runtime_env(trainer: Any) -> None:
    """打印 Trainer 启动时的运行环境概况:device / torch / 精度 / 分布式。

    Args:
        trainer: 已构造好的 ``Trainer`` 实例,需要其暴露
            ``device`` / ``precision`` / ``precision_plugin`` / ``strategy`` 等字段。
    """
    # Only the global-zero rank prints; under DDP the other ranks would otherwise interleave
    # duplicate blocks into the same log file.
    if not trainer.strategy.is_global_zero:
        return
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

    device = trainer.device
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(idx)
        total_mem_gib = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        cuda_ver = torch.version.cuda or "unknown"
        cudnn_ver = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
        device_desc = (
            f"cuda:{idx} ({gpu_name}, {total_mem_gib:.1f} GiB) "
            f"| CUDA {cuda_ver} | cuDNN {cudnn_ver}"
        )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            device_desc += f" | CUDA_VISIBLE_DEVICES={visible}"
    elif device.type == "mps":
        device_desc = "mps (Apple Silicon)"
    else:
        device_desc = f"cpu ({os.cpu_count()} logical cores)"

    precision = str(trainer.precision)
    amp_mode = "off"
    if precision in {"16", "16-mixed"}:
        amp_mode = "fp16-mixed"
    elif precision in {"bf16", "bf16-mixed"}:
        amp_mode = "bf16-mixed"

    grad_scaler_enabled = getattr(trainer.precision_plugin, "scaler", None) is not None

    logging.info("=== TrainFlow runtime ===")
    logging.info(f"  python : {py_ver}  |  torch : {torch_ver}  |  os : {os_info}")
    logging.info(f"  device : {device_desc}")
    logging.info(
        f"  precision : {precision}  |  autocast : {amp_mode}  "
        f"|  grad_scaler : {grad_scaler_enabled}"
    )
    logging.info(
        f"  strategy : {trainer.strategy.__class__.__name__}  "
        f"|  global_zero : {trainer.strategy.is_global_zero}  "
        f"|  world_size : {_dist_world_size()}"
    )
    logging.info(
        f"  max_epochs : {trainer.max_epochs}  "
        f"|  grad_accum : {trainer.gradient_accumulation_steps}  "
        f"|  grad_clip : {trainer.gradient_clip_val}  "
        f"|  compile : {trainer.compiler}"
    )
    logging.info("=========================")


def _print_module_parameter_summary(
    module: nn.Module,
    *,
    name: str = "model",
    max_depth: int = 3,
) -> None:
    """打印参数量摘要,按 ``max_depth`` 层级递归展开各子模块。

    Args:
        module: 顶层模块。
        name: 顶层显示名。
        max_depth: 递归深度;``1`` 仅展开直接子模块,``2`` 多展开一层,以此类推。
            自身永远会被打印,只对子模块的展开起作用。
    """

    def _stats(m: nn.Module) -> tuple[int, int, int]:
        params = list(m.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        bytes_ = sum(p.numel() * p.element_size() for p in params)
        return total, trainable, bytes_

    # Restrict to global-zero so the summary is logged once under DDP.
    from trainflow._rank_zero import is_global_zero as _is_global_zero

    if not _is_global_zero():
        return

    def _line(prefix: str, label: str, total: int, trainable: int, bytes_: int) -> str:
        mib = bytes_ / (1024**2)
        return (
            f"{prefix}{label}: {total:,} params "
            f"({trainable:,} trainable), ~{mib:.2f} MiB"
        )

    n_total, n_trainable, n_bytes = _stats(module)
    logging.info(
        f"[{name}] params: total={n_total:,} trainable={n_trainable:,} "
        f"weight_bytes≈{n_bytes:,} (~{n_bytes / (1024**2):.2f} MiB)"
    )

    def _walk(m: nn.Module, depth: int, indent: str) -> None:
        if depth >= max_depth:
            return
        children = [(n, c) for n, c in m.named_children() if any(True for _ in c.parameters())]
        last_idx = len(children) - 1
        for i, (child_name, child) in enumerate(children):
            total, trainable, bytes_ = _stats(child)
            branch = "└─" if i == last_idx else "├─"
            label = f"{child_name} ({type(child).__name__})"
            logging.info(_line(f"{indent}{branch} ", label, total, trainable, bytes_))
            next_indent = indent + ("   " if i == last_idx else "│  ")
            _walk(child, depth + 1, next_indent)

    _walk(module, depth=0, indent="  ")


def _maybe_seed(cfg: DictConfig) -> None:
    """Seed all RNGs when a ``seed`` is configured (top-level or under ``trainflow``)."""
    seed = OmegaConf.select(cfg, "seed")
    if seed is None:
        seed = OmegaConf.select(cfg, "trainflow.seed")
    if seed is None:
        return
    from trainflow.seed import seed_everything

    seed_everything(int(seed))
    logging.info(f"seeded everything with seed={int(seed)}")


def run_fit(cfg: DictConfig) -> None:
    """Instantiate trainflow and run ``fit``. Resume when ``resume_checkpoint`` is set or
    ``auto_load_checkpoint`` is true with ``checkpoint_path``.
    """
    from trainflow.distributed import setup_distributed, teardown_distributed

    setup_distributed()
    try:
        logging.info("start train...")
        _maybe_seed(cfg)
        trainer, model, datamodule = instantiate_trainflow(cfg)
        _print_runtime_env(trainer)
        if isinstance(model, nn.Module):
            _print_module_parameter_summary(model)
        ckpt = fit_resume_checkpoint_path(cfg)
        if ckpt is None:
            trainer.fit(model, datamodule)
            return
        strict, weights_only = resolve_strict_weights_only(cfg)
        trainer.fit(
            model,
            datamodule,
            ckpt_path=ckpt,
            ckpt_strict=strict,
            ckpt_weights_only=weights_only,
        )
        logging.info("train done!")
    finally:
        teardown_distributed()

def run_validate(cfg: DictConfig) -> None:
    from trainflow.distributed import setup_distributed, teardown_distributed

    setup_distributed()
    try:
        logging.info("start validate...")
        trainer, model, datamodule = instantiate_trainflow(cfg)
        _print_runtime_env(trainer)
        trainer.model = model
        if isinstance(model, nn.Module):
            _print_module_parameter_summary(model)
        trainer.datamodule = datamodule
        ckpt = _select_resume_checkpoint(cfg)
        if ckpt is None:
            raise ValueError("Missing config key `resume_checkpoint` (required for validate).")
        strict, weights_only = resolve_strict_weights_only(cfg)
        trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
        metrics = trainer.validate()
        logging.info("validate done!")
        return metrics
    finally:
        teardown_distributed()


def run_test(cfg: DictConfig) -> None:
    from trainflow.distributed import setup_distributed, teardown_distributed

    setup_distributed()
    try:
        logging.info("start test...")
        trainer, model, datamodule = instantiate_trainflow(cfg)
        _print_runtime_env(trainer)
        trainer.model = model
        if isinstance(model, nn.Module):
            _print_module_parameter_summary(model)
        trainer.datamodule = datamodule
        ckpt = _select_resume_checkpoint(cfg)
        if ckpt is None:
            raise ValueError("Missing config key `resume_checkpoint` (required for test).")
        strict, weights_only = resolve_strict_weights_only(cfg)
        trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
        metrics = trainer.test()
        logging.info("test done!")
        return metrics
    finally:
        teardown_distributed()


def run_predict(cfg: DictConfig) -> list[Any]:
    from trainflow.distributed import setup_distributed, teardown_distributed

    setup_distributed()
    try:
        logging.info("start predict...")
        trainer, model, datamodule = instantiate_trainflow(cfg)
        _print_runtime_env(trainer)
        trainer.model = model
        if isinstance(model, nn.Module):
            _print_module_parameter_summary(model)
        trainer.datamodule = datamodule
        ckpt = _select_resume_checkpoint(cfg)
        if ckpt is None:
            raise ValueError("Missing config key `resume_checkpoint` (required for predict).")
        strict, weights_only = resolve_strict_weights_only(cfg)
        trainer.load_checkpoint(ckpt, strict=strict, weights_only=weights_only)
        outputs = trainer.predict()
        logging.info("predict done!")
        return outputs
    finally:
        teardown_distributed()