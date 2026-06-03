from __future__ import annotations

import importlib
import os
import random
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._rank_zero import rank_zero_warn
from .callbacks import Callback
from .data import DataModule
from .distributed import clone_dataloader_with_sampler, is_distributed, make_distributed_sampler
from .precision import Precision, build_precision
from .loggers import Logger, LoggerCollection, NoOpLogger
from .metrics import MetricCollector
from .model import TrainableModel
from .strategies import Strategy, build_strategy

# Bump when the checkpoint schema changes in a backward-incompatible way.
CHECKPOINT_VERSION = 1


def _numpy_pickling_safe_globals() -> list[Any]:
    """Types referenced when unpickling NumPy arrays (e.g. ``np.random.get_state()`` in checkpoints)."""
    extra: list[Any] = [np.ndarray, np.dtype]
    for mod_name in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            mod = importlib.import_module(mod_name)
            extra.append(mod._reconstruct)
            break
        except (ImportError, AttributeError):
            continue
    # NumPy 2.x / 1.26+: concrete scalar dtypes are classes under ``numpy.dtypes`` (e.g. UInt32DType for
    # MT19937 key array); ``weights_only`` requires them in ``safe_globals``, not only ``numpy.dtype``.
    try:
        import numpy.dtypes as np_dtypes

        for name in dir(np_dtypes):
            obj = getattr(np_dtypes, name, None)
            if isinstance(obj, type) and name.endswith("DType"):
                extra.append(obj)
    except ImportError:
        pass
    return extra


def _weights_only_load_context():
    """Allow NumPy objects pickled inside checkpoints when using ``weights_only=True`` (PyTorch ≥2.6)."""
    try:
        from torch.serialization import safe_globals
    except ImportError:  # pragma: no cover — very old PyTorch
        return nullcontext()
    return safe_globals(_numpy_pickling_safe_globals())


def _torch_load_checkpoint(path: Path | str, *, map_location: Any, weights_only: bool):
    ctx = _weights_only_load_context() if weights_only else nullcontext()
    try:
        with ctx:
            return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # pragma: no cover — older PyTorch without weights_only
        return torch.load(path, map_location=map_location)


@dataclass
class OptimizerBundle:
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[Any]


@dataclass
class SchedulerConfig:
    """Normalised scheduler entry.

    ``interval`` is ``"step"`` (per optimizer step, the existing default) or ``"epoch"``.
    ``monitor`` names the metric fed to ``ReduceLROnPlateau.step(metric)``.
    """

    scheduler: Any
    interval: str = "step"
    frequency: int = 1
    monitor: str | None = None

    @property
    def is_plateau(self) -> bool:
        return isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


class Trainer:
    def __init__(
        self,
        *,
        max_epochs: int = 1,
        accelerator: str = "auto",
        devices: int | str | None = None,
        precision: str | int = "32",
        strategy: str | Strategy | None = None,
        gradient_accumulation_steps: int = 1,
        gradient_clip_val: float | None = None,
        log_every_n_steps: int = 50,
        callbacks: list[Callback] | None = None,
        logger: Logger | list[Logger] | None = None,
        compiler: bool = False,
    ) -> None:
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = str(precision)
        self.precision_plugin: Precision = build_precision(precision)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_steps = log_every_n_steps
        self.callbacks = callbacks or []
        if logger is None:
            self.logger: Logger = NoOpLogger()
        elif isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = LoggerCollection(logger)
        self.compiler = compiler

        self.strategy = strategy if isinstance(strategy, Strategy) else build_strategy(strategy)
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.stage = "init"
        self.current_metrics: dict[str, float] = {}
        self.model: TrainableModel
        self.wrapped_model: torch.nn.Module
        self.datamodule: DataModule
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[Any] = []
        self.scheduler_configs: list[SchedulerConfig] = []
        self._active_collector: MetricCollector | None = None
        self.device = self._select_device()

    def fit(
        self,
        model: TrainableModel,
        datamodule: DataModule,
        *,
        ckpt_path: str | Path | None = None,
        ckpt_strict: bool = True,
        ckpt_weights_only: bool = False,
    ) -> None:
        """Train ``model`` on ``datamodule``.

        When ``ckpt_path`` is set, **model weights** are loaded **before** ``torch.compile`` (if
        enabled) inside ``_setup_fit``, so ``load_state_dict`` always targets the plain module.
        Training state (epoch, optimizers, schedulers, callbacks, RNG) is restored **after**
        ``_setup_fit`` when ``ckpt_weights_only=False``. Callbacks still see the full resumed
        state before ``on_fit_start``.

        - ``ckpt_weights_only=True``: load ``model_state_dict`` only; optimizer/scheduler stay
          freshly configured on the loaded weights.
        - ``ckpt_weights_only=False``: full resume (epoch, step, optimizer, schedulers,
          callbacks state, RNG) as saved by :meth:`save_checkpoint`.
        """
        self.model = model
        self.datamodule = datamodule
        self.stage = "fit"
        resume_state: dict[str, Any] | None = None
        if ckpt_path is not None:
            resume_state = _torch_load_checkpoint(
                ckpt_path, map_location="cpu", weights_only=ckpt_weights_only
            )
            self._apply_checkpoint_model_state(resume_state, strict=ckpt_strict)
        self._setup_fit()
        if ckpt_path is not None and not ckpt_weights_only and resume_state is not None:
            self._restore_training_state_from_checkpoint(resume_state)
        try:
            self._call("on_fit_start")
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                self._train_epoch()
                self._validate_epoch()
                # Agree on stopping across all ranks so DDP collective ops never desync/hang.
                if self.strategy.reduce_bool_any(self.should_stop):
                    self.should_stop = True
                    break
            self._call("on_fit_end")
        except BaseException as exc:
            # Give callbacks a chance to react (e.g. emergency checkpoint) and surface the error.
            self._call("on_exception", exc)
            raise
        finally:
            self.logger.finalize()
            self.stage = "init"

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.model._trainer = self
        self.datamodule.setup(stage="validate")
        previous_stage = self.stage
        if previous_stage != "fit":
            self.stage = "validate"
        try:
            metrics = self._run_eval_loop(stage="val")
        finally:
            self.stage = previous_stage
        if metrics:
            # _run_eval_loop already reduced across ranks; avoid a second redundant all-reduce.
            self.log(metrics, reduce=False)
            self.logger.finalize()
        return metrics

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.model._trainer = self
        self.datamodule.setup(stage="test")
        previous_stage = self.stage
        self.stage = "test"
        try:
            metrics = self._run_eval_loop(stage="test")
        finally:
            self.stage = previous_stage
        if metrics:
            # _run_eval_loop already reduced across ranks; avoid a second redundant all-reduce.
            self.log(metrics, reduce=False)
            self.logger.finalize()
        return metrics

    @torch.no_grad()
    def predict(self) -> list[Any]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.model._trainer = self
        self.datamodule.setup(stage="predict")
        previous_stage = self.stage
        self.stage = "predict"
        self.model.eval()
        dl = self._prepare_dataloader(self.datamodule.predict_dataloader(), shuffle=False)
        outputs: list[Any] = []
        self._call("on_predict_epoch_start")
        try:
            for batch_idx, batch in enumerate(dl):
                self._call("on_predict_batch_start", batch, batch_idx)
                batch = self._to_device(batch)
                output = self.model.predict_step(batch)
                outputs.append(output)
                self._call("on_predict_batch_end", output, batch, batch_idx)
            self._call("on_predict_epoch_end")
        finally:
            self.stage = previous_stage
        return outputs

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        callback_states = {cb.__class__.__name__: cb.state_dict() for cb in self.callbacks}
        state = {
            "version": CHECKPOINT_VERSION,
            # ``epoch`` is the next epoch to run on resume (i.e. completed epochs). Checkpoints are
            # written after an epoch finishes, so storing ``current_epoch + 1`` lets ``fit`` resume
            # at the following epoch instead of repeating the one just completed.
            "epoch": self.current_epoch + 1,
            "global_step": self.global_step,
            "model_state_dict": self._unwrap_model_for_state_dict().state_dict(),
            "optimizer_state_dict": [opt.state_dict() for opt in self.optimizers],
            "lr_scheduler_state_dict": [sch.state_dict() for sch in self.schedulers if hasattr(sch, "state_dict")],
            "precision_state": self.precision_plugin.state_dict(),
            "callback_states": callback_states,
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        self._atomic_torch_save(state, path)

    @staticmethod
    def _atomic_torch_save(state: dict[str, Any], path: Path) -> None:
        """Write to a temp file in the same dir then ``os.replace`` so a crash never leaves a
        half-written (corrupt) checkpoint at ``path``."""
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def load_checkpoint(self, path: str | Path, strict: bool = True, weights_only: bool = False) -> None:
        state = _torch_load_checkpoint(path, map_location="cpu", weights_only=weights_only)
        self._apply_checkpoint_model_state(state, strict=strict)
        if weights_only:
            return
        self._restore_training_state_from_checkpoint(state)

    def _unwrap_model_for_state_dict(self) -> TrainableModel:
        """Prefer the inner module when ``self.model`` is ``torch.compile``-wrapped."""
        inner = getattr(self.model, "_orig_mod", None)
        if inner is not None:
            return inner  # type: ignore[return-value]
        return self.model

    def _apply_checkpoint_model_state(self, state: dict[str, Any], *, strict: bool) -> None:
        self._unwrap_model_for_state_dict().load_state_dict(state["model_state_dict"], strict=strict)

    def _restore_training_state_from_checkpoint(self, state: dict[str, Any]) -> None:
        self.current_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        for opt, opt_state in zip(self.optimizers, state.get("optimizer_state_dict", [])):
            opt.load_state_dict(opt_state)
        for sch, sch_state in zip(self.schedulers, state.get("lr_scheduler_state_dict", [])):
            if hasattr(sch, "load_state_dict"):
                sch.load_state_dict(sch_state)
        precision_state = state.get("precision_state")
        if precision_state:
            self.precision_plugin.load_state_dict(precision_state)
        callback_states = state.get("callback_states", {})
        for cb in self.callbacks:
            if cb.__class__.__name__ in callback_states:
                cb.load_state_dict(callback_states[cb.__class__.__name__])
        rs = state.get("random_state")
        if rs:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            cuda_state = rs.get("cuda")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)

    def log(self, metrics: dict[str, float], *, reduce: bool = True) -> None:
        """Reduce ``metrics`` across ranks (unless ``reduce=False``), store and forward to the logger.

        Pass ``reduce=False`` for values already reduced by ``reduce_metrics`` (e.g. the dict returned
        by ``_run_eval_loop``) to avoid a redundant second cross-rank all-reduce.
        """
        reduced = self.strategy.reduce_metrics(metrics) if reduce else metrics
        self.current_metrics.update(reduced)
        if self.strategy.is_global_zero:
            self.logger.log_metrics(reduced, step=self.global_step)

    def _current_lr_metrics(self, prefix: str) -> dict[str, float]:
        """当前各 optimizer param_group 的学习率，便于写入 CSV / TensorBoard。"""
        out: dict[str, float] = {}
        if not self.optimizers:
            return out
        for oi, opt in enumerate(self.optimizers):
            base = f"{prefix}/lr" if len(self.optimizers) == 1 else f"{prefix}/lr_opt{oi}"
            for pi, pg in enumerate(opt.param_groups):
                key = base if len(opt.param_groups) == 1 else f"{base}_pg{pi}"
                out[key] = float(pg["lr"])
        return out

    def _setup_fit(self) -> None:
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")
        self.model.to(self.device)
        if self.compiler and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        # Bind on the unwrapped module so ``model.log`` works whether or not ``torch.compile`` wraps.
        self._unwrap_model_for_state_dict()._trainer = self
        self.wrapped_model = self.strategy.prepare_model(self.model)
        self._configure_optimizers()
        self.strategy.setup(self)

    def _configure_optimizers(self) -> None:
        configured = self.model.configure_optimizers()
        opts, raw_schedulers = self._parse_optimizer_config(configured)
        self.optimizers = [o for o in opts if o is not None]
        self.scheduler_configs = [
            self._normalize_scheduler(s) for s in raw_schedulers if s is not None
        ]
        self.schedulers = [cfg.scheduler for cfg in self.scheduler_configs]

    @staticmethod
    def _parse_optimizer_config(configured: Any) -> tuple[list[Any], list[Any]]:
        """Normalise ``configure_optimizers()`` return into ``(optimizers, schedulers)``.

        Accepts:
        - a single ``Optimizer``;
        - a list/tuple of ``Optimizer`` (no schedulers);
        - a 2-tuple ``(optimizers, schedulers)``;
        - a dict ``{"optimizer": ..., "lr_scheduler": ...}`` (Lightning-style, recommended).
        """
        def as_list(x: Any) -> list[Any]:
            return list(x) if isinstance(x, (list, tuple)) else [x]

        if isinstance(configured, torch.optim.Optimizer):
            return [configured], []
        if isinstance(configured, dict):
            opts = configured.get("optimizer") or configured.get("optimizers")
            if opts is None:
                raise ValueError("configure_optimizers() dict must include `optimizer`.")
            schs = configured.get("lr_scheduler") or configured.get("schedulers") or []
            return as_list(opts), as_list(schs)
        if isinstance(configured, (list, tuple)):
            if configured and all(isinstance(o, torch.optim.Optimizer) for o in configured):
                return list(configured), []
            if len(configured) == 2:
                opts, schs = configured
                return as_list(opts), as_list(schs)
        raise TypeError(
            "Unsupported configure_optimizers() return type. Return an Optimizer, a list of "
            "Optimizers, a (optimizers, schedulers) tuple, or a dict with `optimizer`/`lr_scheduler`."
        )

    @staticmethod
    def _normalize_scheduler(entry: Any) -> SchedulerConfig:
        """Accept a bare scheduler or a Lightning-style ``{"scheduler": ..., "interval": ...}`` dict.

        A bare scheduler defaults to ``interval="step"``, preserving the previous per-step behaviour.
        """
        if isinstance(entry, SchedulerConfig):
            return entry
        if isinstance(entry, dict):
            scheduler = entry.get("scheduler")
            if scheduler is None:
                raise ValueError("Scheduler config dict must include `scheduler`.")
            interval = str(entry.get("interval", "step"))
            if interval not in ("step", "epoch"):
                raise ValueError(f"Scheduler interval must be 'step' or 'epoch', got {interval!r}.")
            return SchedulerConfig(
                scheduler=scheduler,
                interval=interval,
                frequency=max(1, int(entry.get("frequency", 1))),
                monitor=entry.get("monitor"),
            )
        return SchedulerConfig(scheduler=entry)

    def _step_schedulers(self, interval: str) -> None:
        for cfg in self.scheduler_configs:
            if cfg.interval != interval:
                continue
            scheduler = cfg.scheduler
            if not hasattr(scheduler, "step"):
                continue
            if cfg.is_plateau:
                monitor = cfg.monitor or "val/loss"
                if monitor not in self.current_metrics:
                    continue
                scheduler.step(self.current_metrics[monitor])
            else:
                scheduler.step()

    def _prepare_dataloader(self, loader: Any, *, shuffle: bool, set_epoch: bool = False) -> Any:
        """Centrally inject a ``DistributedSampler`` under DDP so DataModules stay distribution-agnostic.

        Off DDP (``is_distributed()`` False) the loader is returned unchanged — single-device
        behaviour is bit-for-bit identical. If the loader already uses a ``DistributedSampler``
        (e.g. a DataModule that injects its own), only ``set_epoch`` is applied and no cloning
        happens, so user-managed sharding is respected. Non-``DataLoader`` iterables pass through.
        """
        from torch.utils.data import DataLoader, DistributedSampler

        if not isinstance(loader, DataLoader):
            return loader
        if not is_distributed():
            return loader

        existing = loader.sampler
        if isinstance(existing, DistributedSampler):
            if set_epoch:
                existing.set_epoch(self.current_epoch)
            return loader

        sampler = make_distributed_sampler(
            loader.dataset, shuffle=shuffle, drop_last=loader.drop_last
        )
        if sampler is None:
            return loader
        if set_epoch:
            sampler.set_epoch(self.current_epoch)
        return clone_dataloader_with_sampler(loader, sampler)

    def _train_epoch(self) -> None:
        self.model.train()
        self._call("on_train_epoch_start")
        self.datamodule.set_epoch(self.current_epoch)
        collector = MetricCollector("train")
        self._active_collector = collector
        dataloader = self._prepare_dataloader(
            self.datamodule.train_dataloader(), shuffle=True, set_epoch=True
        )
        num_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
        pending_grad = False
        try:
            for batch_idx, batch in enumerate(dataloader):
                batch = self._to_device(batch)
                self._call("on_train_batch_start", batch, batch_idx)
                accumulation_index = (batch_idx % self.gradient_accumulation_steps) + 1
                is_last_batch = num_batches is not None and batch_idx == num_batches - 1
                sync_context = (
                    self.strategy.no_sync_context(self.wrapped_model)
                    if accumulation_index < self.gradient_accumulation_steps and not is_last_batch
                    else nullcontext()
                )
                with sync_context:
                    with self.precision_plugin.autocast_context(self.device):
                        output = self._run_train_step(batch)
                        loss, metrics = self._parse_step_output(output)
                    loss = loss / self.gradient_accumulation_steps
                    self._call("on_before_backward", loss)
                    self.model.on_before_backward(loss)
                    self.precision_plugin.backward(loss, self.strategy)
                    self.model.on_after_backward()
                    self._call("on_after_backward")
                    pending_grad = True

                should_step = accumulation_index == self.gradient_accumulation_steps or is_last_batch
                if should_step:
                    self._optimizer_step()
                    pending_grad = False
                self.global_step += 1
                self._call("on_train_batch_end", output, batch, batch_idx)
                if self.global_step % self.log_every_n_steps == 0:
                    row = {f"train/{k}": float(v) for k, v in metrics.items()} if metrics else {}
                    row.update(collector.step_metrics())
                    if row:
                        row.update(self._current_lr_metrics("train"))
                        row.update({"train/epoch": self.current_epoch})
                        self.log(row)
            if pending_grad:
                self._optimizer_step()
            # Epoch-aggregated declarative metrics (reduced across ranks inside ``log``).
            epoch_metrics = collector.epoch_metrics()
            if epoch_metrics:
                epoch_metrics["train/epoch"] = self.current_epoch
                self.log(epoch_metrics)
        finally:
            self._active_collector = None
        self._call("on_train_epoch_end")

    def _optimizer_step(self) -> None:
        self._call("on_before_optimizer_step")
        clip_fn = None
        if self.gradient_clip_val is not None:
            clip_fn = self._clip_optimizer_grads
        stepped = True
        for opt in self.optimizers:
            applied = self.precision_plugin.optimizer_step(opt, clip_fn=clip_fn)
            stepped = stepped and applied
            opt.zero_grad(set_to_none=True)
        # When AMP detects inf/NaN grads it skips ``optimizer.step`` (weights unchanged), so per-step
        # schedulers must not advance either.
        if stepped:
            self._step_schedulers("step")
        self._call("on_after_optimizer_step")

    def _clip_optimizer_grads(self, opt: torch.optim.Optimizer) -> None:
        """Clip only the parameters owned by ``opt`` (not the whole model)."""
        params = [p for group in opt.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, self.gradient_clip_val)

    @torch.no_grad()
    def _validate_epoch(self) -> None:
        self.datamodule.setup(stage="validate")
        metrics = self._run_eval_loop(stage="val")
        if metrics:
            # Already reduced across ranks inside _run_eval_loop.
            self.log(metrics, reduce=False)
        self._step_schedulers("epoch")

    @torch.no_grad()
    def _run_eval_loop(self, stage: str) -> dict[str, float]:
        self.model.eval()
        if stage == "val":
            epoch_start_hook, epoch_end_hook = "on_validation_epoch_start", "on_validation_epoch_end"
            batch_start_hook, batch_end_hook = "on_validation_batch_start", "on_validation_batch_end"
            dl = self._prepare_dataloader(self.datamodule.val_dataloader(), shuffle=False)
            step_fn = self.model.validation_step
        elif stage == "test":
            epoch_start_hook, epoch_end_hook = "on_test_epoch_start", "on_test_epoch_end"
            batch_start_hook, batch_end_hook = "on_test_batch_start", "on_test_batch_end"
            dl = self._prepare_dataloader(self.datamodule.test_dataloader(), shuffle=False)
            step_fn = self.model.test_step
        else:
            raise ValueError(f"Unknown eval stage: {stage}")
        self._call(epoch_start_hook)
        collector = MetricCollector(stage)
        self._active_collector = collector
        agg: dict[str, list[float]] = {}
        try:
            for batch_idx, batch in enumerate(dl):
                batch = self._to_device(batch)
                self._call(batch_start_hook, batch, batch_idx)
                with self.precision_plugin.autocast_context(self.device):
                    output = step_fn(batch)
                    _, metrics = self._parse_step_output(output)
                for key, value in metrics.items():
                    agg.setdefault(key, []).append(float(value))
                self._call(batch_end_hook, output, batch, batch_idx)
        finally:
            self._active_collector = None
        # Merge return-dict metrics (mean over batches) with declarative epoch metrics.
        reduced = {f"{stage}/{k}": float(np.mean(v)) for k, v in agg.items() if v}
        reduced.update(collector.epoch_metrics())
        # Aggregate across ranks BEFORE callbacks/checkpoint logic read ``current_metrics`` so every
        # rank observes identical, dataset-wide metrics. Single-device ``reduce_metrics`` is identity,
        # so this is bit-for-bit unchanged off DDP. The ``{stage}/epoch`` key is added afterwards so
        # it is never run through the cross-rank average.
        reduced = self.strategy.reduce_metrics(reduced)
        reduced.update({f"{stage}/epoch": self.current_epoch})
        self.current_metrics.update(reduced)
        self._call(epoch_end_hook)
        return reduced

    def _call(self, hook: str, *args: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook, None)
            if callable(fn):
                fn(self, *args)

    def _collect_metric(
        self,
        name: str,
        value: float,
        *,
        on_step: bool,
        on_epoch: bool,
        reduce_fx: str,
        prog_bar: bool,
    ) -> None:
        """Receive a ``model.log(...)`` call and route it to the active stage collector."""
        if self._active_collector is None:
            return
        self._active_collector.log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fx=reduce_fx,
            prog_bar=prog_bar,
        )

    def _run_train_step(self, batch: Any) -> Any:
        """Run ``training_step`` so DDP gradient sync is armed.

        Single-device (``wrapped_model is model``): call ``training_step`` directly so behaviour is
        bit-for-bit unchanged. Under DDP the computation must run inside
        ``DistributedDataParallel.forward`` so ``prepare_for_backward`` is armed and gradients are
        all-reduced. We temporarily redirect the module's ``forward`` to ``training_step`` (the
        standard forward-redirection pattern); the redirected forward restores the real ``forward``
        before invoking the step so models whose ``training_step`` calls ``self.forward`` do not
        recurse.
        """
        if self.wrapped_model is self.model:
            return self.model.training_step(batch)

        original_forward = self.model.forward

        def _redirected_forward(*_args: Any, **_kwargs: Any) -> Any:
            self.model.forward = original_forward  # type: ignore[method-assign]
            try:
                return self.model.training_step(batch)
            finally:
                self.model.forward = _redirected_forward  # type: ignore[method-assign, assignment]

        self.model.forward = _redirected_forward  # type: ignore[method-assign, assignment]
        try:
            return self.wrapped_model(batch)
        finally:
            self.model.forward = original_forward  # type: ignore[method-assign]

    def _parse_step_output(self, output: Any) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(output, torch.Tensor):
            return output, {"loss": float(output.detach().item())}
        if isinstance(output, dict):
            if "loss" not in output:
                raise ValueError("Step output dict must include `loss`.")
            loss = output["loss"]
            metrics = {}
            for key, value in output.items():
                if key == "loss":
                    metrics[key] = float(loss.detach().item())
                elif isinstance(value, torch.Tensor) and value.ndim == 0:
                    metrics[key] = float(value.detach().item())
                elif isinstance(value, (float, int)):
                    metrics[key] = float(value)
            return loss, metrics
        raise TypeError("Step output must be tensor or dict containing `loss`.")

    def _select_device(self) -> torch.device:
        if self.accelerator == "cpu":
            return torch.device("cpu")
        if self.accelerator == "mps":
            # Explicit opt-in only; the ``auto`` path below stays CPU-unless-CUDA so existing
            # (non-MPS) behaviour is unchanged.
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if torch.cuda.is_available():
            self._validate_devices()
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _validate_devices(self) -> None:
        """Validate the configured ``devices`` count against ``WORLD_SIZE`` and warn on mismatch.

        ``devices`` is treated purely as a device *count* sanity check (the actual device binding
        still follows ``LOCAL_RANK``). ``devices=None`` returns immediately, so the default path is
        unchanged; non-integer values are silently ignored.
        """
        if self.devices is None:
            return
        try:
            requested = int(self.devices)
        except (TypeError, ValueError):
            return
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        mismatch = (world_size > 1 and world_size != requested) or (
            world_size == 1 and requested > 1
        )
        if mismatch:
            rank_zero_warn(
                f"Configured devices={requested} does not match WORLD_SIZE={world_size}; "
                "device binding follows LOCAL_RANK. Launch with "
                f"`torchrun --nproc_per_node={requested}` to use {requested} devices.",
                UserWarning,
                stacklevel=2,
            )

    def _ensure_model_and_datamodule(self) -> None:
        if not hasattr(self, "model") or not hasattr(self, "datamodule"):
            raise RuntimeError(
                "Trainer has no model or datamodule. Call fit(model, datamodule) first, "
                "or assign trainer.model and trainer.datamodule before validate/test/predict."
            )

    def _to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=self._non_blocking(batch))
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
            return type(batch)(*(self._to_device(v) for v in batch))
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        # Custom batch objects (dataclasses, PyG Data, etc.) that implement ``.to(device)``.
        to_fn = getattr(batch, "to", None)
        if callable(to_fn):
            try:
                return to_fn(self.device)
            except (TypeError, RuntimeError):
                return batch
        return batch

    def _non_blocking(self, tensor: torch.Tensor) -> bool:
        """``non_blocking`` only helps for pinned CPU tensors moving to CUDA."""
        return (
            self.device.type == "cuda"
            and tensor.device.type == "cpu"
            and tensor.is_pinned()
        )
