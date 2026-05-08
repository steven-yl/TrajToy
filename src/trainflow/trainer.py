from __future__ import annotations

import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .callbacks import Callback
from .data import DataModule
from .loggers import Logger, LoggerCollection, NoOpLogger
from .model import TrainableModel
from .strategies import Strategy, build_strategy


def _make_grad_scaler(enabled: bool):
    """Prefer ``torch.amp.GradScaler`` (2.x), fall back to ``torch.cuda.amp.GradScaler``."""
    try:
        from torch import amp as torch_amp

        return torch_amp.GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError, AttributeError):  # pragma: no cover
        from torch.cuda.amp import GradScaler as CudaGradScaler

        return CudaGradScaler(enabled=enabled)


def _torch_load_checkpoint(path: Path | str, *, map_location: Any, weights_only: bool):
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:  # pragma: no cover — older PyTorch without weights_only
        return torch.load(path, map_location=map_location)


@dataclass
class OptimizerBundle:
    optimizers: list[torch.optim.Optimizer]
    schedulers: list[Any]


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
        self.current_metrics: dict[str, float] = {}
        self.model: TrainableModel
        self.wrapped_model: torch.nn.Module
        self.datamodule: DataModule
        self.optimizers: list[torch.optim.Optimizer] = []
        self.schedulers: list[Any] = []
        self.scaler = _make_grad_scaler(self._use_grad_scaler())
        self.device = self._select_device()

    def fit(self, model: TrainableModel, datamodule: DataModule) -> None:
        self.model = model
        self.datamodule = datamodule
        self._setup_fit()
        self._call("on_fit_start")
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self._train_epoch()
            self._validate_epoch()
            if self.should_stop:
                break
        self._call("on_fit_end")
        self.logger.finalize()

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.datamodule.setup(stage="validate")
        return self._run_eval_loop(stage="val")

    @torch.no_grad()
    def test(self) -> dict[str, float]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.datamodule.setup(stage="test")
        return self._run_eval_loop(stage="test")

    @torch.no_grad()
    def predict(self) -> list[Any]:
        self._ensure_model_and_datamodule()
        self.model.to(self.device)
        self.datamodule.setup(stage="predict")
        self.model.eval()
        dl = self.datamodule.predict_dataloader()
        outputs = []
        for batch in dl:
            outputs.append(self.model.predict_step(self._to_device(batch)))
        return outputs

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        callback_states = {cb.__class__.__name__: cb.state_dict() for cb in self.callbacks}
        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": [opt.state_dict() for opt in self.optimizers],
            "lr_scheduler_state_dict": [sch.state_dict() for sch in self.schedulers if hasattr(sch, "state_dict")],
            "callback_states": callback_states,
            "random_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str | Path, strict: bool = True, weights_only: bool = False) -> None:
        state = _torch_load_checkpoint(path, map_location="cpu", weights_only=weights_only)
        self.model.load_state_dict(state["model_state_dict"], strict=strict)
        if weights_only:
            return
        self.current_epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("global_step", 0))
        for opt, opt_state in zip(self.optimizers, state.get("optimizer_state_dict", [])):
            opt.load_state_dict(opt_state)
        for sch, sch_state in zip(self.schedulers, state.get("lr_scheduler_state_dict", [])):
            if hasattr(sch, "load_state_dict"):
                sch.load_state_dict(sch_state)
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

    def log(self, metrics: dict[str, float]) -> None:
        reduced = self.strategy.reduce_metrics(metrics)
        self.current_metrics.update(reduced)
        if self.strategy.is_global_zero:
            self.logger.log_metrics(reduced, step=self.global_step)

    def _setup_fit(self) -> None:
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="fit")
        self.model.to(self.device)
        if self.compiler and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.wrapped_model = self.strategy.prepare_model(self.model)
        self._configure_optimizers()
        self.strategy.setup(self)

    def _configure_optimizers(self) -> None:
        configured = self.model.configure_optimizers()
        if isinstance(configured, torch.optim.Optimizer):
            self.optimizers = [configured]
            self.schedulers = []
            return
        if isinstance(configured, (list, tuple)):
            if configured and isinstance(configured[0], torch.optim.Optimizer):
                self.optimizers = list(configured)
                self.schedulers = []
                return
            if len(configured) == 2:
                opts, schs = configured
                self.optimizers = opts if isinstance(opts, list) else [opts]
                self.schedulers = schs if isinstance(schs, list) else [schs]
                return
        if isinstance(configured, dict):
            opts = configured.get("optimizer") or configured.get("optimizers")
            schs = configured.get("lr_scheduler") or configured.get("schedulers") or []
            self.optimizers = opts if isinstance(opts, list) else [opts]
            self.schedulers = schs if isinstance(schs, list) else [schs]
            self.optimizers = [o for o in self.optimizers if o is not None]
            self.schedulers = [s for s in self.schedulers if s is not None]
            return
        raise TypeError("Unsupported configure_optimizers() return type.")

    def _train_epoch(self) -> None:
        self.model.train()
        self._call("on_train_epoch_start")
        dataloader = self.datamodule.train_dataloader()
        for batch_idx, batch in enumerate(dataloader):
            batch = self._to_device(batch)
            self._call("on_train_batch_start", batch, batch_idx)
            accumulation_index = (batch_idx % self.gradient_accumulation_steps) + 1
            sync_context = (
                self.strategy.no_sync_context(self.wrapped_model)
                if accumulation_index < self.gradient_accumulation_steps
                else nullcontext()
            )
            with sync_context:
                with self._autocast_context():
                    output = self.model.training_step(batch)
                    loss, metrics = self._parse_step_output(output)
                loss = loss / self.gradient_accumulation_steps
                self._call("on_before_backward", loss)
                self.model.on_before_backward(loss)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    self.strategy.backward(loss)
                self.model.on_after_backward()
                self._call("on_after_backward")

            should_step = accumulation_index == self.gradient_accumulation_steps
            if should_step:
                self._optimizer_step()
            self.global_step += 1
            self._call("on_train_batch_end", output, batch, batch_idx)
            if metrics and self.global_step % self.log_every_n_steps == 0:
                self.log({f"train/{k}": float(v) for k, v in metrics.items()})
        self._call("on_train_epoch_end")

    def _optimizer_step(self) -> None:
        self._call("on_before_optimizer_step")
        for opt in self.optimizers:
            if self.scaler.is_enabled():
                if self.gradient_clip_val is not None:
                    self.scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                self.scaler.step(opt)
            else:
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                opt.step()
            opt.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.update()
        for scheduler in self.schedulers:
            if hasattr(scheduler, "step"):
                scheduler.step()
        self._call("on_after_optimizer_step")

    @torch.no_grad()
    def _validate_epoch(self) -> None:
        self.datamodule.setup(stage="validate")
        metrics = self._run_eval_loop(stage="val")
        if metrics:
            self.log(metrics)

    @torch.no_grad()
    def _run_eval_loop(self, stage: str) -> dict[str, float]:
        self.model.eval()
        if stage == "val":
            epoch_start_hook, epoch_end_hook = "on_validation_epoch_start", "on_validation_epoch_end"
            batch_start_hook, batch_end_hook = "on_validation_batch_start", "on_validation_batch_end"
            dl = self.datamodule.val_dataloader()
            step_fn = self.model.validation_step
        elif stage == "test":
            epoch_start_hook, epoch_end_hook = "on_test_epoch_start", "on_test_epoch_end"
            batch_start_hook, batch_end_hook = "on_test_batch_start", "on_test_batch_end"
            dl = self.datamodule.test_dataloader()
            step_fn = self.model.test_step
        else:
            raise ValueError(f"Unknown eval stage: {stage}")
        self._call(epoch_start_hook)
        agg: dict[str, list[float]] = {}
        for batch_idx, batch in enumerate(dl):
            batch = self._to_device(batch)
            self._call(batch_start_hook, batch, batch_idx)
            with self._autocast_context():
                output = step_fn(batch)
                _, metrics = self._parse_step_output(output)
            for key, value in metrics.items():
                agg.setdefault(key, []).append(float(value))
            self._call(batch_end_hook, output, batch, batch_idx)
        reduced = {f"{stage}/{k}": float(np.mean(v)) for k, v in agg.items() if v}
        self.current_metrics.update(reduced)
        self._call(epoch_end_hook)
        return reduced

    def _call(self, hook: str, *args: Any) -> None:
        for callback in self.callbacks:
            fn = getattr(callback, hook, None)
            if callable(fn):
                fn(self, *args)

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
                elif isinstance(value, torch.Tensor):
                    metrics[key] = float(value.detach().item())
                elif isinstance(value, (float, int)):
                    metrics[key] = float(value)
            return loss, metrics
        raise TypeError("Step output must be tensor or dict containing `loss`.")

    def _autocast_context(self):
        if self.precision in {"16", "16-mixed"} and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.precision in {"bf16", "bf16-mixed"} and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _use_grad_scaler(self) -> bool:
        return self.precision in {"16", "16-mixed"} and torch.cuda.is_available()

    def _select_device(self) -> torch.device:
        if self.accelerator == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

    def _ensure_model_and_datamodule(self) -> None:
        if not hasattr(self, "model") or not hasattr(self, "datamodule"):
            raise RuntimeError(
                "Trainer has no model or datamodule. Call fit(model, datamodule) first, "
                "or assign trainer.model and trainer.datamodule before validate/test/predict."
            )

    def _to_device(self, batch: Any) -> Any:
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(v) for v in batch)
        return batch
