"""Precision plugins: unify autocast, gradient scaling and backward across device types.

This replaces the CUDA-only autocast/GradScaler logic that used to live inline in ``Trainer``.
Precision is selected from a string and adapts to the actual device:

- ``"32"`` / ``"32-true"``: full precision, no autocast, no scaler.
- ``"16"`` / ``"16-mixed"``: fp16 autocast; GradScaler enabled only on CUDA (fp16 on CPU/MPS runs
  autocast without a scaler, matching PyTorch's support matrix).
- ``"bf16"`` / ``"bf16-mixed"``: bfloat16 autocast (CUDA/CPU/MPS), no scaler needed.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch


def _make_grad_scaler(enabled: bool):
    """Prefer ``torch.amp.GradScaler`` (2.x), fall back to ``torch.cuda.amp.GradScaler``."""
    try:
        from torch import amp as torch_amp

        return torch_amp.GradScaler("cuda", enabled=enabled)
    except (ImportError, TypeError, AttributeError):  # pragma: no cover
        from torch.cuda.amp import GradScaler as CudaGradScaler

        return CudaGradScaler(enabled=enabled)


class Precision:
    """Full-precision (fp32) plugin and base class. No autocast, no scaling."""

    def __init__(self, precision: str = "32") -> None:
        self.precision = precision

    def autocast_context(self, device: torch.device):
        return nullcontext()

    @property
    def scaler(self):
        return None

    def backward(self, loss: torch.Tensor, strategy) -> None:
        strategy.backward(loss)

    def optimizer_step(self, optimizer, *, clip_fn=None) -> bool:
        """Step ``optimizer``. Returns True if the step was applied (always True without scaling)."""
        if clip_fn is not None:
            clip_fn(optimizer)
        optimizer.step()
        return True

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


class MixedPrecision(Precision):
    """fp16/bf16 autocast with optional GradScaler (fp16 on CUDA only)."""

    def __init__(self, precision: str, dtype: torch.dtype) -> None:
        super().__init__(precision)
        self.dtype = dtype
        # GradScaler is meaningful only for fp16 on CUDA.
        self._use_scaler = dtype == torch.float16 and torch.cuda.is_available()
        self._scaler = _make_grad_scaler(self._use_scaler)

    def autocast_context(self, device: torch.device):
        device_type = device.type
        # autocast supports cuda/cpu/mps; anything else → no-op.
        if device_type in ("cuda", "cpu", "mps"):
            # CPU autocast only supports bf16 reliably; fp16-on-cpu falls back to no-op.
            if device_type == "cpu" and self.dtype == torch.float16:
                return nullcontext()
            return torch.autocast(device_type=device_type, dtype=self.dtype)
        return nullcontext()

    @property
    def scaler(self):
        return self._scaler if self._use_scaler else None

    def backward(self, loss: torch.Tensor, strategy) -> None:
        if self._use_scaler:
            self._scaler.scale(loss).backward()
        else:
            strategy.backward(loss)

    def optimizer_step(self, optimizer, *, clip_fn=None) -> bool:
        if not self._use_scaler:
            if clip_fn is not None:
                clip_fn(optimizer)
            optimizer.step()
            return True
        if clip_fn is not None:
            self._scaler.unscale_(optimizer)
            clip_fn(optimizer)
        scale_before = self._scaler.get_scale()
        self._scaler.step(optimizer)
        self._scaler.update()
        # If the scale dropped, AMP found inf/NaN grads and skipped the step.
        return self._scaler.get_scale() >= scale_before

    def state_dict(self) -> dict:
        return {"scaler": self._scaler.state_dict()} if self._use_scaler else {}

    def load_state_dict(self, state: dict) -> None:
        if self._use_scaler and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])


def build_precision(precision: str | int) -> Precision:
    p = str(precision)
    if p in ("16", "16-mixed"):
        return MixedPrecision(p, torch.float16)
    if p in ("bf16", "bf16-mixed"):
        return MixedPrecision(p, torch.bfloat16)
    if p in ("32", "32-true"):
        return Precision(p)
    raise ValueError(
        f"Unsupported precision {precision!r}; expected one of "
        "32, 32-true, 16, 16-mixed, bf16, bf16-mixed."
    )
