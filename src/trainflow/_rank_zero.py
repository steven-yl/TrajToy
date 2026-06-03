"""Rank-zero utilities.

Provide a single, reusable way to restrict side effects (logging, file writes, prints) to the
global-zero rank, so callbacks and user code do not each re-implement ``if not is_global_zero``.
All helpers are no-ops-aware: when distribution is not initialised, ``is_global_zero`` is ``True``
and everything runs normally (single-device path unchanged).
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, Callable, TypeVar

import torch.distributed as dist

_logger = logging.getLogger("trainflow")

F = TypeVar("F", bound=Callable[..., Any])


def global_rank() -> int:
    """Current global rank (0 when distribution is unavailable / uninitialised)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_global_zero() -> bool:
    return global_rank() == 0


def rank_zero_only(fn: F) -> F:
    """Decorate ``fn`` so it only executes on global rank 0; other ranks return ``None``."""

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if is_global_zero():
            return fn(*args, **kwargs)
        return None

    return wrapped  # type: ignore[return-value]


@rank_zero_only
def rank_zero_info(*args: Any, **kwargs: Any) -> None:
    _logger.info(*args, **kwargs)


@rank_zero_only
def rank_zero_warn(message: str, category: type[Warning] = UserWarning, stacklevel: int = 2) -> None:
    warnings.warn(message, category, stacklevel=stacklevel)


@rank_zero_only
def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)
