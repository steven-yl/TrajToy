"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, workers: bool = True) -> int:
    """Seed Python, NumPy and torch (CPU + CUDA) RNGs.

    When ``workers`` is True, also export ``TRAINFLOW_SEED_WORKERS`` so :func:`worker_init_fn` can
    derive per-worker seeds deterministically. Returns the seed for convenience.
    """
    os.environ["TRAINFLOW_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if workers:
        os.environ["TRAINFLOW_SEED_WORKERS"] = "1"
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that gives each worker a distinct, deterministic seed.

    Combines the global seed, worker id, and (under DDP) the rank so different workers/ranks draw
    independent — yet reproducible — augmentation randomness.
    """
    base = os.environ.get("TRAINFLOW_GLOBAL_SEED")
    if base is None:
        return
    rank = 0
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    seed = (int(base) + worker_id + rank * 1000) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
