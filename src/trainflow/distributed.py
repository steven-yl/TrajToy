"""Distributed helpers for TrainFlow.

Single source of truth for distribution probing and ``DistributedSampler`` construction.
All helpers degrade to single-device behaviour when ``torch.distributed`` is unavailable or
uninitialised, so single-device code paths stay bit-for-bit unchanged.
"""

from __future__ import annotations

import inspect
import logging
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


def setup_distributed(timeout_seconds: int = 1800) -> bool:
    """Initialise the process group when launched under a multi-rank launcher (e.g. ``torchrun``).

    Single source of truth for bringing ``torch.distributed`` up. Behaviour:

    - ``WORLD_SIZE`` unset or ``<= 1``: no-op, returns ``False`` (single-device path unchanged).
    - process group already initialised: no-op, returns ``True`` (idempotent).
    - otherwise: binds the current process to ``cuda:LOCAL_RANK`` (when CUDA is available) **before**
      ``init_process_group`` so collectives and ``DistributedDataParallel`` use the right device,
      then initialises the group (``nccl`` on CUDA, ``gloo`` on CPU).

    Reads ``WORLD_SIZE`` / ``RANK`` / ``LOCAL_RANK`` from the environment, exactly as ``torchrun``
    populates them. Returns ``True`` when a real multi-rank group is active afterwards.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not dist.is_available():
        logging.warning("WORLD_SIZE=%s but torch.distributed is unavailable; running single-device.", world_size)
        return False
    if dist.is_initialized():
        return True

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Bind the device BEFORE init so the NCCL backend and current_device() agree per rank.
        torch.cuda.set_device(local_rank)
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    logging.info(
        "distributed initialised: backend=%s rank=%s/%s local_rank=%s device=%s",
        backend,
        rank,
        world_size,
        local_rank,
        f"cuda:{local_rank}" if use_cuda else "cpu",
    )
    return True


def teardown_distributed() -> None:
    """Destroy the process group if one was initialised. Safe to call unconditionally."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_distributed() -> bool:
    """True only when a real multi-rank process group is initialised.

    Mirrors the ``isDistributed()`` predicate from the bugfix spec: distribution is considered
    active only when ``torch.distributed`` is available, initialised, and ``world_size > 1``.
    """
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def make_distributed_sampler(
    dataset: Dataset,
    *,
    shuffle: bool,
    drop_last: bool = False,
) -> Optional[DistributedSampler]:
    """Return a ``DistributedSampler`` under DDP, otherwise ``None``.

    When ``None`` is returned, callers keep their original (single-device) ``DataLoader``
    configuration unchanged.
    """
    if not is_distributed():
        return None
    return DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)


# Attributes that can be re-supplied to ``DataLoader(...)`` to clone an existing loader. We read
# these off the instance (DataLoader stores them as same-named attributes).
_DATALOADER_KWARGS = (
    "batch_size",
    "num_workers",
    "collate_fn",
    "pin_memory",
    "drop_last",
    "timeout",
    "worker_init_fn",
    "multiprocessing_context",
    "generator",
    "prefetch_factor",
    "persistent_workers",
    "pin_memory_device",
)


def clone_dataloader_with_sampler(loader: DataLoader, sampler) -> DataLoader:
    """Rebuild ``loader`` with ``sampler`` injected (``shuffle`` disabled, sampler owns ordering).

    Preserves the loader's other constructor arguments. Loaders driven by a custom ``batch_sampler``
    are returned unchanged (cannot combine ``sampler`` with ``batch_sampler``).
    """
    # A custom batch_sampler is mutually exclusive with sampler; don't touch those loaders.
    default_bs = getattr(loader, "batch_sampler", None)
    from torch.utils.data import BatchSampler

    if default_bs is not None and not isinstance(default_bs, BatchSampler):
        return loader

    valid = set(inspect.signature(DataLoader.__init__).parameters)
    kwargs = {}
    for name in _DATALOADER_KWARGS:
        if name not in valid:
            continue
        if not hasattr(loader, name):
            continue
        value = getattr(loader, name)
        # prefetch_factor is only accepted when num_workers > 0.
        if name == "prefetch_factor" and getattr(loader, "num_workers", 0) == 0:
            continue
        kwargs[name] = value
    return DataLoader(loader.dataset, shuffle=False, sampler=sampler, **kwargs)

