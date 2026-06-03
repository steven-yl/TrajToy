"""Distributed helpers for TrainFlow.

Single source of truth for distribution probing and ``DistributedSampler`` construction.
All helpers degrade to single-device behaviour when ``torch.distributed`` is unavailable or
uninitialised, so single-device code paths stay bit-for-bit unchanged.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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

