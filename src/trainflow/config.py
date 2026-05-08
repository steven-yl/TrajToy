from __future__ import annotations

import importlib
from typing import Any

try:
    from hydra.utils import instantiate as hydra_instantiate
except ImportError:  # pragma: no cover
    hydra_instantiate = None


def import_from_path(path: str) -> type:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def instantiate(config: dict[str, Any] | None) -> Any:
    """Instantiate from a plain dict using ``class_path`` + ``init_args``.

    For Hydra YAML, prefer ``hydra.utils.instantiate`` or :func:`instantiate_hydra_node`
    with ``_target_`` / ``_partial_`` / ``_convert_``.
    """
    if config is None:
        return None
    class_path = config.get("class_path")
    if not class_path:
        raise ValueError("Missing `class_path` in config.")
    init_args = config.get("init_args", {})
    cls = import_from_path(class_path)
    return cls(**init_args)


def instantiate_many(configs: list[dict[str, Any]] | None) -> list[Any]:
    return [instantiate(cfg) for cfg in (configs or [])]


def instantiate_hydra_node(config: Any, *, recursive: bool = True) -> Any:
    """Call Hydra's ``instantiate`` on an OmegaConf node (``_target_``, recursive children)."""
    if hydra_instantiate is None:
        raise ImportError("hydra-core is required for instantiate_hydra_node")
    return hydra_instantiate(config, _recursive_=recursive)
