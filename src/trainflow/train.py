"""Hydra entry for TrainFlow: ``python -m trainflow.train`` (ensure ``src`` on PYTHONPATH)."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trainflow.hydra_build import run_fit

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_fit(cfg)


if __name__ == "__main__":
    main()
