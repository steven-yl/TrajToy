"""Hydra entry for IL on TrainFlow: ``python -m il.train``."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trainflow.hydra_build import run_fit, run_validate, run_test, run_predict

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.run_mode == "fit":
        print("Running fit...")
        run_fit(cfg)
    elif cfg.run_mode == "validate":
        print("Running validate...")
        run_validate(cfg)
    elif cfg.run_mode == "test":
        print("Running test...")
        run_test(cfg)
    elif cfg.run_mode == "predict":
        print("Running predict...")
        run_predict(cfg)
    print("Done")

if __name__ == "__main__":
    main()
