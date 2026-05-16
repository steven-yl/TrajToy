"""Hydra entry for IL on TrainFlow: ``python -m il.train``."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from il.evaluation.close_eval import instantiate_close_eval

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


@hydra.main(version_base=None, config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))
    close_eval = instantiate_close_eval(cfg)
    close_eval.evaluate_closed_loop(cfg)
    logging.info("Done!")

if __name__ == "__main__":
    main()
