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

from il.training.close_val_mlp import run_close_eval

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)


def _run_mode(cfg: DictConfig) -> str:
    rm = OmegaConf.select(cfg, "run_mode")
    if rm is None:
        rm = OmegaConf.select(cfg, "training.run_mode")
    if rm is None:
        raise ValueError("缺少 run_mode：在配置根节点或 training.run_mode 中设置。")
    return str(rm)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    mode = _run_mode(cfg)
    if mode in ["train", "validate", "test", "predict"]:
        # TrainFlow 期望 trainflow 位于传入的 cfg 根节点；Hydra 使用 training@ 打包时常为 cfg.training
        tf_cfg = cfg if OmegaConf.select(cfg, "trainflow") is not None else cfg.training
        if mode == "train":
            run_fit(tf_cfg)
        elif mode == "validate":
            run_validate(tf_cfg)
        elif mode == "test":
            run_test(tf_cfg)
        elif mode == "predict":
            run_predict(tf_cfg)
        print("Done!")
    elif mode == "close_eval":
        run_close_eval(cfg)
        print("Done!")
    else:
        raise ValueError(f"Unknown run mode: {mode}")

if __name__ == "__main__":
    main()
