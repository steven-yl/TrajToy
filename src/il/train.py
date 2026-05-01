"""Hydra 训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
from hydra.utils import instantiate

# Support `python ./src/il/train.py` execution by ensuring `src` is on sys.path.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

# ── Hydra 入口 ───────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra 入口。"""
    # 打印配置
    print(OmegaConf.to_yaml(cfg))
    trainer = instantiate(cfg.training.trainer)
    trainer(cfg.training)

if __name__ == "__main__":
    main()
