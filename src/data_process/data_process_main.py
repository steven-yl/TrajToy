"""Hydra 统一入口：数据生产 + 预处理 + 质量分析。

用法:
    python -m data_process.data_process_main                         # 生产 + 预处理
    python -m data_process.data_process_main run_mode=create         # 仅生产
    python -m data_process.data_process_main run_mode=preprocess   # 仅预处理
    python -m data_process.data_process_main run_mode=analyze        # 仅数据分析（读 preprocess 输出）
    python -m data_process.data_process_main run_mode=all            # 生产 + 预处理（默认）
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from data_process.process.data_creator import DataCreator
from data_process.process.data_analysis import analyze_directory
from data_process.process.data_preprocess import preprocess_directory
from sim_env.vehicle_controller import VehicleMPC
from sim_env import (
    RoadVehicleEnv,
)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    mode = cfg.get("run_mode", "all")

    if mode in ("all", "create"):
        print("=" * 60)
        print("数据生产")
        print("=" * 60)
        env = RoadVehicleEnv.bulid_from_config(cfg.env)
        controller = VehicleMPC.bulid_from_config(cfg.controller)
        creator = DataCreator.bulid_from_config(env, controller, cfg.creator)
        creator.create_data()

    if mode in ("all", "preprocess"):
        print("\n" + "=" * 60)
        print("数据预处理")
        print("=" * 60)
        preprocess_directory(cfg)

    if mode == "analyze":
        print("\n" + "=" * 60)
        print("数据集质量分析")
        print("=" * 60)
        analyze_directory(cfg)


if __name__ == "__main__":
    main()
