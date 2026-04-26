"""测试配置：将 src 目录加入 sys.path，提供公共 fixture。"""

import sys
import os
import subprocess

# 将 src 目录加入 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from sim_env import (
    VehicleState,
    VehicleParams,
    VehicleModel,
    ModelType,
    IntegratorType,
    RoadSegmentType,
    SegmentSpec,
    RoadGenerationConfig,
    RoadModel,
    RewardWeights,
    RewardModel,
    EnvConfig,
    RoadVehicleEnv,
)


def _torch_importable() -> bool:
    """在子进程中探测 torch 是否可导入，避免主进程崩溃。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


if not _torch_importable():
    # 本机环境下 torch 导入会触发 abort 时，跳过 RL 测试文件，保证其余测试可运行。
    collect_ignore = ["test_rl.py"]


@pytest.fixture
def default_vehicle_params():
    """默认车辆参数。"""
    return VehicleParams()


@pytest.fixture
def default_vehicle_model():
    """默认运动学 + 欧拉积分车辆模型。"""
    return VehicleModel()


@pytest.fixture
def default_road_config():
    """默认道路生成配置（仅直道和弯道）。"""
    return RoadGenerationConfig(
        num_random_segments=3,
        num_lanes=2,
        segment_weights={
            RoadSegmentType.STRAIGHT: 5.0,
            RoadSegmentType.CURVE: 5.0,
            RoadSegmentType.INTERSECTION: 0.0,
            RoadSegmentType.SPLIT: 0.0,
            RoadSegmentType.MERGE: 0.0,
        },
    )


@pytest.fixture
def straight_road_model():
    """仅包含直道的道路模型。"""
    model = RoadModel()
    model.generate_fixed([
        SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
    ])
    return model


@pytest.fixture
def default_reward_model():
    """默认奖励模型。"""
    return RewardModel()


@pytest.fixture
def simple_env_config():
    """简单环境配置（短道路、少步数）。"""
    return EnvConfig(
        dt=0.1,
        max_steps=50,
        init_speed=5.0,
        road_points_ahead=10,
        include_road_obs=True,
        road_config=RoadGenerationConfig(
            num_random_segments=2,
            num_lanes=1,
            loop_segments=False,
            segment_weights={
                RoadSegmentType.STRAIGHT: 1.0,
                RoadSegmentType.CURVE: 0.0,
                RoadSegmentType.INTERSECTION: 0.0,
                RoadSegmentType.SPLIT: 0.0,
                RoadSegmentType.MERGE: 0.0,
            },
            straight_length_range=(50.0, 100.0),
        ),
    )


@pytest.fixture
def simple_env(simple_env_config):
    """简单仿真环境实例。"""
    env = RoadVehicleEnv(simple_env_config)
    yield env
    env.close()
