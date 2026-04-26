"""奖励模型：车道保持、速度跟踪、安全惩罚。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

@dataclass
class RewardWeights:
    """奖励函数权重配置。"""

    lane_keeping: float = 1.0  # 车道保持奖励权重
    speed_tracking: float = 0.5  # 速度跟踪奖励权重
    heading_error: float = 0.3  # 朝向误差惩罚权重
    action_penalty: float = 0.01  # 动作幅度惩罚权重
    off_road_penalty: float = 10.0  # 出界惩罚


@dataclass
class RewardModelConfig:
    """奖励模型配置。"""

    weights: RewardWeights = field(default_factory=RewardWeights)
    target_speed: float = 5.0  # 目标速度 (m/s)


class RewardModel:
    """奖励计算器。"""
    @staticmethod
    def bulid_from_config(cfg: DictConfig) -> RewardModel:
        return RewardModel(cfg)

    def __init__(self, cfg: RewardModelConfig | None = None) -> None:
        self._config = cfg or RewardModelConfig()

    @property
    def target_speed(self) -> float:
        return self._config.target_speed

    @target_speed.setter
    def target_speed(self, value: float) -> None:
        self._config.target_speed = value

    @property
    def weights(self) -> RewardWeights:
        return self._config.weights

    def compute(
        self,
        lateral_offset: float,
        heading_error: float,
        speed: float,
        action: np.ndarray,
        off_road: bool,
        lane_width: float = 3.5,
    ) -> tuple[float, dict[str, float]]:
        """计算单步奖励。

        Args:
            lateral_offset: 横向偏移 (m)
            heading_error: 朝向误差 (rad)
            speed: 当前速度 (m/s)
            action: 动作 [a, omega]
            off_road: 是否出界
            lane_width: 车道宽度 (m)

        Returns:
            (total_reward, reward_components)
        """
        w = self._config.weights

        # 车道保持：高斯衰减
        r_lane = np.exp(-0.5 * (lateral_offset / (lane_width * 0.3)) ** 2)

        # 速度跟踪：越接近目标速度越好
        speed_err = abs(speed - self._config.target_speed) / max(self._config.target_speed, 1.0)
        r_speed = max(1.0 - speed_err, 0.0)

        # 朝向误差惩罚
        r_heading = -abs(heading_error)

        # 动作幅度惩罚
        action = np.asarray(action, dtype=np.float64)
        r_action = -float(np.sum(action ** 2))

        # 出界惩罚
        r_offroad = -1.0 if off_road else 0.0

        total = (
            w.lane_keeping * r_lane
            + w.speed_tracking * r_speed
            + w.heading_error * r_heading
            + w.action_penalty * r_action
            + w.off_road_penalty * r_offroad
        )

        components = {
            "lane_keeping_reward": w.lane_keeping * r_lane,
            "speed_tracking_reward": w.speed_tracking * r_speed,
            "heading_error_reward": w.heading_error * r_heading,
            "action_penalty_reward": w.action_penalty * r_action,
            "off_road_reward": w.off_road_penalty * r_offroad,
        }

        return float(total), components
