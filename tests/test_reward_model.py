"""奖励模型单元测试。"""

import numpy as np
import pytest

from sim_env.reward_model import RewardWeights, RewardModelConfig, RewardModel


class TestRewardWeights:
    """奖励权重配置测试。"""

    def test_defaults(self):
        weights = RewardWeights()
        assert weights.lane_keeping == 1.0
        assert weights.speed_tracking == 0.5
        assert weights.heading_error == 0.3
        assert weights.action_penalty == 0.01
        assert weights.off_road_penalty == 10.0

    def test_custom(self):
        weights = RewardWeights(lane_keeping=2.0, off_road_penalty=20.0)
        assert weights.lane_keeping == 2.0
        assert weights.off_road_penalty == 20.0


class TestRewardModel:
    """奖励模型测试。"""

    def test_perfect_driving(self):
        """完美驾驶（居中、目标速度、零朝向误差）应获得高奖励。"""
        model = RewardModel(RewardModelConfig(target_speed=10.0))
        reward, components = model.compute(
            lateral_offset=0.0,
            heading_error=0.0,
            speed=10.0,
            action=np.array([0.0, 0.0]),
            off_road=False,
        )
        assert reward > 0
        assert components["lane_keeping_reward"] > 0
        assert components["speed_tracking_reward"] > 0
        assert components["off_road_reward"] == 0.0

    def test_off_road_penalty(self):
        """出界应受到大惩罚。"""
        model = RewardModel()
        reward_on, _ = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        reward_off, components = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=True,
        )
        assert reward_off < reward_on
        assert components["off_road_reward"] < 0

    def test_lateral_offset_reduces_reward(self):
        """横向偏移越大，车道保持奖励越低。"""
        model = RewardModel()
        _, comp_center = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        _, comp_offset = model.compute(
            lateral_offset=3.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        assert comp_center["lane_keeping_reward"] > comp_offset["lane_keeping_reward"]

    def test_speed_tracking(self):
        """速度偏离目标越大，速度跟踪奖励越低。"""
        model = RewardModel(RewardModelConfig(target_speed=15.0))
        _, comp_match = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=15.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        _, comp_slow = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=5.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        assert comp_match["speed_tracking_reward"] > comp_slow["speed_tracking_reward"]

    def test_heading_error_penalty(self):
        """朝向误差应产生负奖励。"""
        model = RewardModel()
        _, comp_aligned = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        _, comp_misaligned = model.compute(
            lateral_offset=0.0, heading_error=1.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        assert comp_aligned["heading_error_reward"] > comp_misaligned["heading_error_reward"]

    def test_action_penalty(self):
        """大动作应受到惩罚。"""
        model = RewardModel()
        _, comp_zero = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        _, comp_large = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([5.0, 0.5]), off_road=False,
        )
        assert comp_zero["action_penalty_reward"] > comp_large["action_penalty_reward"]

    def test_return_types(self):
        """返回值类型检查。"""
        model = RewardModel()
        reward, components = model.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        assert isinstance(reward, float)
        assert isinstance(components, dict)
        expected_keys = {
            "lane_keeping_reward",
            "speed_tracking_reward",
            "heading_error_reward",
            "action_penalty_reward",
            "off_road_reward",
        }
        assert set(components.keys()) == expected_keys

    def test_custom_weights(self):
        """自定义权重应影响奖励计算。"""
        model_default = RewardModel()
        model_custom = RewardModel(RewardModelConfig(weights=RewardWeights(lane_keeping=10.0)))
        _, comp_default = model_default.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        _, comp_custom = model_custom.compute(
            lateral_offset=0.0, heading_error=0.0, speed=10.0,
            action=np.array([0.0, 0.0]), off_road=False,
        )
        assert comp_custom["lane_keeping_reward"] > comp_default["lane_keeping_reward"]
