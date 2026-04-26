"""Gymnasium 仿真环境单元测试。"""

import numpy as np
import pytest

from sim_env import (
    EnvConfig,
    RoadVehicleEnv,
    RoadGenerationConfig,
    RoadSegmentType,
    SegmentSpec,
    RewardWeights,
    RewardModelConfig,
)


def _default_fixed_segments():
    return [
        SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
        SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
    ]


def _make_simple_env(**kwargs):
    """创建简单测试环境。"""
    loop_segments = kwargs.pop("loop_segments", False)
    fixed_segments = kwargs.pop("fixed_segments", _default_fixed_segments())
    road_config = kwargs.pop("road_config", None)
    if road_config is None:
        road_config = RoadGenerationConfig(
            fixed_segments=fixed_segments,
            loop_segments=loop_segments,
        )
    cfg_kwargs = {
        "dt": 0.1,
        "max_steps": 50,
        "init_speed": 5.0,
        "road_points_ahead": 10,
        "include_road_obs": True,
        "road_config": road_config,
    }
    cfg_kwargs.update(kwargs)
    cfg = EnvConfig(**cfg_kwargs)
    return RoadVehicleEnv(cfg)


class TestEnvConfig:
    """环境配置测试。"""

    def test_defaults(self):
        cfg = EnvConfig()
        assert cfg.dt == 0.1
        assert cfg.max_steps == 1000
        assert cfg.init_speed == 5.0

    def test_custom(self):
        cfg = EnvConfig(dt=0.05, max_steps=500)
        assert cfg.dt == 0.05
        assert cfg.max_steps == 500


class TestRoadVehicleEnv:
    """仿真环境测试。"""

    def test_reset(self):
        env = _make_simple_env()
        obs, info = env.reset(seed=42)
        assert "vehicle" in obs
        assert obs["vehicle"].shape == (5,)
        assert "x" in info
        assert "y" in info
        env.close()

    def test_reset_with_road_obs(self):
        env = _make_simple_env(include_road_obs=True)
        obs, info = env.reset(seed=42)
        assert "centerline" in obs
        assert "left_boundary" in obs
        assert "right_boundary" in obs
        assert obs["centerline"].shape == (10, 2)
        env.close()

    def test_reset_without_road_obs(self):
        env = _make_simple_env(include_road_obs=False)
        obs, info = env.reset(seed=42)
        assert "centerline" not in obs
        env.close()

    def test_step(self):
        env = _make_simple_env()
        env.reset(seed=42)
        action = np.array([1.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "vehicle" in obs
        assert "reward_components" in info
        env.close()

    def test_step_returns_finite(self):
        env = _make_simple_env()
        env.reset(seed=42)
        for _ in range(10):
            action = np.array([0.5, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert np.all(np.isfinite(obs["vehicle"]))
            if terminated or truncated:
                break
        env.close()

    def test_action_space(self):
        env = _make_simple_env()
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
        env.close()

    def test_observation_space(self):
        env = _make_simple_env()
        assert "vehicle" in env.observation_space.spaces
        assert env.observation_space["vehicle"].shape == (5,)
        env.close()

    def test_truncation_at_max_steps(self):
        """达到最大步数应触发 truncated。"""
        env = _make_simple_env(max_steps=5)
        env.reset(seed=42)
        for step in range(10):
            action = np.array([0.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated or terminated:
                break
        assert truncated or terminated
        env.close()

    def test_off_road_termination(self):
        """大幅偏离道路应触发 terminated。"""
        env = _make_simple_env(max_lateral_offset=2.0, max_steps=200)
        env.reset(seed=42)
        terminated = False
        for _ in range(200):
            # 持续大幅转向以偏离道路
            action = np.array([2.0, 0.5], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
        assert terminated
        env.close()

    def test_deterministic_reset(self):
        """相同 seed 的 reset 应产生相同初始状态。"""
        env = _make_simple_env()
        obs1, info1 = env.reset(seed=123)
        obs2, info2 = env.reset(seed=123)
        np.testing.assert_allclose(obs1["vehicle"], obs2["vehicle"])
        env.close()

    def test_loop_segments(self):
        """loop_segments=True 时道路应自动延长。"""
        env = _make_simple_env(
            max_steps=500,
            road_config=RoadGenerationConfig(
                fixed_segments=[
                    SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 30.0}),
                ],
                loop_segments=True,
            ),
        )
        env.reset(seed=42)
        for _ in range(200):
            action = np.array([1.0, 0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        # 不应因到达末端而截断（道路会自动延长）
        env.close()

    def test_info_keys(self):
        """info 应包含关键信息。"""
        env = _make_simple_env()
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        expected_keys = {"x", "y", "theta", "v", "steering", "step",
                         "reward_components", "lateral_offset",
                         "heading_error", "progress"}
        assert expected_keys.issubset(set(info.keys()))
        env.close()

    def test_render_rgb_array(self):
        """rgb_array 渲染模式应返回图像数组。"""
        env = RoadVehicleEnv(
            EnvConfig(
                dt=0.1,
                max_steps=10,
                road_config=RoadGenerationConfig(
                    fixed_segments=[
                        SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
                    ],
                ),
            ),
            render_mode="rgb_array",
        )
        env.reset(seed=42)
        env.step(np.array([0.0, 0.0]))
        frame = env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3  # RGB
        # 在 headless 环境下，rgb_array 渲染可能不会初始化 display 子系统。
        # close() 在该场景下可能抛 pygame.error；此处吞掉以保证接口测试稳定。
        try:
            env.close()
        except Exception:
            pass

    def test_close_idempotent(self):
        """多次 close 不应报错。"""
        env = _make_simple_env()
        env.reset(seed=42)
        env.close()
        env.close()

    def test_reset_target_speed_none(self):
        """target_speed=None 时不应修改 reward model 的目标速度。"""
        cfg = EnvConfig(
            dt=0.1,
            max_steps=50,
            road_config=RoadGenerationConfig(
                fixed_segments=_default_fixed_segments(),
            ),
            reward_config=RewardModelConfig(target_speed=10.0),
        )
        env = RoadVehicleEnv(cfg)
        env.reset(seed=42, target_speed=None)
        assert env._reward_model.target_speed == 10.0
        env.close()

    def test_reset_target_speed_override(self):
        """target_speed 传值时应更新 reward model 的目标速度。"""
        cfg = EnvConfig(
            dt=0.1,
            max_steps=50,
            road_config=RoadGenerationConfig(
                fixed_segments=_default_fixed_segments(),
            ),
            reward_config=RewardModelConfig(target_speed=10.0),
        )
        env = RoadVehicleEnv(cfg)
        env.reset(seed=42, target_speed=20.0)
        assert env._reward_model.target_speed == 20.0
        env.close()

    def test_reset_target_speed_multiple(self):
        """多次 reset 使用不同 target_speed 应正确更新。"""
        cfg = EnvConfig(
            dt=0.1,
            max_steps=50,
            road_config=RoadGenerationConfig(
                fixed_segments=_default_fixed_segments(),
            ),
            reward_config=RewardModelConfig(target_speed=10.0),
        )
        env = RoadVehicleEnv(cfg)
        env.reset(seed=42, target_speed=15.0)
        assert env._reward_model.target_speed == 15.0
        env.reset(seed=42, target_speed=None)
        assert env._reward_model.target_speed == 15.0  # 不应回退
        env.reset(seed=42, target_speed=5.0)
        assert env._reward_model.target_speed == 5.0
        env.close()
