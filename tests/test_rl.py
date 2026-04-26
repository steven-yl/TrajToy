"""PPO 强化学习组件单元测试。"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sim_env import (
    EnvConfig,
    RoadVehicleEnv,
    RoadGenerationConfig,
    RoadSegmentType,
    SegmentSpec,
)
from rl import normal_and_flatten_obs, get_obs_dims, ActorCritic, RolloutBuffer, PPOAgent
from rl.utils.obs_utils import _to_local_coords


def _make_test_ppo_cfg(**agent_overrides):
    """构造与 PPOAgent 当前实现兼容的最小配置。"""
    agent_cfg = {
        "obs_keys": ["vehicle", "centerline", "left_boundary", "right_boundary"],
        "hidden_sizes": [64, 64],
        "lr": 3e-4,
        "steps_per_epoch": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_epochs": 2,
        "batch_size": 8,
        "clip_eps": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
    }
    agent_cfg.update(agent_overrides)
    return OmegaConf.create({"device": "cpu", "agent": agent_cfg})


# ── 坐标转换 ─────────────────────────────────────────────────────────

class TestToLocalCoords:
    """局部坐标转换测试。"""

    def test_identity(self):
        """原点在 (0,0)，朝向 0 时，局部坐标等于全局坐标。"""
        points = np.array([[1.0, 2.0], [3.0, 4.0]])
        origin = np.array([0.0, 0.0])
        result = _to_local_coords(points, origin, 0.0)
        np.testing.assert_allclose(result, points, atol=1e-10)

    def test_translation(self):
        """纯平移。"""
        points = np.array([[5.0, 5.0]])
        origin = np.array([3.0, 2.0])
        result = _to_local_coords(points, origin, 0.0)
        np.testing.assert_allclose(result, [[2.0, 3.0]], atol=1e-10)

    def test_rotation_90(self):
        """朝向 π/2 时，全局 x 方向变为局部 y 反方向。"""
        points = np.array([[1.0, 0.0]])
        origin = np.array([0.0, 0.0])
        result = _to_local_coords(points, origin, np.pi / 2)
        # cos(π/2)=0, sin(π/2)=1
        # local_x = 1*0 + 0*1 = 0
        # local_y = -1*1 + 0*0 = -1
        np.testing.assert_allclose(result, [[0.0, -1.0]], atol=1e-10)

    def test_batch(self):
        """批量转换。"""
        points = np.random.randn(10, 2)
        origin = np.array([1.0, 2.0])
        result = _to_local_coords(points, origin, 0.5)
        assert result.shape == (10, 2)

    def test_3d_input(self):
        """支持 (..., 2) 形状输入。"""
        points = np.random.randn(3, 5, 2)
        origin = np.array([0.0, 0.0])
        result = _to_local_coords(points, origin, 0.0)
        assert result.shape == (3, 5, 2)


# ── 观测处理 ─────────────────────────────────────────────────────────

class TestObsProcessing:
    """观测归一化和展平测试。"""

    @pytest.fixture
    def sample_obs(self):
        return {
            "vehicle": np.array([10.0, 20.0, 0.5, 15.0, 0.1], dtype=np.float32),
            "centerline": np.random.randn(10, 2).astype(np.float32),
            "left_boundary": np.random.randn(10, 2).astype(np.float32),
            "right_boundary": np.random.randn(10, 2).astype(np.float32),
        }

    def test_normal_and_flatten_obs(self, sample_obs):
        obs_keys = ["vehicle", "centerline", "left_boundary", "right_boundary"]
        vehicle_obs, road_obs = normal_and_flatten_obs(sample_obs, obs_keys)
        assert vehicle_obs.shape == (4,)
        assert road_obs.shape == (10 * 2 * 3,)  # 3 road keys × 10 points × 2 coords

    def test_vehicle_obs_normalization(self, sample_obs):
        obs_keys = ["vehicle", "centerline"]
        vehicle_obs, _ = normal_and_flatten_obs(sample_obs, obs_keys)
        # v / 30.0
        assert vehicle_obs[2] == pytest.approx(15.0 / 30.0)
        # steering / (π/3)
        assert vehicle_obs[3] == pytest.approx(0.1 / (np.pi / 3.0))

    def test_empty_road_keys(self, sample_obs):
        vehicle_obs, road_obs = normal_and_flatten_obs(sample_obs, ["vehicle"])
        assert vehicle_obs.shape == (4,)
        assert road_obs.shape == (0,)

    def test_with_lane_dividers(self):
        obs = {
            "vehicle": np.array([0.0, 0.0, 0.0, 10.0, 0.0], dtype=np.float32),
            "centerline": np.zeros((5, 2), dtype=np.float32),
            "lane_dividers": np.zeros((1, 5, 2), dtype=np.float32),
        }
        obs_keys = ["vehicle", "centerline", "lane_dividers"]
        vehicle_obs, road_obs = normal_and_flatten_obs(obs, obs_keys)
        # centerline: 5*2=10, lane_dividers: 1*5*2=10
        assert road_obs.shape == (20,)


class TestGetObsDims:
    """观测维度计算测试。"""

    def test_basic(self):
        env = RoadVehicleEnv(EnvConfig(
            road_points_ahead=10,
            include_road_obs=True,
            road_config=RoadGenerationConfig(
                fixed_segments=[
                    SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
                ],
            ),
        ))
        obs_keys = ["vehicle", "centerline", "left_boundary", "right_boundary"]
        vehicle_dim, road_dim = get_obs_dims(env.observation_space, obs_keys)
        assert vehicle_dim == 4
        assert road_dim == 10 * 2 * 3  # 3 road keys × 10 × 2
        env.close()


# ── ActorCritic 网络 ─────────────────────────────────────────────────

class TestActorCritic:
    """Actor-Critic 网络测试。"""

    @pytest.fixture
    def network(self):
        return ActorCritic(
            vehicle_dim=4,
            road_dim=60,
            act_dim=2,
            hidden_sizes=[64, 64],
        )

    def test_forward(self, network):
        vehicle_obs = torch.randn(1, 4)
        road_obs = torch.randn(1, 60)
        dist, value = network(vehicle_obs, road_obs)
        assert dist.mean.shape == (1, 2)
        assert value.shape == (1,)

    def test_batch_forward(self, network):
        batch_size = 8
        vehicle_obs = torch.randn(batch_size, 4)
        road_obs = torch.randn(batch_size, 60)
        dist, value = network(vehicle_obs, road_obs)
        assert dist.mean.shape == (batch_size, 2)
        assert value.shape == (batch_size,)

    def test_get_action_and_value_sample(self, network):
        vehicle_obs = torch.randn(1, 4)
        road_obs = torch.randn(1, 60)
        action, log_prob, entropy, value = network.get_action_and_value(
            vehicle_obs, road_obs,
        )
        assert action.shape == (1, 2)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_get_action_and_value_evaluate(self, network):
        vehicle_obs = torch.randn(1, 4)
        road_obs = torch.randn(1, 60)
        given_action = torch.randn(1, 2)
        action, log_prob, entropy, value = network.get_action_and_value(
            vehicle_obs, road_obs, action=given_action,
        )
        # 返回的 action 应与给定的相同
        torch.testing.assert_close(action, given_action)

    def test_deterministic_with_same_seed(self, network):
        vehicle_obs = torch.randn(1, 4)
        road_obs = torch.randn(1, 60)
        dist, _ = network(vehicle_obs, road_obs)
        mean1 = dist.mean.detach().clone()
        dist2, _ = network(vehicle_obs, road_obs)
        mean2 = dist2.mean.detach().clone()
        torch.testing.assert_close(mean1, mean2)


# ── RolloutBuffer ────────────────────────────────────────────────────

class TestRolloutBuffer:
    """Rollout Buffer 测试。"""

    @pytest.fixture
    def buffer(self):
        return RolloutBuffer(
            capacity=100,
            vehicle_dim=4,
            road_dim=60,
            act_dim=2,
            device="cpu",
        )

    def test_store_and_ptr(self, buffer):
        for i in range(10):
            buffer.store(
                vehicle_obs=np.random.randn(4).astype(np.float32),
                road_obs=np.random.randn(60).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                reward=float(i),
                done=False,
                log_prob=-0.5,
                value=1.0,
            )
        assert buffer.ptr == 10

    def test_reset(self, buffer):
        buffer.store(
            vehicle_obs=np.zeros(4),
            road_obs=np.zeros(60),
            action=np.zeros(2),
            reward=0.0, done=False, log_prob=0.0, value=0.0,
        )
        assert buffer.ptr == 1
        buffer.reset()
        assert buffer.ptr == 0

    def test_compute_gae(self, buffer):
        for i in range(5):
            buffer.store(
                vehicle_obs=np.zeros(4),
                road_obs=np.zeros(60),
                action=np.zeros(2),
                reward=1.0,
                done=(i == 4),
                log_prob=-0.5,
                value=0.5,
            )
        buffer.compute_gae(last_value=0.0, gamma=0.99, lam=0.95)
        # 优势和回报应被计算
        assert not np.all(buffer.advantages[:5] == 0)
        assert not np.all(buffer.returns[:5] == 0)

    def test_get_batches(self, buffer):
        for _ in range(20):
            buffer.store(
                vehicle_obs=np.random.randn(4).astype(np.float32),
                road_obs=np.random.randn(60).astype(np.float32),
                action=np.random.randn(2).astype(np.float32),
                reward=1.0, done=False, log_prob=-0.5, value=1.0,
            )
        buffer.compute_gae(0.0, 0.99, 0.95)

        batch_count = 0
        total_samples = 0
        for batch in buffer.get_batches(batch_size=8):
            batch_count += 1
            total_samples += batch["vehicle_obs"].shape[0]
            assert "vehicle_obs" in batch
            assert "road_obs" in batch
            assert "actions" in batch
            assert "log_probs" in batch
            assert "advantages" in batch
            assert "returns" in batch
        assert total_samples == 20


# ── PPOAgent ─────────────────────────────────────────────────────────

class TestPPOAgent:
    """PPO 智能体测试。"""

    @pytest.fixture
    def agent_and_env(self):
        env_cfg = EnvConfig(
            dt=0.1,
            max_steps=20,
            init_speed=5.0,
            road_points_ahead=10,
            include_road_obs=True,
            road_config=RoadGenerationConfig(
                fixed_segments=[
                    SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
                    SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
                ],
            ),
        )
        ppo_cfg = _make_test_ppo_cfg(
            steps_per_epoch=32,
            num_epochs=2,
            batch_size=8,
            hidden_sizes=[32, 32],
        )
        env = RoadVehicleEnv(env_cfg)
        agent = PPOAgent(env, ppo_cfg)
        yield agent, env
        env.close()

    def test_predict(self, agent_and_env):
        agent, env = agent_and_env
        obs, info = env.reset(seed=42)
        action = agent.predict(obs, deterministic=True, info=info)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))

    def test_predict_stochastic(self, agent_and_env):
        agent, env = agent_and_env
        obs, info = env.reset(seed=42)
        action = agent.predict(obs, deterministic=False, info=info)
        assert action.shape == (2,)

    def test_collect_rollout(self, agent_and_env):
        agent, env = agent_and_env
        stats = agent.collect_rollout()
        assert "mean_reward" in stats
        assert "num_episodes" in stats
        assert isinstance(stats["mean_reward"], float)

    def test_update(self, agent_and_env):
        agent, env = agent_and_env
        agent.collect_rollout()
        loss_stats = agent.update()
        assert "all_loss" in loss_stats
        assert "pg_loss" in loss_stats
        assert "vf_loss" in loss_stats
        assert "entropy" in loss_stats

    def test_save_and_load(self, agent_and_env, tmp_path):
        agent, env = agent_and_env
        save_path = tmp_path / "test_model.pt"
        agent.save(save_path)
        assert save_path.exists()

        # 创建新 agent 并加载
        new_agent = PPOAgent(env, agent.cfg)
        new_agent.load(save_path)

        # 验证参数一致
        obs, info = env.reset(seed=42)
        action1 = agent.predict(obs, deterministic=True, info=info)
        action2 = new_agent.predict(obs, deterministic=True, info=info)
        np.testing.assert_allclose(action1, action2, atol=1e-5)
