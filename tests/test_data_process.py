"""数据处理模块单元测试。"""

import numpy as np
import pytest

from sim_env import VehicleParams
from data_process.process.data_creator import FrameData, EpisodeData
from data_process.process.data_preprocess import (
    TrainingSample,
    _frame_to_state,
    preprocess_episode,
    _STATE_DIM,
)


# ── FrameData ────────────────────────────────────────────────────────

class TestFrameData:
    """帧数据测试。"""

    def test_creation(self):
        frame = FrameData(
            episode_id=0, step=0, timestamp=0.0,
            x=1.0, y=2.0, theta=0.5, v=10.0, steering=0.1,
            action_accel=1.0, action_omega=0.2,
        )
        assert frame.x == 1.0
        assert frame.v == 10.0
        assert frame.action_accel == 1.0

    def test_optional_fields(self):
        frame = FrameData(
            episode_id=0, step=0, timestamp=0.0,
            x=0.0, y=0.0, theta=0.0, v=0.0, steering=0.0,
            action_accel=0.0, action_omega=0.0,
        )
        assert frame.centerline is None
        assert frame.road_segment_types is None
        assert frame.reward == 0.0


# ── EpisodeData ──────────────────────────────────────────────────────

class TestEpisodeData:
    """Episode 数据测试。"""

    def test_creation(self):
        episode = EpisodeData(
            episode_id=0, seed=42, total_steps=100,
            total_reward=50.0, final_progress=0.8,
            terminated=False, truncated=True,
        )
        assert episode.episode_id == 0
        assert episode.seed == 42
        assert len(episode.frames) == 0

    def test_with_frames(self):
        frames = [
            FrameData(
                episode_id=0, step=i, timestamp=i * 0.1,
                x=float(i), y=0.0, theta=0.0, v=5.0, steering=0.0,
                action_accel=0.5, action_omega=0.0,
            )
            for i in range(10)
        ]
        episode = EpisodeData(
            episode_id=0, seed=42, total_steps=10,
            total_reward=10.0, final_progress=0.5,
            terminated=False, truncated=False,
            frames=frames, dt=0.1,
        )
        assert len(episode.frames) == 10
        assert episode.dt == 0.1


# ── _frame_to_state ──────────────────────────────────────────────────

class TestFrameToState:
    """帧到状态向量转换测试。"""

    def test_output_shape(self):
        frame = FrameData(
            episode_id=0, step=0, timestamp=0.0,
            x=1.0, y=2.0, theta=0.3, v=5.0, steering=0.1,
            action_accel=0.5, action_omega=0.2,
        )
        state = _frame_to_state(frame)
        assert state.shape == (_STATE_DIM,)
        assert state[0] == 1.0  # x
        assert state[1] == 2.0  # y
        assert state[2] == pytest.approx(0.3)  # theta
        assert state[3] == 5.0  # v
        assert state[4] == pytest.approx(0.1)  # steering
        assert state[5] == 0.5  # action_accel
        assert state[6] == pytest.approx(0.2)  # action_omega

    def test_state_dim(self):
        assert _STATE_DIM == 7


# ── preprocess_episode ───────────────────────────────────────────────

def _make_episode(num_frames: int, dt: float = 0.1) -> EpisodeData:
    """创建测试用 episode。"""
    frames = [
        FrameData(
            episode_id=0, step=i, timestamp=i * dt,
            x=float(i), y=float(i * 0.5), theta=0.1 * i,
            v=5.0 + i * 0.1, steering=0.0,
            action_accel=0.5, action_omega=0.0,
            centerline=np.array([[float(j), 0.0] for j in range(5)]),
            left_boundary=np.array([[float(j), 1.0] for j in range(5)]),
            right_boundary=np.array([[float(j), -1.0] for j in range(5)]),
        )
        for i in range(num_frames)
    ]
    return EpisodeData(
        episode_id=0, seed=42, total_steps=num_frames,
        total_reward=10.0, final_progress=0.5,
        terminated=False, truncated=True,
        frames=frames, dt=dt,
        vehicle_params=VehicleParams(),
    )


class TestPreprocessEpisode:
    """Episode 预处理测试。"""

    def test_basic(self):
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        assert len(samples) == 20
        assert all(isinstance(s, TrainingSample) for s in samples)

    def test_history_shape(self):
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        for sample in samples:
            assert sample.history_states.shape == (6, _STATE_DIM)  # H+1
            assert sample.history_mask.shape == (6,)

    def test_future_shape(self):
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        for sample in samples:
            assert sample.future_states.shape == (5, _STATE_DIM)
            assert sample.future_mask.shape == (5,)

    def test_first_frame_history_padding(self):
        """第一帧的历史应大部分被 padding（mask=0）。"""
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        first = samples[0]
        # 第一帧只有当前帧有效，前 5 帧应为 padding
        assert first.history_mask[0] == 0.0  # padding
        assert first.history_mask[-1] == 1.0  # 当前帧

    def test_last_frame_future_padding(self):
        """最后一帧的未来应全部为 padding（mask=0）。"""
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        last = samples[-1]
        assert np.all(last.future_mask == 0.0)

    def test_middle_frame_full_context(self):
        """中间帧应有完整的历史和未来。"""
        episode = _make_episode(20)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        mid = samples[10]
        assert np.all(mid.history_mask == 1.0)
        assert np.all(mid.future_mask == 1.0)

    def test_sample_interval(self):
        """采样间隔应减少样本数。"""
        episode = _make_episode(20)
        samples_1 = preprocess_episode(episode, history_len=5, future_len=5, sample_interval=1)
        samples_5 = preprocess_episode(episode, history_len=5, future_len=5, sample_interval=5)
        assert len(samples_5) < len(samples_1)
        assert len(samples_5) == 4  # 0, 5, 10, 15

    def test_empty_episode(self):
        episode = EpisodeData(
            episode_id=0, seed=0, total_steps=0,
            total_reward=0.0, final_progress=0.0,
            terminated=False, truncated=False,
            frames=[], dt=0.1,
        )
        samples = preprocess_episode(episode)
        assert len(samples) == 0

    def test_single_frame(self):
        episode = _make_episode(1)
        samples = preprocess_episode(episode, history_len=5, future_len=5)
        assert len(samples) == 1
        sample = samples[0]
        assert sample.history_mask[-1] == 1.0
        assert np.all(sample.future_mask == 0.0)

    def test_road_info_preserved(self):
        """道路信息应被保留到样本中。"""
        episode = _make_episode(5)
        samples = preprocess_episode(episode, history_len=2, future_len=2)
        for sample in samples:
            assert sample.centerline is not None
            assert sample.left_boundary is not None
            assert sample.right_boundary is not None

    def test_vehicle_params_preserved(self):
        """车辆参数应被保留。"""
        episode = _make_episode(5)
        samples = preprocess_episode(episode, history_len=2, future_len=2)
        for sample in samples:
            assert sample.vehicle_params is not None
            assert isinstance(sample.vehicle_params, VehicleParams)

    def test_timestamp(self):
        """样本时间戳应为 episode 时间戳 + 帧时间戳。"""
        episode = _make_episode(5, dt=0.1)
        episode.timestamp = 1000.0
        samples = preprocess_episode(episode, history_len=2, future_len=2)
        assert samples[0].timestamp == pytest.approx(1000.0)
        assert samples[1].timestamp == pytest.approx(1000.1)