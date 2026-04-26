"""数据生产器：批量运行仿真环境，采集训练数据。"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from multiprocessing import Pool
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from sim_env import (
    VehicleParams,
)


# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class FrameData:
    """单帧数据。"""

    episode_id: int
    step: int
    timestamp: float
    # 车辆状态
    x: float
    y: float
    theta: float
    v: float
    steering: float
    # 控制量
    action_accel: float
    action_omega: float
    # 道路信息
    centerline: np.ndarray | None = None
    left_boundary: np.ndarray | None = None
    right_boundary: np.ndarray | None = None
    lane_dividers: np.ndarray | None = None
    actual_length_num: int = 0.0
    road_segment_types: list[str] | None = None
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    progress: float = 0.0
    # 奖励
    reward: float = 0.0
    reward_components: dict = field(default_factory=dict)


@dataclass
class EpisodeData:
    """单个 episode 数据。"""

    episode_id: int
    seed: int
    total_steps: int
    total_reward: float
    final_progress: float
    terminated: bool
    truncated: bool
    frames: list[FrameData] = field(default_factory=list)
    vehicle_params: VehicleParams | None = None
    timestamp: float = 0.0
    dt: float = 0.0


@dataclass
class ProductionReport:
    """生产报告。"""

    total_episodes: int = 0
    total_frames: int = 0
    total_time_s: float = 0.0
    fps: float = 0.0
    avg_episode_length: float = 0.0
    avg_reward: float = 0.0
    avg_progress: float = 0.0
    termination_rate: float = 0.0
    truncation_rate: float = 0.0
    output_dir: str = ""
    output_format: str = ""
    episode_files: list[str] = field(default_factory=list)
    created_at: str = ""
    dt: float = 0.0

    def __str__(self) -> str:
        return (
            f"=== 生产报告 ===\n"
            f"  生产时间:    {self.created_at}\n"
            f"  输出目录:    {self.output_dir}\n"
            f"  输出格式:    {self.output_format}\n"
            f"  帧时间间隔:  {self.dt}s\n"
            f"  episodes:    {self.total_episodes}\n"
            f"  frames:      {self.total_frames}\n"
            f"  耗时:        {self.total_time_s:.1f}s\n"
            f"  帧率:        {self.fps:.0f} fps\n"
            f"  平均步数:    {self.avg_episode_length:.0f}\n"
            f"  平均奖励:    {self.avg_reward:.1f}\n"
            f"  平均进度:    {self.avg_progress:.1%}\n"
            f"  平均终止率:  {self.termination_rate:.1%}\n"
            f"  平均截断率:  {self.truncation_rate:.1%}"
        )


# ── 核心逻辑 ──────────────────────────────────────────────────────────
def _run_single_episode(episode_id, cfg, env, controller) -> EpisodeData:
    """运行单个 episode（可被 multiprocessing 调用）。"""
    controller.reset()

    default_action = np.array([0.0, 0.0])

    dt = float(env.config.dt)

    if env.config.random_init and hasattr(env.config.random_config, "random_target_speed_range"):
        low, high = env.config.random_config.random_target_speed_range
        target_speed = np.random.uniform(low, high)
    else:
        target_speed = env.config.reward_config.target_speed
    obs, info = env.reset(seed=episode_id, target_speed=target_speed)
    frames: list[FrameData] = []
    total_reward = 0.0

    timestamp = time.time()
    for step in range(cfg.max_steps_per_episode):
        if controller is not None:
            veh = obs["vehicle"]
            state = np.array([veh[0], veh[1], veh[2], veh[3], info.get("steering", 0.0)])
            ref_path = obs.get("centerline", np.zeros((2, 2)))
            action, _, _ = controller.compute(state, ref_path, target_speed)
        else:
            action = default_action.copy()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % cfg.save_interval == 0:
            frame = FrameData(
                episode_id=episode_id,
                step=step,
                timestamp=step * dt,
                x=info["x"], y=info["y"], theta=info["theta"],
                v=info["v"], steering=info.get("steering", 0.0),
                action_accel=float(action[0]),
                action_omega=float(action[1]),
                lateral_offset=info.get("lateral_offset", 0.0),
                heading_error=info.get("heading_error", 0.0),
                progress=info.get("progress", 0.0),
                reward=reward,
                reward_components=info.get("reward_components", {}),
            )
            if "centerline" in obs:
                frame.centerline = obs["centerline"].copy()
                frame.left_boundary = obs["left_boundary"].copy()
                frame.right_boundary = obs["right_boundary"].copy()
                if "lane_dividers" in obs:
                    frame.lane_dividers = obs["lane_dividers"].copy()
            frame.actual_length_num = info.get("actual_length_num", 0.0)
            raw_seg_types = info.get("road_segment_types", None)
            if raw_seg_types is not None:
                frame.road_segment_types = list(raw_seg_types)
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()

    return EpisodeData(
        episode_id=episode_id,
        seed=episode_id,
        total_steps=info.get("step", step + 1),
        total_reward=total_reward,
        final_progress=info.get("progress", 0.0),
        terminated=terminated,
        truncated=truncated,
        frames=frames,
        vehicle_params=env._vehicle.params,
        timestamp=timestamp,
        dt=dt,
    )


def _run_single_episode_star(args) -> EpisodeData:
    """multiprocessing 参数解包包装器。"""
    return _run_single_episode(*args)


# ── 保存 ──────────────────────────────────────────────────────────────


def _save_episode_pkl(episode: EpisodeData, output_dir: str) -> None:
    path = os.path.join(output_dir, f"episode_{episode.episode_id:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(episode, f, protocol=pickle.HIGHEST_PROTOCOL)


def _frame_to_serializable(frame: FrameData) -> dict:
    """将 FrameData 转为 JSON 可序列化的 dict。"""
    d: dict[str, Any] = {
        "episode_id": frame.episode_id,
        "step": frame.step,
        "timestamp": frame.timestamp,
        "x": frame.x, "y": frame.y, "theta": frame.theta,
        "v": frame.v, "steering": frame.steering,
        "action_accel": frame.action_accel, "action_omega": frame.action_omega,
        "lateral_offset": frame.lateral_offset,
        "heading_error": frame.heading_error,
        "progress": frame.progress,
        "reward": frame.reward,
        "reward_components": frame.reward_components,
    }
    for key in ("centerline", "left_boundary", "right_boundary", "lane_dividers"):
        val = getattr(frame, key, None)
        if val is not None:
            d[key] = val.tolist()
    if frame.road_segment_types is not None:
        d["road_segment_types"] = frame.road_segment_types
    return d


def _save_episode_json(episode: EpisodeData, output_dir: str) -> None:
    path = os.path.join(output_dir, f"episode_{episode.episode_id:04d}.json")
    data = {
        "episode_id": episode.episode_id,
        "seed": episode.seed,
        "total_steps": episode.total_steps,
        "total_reward": episode.total_reward,
        "final_progress": episode.final_progress,
        "terminated": episode.terminated,
        "truncated": episode.truncated,
        "vehicle_params": asdict(episode.vehicle_params) if episode.vehicle_params is not None else None,
        "timestamp": episode.timestamp,
        "dt": episode.dt,
        "frames": [_frame_to_serializable(f) for f in episode.frames],
    }
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


# ── DataCreator 主类 ──────────────────────────────────────────────────
@dataclass
class DataCreatorConfig:
    # creator
    num_episodes: int = 10
    max_steps_per_episode: int = 500
    save_interval: int = 1
    num_workers: int = 1
    output_dir: str = "data/output"
    output_format: str = "pkl"
    parallel: bool = True
    # env
    dt: float = 0.1

class DataCreator:
    """数据生产器。"""

    @staticmethod
    def bulid_from_config(env, controller, cfg: DictConfig) -> DataCreator:
        dc = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        mpc_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(DataCreatorConfig()), dc),
        )
        if not isinstance(mpc_cfg, DataCreatorConfig):
            raise TypeError("Hydra controller 配置无法转换为 MPCConfig")
        return DataCreator(env, controller, mpc_cfg)

    def __init__(self, env, controller, cfg: DataCreatorConfig) -> None:
        self.env = env
        self.controller = controller
        self.config = cfg

    def create_data(self) -> ProductionReport:
        if self.config.parallel:
            self.run_parallel()
        else:
            self.run_serial()


    def run_serial(self) -> ProductionReport:
        """串行生产所有 episode。"""
        t0 = time.monotonic()
        episodes = [
            _run_single_episode(i, self.config, self.env, self.controller)
            for i in tqdm(
                range(self.config.num_episodes),
                desc="Creating episodes",
                unit="ep",
            )
        ]
        elapsed = time.monotonic() - t0
        return self._finalize(episodes, elapsed)

    def run_parallel(self) -> ProductionReport:
        """并发生产（multiprocessing）。"""
        args_list = [(i, self.config, self.env, self.controller) for i in range(self.config.num_episodes)]
        n_workers = min(self.config.num_workers, len(args_list))
        t0 = time.monotonic()
        if n_workers <= 1:
            episodes = [
                _run_single_episode(*args)
                for args in tqdm(args_list, desc="Creating episodes", unit="ep")
            ]
        else:
            with Pool(processes=n_workers) as pool:
                episodes = list(
                    tqdm(
                        pool.imap_unordered(_run_single_episode_star, args_list),
                        total=len(args_list),
                        desc="Creating episodes",
                        unit="ep",
                    ),
                )
                episodes.sort(key=lambda ep: ep.episode_id)
        elapsed = time.monotonic() - t0
        return self._finalize(episodes, elapsed)

    def _finalize(self, episodes: list[EpisodeData], elapsed: float) -> ProductionReport:
        """保存数据 + 生成报告。"""
        output_dir = os.path.join(self.config.output_dir, "creator")
        os.makedirs(output_dir, exist_ok=True)

        episode_files: list[str] = []
        for ep in tqdm(episodes, desc="Saving episodes", unit="ep"):
            if self.config.output_format == "json":
                _save_episode_json(ep, output_dir)
                episode_files.append(f"episode_{ep.episode_id:04d}.json")
            else:
                _save_episode_pkl(ep, output_dir)
                episode_files.append(f"episode_{ep.episode_id:04d}.pkl")

        total_frames = sum(len(ep.frames) for ep in episodes)
        n = len(episodes)
        report = ProductionReport(
            total_episodes=n,
            total_frames=total_frames,
            total_time_s=elapsed,
            fps=total_frames / max(elapsed, 1e-6),
            avg_episode_length=np.mean([ep.total_steps for ep in episodes]) if n else 0,
            avg_reward=np.mean([ep.total_reward for ep in episodes]) if n else 0,
            avg_progress=np.mean([ep.final_progress for ep in episodes]) if n else 0,
            termination_rate=np.mean([ep.terminated for ep in episodes]) if n else 0,
            truncation_rate=np.mean([ep.truncated for ep in episodes]) if n else 0,
            output_dir=os.path.abspath(output_dir),
            output_format=self.config.output_format,
            episode_files=episode_files,
            created_at=datetime.now().isoformat(),
            dt=self.config.dt,
        )

        report_path = os.path.join(self.config.output_dir, "creator_report.json")
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)

        print(report)
        return report
