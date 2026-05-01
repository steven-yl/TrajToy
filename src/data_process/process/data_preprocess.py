"""数据预处理：将原始 EpisodeData 转换为可直接用于训练的样本。

以当前帧为锚点，构造历史状态序列、未来状态序列及对应的道路上下文。
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from data_process.process.data_creator import EpisodeData, FrameData
from sim_env import VehicleParams


# 每帧提取的状态字段，顺序与 readme 一致
_STATE_KEYS = ("x", "y", "theta", "v", "steering", "action_accel", "action_omega")
_STATE_DIM = len(_STATE_KEYS)


@dataclass
class TrainingSample:
    """单条训练样本。"""

    timestamp: float
    dt: float
    history_future_dt: float
    vehicle_params: VehicleParams | None = None
    history_states: np.ndarray = field(default_factory=lambda: np.empty((0, _STATE_DIM)))
    history_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    future_states: np.ndarray = field(default_factory=lambda: np.empty((0, _STATE_DIM)))
    future_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    centerline_max_v: np.ndarray | None = None
    centerline_max_v_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    centerline: np.ndarray | None = None
    centerline_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    left_boundary: np.ndarray | None = None
    left_boundary_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    right_boundary: np.ndarray | None = None
    right_boundary_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    lane_dividers: np.ndarray | None = None
    lane_dividers_mask: np.ndarray = field(default_factory=lambda: np.empty(0))
    road_segment_types: list[str] | None = None
    lateral_offset: float = 0.0
    heading_error: float = 0.0


def _frame_to_state(frame: FrameData) -> np.ndarray:
    """从 FrameData 提取 (7,) 状态向量。"""
    return np.array([getattr(frame, k) for k in _STATE_KEYS], dtype=np.float64)


def preprocess_episode(
    episode: EpisodeData,
    history_len: int = 10,
    future_len: int = 10,
    sample_interval: int = 1,
    history_future_dt: float = 0.1,
) -> list[TrainingSample]:
    """将单个 episode 转换为训练样本列表。"""
    frames = episode.frames
    n = len(frames)
    if n == 0:
        return []

    all_states = np.array([_frame_to_state(f) for f in frames], dtype=np.float64)
    base_dt = float(episode.dt)
    if base_dt <= 0:
        raise ValueError(f"episode.dt 必须为正数，当前为: {episode.dt}")
    if history_future_dt <= 0:
        raise ValueError(f"history_future_dt 必须为正数，当前为: {history_future_dt}")
    state_stride = max(1, int(round(history_future_dt / base_dt)))

    samples: list[TrainingSample] = []
    for i in range(0, n, sample_interval):
        cur = frames[i]

        h_total = history_len + 1
        history = np.zeros((h_total, _STATE_DIM), dtype=np.float64)
        h_mask = np.zeros(h_total, dtype=np.float64)
        hist_indices = [i - k * state_stride for k in range(history_len, -1, -1)]
        h_actual = 0
        for out_idx, src_idx in enumerate(hist_indices):
            if 0 <= src_idx < n:
                history[out_idx] = all_states[src_idx]
                h_mask[out_idx] = 1.0
                h_actual += 1

        future = np.zeros((future_len, _STATE_DIM), dtype=np.float64)
        f_mask = np.zeros(future_len, dtype=np.float64)
        future_indices = [i + (k + 1) * state_stride for k in range(future_len)]
        f_actual = 0
        for out_idx, src_idx in enumerate(future_indices):
            if src_idx >= n:
                break
            future[out_idx] = all_states[src_idx]
            f_mask[out_idx] = 1.0
            f_actual += 1

        # 限速信息：取未来序列速度(v)最大值，并扩展到与 centerline 同长度
        centerline_max_v = np.empty(0, dtype=np.float64)
        future_max_v = 0.0
        if f_actual > 0:
            future_max_v = float(np.max(future[:f_actual, 3]))
            centerline_max_v = np.full((cur.centerline.shape[0],), future_max_v, dtype=np.float64)

        actual = int(cur.actual_length_num) if cur.actual_length_num else 0

        def _make_road_mask(data: np.ndarray | None, valid_length: int) -> np.ndarray:
            if data is None or data.size == 0:
                return np.empty(0, dtype=np.float64)
            if data.ndim == 2:
                mask = np.zeros(data.shape[0], dtype=np.float64)
                mask[:valid_length] = 1.0
                return mask
            elif data.ndim == 3:
                mask = np.zeros(data.shape[:2], dtype=np.float64)
                mask[:, :valid_length] = 1.0
                return mask
            return np.empty(0, dtype=np.float64)

        sample = TrainingSample(
            timestamp=episode.timestamp + cur.timestamp,
            dt=episode.dt,
            history_future_dt=history_future_dt,
            vehicle_params=episode.vehicle_params,
            history_states=history,
            history_mask=h_mask,
            future_states=future,
            future_mask=f_mask,
            centerline_max_v=centerline_max_v,
            centerline_max_v_mask=_make_road_mask(cur.centerline, actual),
            centerline=cur.centerline,
            centerline_mask=_make_road_mask(cur.centerline, actual),
            left_boundary=cur.left_boundary,
            left_boundary_mask=_make_road_mask(cur.left_boundary, actual),
            right_boundary=cur.right_boundary,
            right_boundary_mask=_make_road_mask(cur.right_boundary, actual),
            lane_dividers=cur.lane_dividers,
            lane_dividers_mask=_make_road_mask(cur.lane_dividers, actual),
            road_segment_types=cur.road_segment_types,
            lateral_offset=cur.lateral_offset,
            heading_error=cur.heading_error,
        )
        samples.append(sample)

    return samples


@dataclass
class PreprocessReport:
    """预处理报告。"""

    input_dir: str = ""
    output_dir: str = ""
    history_len: int = 0
    future_len: int = 0
    sample_interval: int = 1
    dt: float = 0.0
    history_future_dt: float = 0.1
    fmt: str = "pkl"
    total_episodes: int = 0
    total_source_frames: int = 0
    total_samples: int = 0
    samples_per_episode: list[int] = field(default_factory=list)
    processed_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    wall_time: float = 0.0
    created_at: str = ""

    def __str__(self) -> str:
        lines = []
        lines.append("=== 预处理报告 ===")
        lines.append(f"  生产时间:        {self.created_at}")
        lines.append(f"  输入目录:        {self.input_dir}")
        lines.append(f"  输出目录:        {self.output_dir}")
        lines.append(f"  历史帧数 H:      {self.history_len}")
        lines.append(f"  未来帧数 F:      {self.future_len}")
        lines.append(f"  采样间隔:        {self.sample_interval}")
        lines.append(f"  帧时间间隔:      {self.dt}s")
        lines.append(f"  历史-未来状态时间间隔: {self.history_future_dt}s")
        lines.append(f"  episode 数:      {self.total_episodes}")
        lines.append(f"  原始总帧数:      {self.total_source_frames}")
        lines.append(f"  生成样本数:      {self.total_samples}")
        lines.append(f"  输出文件数:      {len(self.output_files)}")
        lines.append(f"  耗时:            {self.wall_time:.2f}s")
        return "\n".join(lines)


def preprocess_directory(cfg: DictConfig) -> PreprocessReport:
    """批量预处理目录下所有 episode 文件。

    Args:
        cfg: 完整 DictConfig，使用 cfg.preprocess 子配置。
    """
    pc = cfg.preprocess
    input_dir = os.path.join(pc.input_dir, "creator")
    output_dir = pc.output_dir
    history_len = pc.history_len
    future_len = pc.future_len
    sample_interval = pc.sample_interval
    history_future_dt = pc.history_future_dt
    fmt = pc.fmt

    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "preprocessed")
    os.makedirs(samples_dir, exist_ok=True)

    report = PreprocessReport(
        input_dir=os.path.abspath(input_dir),
        output_dir=os.path.abspath(output_dir),
        history_len=history_len,
        future_len=future_len,
        sample_interval=sample_interval,
        history_future_dt=history_future_dt,
        fmt=fmt,
    )

    t0 = time.perf_counter()

    episode_files = [
        fname
        for fname in sorted(os.listdir(input_dir))
        if fname.startswith("episode_") and fname.endswith(f".{fmt}")
    ]

    dt_values: list[float] = []
    for fname in tqdm(episode_files, desc="Preprocessing episodes", unit="ep"):
        path = os.path.join(input_dir, fname)
        with open(path, "rb") as f:
            episode: EpisodeData = pickle.load(f)

        source_frames = len(episode.frames)
        dt_values.append(float(episode.dt))
        samples = preprocess_episode(episode, history_len, future_len, sample_interval, history_future_dt)

        report.total_episodes += 1
        report.total_source_frames += source_frames
        report.processed_files.append(fname)

        if not samples:
            report.samples_per_episode.append(0)
            continue

        ep_stem = os.path.splitext(fname)[0]
        ep_sample_count = 0
        for sample in tqdm(
            samples,
            desc=f"Saving {ep_stem}",
            unit="sample",
            leave=False,
        ):
            sample_fname = f"{ep_stem}_{sample.timestamp:.6f}.pkl"
            out_path = os.path.join(samples_dir, sample_fname)
            with open(out_path, "wb") as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            report.output_files.append(sample_fname)
            report.total_samples += 1
            ep_sample_count += 1

        report.samples_per_episode.append(ep_sample_count)
    if dt_values:
        report.dt = float(dt_values[0])
        unique_dts = {round(v, 12) for v in dt_values}
        if len(unique_dts) > 1:
            print(f"警告: 检测到多个 episode.dt: {sorted(unique_dts)}，报告中记录首个 dt={report.dt}")
    report.wall_time = time.perf_counter() - t0
    report.created_at = datetime.now().isoformat()

    report_path = os.path.join(output_dir, "preprocess_report.json")
    with open(report_path, "w") as f:
        json.dump(asdict(report), f, ensure_ascii=False, indent=2)

    print(report)
    return report
