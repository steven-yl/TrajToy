"""轨迹预测数据集：加载 TrainingSample pkl 文件，转换为模型输入。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from omegaconf import DictConfig

from data_process.process.data_preprocess import TrainingSample


# ── 坐标工具 ─────────────────────────────────────────────────────────


def _to_local_coords(
    points: np.ndarray, origin: np.ndarray, theta: float,
) -> np.ndarray:
    """绝对坐标 → 以 (origin, theta) 为参考的局部坐标。"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    delta = points - origin
    local_x = delta[..., 0] * cos_t + delta[..., 1] * sin_t
    local_y = -delta[..., 0] * sin_t + delta[..., 1] * cos_t
    return np.stack([local_x, local_y], axis=-1)


def _normalize_angle(angle: float, ref_theta: float) -> float:
    diff = angle - ref_theta
    return float((diff + np.pi) % (2 * np.pi) - np.pi)


# ── pad / truncate 工具 ──────────────────────────────────────────────


def _pad_or_truncate_2d(arr: np.ndarray | None, n: int) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros((n, 2), dtype=np.float32)
    arr = arr.astype(np.float32)
    if arr.shape[0] >= n:
        return arr[:n]
    out = np.zeros((n, 2), dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


def _pad_or_truncate_mask(m: np.ndarray | None, n: int) -> np.ndarray:
    if m is None or m.size == 0:
        return np.zeros(n, dtype=np.bool_)
    m = m.astype(np.bool_)
    if m.shape[0] >= n:
        return m[:n]
    out = np.zeros(n, dtype=np.bool_)
    out[:m.shape[0]] = m
    return out


def _pad_or_truncate_seq(arr: np.ndarray, n: int) -> np.ndarray:
    """按时间维对序列做 pad / truncate，保留特征维。"""
    arr = arr.astype(np.float32)
    if arr.shape[0] >= n:
        return arr[:n]
    out = np.zeros((n, arr.shape[1]), dtype=np.float32)
    out[:arr.shape[0]] = arr
    return out


# ── Dataset ──────────────────────────────────────────────────────────


class TrajectoryDataset(Dataset):
    """轨迹预测数据集。

    输出 dict[str, Tensor]:
      vehicle_params (11,), history (H+1, 7), history_mask (H+1, bool),
      centerline (N, 2), centerline_mask (N, bool),
      left_boundary (N, 2), left_boundary_mask (N, bool),
      right_boundary (N, 2), right_boundary_mask (N, bool),
      lane_dividers (D, N, 2), lane_dividers_mask (D, N, bool),
      future (F, 4), future_mask (F, bool),
      max_v (标量), max_v_mask (标量 bool)
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._dc = cfg.data
        data_dirs = self._resolve_data_dirs()
        files: list[Path] = []
        for data_dir in data_dirs:
            if data_dir.exists():
                files.extend(sorted(data_dir.glob("*.pkl")))
        self._data_dirs = data_dirs
        self._file_list = sorted(files)

    def _resolve_data_dirs(self) -> list[Path]:
        """兼容单目录与多目录配置，统一返回目录列表。"""
        if "data_dirs" in self._dc and self._dc.data_dirs is not None:
            raw_dirs = self._dc.data_dirs
        elif "data_dir" in self._dc and self._dc.data_dir is not None:
            raw_dirs = [self._dc.data_dir]
        else:
            raise ValueError("缺少数据目录配置：请设置 data.data_dirs 或 data.data_dir")

        return [Path(str(d)) for d in raw_dirs]

    def __len__(self) -> int:
        return len(self._file_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        with open(self._file_list[idx], "rb") as f:
            sample: TrainingSample = pickle.load(f)
        return self._convert(sample)

    # ------------------------------------------------------------------

    def _convert(self, s: TrainingSample) -> dict[str, torch.Tensor]:
        dc = self._dc
        # 将序列严格对齐到配置长度，避免模型输出步长与标签步长不一致。
        history_len = int(dc.history_len) + 1
        future_len = int(dc.future_len)

        history = _pad_or_truncate_seq(s.history_states, history_len)
        history_mask = _pad_or_truncate_mask(s.history_mask, history_len)
        ego_xy = history[-1, :2].copy()
        ego_theta = float(history[-1, 2])

        future = _pad_or_truncate_seq(s.future_states, future_len)
        future_mask = _pad_or_truncate_mask(s.future_mask, future_len)

        if dc.use_local_coords:
            history = self._to_local_history(history, ego_xy, ego_theta)
            future = self._to_local_history(future, ego_xy, ego_theta)

        road = self._build_road(s, ego_xy, ego_theta)

        vp = s.vehicle_params
        if vp is not None:
            vp_arr = np.array([
                vp.a_max, vp.omega_max, vp.v_max, vp.wheelbase,
                vp.length, vp.width, vp.height,
                vp.front_overhang, vp.rear_overhang, vp.mass, vp.drag_coeff,
            ], dtype=np.float32)
        else:
            vp_arr = np.zeros(11, dtype=np.float32)

        valid_future = future_mask
        max_v_mask = np.array(np.any(valid_future), dtype=np.bool_)
        if np.any(valid_future):
            max_v = np.array(float(np.max(future[valid_future, 3])), dtype=np.float32)
        else:
            max_v = np.array(0.0, dtype=np.float32)

        return {
            "vehicle_params": torch.from_numpy(vp_arr),
            "history": torch.from_numpy(history),
            "history_mask": torch.from_numpy(history_mask),
            **{k: torch.from_numpy(v) for k, v in road.items()},
            "future": torch.from_numpy(future[:, :4]),
            "future_mask": torch.from_numpy(future_mask),
            "max_v": torch.from_numpy(max_v),
            "max_v_mask": torch.from_numpy(max_v_mask),
        }

    @staticmethod
    def _to_local_history(arr: np.ndarray, ego_xy: np.ndarray, ego_theta: float) -> np.ndarray:
        out = arr.copy()
        out[:, :2] = _to_local_coords(arr[:, :2], ego_xy, ego_theta)
        for i in range(out.shape[0]):
            out[i, 2] = _normalize_angle(arr[i, 2], ego_theta)
        return out

    def _build_road(
        self, s: TrainingSample, ego_xy: np.ndarray, ego_theta: float,
    ) -> dict[str, np.ndarray]:
        N = self._dc.road_points
        D = self._dc.num_lane_dividers
        use_local = self._dc.use_local_coords

        def _line(pts, mask_arr):
            p = _pad_or_truncate_2d(pts, N)
            m = _pad_or_truncate_mask(mask_arr, N)
            if use_local:
                p = _to_local_coords(p, ego_xy, ego_theta).astype(np.float32)
            return p, m

        cl, cl_m = _line(s.centerline, s.centerline_mask)
        lb, lb_m = _line(s.left_boundary, s.left_boundary_mask)
        rb, rb_m = _line(s.right_boundary, s.right_boundary_mask)

        ld = np.zeros((D, N, 2), dtype=np.float32)
        ld_m = np.zeros((D, N), dtype=np.bool_)
        if s.lane_dividers is not None and s.lane_dividers.size > 0:
            src = s.lane_dividers.astype(np.float32)
            if src.ndim == 2:
                src = src[np.newaxis]
            d_actual = min(src.shape[0], D)
            src_mask = s.lane_dividers_mask
            if src_mask is not None and src_mask.size > 0:
                src_mask = src_mask.astype(np.bool_)
                if src_mask.ndim == 1:
                    src_mask = src_mask[np.newaxis]
            for d in range(d_actual):
                ld[d] = _pad_or_truncate_2d(src[d], N)
                if src_mask is not None and d < src_mask.shape[0]:
                    ld_m[d] = _pad_or_truncate_mask(src_mask[d], N)
        if use_local:
            for d in range(D):
                ld[d] = _to_local_coords(ld[d], ego_xy, ego_theta).astype(np.float32)

        return {
            "centerline": cl, "centerline_mask": cl_m,
            "left_boundary": lb, "left_boundary_mask": lb_m,
            "right_boundary": rb, "right_boundary_mask": rb_m,
            "lane_dividers": ld, "lane_dividers_mask": ld_m,
        }


# ── DataLoader 工厂 ──────────────────────────────────────────────────


def create_dataloaders(cfg: DictConfig):
    """创建 train / val / test DataLoader。"""
    ds = TrajectoryDataset(cfg)
    total = len(ds)
    if total == 0:
        if "data_dirs" in cfg.data and cfg.data.data_dirs is not None:
            configured_dirs = list(cfg.data.data_dirs)
        else:
            configured_dirs = [cfg.data.data_dir]
        raise ValueError(f"数据目录 {configured_dirs} 中没有找到 pkl 文件")

    dc = cfg.data
    t_end = int(total * dc.train_ratio)
    v_end = t_end + int(total * dc.val_ratio)

    from torch.utils.data import DataLoader, Subset

    kw = dict(num_workers=dc.num_workers, pin_memory=True)
    train_loader = DataLoader(Subset(ds, range(0, t_end)),
                              batch_size=dc.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(Subset(ds, range(t_end, v_end)),
                            batch_size=dc.batch_size, shuffle=False, **kw)
    test_loader = DataLoader(Subset(ds, range(v_end, total)),
                             batch_size=dc.batch_size, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
