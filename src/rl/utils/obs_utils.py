"""观测空间处理：归一化、局部坐标转换、展平。"""

from __future__ import annotations

import numpy as np


_VEHICLE_OBS_DIM = 4


def _to_local_coords(points: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    """绝对坐标 → 以 (origin, theta) 为参考的局部坐标。"""
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    delta = points - origin
    local_x = delta[..., 0] * cos_t + delta[..., 1] * sin_t
    local_y = -delta[..., 0] * sin_t + delta[..., 1] * cos_t
    return np.stack([local_x, local_y], axis=-1)


def normal_and_flatten_obs(
    obs: dict[str, np.ndarray],
    obs_keys: list[str],
    info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """归一化并展平 Dict 观测，分别返回车辆状态和道路信息。"""
    veh = np.asarray(obs["vehicle"], dtype=np.float32)
    ego_xy = veh[:2]
    ego_theta = veh[2]
    ego_v = veh[3]
    ego_steer = veh[4]

    vehicle_obs = np.array([
        0.0, 0.0,
        ego_v / 30.0,
        ego_steer / (np.pi / 3.0),
    ], dtype=np.float32)

    road_keys = {"centerline", "left_boundary", "right_boundary", "lane_dividers"}
    road_parts: list[np.ndarray] = []
    for key in obs_keys:
        if key not in obs or key not in road_keys:
            continue
        val = np.asarray(obs[key], dtype=np.float32)
        original_shape = val.shape
        points_2d = val.reshape(-1, 2)
        local_pts = _to_local_coords(points_2d, ego_xy, ego_theta) / 50.0
        road_parts.append(local_pts.reshape(original_shape).ravel().astype(np.float32))

    road_obs = np.concatenate(road_parts) if road_parts else np.empty(0, dtype=np.float32)
    return vehicle_obs, road_obs


def get_obs_dims(observation_space, obs_keys: list[str]) -> tuple[int, int]:
    """分别计算车辆状态和道路信息的展平维度。"""
    road_keys = {"centerline", "left_boundary", "right_boundary", "lane_dividers"}
    road_dim = 0
    for key, space in observation_space.spaces.items():
        if key not in obs_keys:
            continue
        if key in road_keys:
            road_dim += int(np.prod(space.shape))
    return _VEHICLE_OBS_DIM, road_dim
