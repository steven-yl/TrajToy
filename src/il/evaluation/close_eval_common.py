"""闭环验证共用：配置解析、模型加载、观测构图、仿真环境。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from il.data.dataset.trajectory_dataset import (
    _normalize_angle,
    _pad_or_truncate_2d,
    _pad_or_truncate_mask,
    _pad_or_truncate_seq,
    _sample_sequence_with_interval,
    _to_local_coords,
)
from sim_env.road_vehicle_env import RoadVehicleEnv
from sim_env.vehicle_controller import VehicleMPC
from trainflow.hydra_build import instantiate_trainer_and_model, resolve_strict_weights_only


def scoped_train_cfg(cfg: DictConfig) -> DictConfig:
    """解析包含 ``trainflow.model`` 的配置节点（根节点或 ``training`` 打包）。"""
    if OmegaConf.select(cfg, "trainflow.model") is not None:
        return cfg
    if OmegaConf.select(cfg, "training.trainflow.model") is not None:
        return cfg.training
    raise ValueError(
        "缺少 trainflow.model：请在 Hydra 配置中提供 trainflow.model，"
        "或使用 training@ 加载含 trainflow 的训练配置。"
    )


def checkpoint_path(scfg: DictConfig) -> Path:
    raw = OmegaConf.select(scfg, "resume_checkpoint")
    if raw is None:
        raw = OmegaConf.select(scfg, "trainer.checkpoint_path")
    if raw is None:
        raise ValueError("缺少 checkpoint：请设置 resume_checkpoint 或 trainer.checkpoint_path")
    path = Path(str(raw))
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {path}")
    return path


def env_cfg(scfg: DictConfig) -> DictConfig:
    node = OmegaConf.select(scfg, "trainflow.env")
    if node is None:
        raise ValueError("缺少 env 配置：设置 trainflow.env（RoadVehicleEnv）")
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def controller_cfg(scfg: DictConfig) -> DictConfig:
    node = OmegaConf.select(scfg, "trainflow.controller")
    if node is None:
        raise ValueError("缺少 controller 配置：设置 trainflow.controller（VehicleMPC）")
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def data_defaults(scfg: DictConfig) -> tuple[bool, int, int]:
    ds = OmegaConf.select(scfg, "trainflow.data.data_set")
    use_local = bool(OmegaConf.select(ds, "use_local_coords", default=True))
    n_div = int(OmegaConf.select(ds, "num_lane_dividers", default=2))
    hist_iv = max(1, int(OmegaConf.select(ds, "history_interval", default=1)))
    return use_local, n_div, hist_iv


def target_speed(scfg: DictConfig, env_node: DictConfig) -> float:
    ts = OmegaConf.select(env_node, "reward_config.target_speed")
    if ts is None:
        ts = OmegaConf.select(scfg, "env.reward_config.target_speed")
    if ts is None:
        ts = OmegaConf.select(scfg, "trainflow.env.reward_config.target_speed")
    return float(ts if ts is not None else 5.0)


def load_model_from_checkpoint(scfg: DictConfig) -> tuple[Any, torch.nn.Module]:
    """实例化 TrainFlow 模型并通过 Trainer 加载 checkpoint。"""
    trainer, model = instantiate_trainer_and_model(scfg)
    trainer.model = model
    ckpt_path = checkpoint_path(scfg)
    strict, weights_only = resolve_strict_weights_only(scfg)
    trainer.load_checkpoint(ckpt_path, strict=strict, weights_only=weights_only)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    print(f"已加载模型权重: {ckpt_path}")
    return trainer, trainer.model


def build_model_inputs(
    history_buf: list[np.ndarray],
    obs: dict[str, Any],
    info: dict[str, Any],
    predictor_cfg: DictConfig,
    use_local_coords: bool,
    num_lane_dividers: int,
    target_speed_fill: float,
    history_interval: int,
) -> dict[str, np.ndarray]:
    """与 ``TrajectoryDataset._convert`` 一致：interval 下采样、pad、mask、局部坐标。"""
    h_slots = int(predictor_cfg.history_len) + 1
    n_points = int(predictor_cfg.road_points)

    hist_raw_buf = np.asarray(history_buf, dtype=np.float32)
    hist_mask_buf = np.ones(hist_raw_buf.shape[0], dtype=np.float32)

    hist_sub, mask_sub = _sample_sequence_with_interval(
        hist_raw_buf, hist_mask_buf, history_interval, from_tail=True,
    )
    if hist_sub is None:
        raise RuntimeError("history_buf 转为序列失败")
    history = _pad_or_truncate_seq(hist_sub, h_slots, from_tail=True)
    history_mask_np = _pad_or_truncate_mask(mask_sub, h_slots, from_tail=True)
    history_mask = history_mask_np.astype(np.float32)

    ego_xy_world = history[-1, :2].copy()
    ego_theta_world = float(history[-1, 2])

    centerline = _pad_or_truncate_2d(np.asarray(obs["centerline"], dtype=np.float32), n_points)
    left_boundary = _pad_or_truncate_2d(np.asarray(obs["left_boundary"], dtype=np.float32), n_points)
    right_boundary = _pad_or_truncate_2d(np.asarray(obs["right_boundary"], dtype=np.float32), n_points)

    n_dividers = int(num_lane_dividers)
    lane_dividers = np.zeros((n_dividers, n_points, 2), dtype=np.float32)
    if "lane_dividers" in obs:
        ld_obs = np.asarray(obs["lane_dividers"], dtype=np.float32)
        d_actual = min(n_dividers, ld_obs.shape[0])
        for d in range(d_actual):
            lane_dividers[d] = _pad_or_truncate_2d(ld_obs[d], n_points)

    actual_len = int(info.get("actual_length_num", n_points))
    actual_len = max(0, min(actual_len, n_points))
    road_mask = np.zeros(n_points, dtype=np.float32)
    road_mask[:actual_len] = 1.0

    centerline_mask = road_mask.copy()
    left_boundary_mask = road_mask.copy()
    right_boundary_mask = road_mask.copy()
    lane_dividers_mask = np.tile(road_mask[None, :], (n_dividers, 1))

    max_v = np.zeros(n_points, dtype=np.float32)
    max_v[:actual_len] = target_speed_fill
    max_v_mask = np.zeros(n_points, dtype=np.float32)
    max_v_mask[:actual_len] = 1.0

    if use_local_coords:
        ego_xy = history[-1, :2].copy()
        ego_theta = float(history[-1, 2])

        history = history.copy()
        history[:, :2] = _to_local_coords(history[:, :2], ego_xy, ego_theta).astype(np.float32)
        for i in range(history.shape[0]):
            history[i, 2] = _normalize_angle(float(history[i, 2]), ego_theta)

        centerline = _to_local_coords(centerline, ego_xy, ego_theta).astype(np.float32)
        left_boundary = _to_local_coords(left_boundary, ego_xy, ego_theta).astype(np.float32)
        right_boundary = _to_local_coords(right_boundary, ego_xy, ego_theta).astype(np.float32)
        for d in range(n_dividers):
            lane_dividers[d] = _to_local_coords(lane_dividers[d], ego_xy, ego_theta).astype(np.float32)

    return {
        "history": history,
        "history_mask": history_mask,
        "centerline": centerline,
        "centerline_mask": centerline_mask,
        "left_boundary": left_boundary,
        "left_boundary_mask": left_boundary_mask,
        "right_boundary": right_boundary,
        "right_boundary_mask": right_boundary_mask,
        "lane_dividers": lane_dividers,
        "lane_dividers_mask": lane_dividers_mask,
        "max_v": max_v,
        "max_v_mask": max_v_mask,
        "ego_pose_world": np.array([ego_xy_world[0], ego_xy_world[1], ego_theta_world], dtype=np.float32),
    }


def build_torch_batch(model_inputs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in model_inputs.items()
        if k != "ego_pose_world"
    }


def local_xy_to_world(pred_xy: np.ndarray, ego_pose_world: np.ndarray) -> np.ndarray:
    ex, ey, etheta = [float(x) for x in ego_pose_world]
    cos_t, sin_t = np.cos(etheta), np.sin(etheta)
    world_xy = np.zeros_like(pred_xy)
    world_xy[:, 0] = ex + pred_xy[:, 0] * cos_t - pred_xy[:, 1] * sin_t
    world_xy[:, 1] = ey + pred_xy[:, 0] * sin_t + pred_xy[:, 1] * cos_t
    return world_xy


def build_eval_env(scfg: DictConfig) -> tuple[RoadVehicleEnv, VehicleMPC]:
    env = RoadVehicleEnv.bulid_from_config(env_cfg(scfg))
    controller = VehicleMPC.bulid_from_config(controller_cfg(scfg))
    return env, controller


def raw_history_capacity(history_len: int, history_interval: int) -> int:
    h = int(history_len)
    s = max(1, int(history_interval))
    return h * s + 1


def init_history_buffer(
    obs: dict[str, Any],
    history_len: int,
    history_interval: int,
) -> list[np.ndarray]:
    veh = obs["vehicle"]
    init_state7 = np.array([veh[0], veh[1], veh[2], veh[3], veh[4], 0.0, 0.0], dtype=np.float32)
    cap = raw_history_capacity(history_len, history_interval)
    return [init_state7.copy() for _ in range(cap)]


def build_road_overlays(obs: dict[str, Any]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = [
        {
            "points": np.asarray(obs["centerline"], dtype=np.float32),
            "color": (0, 220, 255),
            "width": 3,
            "style": "dashed",
            "label": "Road centerline",
        },
        {
            "points": np.asarray(obs["left_boundary"], dtype=np.float32),
            "color": (0, 255, 120),
            "width": 3,
            "style": "solid",
            "label": "Road left boundary",
        },
        {
            "points": np.asarray(obs["right_boundary"], dtype=np.float32),
            "color": (255, 120, 0),
            "width": 3,
            "style": "solid",
            "label": "Road right boundary",
        },
    ]
    if "lane_dividers" in obs:
        lane_dividers = np.asarray(obs["lane_dividers"], dtype=np.float32)
        for i, divider in enumerate(lane_dividers):
            overlays.append(
                {
                    "points": divider,
                    "color": (180, 180, 180),
                    "width": 2,
                    "style": "solid",
                    "label": f"Road lane divider {i + 1}",
                }
            )
    return overlays
