"""IL 模型闭环评估脚本（IL 规划 + MPC 跟踪）。

用法示例:
    python -m il.eva log.save_dir=log/il_20260426_000000 device=cpu
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from il.model import TrajectoryPredictor
from sim_env.road_vehicle_env import RoadVehicleEnv
from sim_env.vehicle_controller import VehicleMPC
from .trainer import TrainerBase


def _to_local_coords(points: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    delta = points - origin
    local_x = delta[..., 0] * cos_t + delta[..., 1] * sin_t
    local_y = -delta[..., 0] * sin_t + delta[..., 1] * cos_t
    return np.stack([local_x, local_y], axis=-1)


def _normalize_angle(angle: float, ref_theta: float) -> float:
    return float((angle - ref_theta + np.pi) % (2 * np.pi) - np.pi)


def _build_model_inputs(
    history_buf: list[np.ndarray],
    obs: dict[str, Any],
    info: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, np.ndarray]:
    dc = cfg.data
    h_len = int(dc.history_len) + 1
    n_points = int(dc.road_points)
    n_dividers = int(dc.num_lane_dividers)

    history = np.asarray(history_buf, dtype=np.float32)
    history_mask = np.ones(h_len, dtype=np.float32)

    ego_xy_world = history[-1, :2].copy()
    ego_theta_world = float(history[-1, 2])

    centerline = np.asarray(obs["centerline"], dtype=np.float32)
    left_boundary = np.asarray(obs["left_boundary"], dtype=np.float32)
    right_boundary = np.asarray(obs["right_boundary"], dtype=np.float32)

    lane_dividers = np.zeros((n_dividers, n_points, 2), dtype=np.float32)
    if "lane_dividers" in obs:
        ld_obs = np.asarray(obs["lane_dividers"], dtype=np.float32)
        d_actual = min(n_dividers, ld_obs.shape[0])
        lane_dividers[:d_actual] = ld_obs[:d_actual]

    actual_len = int(info.get("actual_length_num", n_points))
    actual_len = max(0, min(actual_len, n_points))
    road_mask = np.zeros(n_points, dtype=np.float32)
    road_mask[:actual_len] = 1.0

    centerline_mask = road_mask.copy()
    left_boundary_mask = road_mask.copy()
    right_boundary_mask = road_mask.copy() 
    lane_dividers_mask = np.tile(road_mask[None, :], (n_dividers, 1))

    # 限速条件：优先取环境提供的 max_v；缺失时使用保守默认值 30.0 m/s。
    max_v = np.zeros(n_points, dtype=np.float32)
    max_v[:actual_len] = cfg.env.reward_config.target_speed
    max_v_mask = np.zeros(n_points, dtype=np.float32)
    max_v_mask[:actual_len] = 1.0

    if bool(dc.use_local_coords):
        ego_xy = history[-1, :2].copy()
        ego_theta = float(history[-1, 2])

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


def _build_torch_batch(model_inputs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in model_inputs.items()
        if k != "ego_pose_world"
    }


def _predict_ref_path(
    model: TrajectoryPredictor,
    model_inputs: dict[str, np.ndarray],
    cfg: DictConfig,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        batch = _build_torch_batch(model_inputs, device)
        pred = model(
            history=batch["history"],
            history_mask=batch["history_mask"],
            centerline=batch["centerline"],
            centerline_mask=batch["centerline_mask"],
            left_boundary=batch["left_boundary"],
            left_boundary_mask=batch["left_boundary_mask"],
            right_boundary=batch["right_boundary"],
            right_boundary_mask=batch["right_boundary_mask"],
            lane_dividers=batch["lane_dividers"],
            lane_dividers_mask=batch["lane_dividers_mask"],
            max_v=batch["max_v"],
            max_v_mask=batch["max_v_mask"],
        ).squeeze(0).cpu().numpy()

    pred_xy = pred[:, :2]
    pred_v = pred[:, 3]

    if bool(cfg.data.use_local_coords):
        ex, ey, etheta = [float(x) for x in model_inputs["ego_pose_world"]]
        cos_t, sin_t = np.cos(etheta), np.sin(etheta)
        world_xy = np.zeros_like(pred_xy)
        world_xy[:, 0] = ex + pred_xy[:, 0] * cos_t - pred_xy[:, 1] * sin_t
        world_xy[:, 1] = ey + pred_xy[:, 0] * sin_t + pred_xy[:, 1] * cos_t
        pred_xy = world_xy

    target_speed = float(np.clip(pred_v[1], 0.0, 30.0))
    return pred_xy, target_speed


def _load_il_model(cfg: DictConfig) -> TrajectoryPredictor:
    ckpt_path = Path(cfg.trainer.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    device = torch.device(cfg.device)
    model = TrajectoryPredictor(cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # 兼容不支持 weights_only 参数的旧版 PyTorch
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"已加载模型: {ckpt_path}")
    return model


def _build_eval_env(cfg: DictConfig) -> tuple[RoadVehicleEnv, VehicleMPC]:
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env, resolve=True))
    controller_cfg = OmegaConf.create(OmegaConf.to_container(cfg.controller, resolve=True))

    env = RoadVehicleEnv.bulid_from_config(env_cfg)
    controller = VehicleMPC.bulid_from_config(controller_cfg)
    return env, controller


def _init_history_buffer(obs: dict[str, Any], history_len: int) -> list[np.ndarray]:
    veh = obs["vehicle"]
    init_state7 = np.array([veh[0], veh[1], veh[2], veh[3], veh[4], 0.0, 0.0], dtype=np.float32)
    return [init_state7.copy() for _ in range(history_len + 1)]


def _build_road_overlays(obs: dict[str, Any]) -> list[dict[str, Any]]:
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



def _run_single_episode(
    ep: int,
    model: TrajectoryPredictor,
    env: RoadVehicleEnv,
    controller: VehicleMPC,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[float, float, int]:
    obs, info = env.reset(seed=1000 + ep)
    controller.reset()
    history_len = int(cfg.data.history_len)
    history_buf = _init_history_buffer(obs, history_len)
    expected_buf_len = history_len + 1

    ep_ret = 0.0
    final_progress = 0.0
    steps = 0
    for t in range(int(cfg.env.max_steps)):
        if len(history_buf) != expected_buf_len:
            raise RuntimeError(f"history_buf 长度异常: 期望 {expected_buf_len}, 实际 {len(history_buf)}")

        model_inputs = _build_model_inputs(history_buf, obs, info, cfg)
        ref_path, target_speed = _predict_ref_path(model, model_inputs, cfg, device)

        veh = obs["vehicle"]
        mpc_state = np.array([veh[0], veh[1], veh[2], veh[3], veh[4]], dtype=np.float64)
        action, pred_mpc, ref_mpc = controller.compute(mpc_state, ref_path, target_speed=target_speed)

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        final_progress = float(info.get("progress", 0.0))
        steps = t + 1

        veh_n = obs["vehicle"]
        state7 = np.array(
            [veh_n[0], veh_n[1], veh_n[2], veh_n[3], veh_n[4], action[0], action[1]],
            dtype=np.float32,
        )
        history_buf.pop(0)
        history_buf.append(state7)

        overlays = _build_road_overlays(obs)
        overlays.extend(
            [
                {
                    "points": np.asarray(pred_mpc, dtype=np.float32),
                    "color": (255, 0, 180),
                    "width": 3,
                    "style": "solid",
                    "label": "MPC prediction",
                },
                {
                    "points": np.asarray(ref_mpc, dtype=np.float32),
                    "color": (0, 255, 255),
                    "width": 2,
                    "style": "dashed",
                    "label": "MPC reference",
                },
                {
                    "points": np.asarray(ref_mpc, dtype=np.float32),
                    "color": (200, 255, 255),
                    "width": 3,
                    "style": "solid",
                    "label": "ref_path",
                },
            ]
        )
        env.render(
                info_text=[
                    f"episode={ep + 1}/1",
                    f"step={t + 1}",
                    f"target_speed={target_speed:.2f}",
                    f"progress={final_progress:.3f}",
                ],
                overlays=overlays,
            )
        if terminated or truncated:
            print(f"terminated={terminated}, truncated={truncated}")
            break

    return ep_ret, final_progress, steps


def evaluate_closed_loop(cfg: DictConfig) -> None:
    model = _load_il_model(cfg)
    env, controller = _build_eval_env(cfg)
    device = torch.device(cfg.device)

    try:
        ep_ret, final_progress, steps = _run_single_episode(
            ep=0, model=model, env=env, controller=controller, cfg=cfg, device=device
        )
        print(f"[EP 1] return={ep_ret:.2f}, progress={final_progress:.3f}, steps={steps}")
    finally:
        env.close()

    print("\n=== 闭环验证结果 ===")
    print(f"return   : {ep_ret:.2f}")
    print(f"progress : {final_progress:.3f}")
    print(f"steps    : {steps}")


class MLPCloseEvaluator(TrainerBase):
    """基于 TrainerBase 的闭环评估入口。"""

    def run(self) -> None:
        cfg = self.cfg.training if "training" in self.cfg else self.cfg
        evaluate_closed_loop(cfg)

