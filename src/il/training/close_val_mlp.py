"""MLP 轨迹模型的闭环验证（IL 规划 + MPC 跟踪）。

使用仿真 ``RoadVehicleEnv``、``VehicleMPC``，从 Hydra 读取配置；
模型权重通过 ``trainflow.trainer.Trainer.load_checkpoint`` 加载；
道路/历史特征构造复用 ``TrajectoryDataset`` 中的坐标、pad、以及与训练一致的
``history_interval`` 下采样与 ``history_mask``（前缀无效槽位为 0）。

用法（示例）::

    python -m il.train training=close_eval_mlp_traj
"""

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
from il.model.traj_mlp import TrajMlpTrainableModel
from sim_env.road_vehicle_env import RoadVehicleEnv
from sim_env.vehicle_controller import VehicleMPC
from trainflow.hydra_build import instantiate_trainer_and_model, resolve_strict_weights_only


def _scoped_train_cfg(cfg: DictConfig) -> DictConfig:
    """解析包含 ``trainflow.model`` 的配置节点（根节点或 ``training`` 打包）。"""
    if OmegaConf.select(cfg, "trainflow.model") is not None:
        return cfg
    if OmegaConf.select(cfg, "training.trainflow.model") is not None:
        return cfg.training
    raise ValueError(
        "缺少 trainflow.model：请在 Hydra 配置中提供 trainflow.model，"
        "或使用 training@ 加载含 trainflow 的训练配置。"
    )


def _checkpoint_path(scfg: DictConfig) -> Path:
    raw = OmegaConf.select(scfg, "resume_checkpoint")
    if raw is None:
        raw = OmegaConf.select(scfg, "trainer.checkpoint_path")
    if raw is None:
        raise ValueError("缺少 checkpoint：请设置 resume_checkpoint 或 trainer.checkpoint_path")
    path = Path(str(raw))
    if not path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {path}")
    return path


def _env_cfg(scfg: DictConfig) -> DictConfig:
    node = OmegaConf.select(scfg, "trainflow.env")
    if node is None:
        raise ValueError("缺少 env 配置：设置 trainflow.env（RoadVehicleEnv）")
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def _controller_cfg(scfg: DictConfig) -> DictConfig:
    node = OmegaConf.select(scfg, "trainflow.controller")
    if node is None:
        raise ValueError("缺少 controller 配置：设置 trainflow.controller（VehicleMPC）")
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def _data_defaults(scfg: DictConfig) -> tuple[bool, int, int]:
    ds = OmegaConf.select(scfg, "trainflow.data.data_set")
    use_local = bool(OmegaConf.select(ds, "use_local_coords", default=True))
    n_div = int(OmegaConf.select(ds, "num_lane_dividers", default=2))
    hist_iv = max(1, int(OmegaConf.select(ds, "history_interval", default=1)))
    return use_local, n_div, hist_iv


def _target_speed(scfg: DictConfig, env_cfg: DictConfig) -> float:
    ts = OmegaConf.select(env_cfg, "reward_config.target_speed")
    if ts is None:
        ts = OmegaConf.select(scfg, "env.reward_config.target_speed")
    if ts is None:
        ts = OmegaConf.select(scfg, "trainflow.env.reward_config.target_speed")
    return float(ts if ts is not None else 5.0)


def load_model_from_checkpoint(scfg: DictConfig) -> Any:
    """使用 TrainFlow Trainer 的 checkpoint 逻辑实例化 ``TrajMlpTrainableModel`` 并加载权重。"""
    trainer, model = instantiate_trainer_and_model(scfg)
    trainer.model = model
    ckpt_path = _checkpoint_path(scfg)
    strict, weights_only = resolve_strict_weights_only(scfg)
    trainer.load_checkpoint(ckpt_path, strict=strict, weights_only=weights_only)
    trainer.model.to(trainer.device)
    trainer.model.eval()
    print(f"已加载模型权重: {ckpt_path}")
    return trainer, trainer.model


def _build_model_inputs(
    history_buf: list[np.ndarray],
    obs: dict[str, Any],
    info: dict[str, Any],
    traj_predictor_cfg: DictConfig,
    use_local_coords: bool,
    num_lane_dividers: int,
    target_speed_fill: float,
    history_interval: int,
) -> dict[str, np.ndarray]:
    """与 ``TrajectoryDataset._convert`` 一致：对原始仿真历史做 interval 下采样再 pad + mask。"""
    h_slots = int(traj_predictor_cfg.history_len) + 1
    n_points = int(traj_predictor_cfg.road_points)

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


def _build_torch_batch(model_inputs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(v).unsqueeze(0).to(device)
        for k, v in model_inputs.items()
        if k != "ego_pose_world"
    }


def _predict_ref_path(
    model: TrajMlpTrainableModel,
    model_inputs: dict[str, np.ndarray],
    use_local_coords: bool,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    core = model.traj_mlp
    with torch.no_grad():
        batch = _build_torch_batch(model_inputs, device)
        pred = core(
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

    if use_local_coords:
        ex, ey, etheta = [float(x) for x in model_inputs["ego_pose_world"]]
        cos_t, sin_t = np.cos(etheta), np.sin(etheta)
        world_xy = np.zeros_like(pred_xy)
        world_xy[:, 0] = ex + pred_xy[:, 0] * cos_t - pred_xy[:, 1] * sin_t
        world_xy[:, 1] = ey + pred_xy[:, 0] * sin_t + pred_xy[:, 1] * cos_t
        pred_xy = world_xy

    target_speed = float(np.clip(pred_v[1], 0.0, 30.0))
    return pred_xy, target_speed


def _build_eval_env(scfg: DictConfig) -> tuple[RoadVehicleEnv, VehicleMPC]:
    env = RoadVehicleEnv.bulid_from_config(_env_cfg(scfg))
    controller = VehicleMPC.bulid_from_config(_controller_cfg(scfg))
    return env, controller


def _raw_history_capacity(history_len: int, history_interval: int) -> int:
    """与训练侧一致：保留足够原始步，使 ``from_tail`` 下采样后能填满 ``history_len+1`` 槽位。"""
    h = int(history_len)
    s = max(1, int(history_interval))
    return h * s + 1


def _init_history_buffer(
    obs: dict[str, Any],
    history_len: int,
    history_interval: int,
) -> list[np.ndarray]:
    veh = obs["vehicle"]
    init_state7 = np.array([veh[0], veh[1], veh[2], veh[3], veh[4], 0.0, 0.0], dtype=np.float32)
    cap = _raw_history_capacity(history_len, history_interval)
    return [init_state7.copy() for _ in range(cap)]


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
    model: TrajMlpTrainableModel,
    env: RoadVehicleEnv,
    controller: VehicleMPC,
    scfg: DictConfig,
    traj_predictor_cfg: DictConfig,
    use_local_coords: bool,
    num_lane_dividers: int,
    target_speed_fill: float,
    history_interval: int,
    device: torch.device,
) -> tuple[float, float, int]:
    obs, info = env.reset(seed=1000 + ep)
    controller.reset()
    history_len = int(traj_predictor_cfg.history_len)
    history_buf = _init_history_buffer(obs, history_len, history_interval)
    expected_buf_len = _raw_history_capacity(history_len, history_interval)

    ep_ret = 0.0
    final_progress = 0.0
    steps = 0
    max_steps = int(env.config.max_steps)

    for t in range(max_steps):
        if len(history_buf) != expected_buf_len:
            raise RuntimeError(f"history_buf 长度异常: 期望 {expected_buf_len}, 实际 {len(history_buf)}")

        model_inputs = _build_model_inputs(
            history_buf,
            obs,
            info,
            traj_predictor_cfg,
            use_local_coords,
            num_lane_dividers,
            target_speed_fill,
            history_interval,
        )
        ref_path, target_speed = _predict_ref_path(model, model_inputs, use_local_coords, device)

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
                    "points": np.asarray(ref_path, dtype=np.float32),
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
    print("start evaluate closed loop...")
    print("scoped train cfg:")
    print(OmegaConf.to_yaml(cfg))
    scfg = _scoped_train_cfg(cfg)
    trainer, model = load_model_from_checkpoint(scfg)
    device = trainer.device
    env, controller = _build_eval_env(scfg)

    traj_predictor_cfg = scfg.trainflow.model.traj_predictor
    use_local_coords, num_lane_dividers, history_interval = _data_defaults(scfg)
    env_node = _env_cfg(scfg)
    target_speed_fill = _target_speed(scfg, env_node)

    ep_ret, final_progress, steps = 0.0, 0.0, 0
    try:
        ep_ret, final_progress, steps = _run_single_episode(
            ep=0,
            model=model,
            env=env,
            controller=controller,
            scfg=scfg,
            traj_predictor_cfg=traj_predictor_cfg,
            use_local_coords=use_local_coords,
            num_lane_dividers=num_lane_dividers,
            target_speed_fill=target_speed_fill,
            history_interval=history_interval,
            device=device,
        )
        print(f"[EP 1] return={ep_ret:.2f}, progress={final_progress:.3f}, steps={steps}")
    finally:
        env.close()

    print("\n=== 闭环验证结果 ===")
    print(f"return   : {ep_ret:.2f}")
    print(f"progress : {final_progress:.3f}")
    print(f"steps    : {steps}")


def run_close_eval(cfg: DictConfig) -> None:
    """Hydra / ``il.train`` 在 ``run_mode=close_eval`` 时的入口。"""
    print("start close eval...")
    evaluate_closed_loop(cfg)
    print("close eval done!")