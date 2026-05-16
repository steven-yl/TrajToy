"""闭环验证仿真主循环与结果汇总。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from sim_env.road_vehicle_env import RoadVehicleEnv
from sim_env.vehicle_controller import VehicleMPC

from .close_eval_common import (
    build_eval_env,
    build_model_inputs,
    build_road_overlays,
    data_defaults,
    env_cfg,
    init_history_buffer,
    load_model_from_checkpoint,
    raw_history_capacity,
    scoped_train_cfg,
    target_speed,
)

PredictRefPathFn = Callable[[dict[str, np.ndarray]], tuple[np.ndarray, float]]


def run_single_episode(
    ep: int,
    env: RoadVehicleEnv,
    controller: VehicleMPC,
    predictor_cfg: DictConfig,
    use_local_coords: bool,
    num_lane_dividers: int,
    target_speed_fill: float,
    history_interval: int,
    predict_ref_path: PredictRefPathFn,
    ref_path_label: str = "ref_path",
) -> tuple[float, float, int]:
    obs, info = env.reset(seed=1000 + ep)
    controller.reset()
    history_len = int(predictor_cfg.history_len)
    history_buf = init_history_buffer(obs, history_len, history_interval)
    expected_buf_len = raw_history_capacity(history_len, history_interval)

    ep_ret = 0.0
    final_progress = 0.0
    steps = 0
    max_steps = int(env.config.max_steps)

    for t in range(max_steps):
        if len(history_buf) != expected_buf_len:
            raise RuntimeError(f"history_buf 长度异常: 期望 {expected_buf_len}, 实际 {len(history_buf)}")

        model_inputs = build_model_inputs(
            history_buf,
            obs,
            info,
            predictor_cfg,
            use_local_coords,
            num_lane_dividers,
            target_speed_fill,
            history_interval,
        )
        ref_path, step_target_speed = predict_ref_path(model_inputs)

        veh = obs["vehicle"]
        mpc_state = np.array([veh[0], veh[1], veh[2], veh[3], veh[4]], dtype=np.float64)
        action, pred_mpc, ref_mpc = controller.compute(
            mpc_state, ref_path, target_speed=step_target_speed,
        )

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

        overlays = build_road_overlays(obs)
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
                    "label": ref_path_label,
                },
            ]
        )
        env.render(
            info_text=[
                f"episode={ep + 1}/1",
                f"step={t + 1}",
                f"target_speed={step_target_speed:.2f}",
                f"progress={final_progress:.3f}",
            ],
            overlays=overlays,
        )
        if terminated or truncated:
            print(f"terminated={terminated}, truncated={truncated}")
            break

    return ep_ret, final_progress, steps


def execute_close_eval(
    cfg: DictConfig,
    *,
    resolve_predictor_cfg: Callable[[DictConfig], DictConfig],
    build_predict_fn: Callable[[DictConfig, torch.nn.Module, torch.device], PredictRefPathFn],
    ref_path_label: str,
    result_title: str,
) -> None:
    """加载模型与环境，跑单 episode 闭环并打印指标。"""
    scfg = scoped_train_cfg(cfg)
    trainer, model = load_model_from_checkpoint(scfg)
    device = trainer.device
    env, controller = build_eval_env(scfg)

    predictor_cfg = resolve_predictor_cfg(scfg)
    use_local_coords, num_lane_dividers, history_interval = data_defaults(scfg)
    env_node = env_cfg(scfg)
    target_speed_fill = target_speed(scfg, env_node)
    predict_ref_path = build_predict_fn(scfg, model, device)

    ep_ret, final_progress, steps = 0.0, 0.0, 0
    try:
        ep_ret, final_progress, steps = run_single_episode(
            ep=0,
            env=env,
            controller=controller,
            predictor_cfg=predictor_cfg,
            use_local_coords=use_local_coords,
            num_lane_dividers=num_lane_dividers,
            target_speed_fill=target_speed_fill,
            history_interval=history_interval,
            predict_ref_path=predict_ref_path,
            ref_path_label=ref_path_label,
        )
        print(f"[EP 1] return={ep_ret:.2f}, progress={final_progress:.3f}, steps={steps}")
    finally:
        env.close()

    print(f"\n=== {result_title} ===")
    print(f"return   : {ep_ret:.2f}")
    print(f"progress : {final_progress:.3f}")
    print(f"steps    : {steps}")
