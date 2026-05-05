"""Diffusion（Conditional1DUNet）闭环评估：DDIM 采样轨迹 + MPC 跟踪。

与 ``mlp_close_eval`` 共用场景构图（history / road / max_v）；规划为
``BatchNormalizer`` + ``sample_ddim``（默认全步；可在 ``schedule_cfg.num_inference_steps`` 加速）。

用法示例::

    python -m il.train training=eval_df_close \\
        trainer.checkpoint_path=log/your_run/best_model.pt device=cuda
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from il.model import Conditional1DUNet
from il.training.df_train import prepare_diffusion_schedule, sample_ddpm, sample_ddim
from il.utils.normalizer import BatchNormalizer
from sim_env.road_vehicle_env import RoadVehicleEnv
from sim_env.vehicle_controller import VehicleMPC

from .mlp_close_eval import (
    _build_eval_env,
    _build_model_inputs,
    _build_road_overlays,
    _build_torch_batch,
    _init_history_buffer,
)
from .trainer import TrainerBase


def _predict_ref_path_df(
    model: Conditional1DUNet,
    batch_norm: BatchNormalizer,
    schedule: dict[str, torch.Tensor],
    model_inputs: dict[str, np.ndarray],
    cfg: DictConfig,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    """归一化 → DDIM 采样 → 反归一化 future，再转世界系 xy。"""
    dc = cfg.data
    future_len = int(dc.future_len)
    pred_dim = int(cfg.model.C1DUnet.prediction_state_dim)

    with torch.no_grad():
        batch = _build_torch_batch(model_inputs, device)
        nb = batch_norm.normalize(batch)
        shape = (1, future_len, pred_dim)
        sched_eval = cfg.trainer.schedule_cfg
        ni = sched_eval.get("num_inference_steps", None)
        ddim_kw: dict = {"eta": 0.0, "log_timing": True}
        if ni is not None:
            ddim_kw["num_inference_steps"] = int(ni)
        pred_norm = sample_ddim(model, nb, schedule, shape, **ddim_kw)
        # pred_norm = sample_ddpm(model, nb, schedule, shape, log_timing=True)
        pred = batch_norm.inverse_future(pred_norm).squeeze(0).cpu().numpy()

    pred_xy = pred[:, :2].astype(np.float64)
    pred_v = pred[:, 3]

    if bool(dc.use_local_coords):
        ex, ey, etheta = [float(x) for x in model_inputs["ego_pose_world"]]
        cos_t, sin_t = np.cos(etheta), np.sin(etheta)
        world_xy = np.zeros_like(pred_xy)
        world_xy[:, 0] = ex + pred_xy[:, 0] * cos_t - pred_xy[:, 1] * sin_t
        world_xy[:, 1] = ey + pred_xy[:, 0] * sin_t + pred_xy[:, 1] * cos_t
        pred_xy = world_xy.astype(np.float32)

    v_cap = float(getattr(cfg.env.reward_config, "target_speed", 30.0))
    target_speed = float(np.clip(pred_v[1], 0.0, v_cap))
    return pred_xy.astype(np.float32), target_speed


def _load_df_model(cfg: DictConfig) -> Conditional1DUNet:
    ckpt_path = Path(cfg.trainer.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    device = torch.device(cfg.device)
    model = Conditional1DUNet(cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"已加载 DF 模型: {ckpt_path}")
    return model


def _run_single_episode_df(
    ep: int,
    model: Conditional1DUNet,
    batch_norm: BatchNormalizer,
    schedule: dict[str, torch.Tensor],
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
        ref_path, target_speed = _predict_ref_path_df(
            model, batch_norm, schedule, model_inputs, cfg, device
        )

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
                    "label": "DF ref_path",
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


def evaluate_closed_loop_df(cfg: DictConfig) -> None:
    model = _load_df_model(cfg)
    batch_norm = BatchNormalizer.from_config(cfg.data.normalization)
    device = torch.device(cfg.device)
    schedule = prepare_diffusion_schedule(cfg, device=device)

    env, controller = _build_eval_env(cfg)

    try:
        ep_ret, final_progress, steps = _run_single_episode_df(
            ep=0,
            model=model,
            batch_norm=batch_norm,
            schedule=schedule,
            env=env,
            controller=controller,
            cfg=cfg,
            device=device,
        )
        print(f"[EP 1] return={ep_ret:.2f}, progress={final_progress:.3f}, steps={steps}")
    finally:
        env.close()

    print("\n=== DF 闭环验证结果 ===")
    print(f"return   : {ep_ret:.2f}")
    print(f"progress : {final_progress:.3f}")
    print(f"steps    : {steps}")


class DFCloseEvaluator(TrainerBase):
    """Hydra 入口：``training=eval_df_close``。"""

    def run(self) -> None:
        cfg = self.cfg.training if "training" in self.cfg else self.cfg
        evaluate_closed_loop_df(cfg)
