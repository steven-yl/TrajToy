"""Diffusion 轨迹模型的闭环验证（DDIM/DDPM 采样 + MPC 跟踪）。

场景构图与 MLP 共用；规划为 ``DiffusionPipeline.sample`` + ``Normalizer`` 反变换。

用法::

    python -m il.eval
    # eval@: close_eval_traj_diffusion
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .close_eval import CloseEvalBase
from .close_eval_common import build_torch_batch, data_defaults, env_cfg, local_xy_to_world, target_speed
from .close_eval_loop import PredictRefPathFn, execute_close_eval


def _predictor_dims(predictor_cfg: DictConfig) -> tuple[int, int]:
    future_len = int(OmegaConf.select(predictor_cfg, "future_len", default=25))
    pred_dim = int(OmegaConf.select(predictor_cfg, "prediction_state_dim", default=4))
    return future_len, pred_dim


def _predict_ref_path_df(
    model: torch.nn.Module,
    model_inputs: dict[str, np.ndarray],
    use_local_coords: bool,
    device: torch.device,
    predictor_cfg: DictConfig,
    target_speed_cap: float,
) -> tuple[np.ndarray, float]:
    future_len, pred_dim = _predictor_dims(predictor_cfg)

    with torch.no_grad():
        batch = build_torch_batch(model_inputs, device)
        batch["future"] = torch.zeros(1, future_len, pred_dim, device=device, dtype=torch.float32)
        nb = model.normalizer.apply(batch)
        pred_norm = model.predictor.sample(nb, shape=batch["future"].shape, device=device)
        pred = model.normalizer.inverse_future(pred_norm).squeeze(0).cpu().numpy()

    pred_xy = pred[:, :2]
    if use_local_coords:
        pred_xy = local_xy_to_world(pred_xy, model_inputs["ego_pose_world"])

    step_target_speed = float(np.clip(pred[:, 3][1], 0.0, target_speed_cap))
    return pred_xy.astype(np.float32), step_target_speed


def _build_df_predict_fn(
    scfg: DictConfig, model: torch.nn.Module, device: torch.device,
) -> PredictRefPathFn:
    use_local_coords, _, _ = data_defaults(scfg)
    predictor_cfg = scfg.trainflow.model.predictor.model
    target_speed_cap = target_speed(scfg, env_cfg(scfg))

    def predict(model_inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        return _predict_ref_path_df(
            model,
            model_inputs,
            use_local_coords,
            device,
            predictor_cfg,
            target_speed_cap,
        )

    return predict


class CloseEvalDf(CloseEvalBase):
    def evaluate_closed_loop(self, cfg: DictConfig) -> None:
        execute_close_eval(
            cfg,
            resolve_predictor_cfg=lambda scfg: scfg.trainflow.model.predictor.model,
            build_predict_fn=_build_df_predict_fn,
            ref_path_label="DF ref_path",
            result_title="DF 闭环验证结果",
        )
