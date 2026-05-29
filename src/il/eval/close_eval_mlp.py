"""MLP 轨迹模型的闭环验证（IL 前向 + MPC 跟踪）。

用法::

    python -m il.eval
    # eval@: close_eval_traj_mlp
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from .close_eval import CloseEvalBase
from .close_eval_common import build_torch_batch, data_defaults, local_xy_to_world
from .close_eval_loop import PredictRefPathFn, execute_close_eval


def _predict_ref_path_mlp(
    model: torch.nn.Module,
    model_inputs: dict[str, np.ndarray],
    use_local_coords: bool,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    predictor = model.predictor
    with torch.no_grad():
        batch = build_torch_batch(model_inputs, device)
        pred = predictor(
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
    if use_local_coords:
        pred_xy = local_xy_to_world(pred_xy, model_inputs["ego_pose_world"])

    target_speed = float(np.clip(pred[:, 3][1], 0.0, 30.0))
    return pred_xy, target_speed


def _build_mlp_predict_fn(
    scfg: DictConfig, model: torch.nn.Module, device: torch.device,
) -> PredictRefPathFn:
    use_local_coords, _, _ = data_defaults(scfg)

    def predict(model_inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        return _predict_ref_path_mlp(model, model_inputs, use_local_coords, device)

    return predict


class CloseEvalMlp(CloseEvalBase):
    def evaluate_closed_loop(self, cfg: DictConfig) -> None:
        execute_close_eval(
            cfg,
            resolve_predictor_cfg=lambda scfg: scfg.trainflow.model.predictor,
            build_predict_fn=_build_mlp_predict_fn,
            ref_path_label="ref_path",
            result_title="闭环验证结果",
        )
