"""数据集质量分析：读取预处理后的 ``TrainingSample`` .pkl 并汇总统计。

通过 Hydra 调用：``python -m data_process.data_process_main run_mode=analyze``，
配置见 ``data_process/conf/analysis/default.yaml``（路径默认与 ``preprocess`` 输出对齐）。
开启 ``analysis.plot.enable`` 时，在 ``analysis.plot.output_dir`` 下写出若干 PNG 汇总图
（含 ``02_pose_speed_controls_theta_ego_hist.png`` 与自车系 ``07_xy_trajectories_anchor.png``），
并在 ``analysis_report.json`` 的 ``analysis_plots`` 字段记录输出路径。
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data_process.process.data_preprocess import (
    TrainingSample,
    _STATE_DIM,
    _STATE_KEYS,
)

_STATE_SPEED_IDX = 3
_IDX_X = 0
_IDX_Y = 1
_IDX_THETA = 2
_IDX_STEERING = 4
_IDX_ACTION_ACCEL = 5
_IDX_ACTION_OMEGA = 6


def _wrap_angle_delta(delta: float) -> float:
    """将角度差 wrap 到 (-pi, pi]。"""
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


def world_delta_to_ego_frame(
    dx_world: np.ndarray,
    dy_world: np.ndarray,
    theta_anchor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """世界系下相对锚点的平移 (dx,dy)，绕锚点航向旋转到自车系。

    约定：自车 +x 为车头前进方向，+y 为车身左侧（与常见平面运动学 theta 一致）。
    """
    c = np.cos(theta_anchor)
    s = np.sin(theta_anchor)
    x_ego = dx_world * c + dy_world * s
    y_ego = -dx_world * s + dy_world * c
    return x_ego, y_ego


def _summarize(arr: np.ndarray) -> dict[str, float]:
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    a = arr.astype(np.float64, copy=False)
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
    }


def _mask_valid_ratio(mask_arr: np.ndarray) -> float:
    m = np.asarray(mask_arr, dtype=np.float64)
    if m.size == 0:
        return float("nan")
    return float(np.sum(m > 0)) / float(m.size)


def _finite_1d(values: list[float] | list[int]) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    return a[np.isfinite(a)]


def _write_analysis_plots(
    *,
    plot_cfg: Any,
    report: dict[str, Any],
    expected_h_slots: int,
    expected_f_slots: int,
    h_valid: list[int],
    f_valid: list[int],
    lat: list[float],
    head: list[float],
    mean_speed_hist: list[float],
    mean_speed_fut: list[float],
    steering_vals: list[float],
    accel_vals: list[float],
    omega_vals: list[float],
    theta_rel_vals: list[float],
    dx_vals: list[float],
    dy_vals: list[float],
    trajectory_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cl_ratio: list[float],
    lb_ratio: list[float],
    rb_ratio: list[float],
    ld_ratio: list[float],
    cmv_ratio: list[float],
    road_hist: list[tuple[str, int]],
    max_v_abs_diff: list[float],
) -> tuple[str, list[str]]:
    """将主要序列指标写成 PNG，返回 (输出目录, 文件路径列表)。"""
    out_dir = os.path.abspath(os.path.expandvars(str(plot_cfg.output_dir)))
    dpi = int(plot_cfg.get("dpi", 150))
    bins = int(plot_cfg.get("hist_bins", 72))
    os.makedirs(out_dir, exist_ok=True)

    written: list[str] = []
    n_ok = int(report.get("samples_analyzed", 0))
    if n_ok <= 0:
        return out_dir, written

    def _save(fig: Any, name: str) -> None:
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
        plt.close(fig)

    # 1) 有效历史 / 未来帧数
    ah = _finite_1d(h_valid)
    af = _finite_1d(f_valid)
    if ah.size or af.size:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if ah.size:
            axes[0].hist(ah, bins=min(bins, max(10, int(ah.max()) - int(ah.min()) + 1)), color="#2c7fb8", edgecolor="white", linewidth=0.4)
            axes[0].axvline(expected_h_slots, color="#e34a33", linestyle="--", linewidth=1.2, label=f"expected={expected_h_slots}")
            axes[0].legend(fontsize=8)
        axes[0].set_title("Effective history frames (mask sum)")
        axes[0].set_xlabel("count")
        axes[0].set_ylabel("samples")
        if af.size:
            axes[1].hist(af, bins=min(bins, max(10, int(af.max()) - int(af.min()) + 1)), color="#2ca25f", edgecolor="white", linewidth=0.4)
            axes[1].axvline(expected_f_slots, color="#e34a33", linestyle="--", linewidth=1.2, label=f"expected={expected_f_slots}")
            axes[1].legend(fontsize=8)
        axes[1].set_title("Effective future frames (mask sum)")
        axes[1].set_xlabel("count")
        axes[1].set_ylabel("samples")
        fig.suptitle(f"Dataset analysis (n={n_ok})", fontsize=11, y=1.02)
        _save(fig, "01_effective_history_future_frames.png")

    # 2) 标签量、速度、控制量；相对当前锚点的角度与自车系位置分布（轨迹另存 07），2 列布局
    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(5, 2, hspace=0.42, wspace=0.32)

    def _hist_ax(ax: Any, values: list[float], title: str, xlabel: str = "", *, hist_range: tuple[float, float] | None = None) -> None:
        d = _finite_1d(values)
        if d.size:
            nb = min(bins, 80)
            ax.hist(d, bins=nb, range=hist_range, color="#8856a7", edgecolor="white", linewidth=0.25)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("count", fontsize=8)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax40 = fig.add_subplot(gs[4, 0])
    ax41 = fig.add_subplot(gs[4, 1])

    _hist_ax(ax00, lat, "lateral_offset")
    _hist_ax(ax01, head, "heading_error")
    _hist_ax(ax10, mean_speed_hist, "per-sample mean speed (history valid)", "m/s")
    _hist_ax(ax11, mean_speed_fut, "per-sample mean speed (future valid)", "m/s")
    _hist_ax(ax20, steering_vals, "steering (all valid hist+fut timesteps)", "rad")
    _hist_ax(ax21, accel_vals, "action_accel (all valid timesteps)", "m/s^2")
    _hist_ax(ax30, omega_vals, "action_omega (all valid timesteps)", "rad/s")
    _hist_ax(
        ax31,
        theta_rel_vals,
        "theta - theta_anchor (wrapped, all valid timesteps)",
        "rad",
        hist_range=(-float(np.pi), float(np.pi)),
    )
    _hist_ax(ax40, dx_vals, "x_ego forward (ego frame, all valid timesteps)", "m")
    _hist_ax(ax41, dy_vals, "y_ego left (ego frame, all valid timesteps)", "m")

    fig.suptitle(f"Pose / controls / relative kinematics (n_samples={n_ok})", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save(fig, "02_pose_speed_controls_theta_ego_hist.png")

    n_cur = len(trajectory_snapshots)
    fig_xy = plt.figure(figsize=(8.5, 8.5))
    ax_xy = fig_xy.add_subplot(111)
    color_hist = "#045a8d"
    color_fut = "#238443"
    legend_hist = False
    legend_fut = False
    if n_cur > 0:
        a_min = float(plot_cfg.get("xy_traj_alpha_min", 0.14))
        a_max = float(plot_cfg.get("xy_traj_alpha_max", 0.62))
        a_scale = float(plot_cfg.get("xy_traj_alpha_scale", 130.0))
        lw = float(plot_cfg.get("xy_traj_linewidth", 1.15))
        alpha = float(np.clip(a_scale / float(max(1, n_cur)), a_min, a_max))
        for xs_h, ys_h, xs_f, ys_f in trajectory_snapshots:
            if xs_h.size >= 2 and np.all(np.isfinite(xs_h)) and np.all(np.isfinite(ys_h)):
                ax_xy.plot(
                    xs_h,
                    ys_h,
                    color=color_hist,
                    alpha=alpha,
                    linewidth=lw,
                    label="history" if not legend_hist else None,
                )
                legend_hist = True
            if xs_f.size >= 1 and np.all(np.isfinite(xs_f)) and np.all(np.isfinite(ys_f)):
                xf = np.concatenate([[0.0], xs_f])
                yf = np.concatenate([[0.0], ys_f])
                ax_xy.plot(
                    xf,
                    yf,
                    color=color_fut,
                    alpha=alpha,
                    linewidth=lw,
                    label="future" if not legend_fut else None,
                )
                legend_fut = True
    ax_xy.scatter([0.0], [0.0], c="#e34a33", s=22, zorder=6, label="current anchor", edgecolors="white", linewidths=0.35)
    ax_xy.axhline(0.0, color="0.8", linewidth=0.45, zorder=1)
    ax_xy.axvline(0.0, color="0.8", linewidth=0.45, zorder=1)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.set_title(
        f"(x_ego, y_ego) ego frame at anchor (+x forward, +y left), n_curves={n_cur}",
        fontsize=10,
    )
    ax_xy.set_xlabel("x_ego forward (m)", fontsize=9)
    ax_xy.set_ylabel("y_ego left (m)", fontsize=9)
    ax_xy.legend(loc="upper right", fontsize=8)
    fig_xy.suptitle(f"XY trajectories (ego / anchor frame, n_samples={n_ok})", fontsize=11, y=0.98)
    fig_xy.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig_xy, "07_xy_trajectories_anchor.png")

    # 3) 道路相关 mask 有效比例
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes.ravel()
    series = [
        ("centerline", _finite_1d(cl_ratio)),
        ("left_boundary", _finite_1d(lb_ratio)),
        ("right_boundary", _finite_1d(rb_ratio)),
        ("lane_dividers", _finite_1d(ld_ratio)),
        ("centerline_max_v", _finite_1d(cmv_ratio)),
    ]
    for i, (name, data) in enumerate(series):
        ax = axes_flat[i]
        if data.size:
            ax.hist(data, bins=min(bins, 50), range=(0, 1) if data.size and data.max() <= 1.0 + 1e-6 else None, color="#f99d1c", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name} mask valid ratio")
        ax.set_xlabel("ratio")
        ax.set_ylabel("samples")
    axes_flat[5].axis("off")
    fig.tight_layout()
    _save(fig, "03_road_mask_valid_ratios.png")

    # 4) 道路类型 Top-N
    if road_hist:
        labels = [k[:40] + ("…" if len(k) > 40 else "") for k, _ in road_hist]
        counts = [c for _, c in road_hist]
        fig, ax = plt.subplots(figsize=(10, max(4.0, 0.22 * len(road_hist))))
        y = np.arange(len(labels))
        ax.barh(y, counts, color="#3690c0", edgecolor="white", linewidth=0.3)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("count (token occurrences)")
        ax.set_title("road_segment_types (top bucket)")
        fig.tight_layout()
        _save(fig, "04_road_segment_types_top.png")

    # 5) centerline_max_v 与 future max speed 绝对误差
    md = _finite_1d(max_v_abs_diff)
    if md.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(md, bins=min(bins, 80), color="#756bb1", edgecolor="white", linewidth=0.3)
        atol = float(report.get("centerline_max_v_vs_future_max_speed", {}).get("atol", 1e-5))
        ax.axvline(atol, color="#e34a33", linestyle="--", linewidth=1.2, label=f"atol={atol:g}")
        ax.set_yscale("log")
        ax.set_title("|centerline_max_v[0] - max(v_future)| (compared samples)")
        ax.set_xlabel("abs diff")
        ax.set_ylabel("samples (log)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        _save(fig, "05_centerline_max_v_abs_diff.png")

    # 6) 汇总比例条形图
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = [
        "full_history",
        "full_future",
        "nonfinite_hist",
        "nonfinite_fut",
        "max_v_mismatch",
        "vehicle_params_null",
    ]
    vals = np.asarray(
        [
            float(report.get("full_history_fraction", float("nan"))),
            float(report.get("full_future_fraction", float("nan"))),
            float(report["samples_with_nonfinite_history_states"]) / float(n_ok) if n_ok else float("nan"),
            float(report["samples_with_nonfinite_future_states"]) / float(n_ok) if n_ok else float("nan"),
            float(report.get("centerline_max_v_vs_future_max_speed", {}).get("mismatch_fraction", float("nan"))),
            float(report["samples_vehicle_params_null"]) / float(n_ok) if n_ok else float("nan"),
        ],
        dtype=np.float64,
    )
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.arange(len(cats))
    colors = ["#2c7fb8", "#2ca25f", "#e34a33", "#e34a33", "#fdae61", "#969696"]
    ax.bar(x, vals.tolist(), color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x, cats, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Quality summary (fractions)")
    fig.tight_layout()
    _save(fig, "06_summary_fractions.png")

    return out_dir, written


def analyze_directory(cfg: DictConfig) -> dict[str, Any]:
    """遍历 ``cfg.analysis.samples_dir`` 下样本，写 ``cfg.analysis.output_report_path``。"""
    ac = cfg.analysis
    samples_dir = os.path.abspath(os.path.expandvars(str(ac.samples_dir)))
    out_report = os.path.abspath(os.path.expandvars(str(ac.output_report_path)))
    prep_report_path = os.path.abspath(os.path.expandvars(str(ac.preprocess_report_path)))

    max_samples = ac.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
    verify = bool(ac.get("verify_preprocess_report", True))
    max_v_atol = float(ac.get("centerline_max_v_atol", 1e-5))
    road_top_k = int(ac.get("road_type_histogram_top_k", 40))

    if not os.path.isdir(samples_dir):
        raise FileNotFoundError(f"samples_dir 不存在或不是目录: {samples_dir}")

    fnames = sorted(f for f in os.listdir(samples_dir) if f.endswith(".pkl"))
    if max_samples is not None:
        fnames = fnames[:max_samples]

    pc = cfg.preprocess
    expected_h_slots = int(pc.history_len) + 1
    expected_f_slots = int(pc.future_len)

    h_valid: list[int] = []
    f_valid: list[int] = []
    hfdt: list[float] = []
    dts: list[float] = []
    lat: list[float] = []
    head: list[float] = []
    cl_ratio: list[float] = []
    lb_ratio: list[float] = []
    rb_ratio: list[float] = []
    ld_ratio: list[float] = []
    cmv_ratio: list[float] = []

    nan_hist_count = 0
    nan_fut_count = 0
    samples_failed = 0
    mean_speed_hist: list[float] = []
    mean_speed_fut: list[float] = []

    full_history_count = 0
    full_future_count = 0
    shape_mismatch_history = 0
    shape_mismatch_future = 0
    vehicle_params_null = 0
    road_types_none = 0
    road_types_empty_list = 0
    road_type_counter: Counter[str] = Counter()

    hist_dim_means: list[list[float]] = [[] for _ in range(_STATE_DIM)]
    fut_dim_means: list[list[float]] = [[] for _ in range(_STATE_DIM)]

    max_v_mismatch = 0
    max_v_abs_diff: list[float] = []
    max_v_skipped_no_future_valid = 0
    max_v_skipped_no_centerline_max_v = 0

    steering_vals: list[float] = []
    accel_vals: list[float] = []
    omega_vals: list[float] = []
    theta_rel_vals: list[float] = []
    dx_vals: list[float] = []
    dy_vals: list[float] = []
    trajectory_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    plot_xy_cap = int(OmegaConf.select(ac, "plot.max_xy_trajectories", default=400))

    for fname in tqdm(fnames, desc="Analyzing samples", unit="file"):
        path = os.path.join(samples_dir, fname)
        try:
            with open(path, "rb") as f:
                s: TrainingSample = pickle.load(f)
        except Exception:
            samples_failed += 1
            continue

        if s.vehicle_params is None:
            vehicle_params_null += 1

        h_mask = np.asarray(s.history_mask, dtype=np.float64)
        f_mask = np.asarray(s.future_mask, dtype=np.float64)
        h_sum = int(np.sum(h_mask))
        f_sum = int(np.sum(f_mask))
        h_valid.append(h_sum)
        f_valid.append(f_sum)
        if h_sum == expected_h_slots:
            full_history_count += 1
        if f_sum == expected_f_slots:
            full_future_count += 1

        hfdt.append(float(s.history_future_dt))
        dts.append(float(s.dt))
        lat.append(float(s.lateral_offset))
        head.append(float(s.heading_error))

        hs = np.asarray(s.history_states, dtype=np.float64)
        fs = np.asarray(s.future_states, dtype=np.float64)
        if hs.shape[0] != h_mask.shape[0] or hs.shape[1] != _STATE_DIM:
            shape_mismatch_history += 1
        if fs.shape[0] != f_mask.shape[0] or fs.shape[1] != _STATE_DIM:
            shape_mismatch_future += 1

        hb = h_mask.astype(bool)
        fb = f_mask.astype(bool)
        if hb.any():
            sub = hs[hb]
            if np.any(~np.isfinite(sub)):
                nan_hist_count += 1
            sm = sub[:, _STATE_SPEED_IDX]
            if np.all(np.isfinite(sm)):
                mean_speed_hist.append(float(np.mean(sm)))
            for d in range(_STATE_DIM):
                col = sub[:, d]
                if np.all(np.isfinite(col)):
                    hist_dim_means[d].append(float(np.mean(col)))
        if fb.any():
            sub = fs[fb]
            if np.any(~np.isfinite(sub)):
                nan_fut_count += 1
            sm = sub[:, _STATE_SPEED_IDX]
            if np.all(np.isfinite(sm)):
                mean_speed_fut.append(float(np.mean(sm)))
            for d in range(_STATE_DIM):
                col = sub[:, d]
                if np.all(np.isfinite(col)):
                    fut_dim_means[d].append(float(np.mean(col)))

        if hb.any():
            i_cur_arr = np.flatnonzero(hb)
            i_cur = int(i_cur_arr[-1])
            xc = float(hs[i_cur, _IDX_X])
            yc = float(hs[i_cur, _IDX_Y])
            tc = float(hs[i_cur, _IDX_THETA])
            if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(tc):
                c_t = float(np.cos(tc))
                s_t = float(np.sin(tc))
                for idx in i_cur_arr:
                    row = hs[int(idx), :]
                    if np.all(np.isfinite(row)):
                        steering_vals.append(float(row[_IDX_STEERING]))
                        accel_vals.append(float(row[_IDX_ACTION_ACCEL]))
                        omega_vals.append(float(row[_IDX_ACTION_OMEGA]))
                        theta_rel_vals.append(_wrap_angle_delta(float(row[_IDX_THETA]) - tc))
                        dx_w = float(row[_IDX_X]) - xc
                        dy_w = float(row[_IDX_Y]) - yc
                        dx_vals.append(dx_w * c_t + dy_w * s_t)
                        dy_vals.append(-dx_w * s_t + dy_w * c_t)
                fi = np.flatnonzero(fb)
                for idx in fi:
                    row = fs[int(idx), :]
                    if np.all(np.isfinite(row)):
                        steering_vals.append(float(row[_IDX_STEERING]))
                        accel_vals.append(float(row[_IDX_ACTION_ACCEL]))
                        omega_vals.append(float(row[_IDX_ACTION_OMEGA]))
                        theta_rel_vals.append(_wrap_angle_delta(float(row[_IDX_THETA]) - tc))
                        dx_w = float(row[_IDX_X]) - xc
                        dy_w = float(row[_IDX_Y]) - yc
                        dx_vals.append(dx_w * c_t + dy_w * s_t)
                        dy_vals.append(-dx_w * s_t + dy_w * c_t)

                if len(trajectory_snapshots) < plot_xy_cap:
                    dx_w_h = hs[i_cur_arr, _IDX_X] - xc
                    dy_w_h = hs[i_cur_arr, _IDX_Y] - yc
                    xh, yh = world_delta_to_ego_frame(dx_w_h, dy_w_h, tc)
                    if fi.size:
                        dx_w_f = fs[fi, _IDX_X] - xc
                        dy_w_f = fs[fi, _IDX_Y] - yc
                        if not (np.all(np.isfinite(dx_w_f)) and np.all(np.isfinite(dy_w_f))):
                            xf = np.empty(0, dtype=np.float64)
                            yf = np.empty(0, dtype=np.float64)
                        else:
                            xf, yf = world_delta_to_ego_frame(dx_w_f, dy_w_f, tc)
                    else:
                        xf = np.empty(0, dtype=np.float64)
                        yf = np.empty(0, dtype=np.float64)
                    if xh.size >= 2 and np.all(np.isfinite(xh)) and np.all(np.isfinite(yh)):
                        trajectory_snapshots.append((xh, yh, xf, yf))

        cl_ratio.append(_mask_valid_ratio(s.centerline_mask))
        lb_ratio.append(_mask_valid_ratio(s.left_boundary_mask))
        rb_ratio.append(_mask_valid_ratio(s.right_boundary_mask))
        ld_ratio.append(_mask_valid_ratio(s.lane_dividers_mask))
        cmv_ratio.append(_mask_valid_ratio(s.centerline_max_v_mask))

        if s.road_segment_types is None:
            road_types_none += 1
        elif len(s.road_segment_types) == 0:
            road_types_empty_list += 1
        else:
            for t in s.road_segment_types:
                road_type_counter[str(t)] += 1

        if not fb.any():
            max_v_skipped_no_future_valid += 1
        else:
            cmv = s.centerline_max_v
            arr = None if cmv is None else np.asarray(cmv, dtype=np.float64)
            if arr is None or arr.size == 0:
                max_v_skipped_no_centerline_max_v += 1
            else:
                fv_max = float(np.max(fs[fb, _STATE_SPEED_IDX]))
                cv0 = float(arr.reshape(-1)[0])
                diff = abs(cv0 - fv_max)
                max_v_abs_diff.append(diff)
                if diff > max_v_atol:
                    max_v_mismatch += 1

    n_ok = len(h_valid)
    road_hist = road_type_counter.most_common(max(0, road_top_k))

    report: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "samples_dir": samples_dir,
        "pkl_files_scheduled": len(fnames),
        "samples_analyzed": n_ok,
        "samples_failed_load": samples_failed,
        "expected_history_slots": expected_h_slots,
        "expected_future_slots": expected_f_slots,
        "full_history_fraction": float(full_history_count) / float(n_ok) if n_ok else float("nan"),
        "full_future_fraction": float(full_future_count) / float(n_ok) if n_ok else float("nan"),
        "shape_mismatch_history_states": shape_mismatch_history,
        "shape_mismatch_future_states": shape_mismatch_future,
        "samples_vehicle_params_null": vehicle_params_null,
        "road_segment_types_none": road_types_none,
        "road_segment_types_empty_list": road_types_empty_list,
        "road_segment_types_histogram_top": [{"type": k, "count": c} for k, c in road_hist],
        "samples_with_nonfinite_history_states": nan_hist_count,
        "samples_with_nonfinite_future_states": nan_fut_count,
        "effective_history_frames": _summarize(np.asarray(h_valid, dtype=np.float64)),
        "effective_future_frames": _summarize(np.asarray(f_valid, dtype=np.float64)),
        "history_future_dt": _summarize(np.asarray(hfdt, dtype=np.float64)),
        "dt": _summarize(np.asarray(dts, dtype=np.float64)),
        "lateral_offset": _summarize(np.asarray(lat, dtype=np.float64)),
        "heading_error": _summarize(np.asarray(head, dtype=np.float64)),
        "centerline_mask_valid_ratio": _summarize(np.asarray(cl_ratio, dtype=np.float64)),
        "left_boundary_mask_valid_ratio": _summarize(np.asarray(lb_ratio, dtype=np.float64)),
        "right_boundary_mask_valid_ratio": _summarize(np.asarray(rb_ratio, dtype=np.float64)),
        "lane_dividers_mask_valid_ratio": _summarize(np.asarray(ld_ratio, dtype=np.float64)),
        "centerline_max_v_mask_valid_ratio": _summarize(np.asarray(cmv_ratio, dtype=np.float64)),
        "per_sample_mean_speed_history_valid": _summarize(np.asarray(mean_speed_hist, dtype=np.float64)),
        "per_sample_mean_speed_future_valid": _summarize(np.asarray(mean_speed_fut, dtype=np.float64)),
        "per_sample_mean_state_by_dim_history_valid": {
            _STATE_KEYS[d]: _summarize(np.asarray(hist_dim_means[d], dtype=np.float64))
            for d in range(_STATE_DIM)
        },
        "per_sample_mean_state_by_dim_future_valid": {
            _STATE_KEYS[d]: _summarize(np.asarray(fut_dim_means[d], dtype=np.float64))
            for d in range(_STATE_DIM)
        },
        "centerline_max_v_vs_future_max_speed": {
            "atol": max_v_atol,
            "samples_compared": len(max_v_abs_diff),
            "samples_mismatch": max_v_mismatch,
            "mismatch_fraction": float(max_v_mismatch) / float(len(max_v_abs_diff))
            if max_v_abs_diff
            else float("nan"),
            "abs_diff": _summarize(np.asarray(max_v_abs_diff, dtype=np.float64)),
            "skipped_no_future_valid": max_v_skipped_no_future_valid,
            "skipped_no_centerline_max_v": max_v_skipped_no_centerline_max_v,
        },
        "preprocess_config_resolved": OmegaConf.to_container(cfg.preprocess, resolve=True),
        "analysis_config_resolved": OmegaConf.to_container(ac, resolve=True),
    }

    hf_dt = float(pc.history_future_dt)
    base_dt = float(report["dt"]["mean"]) if report["dt"]["count"] else float("nan")
    if np.isfinite(base_dt) and base_dt > 0:
        report["derived_state_stride"] = max(1, int(round(hf_dt / base_dt)))

    eff_h_max = float(report["effective_history_frames"]["max"]) if report["effective_history_frames"]["count"] else float("nan")
    eff_f_max = float(report["effective_future_frames"]["max"]) if report["effective_future_frames"]["count"] else float("nan")
    if np.isfinite(eff_h_max) and eff_h_max > expected_h_slots:
        report["effective_history_exceeds_config"] = {
            "max_effective_history_frames": eff_h_max,
            "expected_history_slots": expected_h_slots,
        }
    if np.isfinite(eff_f_max) and eff_f_max > expected_f_slots:
        report["effective_future_exceeds_config"] = {
            "max_effective_future_frames": eff_f_max,
            "expected_future_slots": expected_f_slots,
        }

    if verify and os.path.isfile(prep_report_path):
        with open(prep_report_path, encoding="utf-8") as f:
            prep = json.load(f)
        report["preprocess_report_path"] = prep_report_path
        report["preprocess_report_summary"] = {
            k: prep.get(k)
            for k in (
                "total_samples",
                "total_episodes",
                "history_len",
                "future_len",
                "history_future_dt",
                "sample_interval",
                "dt",
            )
        }
        n_expected = int(prep.get("total_samples", -1))
        if max_samples is None and n_expected >= 0 and n_expected != len(fnames):
            report["preprocess_report_mismatch"] = {
                "total_samples_in_report": n_expected,
                "pkl_files_analyzed": len(fnames),
            }
    elif verify:
        report["preprocess_report_note"] = f"未找到预处理报告，跳过校验: {prep_report_path}"

    plot_node = OmegaConf.select(ac, "plot", default=None)
    plot_summary: dict[str, Any] = {"enable": False, "output_dir": None, "files": []}
    if plot_node is not None and bool(OmegaConf.select(plot_node, "enable", default=False)) and n_ok > 0:
        pdir, pfiles = _write_analysis_plots(
            plot_cfg=plot_node,
            report=report,
            expected_h_slots=expected_h_slots,
            expected_f_slots=expected_f_slots,
            h_valid=h_valid,
            f_valid=f_valid,
            lat=lat,
            head=head,
            mean_speed_hist=mean_speed_hist,
            mean_speed_fut=mean_speed_fut,
            steering_vals=steering_vals,
            accel_vals=accel_vals,
            omega_vals=omega_vals,
            theta_rel_vals=theta_rel_vals,
            dx_vals=dx_vals,
            dy_vals=dy_vals,
            trajectory_snapshots=trajectory_snapshots,
            cl_ratio=cl_ratio,
            lb_ratio=lb_ratio,
            rb_ratio=rb_ratio,
            ld_ratio=ld_ratio,
            cmv_ratio=cmv_ratio,
            road_hist=road_hist,
            max_v_abs_diff=max_v_abs_diff,
        )
        plot_summary = {"enable": True, "output_dir": pdir, "files": pfiles}
    report["analysis_plots"] = plot_summary

    out_dir = os.path.dirname(out_report)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"已分析样本目录: {samples_dir}")
    print(f"报告已写入: {out_report}")
    plots = report.get("analysis_plots", {})
    if plots.get("enable") and plots.get("files"):
        print(f"可视化已写入目录: {plots.get('output_dir')}")
        for p in plots["files"]:
            print(f"  - {p}")
    for key in (
        "samples_analyzed",
        "full_history_fraction",
        "full_future_fraction",
        "effective_history_frames",
        "effective_future_frames",
        "history_future_dt",
        "dt",
        "centerline_max_v_vs_future_max_speed",
    ):
        print(f"  {key}: {report[key]}")

    return report
