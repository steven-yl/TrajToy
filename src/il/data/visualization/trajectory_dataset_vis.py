"""TrajectoryDataset 样本可视化。

使用方式:
    from il.data.visualization import TrajectoryDatasetVisualizer

    dataset = TrajectoryDataset(...)
    sample = dataset[0]
    TrajectoryDatasetVisualizer.plot(sample, show=True)
    TrajectoryDatasetVisualizer.plot(sample, save_path="sample_0.png")

    # 带模型预测结果
    sample["pred_future"] = model_pred  # (F, 2) or (F, 4)
    TrajectoryDatasetVisualizer.plot(sample, show=True)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from il.data.visualization.vis_base import VisualizerBase


class TrajectoryDatasetVisualizer(VisualizerBase):
    """可视化 TrajectoryDataset.__getitem__ 返回的 sample dict。

    支持绘制：
    - 历史轨迹（含方向箭头）
    - 未来轨迹 GT（含方向箭头）
    - 模型预测轨迹 pred_future（可选，含方向箭头）
    - 道路中心线
    - 左右道路边界
    - 车道分割线
    - 中心线最大速度（颜色映射）
    - 自车当前位置标记
    """

    # ── 默认配色 ──────────────────────────────────────────────────────

    COLORS = {
        "pred_future": "#e74c3c",    # 鲜红色，区分于 GT future
        "history": "#1f77b4",        # 蓝色
        "future": "#ff7f0e",         # 橙色
        "centerline": "#2ca02c",     # 绿色
        "left_boundary": "#d62728",  # 红色
        "right_boundary": "#9467bd", # 紫色
        "lane_divider": "#8c564b",   # 棕色
        "max_v": "#e377c2",          # 粉色
        "ego": "#17becf",            # 青色
    }

    @classmethod
    def _draw(cls, ax: Axes, data: dict[str, Any], **kwargs) -> None:
        """绘制 TrajectoryDataset 的单条样本。"""
        show_max_v = kwargs.get("show_max_v", True)
        show_arrows = kwargs.get("show_arrows", True)
        arrow_interval = kwargs.get("arrow_interval", 3)

        # 转 numpy
        history = cls._to_numpy(data["history"])          # (H+1, 7)
        history_mask = cls._to_numpy(data["history_mask"])  # (H+1,)
        future = cls._to_numpy(data["future"])            # (F, 4)
        future_mask = cls._to_numpy(data["future_mask"])  # (F,)

        centerline = cls._to_numpy(data["centerline"])          # (N, 2)
        centerline_mask = cls._to_numpy(data["centerline_mask"])  # (N,)
        left_boundary = cls._to_numpy(data["left_boundary"])      # (N, 2)
        left_boundary_mask = cls._to_numpy(data["left_boundary_mask"])
        right_boundary = cls._to_numpy(data["right_boundary"])    # (N, 2)
        right_boundary_mask = cls._to_numpy(data["right_boundary_mask"])

        lane_dividers = cls._to_numpy(data["lane_dividers"])        # (D, N, 2)
        lane_dividers_mask = cls._to_numpy(data["lane_dividers_mask"])  # (D, N)

        max_v = cls._to_numpy(data["max_v"])          # (N, 1) 或 (N,)
        max_v_mask = cls._to_numpy(data["max_v_mask"])  # (N,)

        # ── 绘制道路元素 ─────────────────────────────────────────────

        # 中心线
        cls._plot_polyline(
            ax, centerline, centerline_mask,
            color=cls.COLORS["centerline"], linewidth=1.5,
            linestyle="--", label="Centerline", alpha=0.7,
        )

        # 左边界
        cls._plot_polyline(
            ax, left_boundary, left_boundary_mask,
            color=cls.COLORS["left_boundary"], linewidth=2.0,
            linestyle="-", label="Left Boundary",
        )

        # 右边界
        cls._plot_polyline(
            ax, right_boundary, right_boundary_mask,
            color=cls.COLORS["right_boundary"], linewidth=2.0,
            linestyle="-", label="Right Boundary",
        )

        # 车道分割线
        num_dividers = lane_dividers.shape[0]
        for d_idx in range(num_dividers):
            divider_points = lane_dividers[d_idx]  # (N, 2)
            divider_mask = lane_dividers_mask[d_idx]  # (N,)
            label = "Lane Divider" if d_idx == 0 else None
            cls._plot_polyline(
                ax, divider_points, divider_mask,
                color=cls.COLORS["lane_divider"], linewidth=1.0,
                linestyle=":", label=label, alpha=0.6,
            )

        # ── 绘制最大速度（颜色映射到中心线上）────────────────────────

        if show_max_v:
            cls._draw_max_v_colormap(ax, centerline, centerline_mask, max_v, max_v_mask)

        # ── 绘制历史轨迹 ─────────────────────────────────────────────

        history_xy = history[:, :2]  # (H+1, 2)
        cls._plot_polyline(
            ax, history_xy, history_mask,
            color=cls.COLORS["history"], linewidth=2.0,
            linestyle="-", label="History", marker="o", markersize=3.0,
        )

        # 历史轨迹方向箭头
        if show_arrows:
            valid_history = history[history_mask.astype(bool)]
            for i in range(0, len(valid_history), arrow_interval):
                cls._plot_arrow(
                    ax, valid_history[i, 0], valid_history[i, 1], valid_history[i, 2],
                    length=0.5, color=cls.COLORS["history"], linewidth=1.0,
                )

        # ── 绘制未来轨迹 ─────────────────────────────────────────────

        future_xy = future[:, :2]  # (F, 2)
        cls._plot_polyline(
            ax, future_xy, future_mask,
            color=cls.COLORS["future"], linewidth=2.0,
            linestyle="-", label="Future", marker="s", markersize=3.0,
        )

        # 未来轨迹方向箭头
        if show_arrows:
            valid_future = future[future_mask.astype(bool)]
            for i in range(0, len(valid_future), arrow_interval):
                cls._plot_arrow(
                    ax, valid_future[i, 0], valid_future[i, 1], valid_future[i, 2],
                    length=0.5, color=cls.COLORS["future"], linewidth=1.0,
                )

        # ── 绘制模型预测轨迹（可选）────────────────────────────────────

        pred_future = data.get("pred_future")
        if pred_future is not None:
            pred_np = cls._to_numpy(pred_future)  # (F, 2) or (F, 4)
            pred_xy = pred_np[:, :2]

            cls._plot_polyline(
                ax, pred_xy, future_mask,
                color=cls.COLORS["pred_future"], linewidth=2.5,
                linestyle="--", label="Prediction", marker="D", markersize=3.0,
                alpha=0.9,
            )

            # 预测轨迹方向箭头（需要 theta 信息，即至少 3 列）
            if show_arrows and pred_np.shape[1] >= 3:
                valid_pred = pred_np[future_mask.astype(bool)] if future_mask is not None else pred_np
                for i in range(0, len(valid_pred), arrow_interval):
                    cls._plot_arrow(
                        ax, valid_pred[i, 0], valid_pred[i, 1], valid_pred[i, 2],
                        length=0.5, color=cls.COLORS["pred_future"], linewidth=1.0,
                    )

        # ── 绘制自车当前位置 ─────────────────────────────────────────

        ego_idx = np.where(history_mask.astype(bool))[0]
        if len(ego_idx) > 0:
            ego_state = history[ego_idx[-1]]  # 最后一个有效历史帧 = 当前帧
            ax.scatter(
                ego_state[0], ego_state[1],
                color=cls.COLORS["ego"], s=100, zorder=10,
                marker="^", label="Ego (current)",
            )
            cls._plot_arrow(
                ax, ego_state[0], ego_state[1], ego_state[2],
                length=1.0, color=cls.COLORS["ego"], linewidth=2.0,
            )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    @classmethod
    def _draw_max_v_colormap(
        cls,
        ax: Axes,
        centerline: np.ndarray,
        centerline_mask: np.ndarray,
        max_v: np.ndarray,
        max_v_mask: np.ndarray,
    ) -> None:
        """在中心线上以颜色渐变展示最大速度限制。"""
        mask = centerline_mask.astype(bool)
        max_v_valid = max_v_mask.astype(bool)

        # 取两个 mask 的交集
        combined_mask = mask & max_v_valid.ravel()[:len(mask)]

        valid_points = centerline[combined_mask]
        valid_v = max_v.ravel()[combined_mask]

        if len(valid_points) < 2 or len(valid_v) < 2:
            return

        # 构建 LineCollection 用颜色表示速度
        segments = np.array([
            [valid_points[i], valid_points[i + 1]]
            for i in range(len(valid_points) - 1)
        ])
        segment_colors = (valid_v[:-1] + valid_v[1:]) / 2.0

        norm = Normalize(vmin=valid_v.min(), vmax=valid_v.max())
        line_collection = LineCollection(
            segments, cmap="RdYlGn", norm=norm, linewidth=4.0, alpha=0.5,
        )
        line_collection.set_array(segment_colors)
        ax.add_collection(line_collection)

        # 添加 colorbar
        scalar_map = ScalarMappable(norm=norm, cmap="RdYlGn")
        scalar_map.set_array([])
        colorbar = ax.get_figure().colorbar(scalar_map, ax=ax, shrink=0.6, pad=0.02)
        colorbar.set_label("Max Speed (m/s)", fontsize=9)
