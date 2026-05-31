"""扩散过程可视化：展示去噪各步骤的轨迹变化。

使用方式:
    from il.data.visualization import DiffusionProcessVisualizer

    # sample 中需包含 x_samples（list[Tensor(F, 4)]）以及道路/GT 信息
    # 方式一：直接 plot 单个样本的扩散过程（自动展开为多子图）
    DiffusionProcessVisualizer.plot_diffusion(sample, show=True)
    DiffusionProcessVisualizer.plot_diffusion(sample, save_path="diffusion.png")

    # 方式二：手动构造 list[dict] 后走标准框架
    steps = DiffusionProcessVisualizer.build_step_dicts(sample)
    DiffusionProcessVisualizer.plot(steps, show=True)
    DiffusionProcessVisualizer.log_to_tensorboard(writer, steps, tag="diffusion")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from il.data.visualization.vis_base import VisualizerBase


class DiffusionProcessVisualizer(VisualizerBase):
    """可视化扩散模型去噪过程中各步骤的轨迹。

    每个子图展示一个扩散步骤的轨迹状态，叠加道路中心线和 GT future 作为参考。
    支持 VisualizerBase 的所有能力：plot / save / log_to_tensorboard / to_numpy_image。
    """

    DEFAULT_FIGSIZE = (5, 4)

    COLORS = {
        "x_sample": "#7f7f7f",      # 灰色 - 扩散步骤轨迹
        "future": "#ff7f0e",         # 橙色 - GT future
        "centerline": "#2ca02c",     # 绿色 - 中心线
        "left_boundary": "#d62728",  # 红色
        "right_boundary": "#9467bd", # 紫色
    }

    @classmethod
    def build_step_dicts(cls, data: dict[str, Any]) -> list[dict[str, Any]]:
        """将单个样本中的 x_samples 展开为每步一个 dict 的列表。

        返回的 list[dict] 可直接传给 ``plot()`` / ``log_to_tensorboard()``
        实现单列多行布局。

        Parameters
        ----------
        data : dict
            包含 ``x_samples`` (list[Tensor(F, 4)]) 以及
            ``future``, ``future_mask``, ``centerline``, ``centerline_mask`` 等字段。

        Returns
        -------
        list[dict]
            每个 dict 包含单步绘图所需的全部信息。
        """
        x_samples = data.get("x_samples", [])
        x_samples = list(x_samples.unbind(0)) if isinstance(x_samples, torch.Tensor) else x_samples
        if not x_samples:
            raise ValueError("data 中不包含 x_samples")

        num_steps = len(x_samples)
        step_dicts = []
        for idx, x_sample_tensor in enumerate(x_samples):
            step_dict = {
                "x_sample": cls._to_numpy(x_sample_tensor),
                "future": cls._to_numpy(data["future"]),
                "future_mask": cls._to_numpy(data["future_mask"]),
                "centerline": cls._to_numpy(data["centerline"]),
                "centerline_mask": cls._to_numpy(data["centerline_mask"]),
            }
            # 可选：左右边界
            if "left_boundary" in data:
                step_dict["left_boundary"] = cls._to_numpy(data["left_boundary"])
                step_dict["left_boundary_mask"] = cls._to_numpy(data["left_boundary_mask"])
            if "right_boundary" in data:
                step_dict["right_boundary"] = cls._to_numpy(data["right_boundary"])
                step_dict["right_boundary_mask"] = cls._to_numpy(data["right_boundary_mask"])

            step_dict["_step_index"] = idx
            step_dict["_num_steps"] = num_steps
            step_dicts.append(step_dict)

        return step_dicts

    @classmethod
    def plot_diffusion(
        cls,
        data: dict[str, Any],
        **kwargs,
    ) -> Figure:
        """便捷方法：一步完成扩散过程的多子图可视化。

        内部调用 ``build_step_dicts`` + ``plot``，
        支持 save_path / show / dpi 等所有 VisualizerBase.plot 参数。

        Parameters
        ----------
        data : dict
            包含 x_samples 的样本数据。
        **kwargs
            透传给 VisualizerBase.plot()。

        Returns
        -------
        Figure
        """
        step_dicts = cls.build_step_dicts(data)
        num_steps = len(step_dicts)
        titles = []
        for idx in range(num_steps):
            timestep = num_steps - 1 - idx
            label = f"t={timestep}" if timestep > 0 else "t=0 (final)"
            titles.append(label)

        return cls.plot(step_dicts, title=titles, **kwargs)

    @classmethod
    def _draw(cls, ax: Axes, data: dict[str, Any], **kwargs) -> None:
        """绘制单个扩散步骤：道路背景 + GT future + x_sample 轨迹。"""
        future = data["future"]
        future_mask = data["future_mask"]
        centerline = data["centerline"]
        centerline_mask = data["centerline_mask"]
        x_sample = data["x_sample"]

        # ── 道路背景 ──────────────────────────────────────────────────

        cls._plot_polyline(
            ax, centerline, centerline_mask,
            color=cls.COLORS["centerline"], linewidth=1.0,
            linestyle="--", label="Centerline", alpha=0.5,
        )

        if "left_boundary" in data:
            cls._plot_polyline(
                ax, data["left_boundary"], data["left_boundary_mask"],
                color=cls.COLORS["left_boundary"], linewidth=1.5,
                linestyle="-", label="Left Boundary", alpha=0.4,
            )

        if "right_boundary" in data:
            cls._plot_polyline(
                ax, data["right_boundary"], data["right_boundary_mask"],
                color=cls.COLORS["right_boundary"], linewidth=1.5,
                linestyle="-", label="Right Boundary", alpha=0.4,
            )

        # ── GT future ─────────────────────────────────────────────────

        cls._plot_polyline(
            ax, future[:, :2], future_mask,
            color=cls.COLORS["future"], linewidth=1.5,
            linestyle="-", label="GT", marker="s", markersize=2.0, alpha=0.5,
        )

        # ── 当前步骤的 x_sample 轨迹 ─────────────────────────────────

        cls._plot_polyline(
            ax, x_sample[:, :2], future_mask,
            color=cls.COLORS["x_sample"], linewidth=2.0,
            linestyle="-", label="X Sample", marker="o", markersize=2.5,
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
