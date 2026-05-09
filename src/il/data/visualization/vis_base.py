"""可视化基类：定义通用的绘图框架和工具方法。

同时支持 matplotlib 直接展示/保存 和 TensorBoard 写入。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class VisualizerBase(ABC):
    """可视化基类。

    提供通用的 figure 管理、保存/展示、颜色/样式常量。
    子类只需实现 `_draw(ax, data, **kwargs)` 即可。

    使用方式（不需要实例化，直接类方法调用）:
        SubVisualizer.plot(data, save_path="out.png")
        SubVisualizer.plot(data, show=True)
    """

    DEFAULT_FIGSIZE = (12, 8)
    DEFAULT_DPI = 150

    # ── 公共 API（类方法，不需要实例化） ────────────────────────────────

    @classmethod
    def plot(
        cls,
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        title: str | list[str] | None = None,
        save_path: str | Path | None = None,
        show: bool = False,
        ax: Axes | None = None,
        ncols: int = 2,
        close: bool = True,
        **kwargs,
    ) -> Figure:
        """绘制可视化图（matplotlib）。

        Parameters
        ----------
        data : dict | list[dict]
            单条数据字典，或多条数据的列表。
            传入列表时自动创建多子图网格布局。
        figsize : tuple, optional
            图片尺寸，默认 (12, 8)。传入列表时会自动按子图数量调整。
        dpi : int, optional
            图片分辨率，默认 150。
        title : str | list[str], optional
            图片标题。传入列表时可为每个子图分别指定标题。
        save_path : str | Path, optional
            保存路径，若提供则保存到文件。
        show : bool
            是否调用 plt.show() 展示图片。
        ax : Axes, optional
            外部传入的 Axes（仅 data 为单条 dict 时生效）。
        ncols : int
            多子图时每行的列数，默认 2。
        close : bool
            绘制完成后是否自动关闭 figure（防止内存泄漏）。
            当 show=True 或外部传入 ax 时自动设为 False。
        **kwargs
            传递给子类 `_draw` 的额外参数。

        Returns
        -------
        Figure
            matplotlib Figure 对象。
        """
        dpi = dpi or cls.DEFAULT_DPI

        # ── 单条数据 ─────────────────────────────────────────────────
        if isinstance(data, dict):
            figsize = figsize or cls.DEFAULT_FIGSIZE
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            else:
                fig = ax.get_figure()
                close = False

            cls._draw(ax, data, **kwargs)

            single_title = title[0] if isinstance(title, list) else title
            if single_title:
                ax.set_title(single_title, fontsize=12)

            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)

        # ── 多条数据（列表）────────────────────────────────────────────
        else:
            num_samples = len(data)
            if num_samples == 0:
                fig, _ = plt.subplots(1, 1, figsize=cls.DEFAULT_FIGSIZE, dpi=dpi)
                return fig

            ncols = min(ncols, num_samples)
            nrows = (num_samples + ncols - 1) // ncols
            subplot_w, subplot_h = cls.DEFAULT_FIGSIZE
            figsize = figsize or (subplot_w * ncols, subplot_h * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
            axes_flat = np.asarray(axes).ravel() if num_samples > 1 else [axes]

            titles = title if isinstance(title, list) else [title] * num_samples

            for idx, sample in enumerate(data):
                current_ax = axes_flat[idx]
                cls._draw(current_ax, sample, **kwargs)

                sample_title = titles[idx] if idx < len(titles) else None
                if sample_title:
                    current_ax.set_title(sample_title, fontsize=11)

                current_ax.set_aspect("equal")
                current_ax.grid(True, alpha=0.3)
                current_ax.legend(loc="upper right", fontsize=7)

            # 隐藏多余的子图
            for idx in range(num_samples, len(axes_flat)):
                axes_flat[idx].set_visible(False)

        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        if show:
            close = False
            plt.show()

        if close:
            plt.close(fig)

        return fig

    @classmethod
    def log_to_tensorboard(
        cls,
        writer,
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        tag: str = "visualization",
        global_step: int | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        title: str | list[str] | None = None,
        **kwargs,
    ) -> Figure:
        """将可视化结果写入 TensorBoard。

        Parameters
        ----------
        writer : torch.utils.tensorboard.SummaryWriter
            TensorBoard SummaryWriter 实例。
        data : dict | list[dict]
            单条或多条数据。
        tag : str
            TensorBoard 中的 tag 名称。
        global_step : int, optional
            全局步数，用于 TensorBoard 时间轴。
        figsize : tuple, optional
            图片尺寸。
        dpi : int, optional
            图片分辨率。
        title : str | list[str], optional
            图片标题。
        **kwargs
            传递给子类 `_draw` 的额外参数。

        Returns
        -------
        Figure
            matplotlib Figure 对象（已关闭）。
        """
        fig = cls.plot(data, figsize=figsize, dpi=dpi, title=title, close=False, **kwargs)
        writer.add_figure(tag, fig, global_step=global_step, close=True)
        return fig

    @classmethod
    def plot_to_numpy_image(
        cls,
        data: dict[str, Any] | list[dict[str, Any]],
        *,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        title: str | list[str] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """将可视化结果渲染为 numpy RGB 图像数组。

        可用于手动写入 TensorBoard (writer.add_image) 或其他用途。

        Returns
        -------
        np.ndarray
            shape (H, W, 3)，dtype uint8 的 RGB 图像。
        """
        fig = cls.plot(data, figsize=figsize, dpi=dpi, title=title, close=False, **kwargs)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return image

    # ── 子类必须实现 ──────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def _draw(cls, ax: Axes, data: dict[str, Any], **kwargs) -> None:
        """在给定 Axes 上绘制内容。子类实现此方法。"""
        ...

    # ── 通用绘图工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _plot_polyline(
        ax: Axes,
        points: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        color: str = "black",
        linewidth: float = 1.5,
        linestyle: str = "-",
        label: str | None = None,
        marker: str | None = None,
        markersize: float = 3.0,
        alpha: float = 1.0,
    ) -> None:
        """绘制带 mask 的折线。仅绘制 mask 为 True 的有效点。"""
        if points is None or points.size == 0:
            return
        points = np.asarray(points)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            points = points[mask]
        if points.shape[0] == 0:
            return
        ax.plot(
            points[:, 0], points[:, 1],
            color=color, linewidth=linewidth, linestyle=linestyle,
            label=label, marker=marker, markersize=markersize, alpha=alpha,
        )

    @staticmethod
    def _plot_arrow(
        ax: Axes,
        x: float, y: float, theta: float,
        *,
        length: float = 1.0,
        color: str = "black",
        linewidth: float = 1.5,
    ) -> None:
        """绘制方向箭头。"""
        dx = length * np.cos(theta)
        dy = length * np.sin(theta)
        ax.annotate(
            "", xy=(x + dx, y + dy), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=color, lw=linewidth),
        )

    @staticmethod
    def _to_numpy(tensor_or_array) -> np.ndarray:
        """将 torch.Tensor 或 np.ndarray 统一转为 numpy。"""
        if hasattr(tensor_or_array, "numpy"):
            return tensor_or_array.detach().cpu().numpy()
        return np.asarray(tensor_or_array)

    @staticmethod
    def batch_to_samples(batch: dict[str, Any]) -> list[dict[str, Any]]:
        """将 batch 转换为 sample。"""
        batch_size = batch[list(batch.keys())[0]].shape[0]
        sample_list = [{key: batch[key][i] for key in batch} for i in range(batch_size)]
        return sample_list
