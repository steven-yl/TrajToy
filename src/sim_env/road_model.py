"""道路模型：支持直道、弯道、路口、分合道的几何生成。

每个道路片段由中心线离散点序列表示，边界线和车道线由中心线偏移生成。
支持固定配置生成和随机生成两种模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
from omegaconf import DictConfig

from .exceptions import ContinuityError, RoadConfigError


# ── 枚举与配置 ────────────────────────────────────────────────────────


class RoadSegmentType(Enum):
    """道路片段类型。"""

    STRAIGHT = "straight"
    CURVE = "curve"
    INTERSECTION = "intersection"
    SPLIT = "split"
    MERGE = "merge"


@dataclass
class SegmentSpec:
    """单个道路片段规格。

    params 字段含义因 segment_type 而异：
      STRAIGHT:     {"length": float, "num_lanes": int}
      CURVE:        {"radius": float, "angle_deg": float, "num_lanes": int}  正=左转，负=右转
      INTERSECTION: {"size": float, "num_lanes": int}
      SPLIT:        {"length": float, "angle_deg": float, "lane_change_count": int, "num_lanes": int}
      MERGE:        {"length": float, "angle_deg": float, "lane_change_count": int, "num_lanes": int}
    """

    segment_type: RoadSegmentType
    params: dict = field(default_factory=dict)


# ── 道路几何数据 ──────────────────────────────────────────────────────


@dataclass
class RoadGeometry:
    """一条完整道路的几何数据。

    所有数组形状为 (N, 2)，N 为采样点数。
    """
    all_centerline: np.ndarray  # 中心线点序列
    all_left_boundary: np.ndarray  # 左侧边界线
    all_right_boundary: np.ndarray  # 右侧边界线
    road_segments: list[RoadSegment]
    total_length: float = 0.0  # 道路总长度 (m)


@dataclass
class RoadSegment:
    """单个道路片段的几何数据。

    所有数组形状为 (N, 2)，N 为采样点数。
    """

    centerline: np.ndarray  # 中心线点序列
    left_boundary: np.ndarray  # 左侧边界线
    right_boundary: np.ndarray  # 右侧边界线
    lane_dividers: list[np.ndarray] = field(default_factory=list)  # 车道分隔线
    segment_indices: list[int] = field(default_factory=list)  # 每段起始索引
    segment_types: list[RoadSegmentType] = field(default_factory=list)  # 每段道路类型
    total_length: float = 0.0  # 道路片段总长度 (m)

# ── 片段几何生成 ──────────────────────────────────────────────────────


@dataclass
class RoadGenerationConfig:
    """道路生成配置。"""
    # 基础几何参数
    lane_width: float = 3.5  # 车道宽度 (m)
    num_lanes: int = 1  # 单向车道数 (1=单车道, 2=双车道)
    points_per_meter: float = 1.0  # 中心线采样密度

    # 生成模式控制
    fixed_segments: list[SegmentSpec] | None = None  # 非 None 时优先使用该序列，否则随机生成
    loop_segments: bool = False  # True 时在道路末端前自动延长，避免车辆接近尽头被截断

    # 随机生成配置
    num_random_segments: int = 3
    segment_weights: dict[RoadSegmentType, float] = field(default_factory=lambda: {
        RoadSegmentType.STRAIGHT: 5.0,
        RoadSegmentType.CURVE: 5.0,
        RoadSegmentType.INTERSECTION: 0.0,
        RoadSegmentType.SPLIT: 0.0,
        RoadSegmentType.MERGE: 0.0,
    })  # 各类型采样权重，None 时均匀分布
    straight_length_range: tuple[float, float] = (20.0, 80.0)
    curve_radius_range: tuple[float, float] = (5.0, 80.0)
    curve_angle_range: tuple[float, float] = (20.0, 180.0)
    intersection_size_range: tuple[float, float] = (10.0, 20.0)
    split_merge_length_range: tuple[float, float] = (15.0, 40.0)
    split_merge_angle_range: tuple[float, float] = (10.0, 30.0)


def _generate_straight(
    start_xy: np.ndarray, start_theta: float, length: float, density: float,
) -> tuple[np.ndarray, float]:
    """生成直道中心线点。返回 (points, end_theta)。"""
    n = max(int(length * density), 2)
    t = np.linspace(0, length, n)
    dx = np.cos(start_theta)
    dy = np.sin(start_theta)
    points = np.column_stack([start_xy[0] + dx * t, start_xy[1] + dy * t])
    return points, start_theta


def _generate_curve(
    start_xy: np.ndarray, start_theta: float,
    radius: float, angle_deg: float, density: float,
) -> tuple[np.ndarray, float]:
    """生成弯道中心线点。angle_deg 正=左转，负=右转。返回 (points, end_theta)。"""
    angle_rad = np.radians(angle_deg)
    if abs(angle_rad) < 1e-10:
        # 角度接近0，退化为直道
        return _generate_straight(start_xy, start_theta, 0.0, density)
    
    arc_length = abs(radius * angle_rad)
    n = max(int(arc_length * density), 2)
    sign = np.sign(angle_rad)

    # 圆心在车辆左侧(左转)或右侧(右转)
    cx = start_xy[0] - sign * radius * np.sin(start_theta)
    cy = start_xy[1] + sign * radius * np.cos(start_theta)

    # 起始角（从圆心到起点的方向）
    alpha_start = np.arctan2(start_xy[1] - cy, start_xy[0] - cx)
    alphas = np.linspace(alpha_start, alpha_start + angle_rad, n)

    points = np.column_stack([cx + radius * np.cos(alphas), cy + radius * np.sin(alphas)])
    end_theta = start_theta + angle_rad
    return points, float(end_theta)


def _generate_intersection(
    start_xy: np.ndarray, start_theta: float, size: float, density: float,
) -> tuple[np.ndarray, float]:
    """生成路口（简化为短直道通过区域）。"""
    return _generate_straight(start_xy, start_theta, size, density)


def _generate_split(
    start_xy: np.ndarray, start_theta: float,
    length: float, angle_deg: float, lane_change_count: int, density: float,
) -> tuple[np.ndarray, float]:
    """生成分道（主路保持直行）。TODO: 实现真正的分道几何。"""
    return _generate_straight(start_xy, start_theta, length, density)


def _generate_merge(
    start_xy: np.ndarray, start_theta: float,
    length: float, angle_deg: float, lane_change_count: int, density: float,
) -> tuple[np.ndarray, float]:
    """生成合道（主路保持直行）。TODO: 实现真正的合道几何。"""
    return _generate_straight(start_xy, start_theta, length, density)


def _generate_segment_points(
    spec: SegmentSpec, start_xy: np.ndarray, start_theta: float, density: float,
) -> tuple[np.ndarray, float]:
    """根据片段规格生成中心线点序列。"""
    st = spec.segment_type
    p = spec.params

    if st == RoadSegmentType.STRAIGHT:
        return _generate_straight(start_xy, start_theta, p.get("length", 50.0), density)
    elif st == RoadSegmentType.CURVE:
        return _generate_curve(
            start_xy, start_theta,
            p.get("radius", 50.0), p.get("angle_deg", 45.0), density,
        )
    elif st == RoadSegmentType.INTERSECTION:
        return _generate_intersection(start_xy, start_theta, p.get("size", 15.0), density)
    elif st == RoadSegmentType.SPLIT:
        return _generate_split(
            start_xy, start_theta,
            p.get("length", 25.0), p.get("angle_deg", 15.0),
            p.get("lane_change_count", 1), density,
        )
    elif st == RoadSegmentType.MERGE:
        return _generate_merge(
            start_xy, start_theta,
            p.get("length", 25.0), p.get("angle_deg", 15.0),
            p.get("lane_change_count", 1), density,
        )
    else:
        raise RoadConfigError(f"未知道路片段类型: {st}")


# ── 边界线生成 ────────────────────────────────────────────────────────


def _compute_normals(centerline: np.ndarray) -> np.ndarray:
    """计算中心线每个点的左侧法向量 (N, 2)。"""
    # 前向差分，末尾用后向差分
    tangents = np.zeros_like(centerline)
    tangents[:-1] = centerline[1:] - centerline[:-1]
    tangents[-1] = tangents[-2]
    # 归一化
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    tangents = tangents / lengths
    # 左侧法向量: (-dy, dx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _offset_line(centerline: np.ndarray, normals: np.ndarray, offset: float) -> np.ndarray:
    """沿法向量偏移中心线。offset > 0 为左侧。"""
    return centerline + normals * offset


# ── 道路模型主类 ──────────────────────────────────────────────────────


class RoadModel:
    """道路模型：根据片段序列或随机配置生成完整道路几何。"""
    @staticmethod
    def bulid_from_config(cfg: DictConfig) -> RoadModel:
        return RoadModel(cfg)

    def __init__(self, config: RoadGenerationConfig | None = None) -> None:
        self._config = config or RoadGenerationConfig()
        self._geometry: RoadGeometry | None = None
        self._rng: np.random.Generator = np.random.default_rng()
        self._last_nearest_idx: int | None = None  # 缓存上一次的最近点索引

    @property
    def geometry(self) -> RoadGeometry | None:
        return self._geometry

    @property
    def config(self) -> RoadGenerationConfig:
        return self._config

    # ── 固定生成 ──────────────────────────────────────────────────

    def generate_fixed(self, segments: Sequence[SegmentSpec],
            current_xy: np.ndarray | None = None,
            current_theta: float = 0.0,) -> RoadGeometry:
        """根据给定片段序列生成道路。"""
        if not segments:
            raise RoadConfigError("片段序列不能为空")
        self._geometry = self._build_road(list(segments), current_xy, current_theta)
        return self._geometry

    # ── 随机生成 ──────────────────────────────────────────────────

    def generate_random(self, seed: int | None = None,
            current_xy: np.ndarray | None = None,
            current_theta: float = 0.0,) -> RoadGeometry:
        """根据配置随机生成道路。"""
        self._rng = np.random.default_rng(seed)
        cfg = self._config
        segments = self._random_segment_specs(cfg)
        self._geometry = self._build_road(segments, current_xy, current_theta)
        return self._geometry


    def extend_road(self, segments: Sequence[SegmentSpec] | None = None,
            x: float | None = None, y: float | None = None,
            seed: int | None = None) -> RoadGeometry:
        """在当前道路末端追加新片段，只保留车辆所在segment及之后的历史道路。

        Args:
            segments: 指定片段序列则固定生成，None 则随机生成
            x, y: 车辆当前位置，用于确定保留哪个segment起；
                   未提供时默认保留最后一个segment
        """
        if self._geometry is None:
            raise RoadConfigError("道路尚未生成，无法延长")
        geo = self._geometry
        self._rng = np.random.default_rng(seed)
        # 取最后一个片段的末端位置和朝向
        last_seg = geo.road_segments[-1]
        end_xy = last_seg.centerline[-1].copy()
        if len(last_seg.centerline) >= 2:
            tangent = last_seg.centerline[-1] - last_seg.centerline[-2]
            end_theta = float(np.arctan2(tangent[1], tangent[0]))
        else:
            end_theta = 0.0

        # 根据参数决定固定生成还是随机生成
        if segments is not None:
            new_segments = list(segments)
        else:
            new_segments = self._random_segment_specs(self._config)

        # 构建新路段，从末端开始
        new_geo = self._build_road(new_segments, end_xy, end_theta)

        # 确定保留的segment起始索引
        if x is not None and y is not None:
            _, segment_idx = self.get_nearest_idx(x, y)
        elif len(geo.road_segments) > 1:
            segment_idx = len(geo.road_segments) - 1
        else:
            segment_idx = 0

        # 保留车辆所在segment及之后的片段，再拼接新片段
        kept_segments = geo.road_segments[segment_idx:]
        all_segments = kept_segments + new_geo.road_segments

        # 重新计算总长度
        total_length = sum(seg.total_length for seg in all_segments)

        self._geometry = self._assemble_geometry(all_segments, total_length)
        # 重置缓存（道路结构已变）
        self._last_nearest_idx = None
        return self._geometry

    def generate(
        self,
        seed: int | None = None,
        current_xy: np.ndarray | None = None,
        current_theta: float = 0.0,
    ) -> RoadGeometry:
        """按 :attr:`config` 统一生成道路。

        ``fixed_segments`` 非空则固定序列，否则按 ``RoadGenerationConfig`` 随机采样。
        """
        fixed = self._config.fixed_segments
        if fixed:
            return self.generate_fixed(fixed, current_xy, current_theta)
        return self.generate_random(seed, current_xy, current_theta)

    def maybe_extend_for_loop(
        self,
        x: float,
        y: float,
        progress: float,
        *,
        progress_threshold: float = 0.8,
        extend_seed: int | None = None,
    ) -> bool:
        """若开启 ``loop_segments`` 且进度达到阈值则延长道路。

        固定道路模式时复用 ``fixed_segments`` 作为延长模板；否则在末端追加随机段。

        Returns:
            是否执行了延长（调用方需在 True 时重新查询最近道路信息）。
        """
        if not self._config.loop_segments or progress < progress_threshold:
            return False
        segments = self._config.fixed_segments if self._config.fixed_segments else None
        self.extend_road(segments=segments, x=x, y=y, seed=extend_seed)
        return True

    def _random_segment_specs(self, cfg: RoadGenerationConfig) -> list[SegmentSpec]:
        """随机生成片段规格序列。"""
        types = list(RoadSegmentType)
        if cfg.segment_weights:
            weights = np.array([cfg.segment_weights.get(t, 0.0) for t in types])
        else:
            weights = np.ones(len(types))
        total = weights.sum()
        if total <= 0:
            raise RoadConfigError("片段权重之和必须 > 0")
        probs = weights / total

        specs: list[SegmentSpec] = []
        for _ in range(cfg.num_random_segments):
            seg_type = self._rng.choice(types, p=probs)
            params = self._random_params_for(seg_type, cfg)
            specs.append(SegmentSpec(segment_type=seg_type, params=params))
        return specs

    def _random_params_for(
        self, seg_type: RoadSegmentType, cfg: RoadGenerationConfig,
    ) -> dict:
        """为指定类型随机生成参数。"""
        rng = self._rng
        if seg_type == RoadSegmentType.STRAIGHT:
            length = float(rng.uniform(*cfg.straight_length_range))
            if length <= 0:
                raise RoadConfigError(f"直道长度必须 > 0，得到 {length}")
            return {"length": length}
        elif seg_type == RoadSegmentType.CURVE:
            radius = float(rng.uniform(*cfg.curve_radius_range))
            if radius <= 0:
                raise RoadConfigError(f"弯道半径必须 > 0，得到 {radius}")
            angle = float(rng.uniform(*cfg.curve_angle_range))
            sign = rng.choice([-1.0, 1.0])
            return {"radius": radius, "angle_deg": angle * sign}
        elif seg_type == RoadSegmentType.INTERSECTION:
            size = float(rng.uniform(*cfg.intersection_size_range))
            if size <= 0:
                raise RoadConfigError(f"路口尺寸必须 > 0，得到 {size}")
            return {"size": size}
        elif seg_type in (RoadSegmentType.SPLIT, RoadSegmentType.MERGE):
            length = float(rng.uniform(*cfg.split_merge_length_range))
            if length <= 0:
                raise RoadConfigError(f"分合道长度必须 > 0，得到 {length}")
            angle = float(rng.uniform(*cfg.split_merge_angle_range))
            lane_change_count = rng.integers(1, 4)  # 1-3车道变化
            return {
                "length": length,
                "angle_deg": angle,
                "lane_change_count": lane_change_count,
            }
        return {}

    # ── 内部：构建道路 ────────────────────────────────────────────

    def _build_road(
        self,
        segments: list[SegmentSpec],
        current_xy: np.ndarray | None = None,
        current_theta: float = 0.0,
    ) -> RoadGeometry:
        """从片段序列构建完整道路几何。

        Args:
            segments: 道路片段规格列表
            current_xy: 起始位置，默认 [0, 0]
            current_theta: 起始朝向 (rad)，默认 0
        """
        cfg = self._config
        density = cfg.points_per_meter

        road_segments: list[RoadSegment] = []
        if current_xy is None:
            current_xy = np.array([0.0, 0.0])
        else:
            current_xy = np.asarray(current_xy, dtype=np.float64)

        total_length = 0.0
        num_lanes = 0
        for i, spec in enumerate(segments):
            if i == 0:
                num_lanes = cfg.num_lanes
            points, end_theta = _generate_segment_points(
                spec, current_xy, current_theta, density,
            )
            if points.shape[0] < 2:
                raise RoadConfigError(f"片段 {i} 生成点数不足")

            # 连续性检查（跳过第一段）
            if road_segments:
                prev_end = road_segments[-1].centerline[-1]
                gap = np.linalg.norm(points[0] - prev_end)
                if gap > 0.5:
                    raise ContinuityError(i, f"与前段间距 {gap:.3f}m > 0.5m")
                # 去掉重复的起始点
                # points = points[1:]

            # 计算当前片段的中心线长度
            if points.shape[0] >= 2:
                diffs = np.diff(points, axis=0)
                segment_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
            else:
                segment_length = 0.0

            # 法向量和边界线
            normals = _compute_normals(points)
            half_road = cfg.lane_width * num_lanes / 2.0
            left_boundary = _offset_line(points, normals, half_road)
            right_boundary = _offset_line(points, normals, -half_road)

            # 车道分隔线（双车道时有中心分隔线）
            lane_dividers: list[np.ndarray] = []
            if num_lanes >= 2:
                for k in range(1, num_lanes):
                    div_offset = half_road - k * cfg.lane_width
                    lane_dividers.append(_offset_line(points, normals, div_offset))

            # 创建道路片段
            segment = RoadSegment(
                centerline=points,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                lane_dividers=lane_dividers,
                segment_indices=[0],  # 单个片段的起始索引为0
                segment_types=[spec.segment_type],
                total_length=segment_length,
            )
            road_segments.append(segment)
            
            total_length += segment_length
            current_xy = points[-1].copy()
            current_theta = end_theta
            p = spec.params
            num_lanes = p.get("num_lanes", cfg.num_lanes)

        return self._assemble_geometry(road_segments, total_length)

    def _assemble_geometry(self, segments: list[RoadSegment], total_length: float) -> RoadGeometry:
        """从片段列表组装几何缓存，避免后续重复拼接。"""
        if segments:
            all_centerline = np.concatenate([seg.centerline for seg in segments], axis=0)
            all_left_boundary = np.concatenate([seg.left_boundary for seg in segments], axis=0)
            all_right_boundary = np.concatenate([seg.right_boundary for seg in segments], axis=0)
        else:
            all_centerline = np.array([], dtype=np.float64).reshape(0, 2)
            all_left_boundary = np.array([], dtype=np.float64).reshape(0, 2)
            all_right_boundary = np.array([], dtype=np.float64).reshape(0, 2)

        return RoadGeometry(
            all_centerline=all_centerline,
            all_left_boundary=all_left_boundary,
            all_right_boundary=all_right_boundary,
            road_segments=segments,
            total_length=total_length,
        )

    def _segment_idx_for_global_idx(self, idx: int) -> int:
        """将全局中心线索引映射到所属片段索引。"""
        if self._geometry is None:
            raise RoadConfigError("道路尚未生成")
        cumulative_offset = 0
        for i, seg in enumerate(self._geometry.road_segments):
            seg_length = len(seg.centerline)
            if cumulative_offset <= idx < cumulative_offset + seg_length:
                return i
            cumulative_offset += seg_length
        return max(len(self._geometry.road_segments) - 1, 0)

    def _slice_lane_dividers(
        self, idx: int, end_idx: int, *, pad_to: int | None = None,
    ) -> list[np.ndarray]:
        """提取 [idx, end_idx) 范围内的车道分隔线，并可按末尾点填充到固定长度。"""
        if self._geometry is None:
            raise RoadConfigError("道路尚未生成")

        segments = self._geometry.road_segments
        max_dividers = max(len(seg.lane_dividers) for seg in segments) if segments else 0
        divider_pts: list[np.ndarray] = []

        for divider_idx in range(max_dividers):
            divider_arrays: list[np.ndarray] = []
            cumulative_offset = 0
            for seg in segments:
                seg_length = len(seg.centerline)
                seg_start = cumulative_offset
                seg_end = cumulative_offset + seg_length
                has_overlap = seg_start < end_idx and seg_end > idx
                if has_overlap and divider_idx < len(seg.lane_dividers):
                    local_start = max(0, idx - seg_start)
                    local_end = min(seg_length, end_idx - seg_start)
                    divider_arrays.append(seg.lane_dividers[divider_idx][local_start:local_end])
                cumulative_offset += seg_length

            if divider_arrays:
                div = np.concatenate(divider_arrays, axis=0) if len(divider_arrays) > 1 else divider_arrays[0]
                if pad_to is not None and div.shape[0] > 0 and div.shape[0] < pad_to:
                    pad_len = pad_to - div.shape[0]
                    div = np.concatenate([div, np.tile(div[-1:], (pad_len, 1))], axis=0)
                divider_pts.append(div)
        return divider_pts

    # ── 查询工具 ──────────────────────────────────────────────────
    def get_nearest_idx(
        self, x: float, y: float, search_range: int = 0
    ) -> tuple[int, int]:
        """查询点到中心线的最近点索引和所在segment索引。

        Args:
            x, y: 查询位置
            search_range: > 0 时在缓存索引前后 search_range 范围内搜索，
                          0 则全量搜索

        Returns:
            (nearest_idx, segment_idx)
            nearest_idx: 最近中心线点索引
            segment_idx: 所在道路段索引
        """
        if self._geometry is None:
            raise RoadConfigError("道路尚未生成")

        all_centerline = self._geometry.all_centerline
        point = np.array([x, y])

        # 使用缓存的索引，在前后范围内搜索最近点
        if search_range > 0 and self._last_nearest_idx is not None:
            start_idx = max(0, self._last_nearest_idx - search_range)
            end_idx = min(len(all_centerline), self._last_nearest_idx + search_range + 1)
            cl_subset = all_centerline[start_idx:end_idx]
            dists_subset = np.linalg.norm(cl_subset - point, axis=1)
            local_idx = int(np.argmin(dists_subset))
            idx = start_idx + local_idx
        else:
            # 全量搜索
            dists = np.linalg.norm(all_centerline - point, axis=1)
            idx = int(np.argmin(dists))

        # 缓存索引
        self._last_nearest_idx = idx

        return idx, self._segment_idx_for_global_idx(idx)

    def get_nearest_centerline_info(
        self, x: float, y: float, search_range: int = 0
    ) -> tuple[float, float, float]:
        """查询点到中心线的最近信息。

        Returns:
            (lateral_offset, heading_at_nearest, progress)
            lateral_offset: 横向偏移，正=在中心线左侧
            heading_at_nearest: 最近点处道路朝向 (rad)
            progress: 沿中心线的纵向进度 [0, 1]
        """
        idx, _ = self.get_nearest_idx(x, y, search_range)

        cl = self._geometry.all_centerline

        # 道路朝向
        if idx < len(cl) - 1:
            tangent = cl[idx + 1] - cl[idx]
        else:
            tangent = cl[idx] - cl[idx - 1]
        heading = float(np.arctan2(tangent[1], tangent[0]))

        # 横向偏移（带符号）
        point = np.array([x, y])
        to_point = point - cl[idx]
        normal_left = np.array([-np.sin(heading), np.cos(heading)])
        lateral = float(np.dot(to_point, normal_left))

        # 纵向进度
        progress = idx / max(len(cl) - 1, 1)

        return lateral, heading, progress

    def get_road_segment_ahead(
        self, x: float, y: float, num_points: int = 20, pad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], int, int, float, list[RoadSegmentType]]:
        """获取当前位置前方固定数量的道路点（绝对坐标）。

        从最近点开始，沿中心线向前取 num_points 个点。

        Args:
            x, y: 查询位置
            num_points: 返回的道路点数
            pad: True 时不足 num_points 的部分用末尾点填充，
                 False 时按实际剩余点数返回（长度可能 < num_points）

        Returns:
            (centerline_pts, left_boundary_pts, right_boundary_pts,
             lane_divider_pts_list, nearest_idx, actual_length_num, segment_length_m,
             road_segment_types)
            pad=True  时各数组形状为 (num_points, 2)
            pad=False 时各数组形状为 (actual_length_num, 2)
            lane_divider_pts_list: 每条车道线同形状，单车道时为空列表
            nearest_idx: 最近中心线点索引
            actual_length_num: 实际有效点数（不含填充），<= num_points
            segment_length_m: 实际有效段的中心线长度 (m)
            road_segment_types: [idx, end_idx) 范围覆盖到的所有道路段类型（按顺序去重）
        """
        if self._geometry is None:
            raise RoadConfigError("道路尚未生成")

        all_centerline = self._geometry.all_centerline
        n_total = len(all_centerline)

        # 找最近点索引
        dists = np.linalg.norm(all_centerline - np.array([x, y]), axis=1)
        idx = int(np.argmin(dists))

        # 从 idx 开始取 num_points 个点
        end_idx = min(idx + num_points, n_total)
        actual = end_idx - idx

        # 实际有效段的中心线长度 (m)
        if actual >= 2:
            seg_diffs = np.diff(all_centerline[idx:end_idx], axis=0)
            segment_length_m = float(np.sum(np.linalg.norm(seg_diffs, axis=1)))
        else:
            segment_length_m = 0.0

        # 收集 [idx, end_idx) 覆盖到的所有道路段类型（保持顺序，去重）
        road_segment_types: list[RoadSegmentType] = []
        cumulative_offset = 0
        for seg in self._geometry.road_segments:
            seg_length = len(seg.centerline)
            seg_start = cumulative_offset
            seg_end = cumulative_offset + seg_length
            if seg_start < end_idx and seg_end > idx:
                for seg_type in seg.segment_types:
                    if not road_segment_types or road_segment_types[-1] != seg_type:
                        road_segment_types.append(seg_type)
            cumulative_offset += seg_length

        # 提取数据
        all_left = self._geometry.all_left_boundary
        all_right = self._geometry.all_right_boundary
        cl_pts = all_centerline[idx:end_idx]
        left_pts = all_left[idx:end_idx]
        right_pts = all_right[idx:end_idx]
        divider_pts = self._slice_lane_dividers(idx, end_idx)

        # 如果需要填充
        if pad and actual < num_points:
            padding_len = num_points - actual
            if n_total > 0:
                # 用最后一个点填充
                cl_padding = np.tile(all_centerline[-1:], (padding_len, 1))
                left_padding = np.tile(all_left[-1:], (padding_len, 1))
                right_padding = np.tile(all_right[-1:], (padding_len, 1))
                
                cl_pts = np.concatenate([cl_pts, cl_padding], axis=0)
                left_pts = np.concatenate([left_pts, left_padding], axis=0)
                right_pts = np.concatenate([right_pts, right_padding], axis=0)
                
                # 填充车道分隔线
                divider_pts = self._slice_lane_dividers(idx, end_idx, pad_to=num_points)

        return cl_pts, left_pts, right_pts, divider_pts, idx, actual, segment_length_m, road_segment_types

def plot_road(ax, geo, title="", show_dividers=True):
    """在 ax 上绘制道路几何。"""
    first_cl = True
    first_lb = True
    first_rb = True
    first_div = True
    for i, seg in enumerate(geo.road_segments):
        ax.plot(seg.centerline[:, 0], seg.centerline[:, 1], "y--", lw=1, label="中心线" if first_cl else None)
        first_cl = False
        ax.plot(seg.left_boundary[:, 0], seg.left_boundary[:, 1], "w-", lw=2, label="左边界" if first_lb else None)
        first_lb = False
        ax.plot(seg.right_boundary[:, 0], seg.right_boundary[:, 1], "w-", lw=2, label="右边界" if first_rb else None)
        first_rb = False
        if show_dividers:
            for div in seg.lane_dividers:
                ax.plot(div[:, 0], div[:, 1], "w:", lw=1, label="车道线" if first_div else None)
                first_div = False

        pt = seg.centerline[0]
        ax.plot(pt[0], pt[1], "ro", ms=6)
        ax.annotate(f"S{i}", (pt[0], pt[1]), fontsize=8,
                    color="red", textcoords="offset points", xytext=(5, 5))

    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_facecolor("#3a3a3a")
    ax.legend(fontsize=8, loc="upper left")