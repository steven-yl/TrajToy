"""道路模型单元测试。"""

import numpy as np
import pytest

from sim_env.road_model import (
    RoadSegmentType,
    SegmentSpec,
    RoadGenerationConfig,
    RoadModel,
    _generate_straight,
    _generate_curve,
)
from sim_env.exceptions import RoadConfigError


# ── 片段几何生成函数 ─────────────────────────────────────────────────

class TestGenerateStraight:
    """直道生成测试。"""

    def test_basic(self):
        start = np.array([0.0, 0.0])
        points, end_theta = _generate_straight(start, 0.0, 10.0, 1.0)
        assert points.shape[1] == 2
        assert points.shape[0] >= 2
        assert end_theta == pytest.approx(0.0)
        # 终点应在 (10, 0) 附近
        np.testing.assert_allclose(points[-1], [10.0, 0.0], atol=0.5)

    def test_angled(self):
        start = np.array([0.0, 0.0])
        theta = np.pi / 4
        points, end_theta = _generate_straight(start, theta, 10.0, 1.0)
        assert end_theta == pytest.approx(theta)
        expected_end = np.array([10.0 * np.cos(theta), 10.0 * np.sin(theta)])
        np.testing.assert_allclose(points[-1], expected_end, atol=0.5)

    def test_density(self):
        start = np.array([0.0, 0.0])
        points_low, _ = _generate_straight(start, 0.0, 10.0, 0.5)
        points_high, _ = _generate_straight(start, 0.0, 10.0, 2.0)
        assert points_high.shape[0] > points_low.shape[0]


class TestGenerateCurve:
    """弯道生成测试。"""

    def test_left_turn(self):
        start = np.array([0.0, 0.0])
        points, end_theta = _generate_curve(start, 0.0, 10.0, 90.0, 1.0)
        assert points.shape[1] == 2
        assert points.shape[0] >= 2
        # 左转 90 度，end_theta 应约为 π/2
        assert end_theta == pytest.approx(np.pi / 2, abs=0.1)

    def test_right_turn(self):
        start = np.array([0.0, 0.0])
        points, end_theta = _generate_curve(start, 0.0, 10.0, -90.0, 1.0)
        assert end_theta == pytest.approx(-np.pi / 2, abs=0.1)

    def test_zero_angle(self):
        """零角度弯道退化为点。"""
        start = np.array([5.0, 5.0])
        points, end_theta = _generate_curve(start, 0.0, 10.0, 0.0, 1.0)
        assert points.shape[0] >= 1


# ── RoadModel ────────────────────────────────────────────────────────

class TestRoadModel:
    """道路模型测试。"""

    @staticmethod
    def _assert_geometry_cache_consistent(geo):
        """断言 RoadGeometry 缓存字段与分段拼接结果一致。"""
        expected_centerline = np.concatenate([seg.centerline for seg in geo.road_segments], axis=0)
        expected_left = np.concatenate([seg.left_boundary for seg in geo.road_segments], axis=0)
        expected_right = np.concatenate([seg.right_boundary for seg in geo.road_segments], axis=0)
        np.testing.assert_allclose(geo.all_centerline, expected_centerline)
        np.testing.assert_allclose(geo.all_left_boundary, expected_left)
        np.testing.assert_allclose(geo.all_right_boundary, expected_right)

    def test_default_config(self):
        model = RoadModel()
        cfg = model.config
        assert cfg.num_random_segments == 3
        assert cfg.lane_width == 3.5

    def test_custom_config(self):
        cfg = RoadGenerationConfig(num_random_segments=5, lane_width=4.0, num_lanes=2)
        model = RoadModel(cfg)
        assert model.config.num_random_segments == 5
        assert model.config.num_lanes == 2

    def test_generate_fixed_straight(self):
        model = RoadModel()
        segments = [
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ]
        geo = model.generate_fixed(segments)
        assert geo is not None
        assert len(geo.road_segments) == 1
        assert geo.total_length > 0
        self._assert_geometry_cache_consistent(geo)

    def test_generate_fixed_mixed(self):
        model = RoadModel()
        segments = [
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 30.0}),
            SegmentSpec(RoadSegmentType.CURVE, {"radius": 20.0, "angle_deg": 45.0}),
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 30.0}),
        ]
        geo = model.generate_fixed(segments)
        assert len(geo.road_segments) == 3
        assert geo.total_length > 0
        self._assert_geometry_cache_consistent(geo)

    def test_generate_random_deterministic(self):
        """相同 seed 应产生相同道路。"""
        model1 = RoadModel()
        model2 = RoadModel()
        geo1 = model1.generate_random(seed=42)
        geo2 = model2.generate_random(seed=42)
        assert len(geo1.road_segments) == len(geo2.road_segments)
        np.testing.assert_allclose(geo1.all_centerline, geo2.all_centerline)
        np.testing.assert_allclose(geo1.all_left_boundary, geo2.all_left_boundary)
        np.testing.assert_allclose(geo1.all_right_boundary, geo2.all_right_boundary)
        for seg1, seg2 in zip(geo1.road_segments, geo2.road_segments):
            np.testing.assert_allclose(seg1.centerline, seg2.centerline)

    def test_generate_random_different_seeds(self):
        """不同 seed 应产生不同道路。"""
        model = RoadModel()
        geo1 = model.generate_random(seed=1)
        geo2 = model.generate_random(seed=999)
        # 至少有一个片段的中心线形状或数值不同
        all_same = True
        for seg1, seg2 in zip(geo1.road_segments, geo2.road_segments):
            if seg1.centerline.shape != seg2.centerline.shape:
                all_same = False
                break
            if not np.allclose(seg1.centerline, seg2.centerline):
                all_same = False
                break
        assert not all_same

    def test_geometry_property(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ])
        geo = model.geometry
        assert geo is not None

    def test_get_nearest_idx(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
        ])
        idx, seg_idx = model.get_nearest_idx(50.0, 0.0)
        assert idx > 0
        assert seg_idx == 0

    def test_get_nearest_centerline_info(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
        ])
        lateral, heading, progress = model.get_nearest_centerline_info(50.0, 1.0)
        # 在直道上方 1m，横向偏移应约为 1.0
        assert lateral == pytest.approx(1.0, abs=0.5)
        # 直道朝向应约为 0
        assert heading == pytest.approx(0.0, abs=0.1)
        # 进度应在 0~1 之间
        assert 0.0 < progress < 1.0

    def test_get_road_segment_ahead(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 100.0}),
        ])
        result = model.get_road_segment_ahead(0.0, 0.0, num_points=20, pad=True)
        centerline, left, right, dividers, nearest_idx, actual, seg_len, seg_types = result
        assert centerline.shape == (20, 2)
        assert left.shape == (20, 2)
        assert right.shape == (20, 2)
        assert actual <= 20
        assert seg_len > 0

    def test_get_road_segment_ahead_no_pad(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 10.0}),
        ])
        result = model.get_road_segment_ahead(0.0, 0.0, num_points=1000, pad=False)
        centerline = result[0]
        actual = result[5]
        assert centerline.shape[0] == actual

    def test_extend_road(self):
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ])
        original_length = model.geometry.total_length
        model.extend_road([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ])
        assert model.geometry.total_length > original_length
        self._assert_geometry_cache_consistent(model.geometry)

    def test_extend_road_random(self):
        model = RoadModel()
        model.generate_random(seed=42)
        original_segments = len(model.geometry.road_segments)
        model.extend_road()
        assert len(model.geometry.road_segments) > original_segments
        self._assert_geometry_cache_consistent(model.geometry)

    def test_road_not_generated_error(self):
        model = RoadModel()
        with pytest.raises(RoadConfigError):
            model.get_nearest_idx(0.0, 0.0)

    def test_multi_lane_dividers(self):
        """双车道应有车道分隔线。"""
        cfg = RoadGenerationConfig(num_lanes=2)
        model = RoadModel(cfg)
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ])
        seg = model.geometry.road_segments[0]
        assert len(seg.lane_dividers) >= 1

    def test_boundaries_wider_than_centerline(self):
        """左右边界应在中心线两侧。"""
        model = RoadModel()
        model.generate_fixed([
            SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50.0}),
        ])
        seg = model.geometry.road_segments[0]
        # 直道沿 x 轴，左边界 y > 中心线 y > 右边界 y
        mid_idx = len(seg.centerline) // 2
        assert seg.left_boundary[mid_idx, 1] > seg.centerline[mid_idx, 1]
        assert seg.right_boundary[mid_idx, 1] < seg.centerline[mid_idx, 1]
