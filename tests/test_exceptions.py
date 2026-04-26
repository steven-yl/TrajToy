"""异常类单元测试。"""

import pytest

from sim_env.exceptions import (
    RoadVehicleError,
    ContinuityError,
    RoadConfigError,
    SerializationError,
)


class TestExceptionHierarchy:
    """异常继承关系测试。"""

    def test_continuity_error_is_road_vehicle_error(self):
        err = ContinuityError(0, "test")
        assert isinstance(err, RoadVehicleError)

    def test_road_config_error_is_road_vehicle_error(self):
        err = RoadConfigError("test")
        assert isinstance(err, RoadVehicleError)

    def test_serialization_error_is_road_vehicle_error(self):
        err = SerializationError("test")
        assert isinstance(err, RoadVehicleError)

    def test_all_are_exceptions(self):
        for exc_class in [RoadVehicleError, ContinuityError, RoadConfigError, SerializationError]:
            assert issubclass(exc_class, Exception)


class TestContinuityError:
    """连续性错误测试。"""

    def test_segment_index(self):
        err = ContinuityError(3, "gap too large")
        assert err.segment_index == 3

    def test_message_format(self):
        err = ContinuityError(2, "与前段间距 1.5m > 0.5m")
        assert "片段 2" in str(err)
        assert "连续性错误" in str(err)

    def test_raise_and_catch(self):
        with pytest.raises(ContinuityError) as exc_info:
            raise ContinuityError(1, "test error")
        assert exc_info.value.segment_index == 1


class TestRoadConfigError:
    """道路配置错误测试。"""

    def test_message(self):
        err = RoadConfigError("invalid config")
        assert "invalid config" in str(err)

    def test_raise_and_catch(self):
        with pytest.raises(RoadConfigError):
            raise RoadConfigError("bad config")


class TestSerializationError:
    """序列化错误测试。"""

    def test_message(self):
        err = SerializationError("cannot serialize")
        assert "cannot serialize" in str(err)

    def test_catch_as_base(self):
        with pytest.raises(RoadVehicleError):
            raise SerializationError("test")
