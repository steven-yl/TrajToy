"""车辆模型单元测试。"""

import numpy as np
import pytest

from sim_env.vehicle_model import (
    VehicleState,
    VehicleParams,
    VehicleModel,
    VehicleModelConfig,
    ModelType,
    IntegratorType,
    _wrap_angle,
)


# ── _wrap_angle ──────────────────────────────────────────────────────

class TestWrapAngle:
    """角度归一化函数测试。"""

    def test_zero(self):
        assert _wrap_angle(0.0) == pytest.approx(0.0)

    def test_pi(self):
        assert _wrap_angle(np.pi) == pytest.approx(-np.pi, abs=1e-10)

    def test_negative_pi(self):
        assert _wrap_angle(-np.pi) == pytest.approx(-np.pi, abs=1e-10)

    def test_two_pi(self):
        assert _wrap_angle(2 * np.pi) == pytest.approx(0.0, abs=1e-10)

    def test_large_positive(self):
        result = _wrap_angle(3 * np.pi)
        assert -np.pi <= result <= np.pi

    def test_large_negative(self):
        result = _wrap_angle(-5 * np.pi)
        assert -np.pi <= result <= np.pi

    def test_small_positive(self):
        assert _wrap_angle(0.5) == pytest.approx(0.5)

    def test_small_negative(self):
        assert _wrap_angle(-0.5) == pytest.approx(-0.5)


# ── VehicleState ─────────────────────────────────────────────────────

class TestVehicleState:
    """车辆状态数据类测试。"""

    def test_creation(self):
        state = VehicleState(x=1.0, y=2.0, theta=0.5, v=10.0)
        assert state.x == 1.0
        assert state.y == 2.0
        assert state.theta == 0.5
        assert state.v == 10.0
        assert state.steering == 0.0

    def test_frozen(self):
        state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
        with pytest.raises(AttributeError):
            state.x = 1.0

    def test_custom_steering(self):
        state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0, steering=0.3)
        assert state.steering == 0.3


# ── VehicleParams ────────────────────────────────────────────────────

class TestVehicleParams:
    """车辆参数配置测试。"""

    def test_defaults(self):
        params = VehicleParams()
        assert params.a_max == 5.0
        assert params.omega_max == 0.5
        assert params.v_max == 30.0
        assert params.wheelbase == 2.5
        assert params.length == 4.5
        assert params.width == 1.8
        assert params.mass == 1500.0

    def test_custom(self):
        params = VehicleParams(a_max=3.0, v_max=20.0, wheelbase=3.0)
        assert params.a_max == 3.0
        assert params.v_max == 20.0
        assert params.wheelbase == 3.0


# ── VehicleModel ─────────────────────────────────────────────────────

class TestVehicleModel:
    """车辆模型测试。"""

    def test_default_init(self):
        model = VehicleModel()
        assert model.model_type == ModelType.KINEMATIC
        assert model.integrator == IntegratorType.EULER
        state = model.state
        assert state.x == 0.0
        assert state.v == 0.0

    def test_reset(self):
        model = VehicleModel()
        state = model.reset(x=10.0, y=20.0, theta=1.0, v=5.0)
        assert state.x == 10.0
        assert state.y == 20.0
        assert state.theta == pytest.approx(1.0)
        assert state.v == 5.0
        assert state.steering == 0.0

    def test_reset_wraps_angle(self):
        model = VehicleModel()
        state = model.reset(x=0.0, y=0.0, theta=4 * np.pi, v=0.0)
        assert -np.pi <= state.theta <= np.pi

    def test_clip_action(self):
        params = VehicleParams(a_max=3.0, omega_max=0.4)
        model = VehicleModel(cfg=VehicleModelConfig(vehicle_params=params))
        clipped = model.clip_action(np.array([10.0, -1.0]))
        assert clipped[0] == pytest.approx(3.0)
        assert clipped[1] == pytest.approx(-0.4)

    def test_clip_action_within_range(self):
        model = VehicleModel()
        action = np.array([1.0, 0.2])
        clipped = model.clip_action(action)
        np.testing.assert_allclose(clipped, action)

    def test_step_invalid_dt(self):
        model = VehicleModel()
        model.reset(0, 0, 0, 5.0)
        with pytest.raises(ValueError, match="dt"):
            model.step(np.array([0.0, 0.0]), dt=0.0)
        with pytest.raises(ValueError, match="dt"):
            model.step(np.array([0.0, 0.0]), dt=-0.1)

    def test_step_invalid_action_shape(self):
        model = VehicleModel()
        model.reset(0, 0, 0, 5.0)
        with pytest.raises(ValueError, match="动作形状"):
            model.step(np.array([1.0, 2.0, 3.0]), dt=0.1)

    def test_step_straight_euler(self):
        """直行：零转向，匀速前进。"""
        model = VehicleModel(
            cfg=VehicleModelConfig(
                model_type=ModelType.KINEMATIC,
                integrator_type=IntegratorType.EULER,
            ),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=10.0)
        state = model.step(np.array([0.0, 0.0]), dt=1.0)
        assert state.x == pytest.approx(10.0, abs=0.01)
        assert state.y == pytest.approx(0.0, abs=0.01)
        assert state.v == pytest.approx(10.0, abs=0.01)

    def test_step_acceleration_euler(self):
        """加速：速度应增加。"""
        model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.EULER),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=0.0)
        state = model.step(np.array([2.0, 0.0]), dt=1.0)
        assert state.v == pytest.approx(2.0, abs=0.01)

    def test_step_speed_clamped(self):
        """速度不应超过 v_max。"""
        params = VehicleParams(v_max=10.0)
        model = VehicleModel(
            cfg=VehicleModelConfig(
                vehicle_params=params,
                integrator_type=IntegratorType.EULER,
            ),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=9.0)
        state = model.step(np.array([5.0, 0.0]), dt=1.0)
        assert state.v <= params.v_max + 1e-6

    def test_step_speed_non_negative(self):
        """速度不应为负。"""
        model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.EULER),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=1.0)
        state = model.step(np.array([-5.0, 0.0]), dt=1.0)
        assert state.v >= 0.0

    def test_step_rk4_straight(self):
        """RK4 积分直行。"""
        model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.RK4),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=10.0)
        state = model.step(np.array([0.0, 0.0]), dt=1.0)
        assert state.x == pytest.approx(10.0, abs=0.1)
        assert state.y == pytest.approx(0.0, abs=0.1)

    def test_step_turning(self):
        """转向：朝向角应变化。"""
        model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.EULER),
        )
        model.reset(x=0.0, y=0.0, theta=0.0, v=5.0)
        # 第一步：施加转向速度让 steering 变为非零
        model.step(np.array([0.0, 0.3]), dt=1.0)
        assert model.state.steering != 0.0
        # 第二步：此时 steering 非零，theta 应发生变化
        model.step(np.array([0.0, 0.0]), dt=1.0)
        assert model.state.theta != 0.0

    def test_dynamic_model_drag(self):
        """动力学模型：空气阻力应减缓加速。"""
        kin_model = VehicleModel(
            cfg=VehicleModelConfig(model_type=ModelType.KINEMATIC),
        )
        dyn_model = VehicleModel(
            cfg=VehicleModelConfig(model_type=ModelType.DYNAMIC),
        )

        kin_model.reset(0, 0, 0, 20.0)
        dyn_model.reset(0, 0, 0, 20.0)

        kin_state = kin_model.step(np.array([2.0, 0.0]), dt=0.1)
        dyn_state = dyn_model.step(np.array([2.0, 0.0]), dt=0.1)

        # 动力学模型有阻力，速度增量应更小
        assert dyn_state.v < kin_state.v

    def test_euler_vs_rk4_consistency(self):
        """欧拉和 RK4 在小步长下结果应接近。"""
        euler_model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.EULER),
        )
        rk4_model = VehicleModel(
            cfg=VehicleModelConfig(integrator_type=IntegratorType.RK4),
        )

        euler_model.reset(0, 0, 0, 10.0)
        rk4_model.reset(0, 0, 0, 10.0)

        small_dt = 0.01
        action = np.array([1.0, 0.1])
        euler_state = euler_model.step(action, small_dt)
        rk4_state = rk4_model.step(action, small_dt)

        assert euler_state.x == pytest.approx(rk4_state.x, abs=0.01)
        assert euler_state.y == pytest.approx(rk4_state.y, abs=0.01)
        assert euler_state.v == pytest.approx(rk4_state.v, abs=0.01)

    def test_multiple_steps(self):
        """多步仿真不应崩溃。"""
        model = VehicleModel()
        model.reset(0, 0, 0, 5.0)
        for _ in range(100):
            state = model.step(np.array([0.5, 0.1]), dt=0.1)
        assert np.isfinite(state.x)
        assert np.isfinite(state.y)
        assert np.isfinite(state.v)
