"""车辆模型：支持运动学/动力学模型，欧拉/RK4积分。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from omegaconf import DictConfig, OmegaConf

# ── 数据结构 ──────────────────────────────────────────────────────────


@dataclass
class VehicleState:
    """车辆状态（不可变）。"""

    x: float  # 位置 x (m)
    y: float  # 位置 y (m)
    theta: float  # 朝向角 (rad), [-π, π]
    v: float  # 纵向速度 (m/s)
    steering: float = 0.0  # 前轮转向角 (rad)


@dataclass
class VehicleParams:
    """车辆参数配置。"""

    a_max: float = 5.0  # 最大加速度 (m/s²)
    omega_max: float = 0.5  # 最大转向速度 (rad/s)
    v_max: float = 30.0  # 最大速度 (m/s)
    wheelbase: float = 2.5  # 轴距 (m)
    length: float = 4.5  # 车身长度 (m)
    width: float = 1.8  # 车身宽度 (m)
    height: float = 1.5  # 车身高度 (m)
    front_overhang: float = 0.8  # 前悬 (m)
    rear_overhang: float = 1.2  # 后悬 (m)
    mass: float = 1500.0  # 质量 (kg)，动力学模型使用
    drag_coeff: float = 0.3  # 空气阻力系数，动力学模型使用


class ModelType(Enum):
    """车辆模型类型。"""

    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"


class IntegratorType(Enum):
    """积分器类型。"""

    EULER = "euler"
    RK4 = "rk4"


# ── 工具函数 ─────────────────────────────────────────────────────────


def _wrap_angle(theta: float) -> float:
    """将角度归一化到 [-π, π]。"""
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


# ── 车辆模型 ─────────────────────────────────────────────────────────
@dataclass
class VehicleModelConfig:
    # 省略该字段时用 default_factory 得到默认参数；显式传 None 表示「尚未指定」，由 VehicleModel 再补默认
    vehicle_params: VehicleParams | None = field(default_factory=VehicleParams)
    model_type: ModelType = ModelType.KINEMATIC
    integrator_type: IntegratorType = IntegratorType.EULER


class VehicleModel:
    """车辆模型，支持运动学/动力学 × 欧拉/RK4 组合。

    动作空间: [acceleration, steering_rate]
    - acceleration: 纵向加速度 (m/s²)
    - steering_rate: 转向角速度 (rad/s)，积分得到前轮转向角 δ

    运动学自行车模型 (无阻力):
        dδ/dt    = ω  (转向角速度)
        dx/dt    = v * cos(θ)
        dy/dt    = v * sin(θ)
        dθ/dt    = v * tan(δ) / L  (L 为轴距)
        dv/dt    = a

    动力学自行车模型 (含空气阻力):
        dδ/dt    = ω
        dx/dt    = v * cos(θ)
        dy/dt    = v * sin(θ)
        dθ/dt    = v * tan(δ) / L
        dv/dt    = a - C_d * v² / m
    """

    @staticmethod
    def bulid_from_config(cfg: DictConfig) -> VehicleModel:
        return VehicleModel(cfg)
 
    def __init__(
        self,
        cfg: VehicleModelConfig | None = None,
    ) -> None:
        base = cfg or VehicleModelConfig()
        if base.vehicle_params is None:
            base = replace(base, vehicle_params=VehicleParams())
        self._config = base
        self._state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0, steering=0.0)

    # ── 属性 ──────────────────────────────────────────────────────

    @property
    def state(self) -> VehicleState:
        """返回当前车辆状态（不可变副本）。"""
        return self._state

    @property
    def vehicle_params(self) -> VehicleParams:
        p = self._config.vehicle_params
        assert p is not None
        return p

    @property
    def params(self) -> VehicleParams:
        """与 :attr:`vehicle_params` 同义，兼容旧代码中的 ``model.params`` 写法。"""
        return self.vehicle_params

    @property
    def model_type(self) -> ModelType:
        return self._config.model_type

    @property
    def integrator(self) -> IntegratorType:
        return self._config.integrator_type

    # ── 公共 API ──────────────────────────────────────────────────

    def reset(
        self, x: float, y: float, theta: float, v: float = 0.0
    ) -> VehicleState:
        """重置车辆状态。"""
        self._state = VehicleState(x=x, y=y, theta=_wrap_angle(theta), v=v, steering=0.0)
        return self._state

    def step(self, action: np.ndarray, dt: float) -> VehicleState:
        """执行一步仿真。

        Args:
            action: [acceleration, steering_rate] 形状 (2,)
            dt: 时间步长 (s)，必须 > 0

        Returns:
            更新后的车辆状态
        """
        if dt <= 0:
            raise ValueError(f"dt 必须 > 0，实际为 {dt}")

        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError(f"动作形状应为 (2,)，实际为 {action.shape}")

        clipped = self.clip_action(action)
        a_cmd = float(clipped[0])
        omega_cmd = float(clipped[1])

        if self._config.integrator_type == IntegratorType.EULER:
            self._step_euler(a_cmd, omega_cmd, dt)
        else:
            self._step_rk4(a_cmd, omega_cmd, dt)

        return self._state

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        """将动作裁剪到允许范围。"""
        action = np.asarray(action, dtype=np.float64)
        p = self._config.vehicle_params
        return np.array(
            [
                np.clip(action[0], -p.a_max, p.a_max),
                np.clip(action[1], -p.omega_max, p.omega_max),
            ],
            dtype=np.float64,
        )

    # ── 内部：状态导数 ────────────────────────────────────────────

    def _derivatives(
        self, x: float, y: float, theta: float, v: float,
        steering: float, a_cmd: float, omega_cmd: float,
    ) -> tuple[float, float, float, float, float]:
        """计算状态导数，返回 (dx, dy, dtheta, dv, dsteering)。"""
        p = self._config.vehicle_params

        # 两种模型共用自行车运动学：dθ/dt = v * tan(δ) / L
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = v * np.tan(steering) / p.wheelbase if p.wheelbase > 0 else 0.0
        dsteering = omega_cmd

        if self._config.model_type == ModelType.KINEMATIC:
            dv = a_cmd
        else:
            # 动力学：含空气阻力
            dv = a_cmd - p.drag_coeff * v * abs(v) / p.mass

        return dx, dy, dtheta, dv, dsteering

    # ── 内部：欧拉积分 ───────────────────────────────────────────

    def _step_euler(self, a_cmd: float, omega_cmd: float, dt: float) -> None:
        s = self._state
        p = self._config.vehicle_params

        dx, dy, dtheta, dv, ds = self._derivatives(
            s.x, s.y, s.theta, s.v, s.steering, a_cmd, omega_cmd,
        )

        v_new = float(np.clip(s.v + dv * dt, 0.0, p.v_max))
        theta_new = _wrap_angle(s.theta + dtheta * dt)
        x_new = s.x + dx * dt
        y_new = s.y + dy * dt
        steering_new = s.steering + ds * dt

        self._state = VehicleState(x=x_new, y=y_new, theta=theta_new, v=v_new, steering=steering_new)

    # ── 内部：RK4 积分 ───────────────────────────────────────────

    def _step_rk4(self, a_cmd: float, omega_cmd: float, dt: float) -> None:
        s = self._state
        p = self._config.vehicle_params
        steer = s.steering

        def derivs(x: float, y: float, th: float, v: float, st: float):
            return self._derivatives(x, y, th, v, st, a_cmd, omega_cmd)

        k1 = derivs(s.x, s.y, s.theta, s.v, steer)
        k2 = derivs(
            s.x + k1[0] * dt / 2, s.y + k1[1] * dt / 2,
            s.theta + k1[2] * dt / 2, s.v + k1[3] * dt / 2,
            steer + k1[4] * dt / 2,
        )
        k3 = derivs(
            s.x + k2[0] * dt / 2, s.y + k2[1] * dt / 2,
            s.theta + k2[2] * dt / 2, s.v + k2[3] * dt / 2,
            steer + k2[4] * dt / 2,
        )
        k4 = derivs(
            s.x + k3[0] * dt, s.y + k3[1] * dt,
            s.theta + k3[2] * dt, s.v + k3[3] * dt,
            steer + k3[4] * dt,
        )

        # 加权平均
        def rk4_combine(y0: float, k1v: float, k2v: float, k3v: float, k4v: float):
            return y0 + (k1v + 2 * k2v + 2 * k3v + k4v) * dt / 6

        x_new = rk4_combine(s.x, k1[0], k2[0], k3[0], k4[0])
        y_new = rk4_combine(s.y, k1[1], k2[1], k3[1], k4[1])
        theta_new = _wrap_angle(rk4_combine(s.theta, k1[2], k2[2], k3[2], k4[2]))
        v_new = float(np.clip(
            rk4_combine(s.v, k1[3], k2[3], k3[3], k4[3]), 0.0, p.v_max
        ))
        steering_new = rk4_combine(steer, k1[4], k2[4], k3[4], k4[4])

        self._state = VehicleState(x=x_new, y=y_new, theta=theta_new, v=v_new, steering=steering_new)
