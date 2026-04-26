"""车辆 MPC 控制器：基于 CasADi 的模型预测控制。

使用运动学自行车模型，跟踪给定的参考路径点。
状态: [x, y, theta, v, steering]
控制: [acceleration, steering_rate]
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import casadi as ca
import numpy as np

from .vehicle_model import VehicleParams
from omegaconf import DictConfig, OmegaConf

@dataclass
class MPCConfig:
    """MPC 控制器配置。"""

    # 车辆参数（用于动力学约束和状态边界）
    vehicle_params: VehicleParams | None = field(default_factory=VehicleParams)

    # 预测
    horizon: int = 20       # 预测步数 N
    dt: float = 0.1         # 预测时间步长 (s)

    # 目标速度
    target_speed: float = 5.0  # m/s

    # 代价权重
    w_pos: float = 10.0     # 位置跟踪权重
    w_heading: float = 5.0  # 朝向跟踪权重
    w_speed: float = 2.0    # 速度跟踪权重
    w_accel: float = 0.1    # 加速度惩罚
    w_omega: float = 0.1    # 转向速度惩罚
    w_d_accel: float = 1.0  # 加速度变化率惩罚
    w_d_omega: float = 1.0  # 转向速度变化率惩罚

class VehicleMPC:
    """基于 CasADi 的车辆 MPC 控制器。

    跟踪参考路径（中心线点序列），输出 [acceleration, steering_rate]。
    内部使用运动学自行车模型做预测。
    """
    @staticmethod
    def bulid_from_config(cfg: DictConfig) -> VehicleMPC:
        """根据配置构建 MPC 控制器，保留 structured config 的默认值行为。"""
        ec = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        mpc_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(MPCConfig()), ec),
        )
        if not isinstance(mpc_cfg, MPCConfig):
            raise TypeError("Hydra controller 配置无法转换为 MPCConfig")
        return VehicleMPC(mpc_cfg)

    def __init__(
        self,
        mpc_config: MPCConfig | None = None,
    ) -> None:
        base = mpc_config or MPCConfig()
        if base.vehicle_params is None:
            base = replace(base, vehicle_params=VehicleParams())
        self._cfg = base

        self._solver = None
        self._prev_u = np.zeros(2)
        self._build_solver()

    @property
    def config(self) -> MPCConfig:
        return self._cfg

    def _build_solver(self) -> None:
        """构建 CasADi NLP 求解器。"""
        cfg = self._cfg
        vp = self._cfg.vehicle_params
        N = cfg.horizon
        dt = cfg.dt
        nx, nu = 5, 2  # 状态维度, 控制维度

        # ── 符号变量 ──
        # 决策变量: 所有状态 + 所有控制
        X = ca.SX.sym("X", nx, N + 1)  # 状态轨迹
        U = ca.SX.sym("U", nu, N)  # 控制轨迹

        # 参数: 初始状态 + 参考路径点 + 参考朝向 + 上一步控制 + 目标速度
        # P = [x0(5), ref_x(N+1), ref_y(N+1), ref_theta(N+1), u_prev(2), v_target(1)]
        n_params = nx + 3 * (N + 1) + nu + 1
        P = ca.SX.sym("P", n_params)

        # 解包参数
        x0 = P[:nx]
        ref_x = P[nx: nx + N + 1]
        ref_y = P[nx + N + 1: nx + 2 * (N + 1)]
        ref_theta = P[nx + 2 * (N + 1): nx + 3 * (N + 1)]
        u_prev = P[nx + 3 * (N + 1): nx + 3 * (N + 1) + nu]
        v_target = P[-1]

        # ── 代价函数和约束 ──
        cost = 0
        g = []  # 等式约束 (动力学)
        lbg = []
        ubg = []

        # 初始状态约束
        g.append(X[:, 0] - x0)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            x_pos, y_pos, theta, v, steer = xk[0], xk[1], xk[2], xk[3], xk[4]
            a_cmd, omega_cmd = uk[0], uk[1]

            # 跟踪代价
            cost += cfg.w_pos * ((x_pos - ref_x[k]) ** 2 + (y_pos - ref_y[k]) ** 2)
            cost += cfg.w_heading * (theta - ref_theta[k]) ** 2
            cost += cfg.w_speed * (v - v_target) ** 2

            # 控制代价
            cost += cfg.w_accel * a_cmd ** 2
            cost += cfg.w_omega * omega_cmd ** 2

            # 控制变化率代价
            if k == 0:
                cost += cfg.w_d_accel * (a_cmd - u_prev[0]) ** 2
                cost += cfg.w_d_omega * (omega_cmd - u_prev[1]) ** 2
            else:
                cost += cfg.w_d_accel * (a_cmd - U[0, k - 1]) ** 2
                cost += cfg.w_d_omega * (omega_cmd - U[1, k - 1]) ** 2

            # 运动学自行车模型 (欧拉积分)
            x_next = x_pos + v * ca.cos(theta) * dt
            y_next = y_pos + v * ca.sin(theta) * dt
            theta_next = theta + v * ca.tan(steer) / vp.wheelbase * dt
            v_next = v + a_cmd * dt
            steer_next = steer + omega_cmd * dt

            xk_next = ca.vertcat(x_next, y_next, theta_next, v_next, steer_next)

            # 动力学等式约束
            g.append(X[:, k + 1] - xk_next)
            lbg += [0.0] * nx
            ubg += [0.0] * nx

        # 终端代价
        cost += cfg.w_pos * 2 * (
            (X[0, N] - ref_x[N]) ** 2 + (X[1, N] - ref_y[N]) ** 2
        )
        cost += cfg.w_heading * 2 * (X[2, N] - ref_theta[N]) ** 2

        # ── 决策变量边界 ──
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        # 状态边界
        lbx = []
        ubx = []
        for _ in range(N + 1):
            lbx += [-1e6, -1e6, -2 * np.pi, 0.0, -np.pi / 3]
            ubx += [1e6, 1e6, 2 * np.pi, vp.v_max, np.pi / 3]
        # 控制边界
        for _ in range(N):
            lbx += [-vp.a_max, -vp.omega_max]
            ubx += [vp.a_max, vp.omega_max]

        # ── 构建求解器 ──
        nlp = {"f": cost, "x": opt_vars, "g": g, "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
        self._solver = ca.nlpsol("mpc", "ipopt", nlp, opts)
        self._nx = nx
        self._nu = nu
        self._n_params = n_params
        self._lbx = lbx
        self._ubx = ubx
        self._lbg = lbg
        self._ubg = ubg
        self._warm_x0 = None

    def compute(
        self,
        state: np.ndarray,
        ref_path: np.ndarray,
        target_speed: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 MPC 控制量。

        Args:
            state: 当前状态 [x, y, theta, v, steering]，形状 (5,)
            ref_path: 参考路径点 (N+1, 2)，每行 [x, y]。
                      如果点数不足 N+1，用最后一个点填充。
            target_speed: 目标速度 (m/s)，None 则使用 MPCConfig.target_speed

        Returns:
            (action, predicted_trajectory, ref_path)
            action: [acceleration, steering_rate]，形状 (2,)
            predicted_trajectory: 预测轨迹 (N+1, 2)
            ref_path: 重采样后的参考路径 (N+1, 2)
        """
        N = self._cfg.horizon
        nx, nu = self._nx, self._nu

        state = np.asarray(state, dtype=np.float64).flatten()
        ref_path = np.asarray(ref_path, dtype=np.float64)
        v_target = target_speed if target_speed is not None else self._cfg.target_speed

        # 按目标速度重采样参考路径：每步期望前进 v_target * dt
        ref_path = self._resample_ref_path(ref_path, v_target, N + 1, self._cfg.dt)

        # 计算参考朝向（相邻点的方向）
        ref_theta = np.zeros(N + 1)
        for i in range(N):
            dx = ref_path[i + 1, 0] - ref_path[i, 0]
            dy = ref_path[i + 1, 1] - ref_path[i, 1]
            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                ref_theta[i] = np.arctan2(dy, dx)
            elif i > 0:
                ref_theta[i] = ref_theta[i - 1]  # 继承前一个朝向
        ref_theta[N] = ref_theta[N - 1]

        # 组装参数向量
        p = np.concatenate([
            state,
            ref_path[:, 0],  # ref_x
            ref_path[:, 1],  # ref_y
            ref_theta,
            self._prev_u,
            [v_target],
        ])

        # 初始猜测
        if self._warm_x0 is not None:
            x0_guess = self._warm_x0
        else:
            # 用当前状态铺满 + 零控制
            x_init = np.tile(state, N + 1)
            u_init = np.zeros(nu * N)
            x0_guess = np.concatenate([x_init, u_init])

        # 求解
        sol = self._solver(
            x0=x0_guess,
            p=p,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=self._lbg,
            ubg=self._ubg,
        )

        opt = np.array(sol["x"]).flatten()

        # 解包
        X_opt = opt[: nx * (N + 1)].reshape(N + 1, nx)
        U_opt = opt[nx * (N + 1):].reshape(N, nu)

        # 第一步控制量
        action = U_opt[0]
        self._prev_u = action.copy()

        # 暖启动：把解向前移一步
        warm_x = np.concatenate([X_opt[1:].flatten(), X_opt[-1:].flatten()])
        warm_u = np.concatenate([U_opt[1:].flatten(), U_opt[-1:].flatten()])
        self._warm_x0 = np.concatenate([warm_x, warm_u])

        predicted_xy = X_opt[:, :2]
        return action, predicted_xy, ref_path

    def reset(self) -> None:
        """重置控制器状态（暖启动缓存和上一步控制量）。"""
        self._prev_u = np.zeros(2)
        self._warm_x0 = None

    @staticmethod
    def _resample_ref_path(
        ref_path: np.ndarray, v_target: float, n_points: int, dt: float | None = None,
    ) -> np.ndarray:
        """按目标速度重采样参考路径。

        将空间等距的参考点转换为时间等距：每步期望前进 v_target * dt 米。
        这样 MPC 第 k 步的参考点就是 "k*dt 秒后车辆应该在的位置"。
        """
        if dt is None:
            dt = 0.1  # fallback

        # 计算参考路径的累计弧长
        diffs = np.diff(ref_path, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        # 每步期望前进的距离
        step_dist = max(v_target * dt, 0.01)
        target_dists = np.array([i * step_dist for i in range(n_points)])

        # 在累计弧长上插值
        resampled = np.zeros((n_points, 2))
        for i, d in enumerate(target_dists):
            if d >= cum_length[-1]:
                resampled[i] = ref_path[-1]
            else:
                idx = np.searchsorted(cum_length, d, side="right") - 1
                idx = max(0, min(idx, len(ref_path) - 2))
                seg_len = cum_length[idx + 1] - cum_length[idx]
                if seg_len < 1e-12:
                    resampled[i] = ref_path[idx]
                else:
                    t = (d - cum_length[idx]) / seg_len
                    resampled[i] = ref_path[idx] * (1 - t) + ref_path[idx + 1] * t

        return resampled
