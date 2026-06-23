"""仿真 Gymnasium 环境。

观测空间（全部为绝对值）:
  车辆状态: [x, y, theta, v]
  道路状态: 前方 N 个点的中心线、左边界、右边界绝对坐标 (展平)

动作空间: [acceleration, steering_rate]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
from omegaconf import DictConfig, OmegaConf

import gymnasium as gym
from gymnasium import spaces

from .vehicle_model import VehicleModel, VehicleModelConfig
from .road_model import RoadGenerationConfig, RoadModel, RoadSegmentType
from .reward_model import RewardModel, RewardModelConfig, RewardWeights


def _wrap_angle(a: float) -> float:
    """将角度归一化到 [-pi, pi]。"""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# 用于 random_*_range：numpy.uniform 的闭区间 [lo, hi]
FloatRange = tuple[float, float]

@dataclass
class EnvRandomConfig:
    random_init_speed_range: FloatRange = (0.0, 10.0)
    random_init_lateral_offset_range: FloatRange = (-1.0, 1.0)
    random_init_heading_offset_range: FloatRange = (-0.2, 0.2)
    random_target_speed_range: FloatRange = (0.0, 10.0)

@dataclass
class EnvConfig:
    """道路-车辆 Gymnasium 环境的全局配置。

    分组：仿真步进 → 车辆模型（VehicleModelConfig）→ 初值（含随机化）→ 道路 → 观测
    → 奖励 → 出界/步数终止判据 → Pygame 渲染。单位：长度 m、时间 s、角度 rad。
    """

    # --- 仿真步进 ---
    dt: float = 0.1
    max_steps: int = 1000

    # --- 车辆模型与数值积分（与 :class:`VehicleModel` 共用）---
    vehicle_config: VehicleModelConfig = field(default_factory=VehicleModelConfig)

    random_init: bool = False
    random_config: EnvRandomConfig = field(default_factory=EnvRandomConfig)
    # --- 初值；random_init 为 False 时使用本组，为 True 时由 random_*_range 采样覆盖 ---
    init_speed: float = 5.0
    init_lateral_offset: float = 0.0  # m，相对首点法向，左正右负
    init_heading_offset: float = 0.0  # rad，相对首段道路切向

    # --- 道路（片段与是否循环延长见 :class:`RoadGenerationConfig`）---
    road_config: RoadGenerationConfig | None = None

    # --- 观测：前方弧长采样 N 个几何点；可关闭道路分支以作消融 ---
    road_points_ahead: int = 20
    include_road_obs: bool = True

    # --- 奖励（None 为 RewardModel 内默认）---
    reward_config: RewardModelConfig | None = None

    # --- 终止：侧向超出视为 off-road（与奖励中 off_road 一致时体验更好）---
    max_lateral_offset: float = 3.5

    # --- 渲染：以自车为画面中心，世界范围约 (W/scale) × (H/scale) m；scale 为 px/m ---
    render_width: int = 800
    render_height: int = 600
    render_scale: float = 10.0

    # 使用 str | None 以兼容当前 OmegaConf 版本（其 structured config 不支持 Literal）
    render_mode: str | None = None
    auto_render: bool = False
    render_fps: int = 30

    # --- 视频保存：save_video=True 时在 render()/step() 中自动写入帧 ---
    save_video: bool = False
    video_path: str | None = None  # None 时每个 reset 在 video_dir 下自动生成
    video_dir: str = "videos"

class RoadVehicleEnv(gym.Env):
    """自动驾驶仿真环境。

    观测（绝对值）:
      [x, y, theta, v,
       cl_x0, cl_y0, ...,   # 中心线 N 个点
       lb_x0, lb_y0, ...,   # 左边界 N 个点
       rb_x0, rb_y0, ...,   # 右边界 N 个点
       dv_x0, dv_y0, ...,   # 车道线 N 个点 (每条, 单车道时无)
       actual_num_points]    # 实际有效点数

    动作: [acceleration, steering_rate]
    """

    @staticmethod
    def bulid_from_config(cfg: DictConfig) -> RoadVehicleEnv:
        """根据 Hydra 配置构建环境，保留结构化配置能力。"""
        ec = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # 允许 YAML 使用枚举名（STRAIGHT）或枚举值（straight）
        if ec.get("road_config") and ec.road_config.get("segment_weights"):
            normalized_weights = {}
            for k, v in ec.road_config.segment_weights.items():
                normalized_weights[RoadSegmentType(k) if isinstance(k, str) and k in [e.value for e in RoadSegmentType]
                else RoadSegmentType[k]] = float(v)
            ec.road_config.segment_weights = normalized_weights

        # 通过 structured schema 保持类型约束与默认值行为
        env_cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(EnvConfig()), ec))

        if not isinstance(env_cfg, EnvConfig):
            raise TypeError("Hydra env 配置无法转换为 EnvConfig")

        return RoadVehicleEnv(env_cfg)

    def __init__(
        self,
        config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self._cfg = config or EnvConfig()
        self._vehicle = VehicleModel.bulid_from_config(cfg=self._cfg.vehicle_config)
        self._road = RoadModel.bulid_from_config(self._cfg.road_config)
        self._reward_model = RewardModel.bulid_from_config(self._cfg.reward_config)

        vp = self._vehicle.params
        n_road = self._cfg.road_points_ahead

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-vp.a_max, -vp.omega_max], dtype=np.float32),
            high=np.array([vp.a_max, vp.omega_max], dtype=np.float32),
            dtype=np.float32,
        )

        # 观测空间 (Dict):
        #   vehicle: [x, y, theta, v]
        #   road (可选): centerline, left_boundary, right_boundary 各 (N, 2)
        #                lane_dividers: (num_dividers, N, 2)
        #                nearest_idx, actual_num_points 各标量
        road_cfg = self._road.config
        num_dividers = max(road_cfg.num_lanes - 1, 0)
        self._num_dividers = num_dividers

        obs_spaces = {
            "vehicle": spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32),
        }
        if self._cfg.include_road_obs:
            obs_spaces["centerline"] = spaces.Box(
                -np.inf, np.inf, shape=(n_road, 2), dtype=np.float32,
            )
            obs_spaces["left_boundary"] = spaces.Box(
                -np.inf, np.inf, shape=(n_road, 2), dtype=np.float32,
            )
            obs_spaces["right_boundary"] = spaces.Box(
                -np.inf, np.inf, shape=(n_road, 2), dtype=np.float32,
            )
            if num_dividers > 0:
                obs_spaces["lane_dividers"] = spaces.Box(
                    -np.inf, np.inf, shape=(num_dividers, n_road, 2), dtype=np.float32,
                )
        self.observation_space = spaces.Dict(obs_spaces)

        self._step_count = 0
        self._seed = 0
        self._screen = None
        self._clock = None
        self._video_writer = None
        self._video_path_current: str | None = None

    @property
    def config(self) -> EnvConfig:
        return self._cfg
    # ── Gymnasium API ─────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
        target_speed: float| None = None,
        init_speed: float | None = None,
        init_lateral_offset: float | None = None,
        init_heading_offset: float | None = None,
        random_reset: bool = False,
        options: dict | None = None):
        super().reset(seed=seed)
        self._seed = seed

        if random_reset:
            # 随机初始化速度
            init_speed = np.random.uniform(
                self._cfg.random_config.random_init_speed_range[0], 
                self._cfg.random_config.random_init_speed_range[1]
            )
            init_lateral_offset = np.random.uniform(
                self._cfg.random_config.random_init_lateral_offset_range[0], 
                self._cfg.random_config.random_init_lateral_offset_range[1]
            )
            init_heading_offset = np.random.uniform(
                self._cfg.random_config.random_init_heading_offset_range[0], 
                self._cfg.random_config.random_init_heading_offset_range[1]
            )
            target_speed = np.random.uniform(
                self._cfg.random_config.random_target_speed_range[0], 
                self._cfg.random_config.random_target_speed_range[1]
            )

        if init_speed is None:
            init_speed = self._cfg.init_speed
        if init_lateral_offset is None:
            init_lateral_offset = self._cfg.init_lateral_offset
        if init_heading_offset is None:
            init_heading_offset = self._cfg.init_heading_offset

        if target_speed is not None:
            self._reward_model.target_speed = target_speed

        self._road.generate(seed=seed)

        geo = self._road.geometry
        # 获取第一个片段的起始点
        if geo.road_segments:
            first_segment = geo.road_segments[0]
            start_xy = first_segment.centerline[0].copy()
            if len(first_segment.centerline) > 1:
                tangent = first_segment.centerline[1] - first_segment.centerline[0]
                start_theta = float(np.arctan2(tangent[1], tangent[0]))
            else:
                start_theta = 0.0
        else:
            start_xy = np.array([0.0, 0.0])
            start_theta = 0.0

        # 应用横向偏移：沿道路法线方向偏移（左正右负）
        if init_lateral_offset != 0.0:
            normal_left = np.array([-np.sin(start_theta), np.cos(start_theta)])
            start_xy = start_xy + init_lateral_offset * normal_left

        # 应用角度偏移
        start_theta = start_theta + init_heading_offset

        self._vehicle.reset(
            x=float(start_xy[0]),
            y=float(start_xy[1]),
            theta=start_theta,
            v=init_speed,
        )
        self._step_count = 0

        obs, road_info = self._get_obs_and_info()
        info = self._get_info() | road_info

        if self._cfg.save_video:
            self._start_video_recording(seed=seed, options=options)
            self.render()

        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        self._vehicle.step(action, self._cfg.dt)
        self._step_count += 1

        obs, road_info = self._get_obs_and_info()

        # 用 road info 计算奖励和终止条件
        vs = self._vehicle.state
        lateral, road_heading, progress = self._road.get_nearest_centerline_info(
            vs.x, vs.y,
        )
        heading_err = _wrap_angle(vs.theta - road_heading)

        off_road = abs(lateral) > self._cfg.max_lateral_offset
        reached_end = progress >= 0.99

        # 自动延长道路（逻辑在 RoadModel.maybe_extend_for_loop）
        extend_seed = hash((self._seed, self._step_count)) % (2**31)
        if self._road.maybe_extend_for_loop(vs.x, vs.y, progress, extend_seed=extend_seed):
            lateral, road_heading, progress = self._road.get_nearest_centerline_info(
                vs.x, vs.y,
            )
            heading_err = _wrap_angle(vs.theta - road_heading)
            reached_end = False

        truncated = self._step_count >= self._cfg.max_steps or reached_end
        terminated = off_road

        reward, reward_info = self._reward_model.compute(
            lateral_offset=lateral,
            heading_error=heading_err,
            speed=vs.v,
            action=action,
            off_road=off_road,
            lane_width=self._road.config.lane_width,
        )
        if reached_end and not off_road:
            reward += 50.0

        info = self._get_info() | road_info
        info["reward_components"] = reward_info
        info["lateral_offset"] = lateral
        info["heading_error"] = heading_err
        info["progress"] = progress
        if self._cfg.auto_render or (self._cfg.save_video and self._cfg.render_mode is None):
            self.render()
        return obs, float(reward), terminated, truncated, info

    # ── 观测构建 ──────────────────────────────────────────────────

    def _get_obs_and_info(self) -> tuple[dict[str, np.ndarray], dict]:
        """构建观测字典 + 道路附加信息。"""
        vs = self._vehicle.state

        obs = {
            "vehicle": np.array([vs.x, vs.y, vs.theta, vs.v, vs.steering], dtype=np.float32),
        }

        if not self._cfg.include_road_obs:
            return obs, {}

        # 前方道路点 (绝对坐标)
        cl_pts, left_pts, right_pts, div_pts, nearest_idx, actual_length_num, segment_length_m, road_segment_types = (
            self._road.get_road_segment_ahead(
                vs.x, vs.y, num_points=self._cfg.road_points_ahead, pad=True,
            )
        )

        obs["centerline"] = cl_pts.astype(np.float32)
        obs["left_boundary"] = left_pts.astype(np.float32)
        obs["right_boundary"] = right_pts.astype(np.float32)
        if div_pts:
            obs["lane_dividers"] = np.stack(div_pts).astype(np.float32)

        info = {
            "actual_length_num": np.float32(actual_length_num),
            "segment_length_m": np.float32(segment_length_m),
            "road_segment_types": [t.value for t in road_segment_types],
        }

        return obs, info

    def _get_info(self) -> dict:
        vs = self._vehicle.state
        info = {
            "x": vs.x, "y": vs.y, "theta": vs.theta, "v": vs.v,
            "steering": vs.steering, "step": self._step_count,
        }
        return info

    # ── 视频保存 ──────────────────────────────────────────────────

    @property
    def video_path_written(self) -> str | None:
        """最近一次已完成或进行中的视频文件路径。"""
        return self._video_path_current

    def stop_video(self) -> str | None:
        """结束当前视频录制并落盘。返回已写入的文件路径。"""
        return self._finalize_video()

    def _resolve_video_path(self, seed: int | None, options: dict | None) -> str:
        if options and options.get("video_path"):
            return str(options["video_path"])
        if self._cfg.video_path:
            return str(self._cfg.video_path)
        video_dir = Path(self._cfg.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        seed_part = f"seed{seed}" if seed is not None else "noseed"
        ts = int(time.time() * 1000)
        return str(video_dir / f"episode_{seed_part}_{ts}.mp4")

    def _start_video_recording(self, *, seed: int | None, options: dict | None) -> None:
        self._finalize_video()
        path = self._resolve_video_path(seed, options)
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "save_video=True 需要 imageio，请安装: pip install imageio imageio-ffmpeg",
            ) from exc

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._video_writer = imageio.get_writer(path, fps=int(self._cfg.render_fps))
        self._video_path_current = path

    def _append_video_frame(self, frame: np.ndarray) -> None:
        if self._video_writer is None:
            return
        self._video_writer.append_data(np.asarray(frame, dtype=np.uint8))

    def _finalize_video(self) -> str | None:
        path = self._video_path_current
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        return path

    # ── 渲染 ──────────────────────────────────────────────────────

    def render(
        self,
        info_text: list[str] | None = None,
        overlays: list[dict] | None = None,
    ):
        """渲染当前帧。

        Args:
            progress: 道路纵向进度 [0, 1]
            overlays: 叠加轨迹列表，每项为 dict:
                {"points": ndarray(M,2), "color": (R,G,B), "width": int,
                 "style": "solid"/"dashed"/"dots", "label": str}
        """
        if self._cfg.render_mode is None and not self._cfg.save_video:
            return None
        try:
            import pygame
        except ImportError:
            return None

        W, H = self._cfg.render_width, self._cfg.render_height
        if self._screen is None:
            pygame.init()
            if self._cfg.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((W, H))
            else:
                self._screen = pygame.Surface((W, H))
        if self._clock is None:
            self._clock = pygame.time.Clock()

        surf = pygame.Surface((W, H))
        surf.fill((50, 50, 50))

        geo = self._road.geometry
        if geo is not None:
            vs = self._vehicle.state
            cx, cy = vs.x, vs.y
            scale = self._cfg.render_scale

            def to_screen(px: float, py: float) -> tuple[int, int]:
                return int(W / 2 + (px - cx) * scale), int(H / 2 - (py - cy) * scale)

            # 渲染所有道路片段
            seg_font = pygame.font.SysFont("monospace", max(12, H // 50))
            for seg_i, segment in enumerate(geo.road_segments):
                # 渲染边界
                for boundary in [segment.left_boundary, segment.right_boundary]:
                    pts = [to_screen(p[0], p[1]) for p in boundary]
                    if len(pts) >= 2:
                        pygame.draw.lines(surf, (255, 255, 255), False, pts, 2)

                # 渲染中心线（稀疏采样）
                cl_pts = [to_screen(p[0], p[1]) for p in segment.centerline[::3]]
                if len(cl_pts) >= 2:
                    pygame.draw.lines(surf, (255, 255, 0), False, cl_pts, 1)

                # 片段起点序号标记
                sp = to_screen(segment.centerline[0][0], segment.centerline[0][1])
                pygame.draw.circle(surf, (255, 100, 0), sp, 5)
                lbl = seg_font.render(str(seg_i), True, (255, 255, 255))
                surf.blit(lbl, (sp[0] + 6, sp[1] - 6))

                # 渲染车道分隔线（虚线）
                for divider in segment.lane_dividers:
                    d_pts = [to_screen(p[0], p[1]) for p in divider[::2]]
                    for k in range(0, len(d_pts) - 1, 2):
                        pygame.draw.line(surf, (200, 200, 200), d_pts[k], d_pts[k + 1], 1)

            car_screen = to_screen(vs.x, vs.y)

            # 车辆 BOX
            car_length_px = int(self._vehicle.params.length * scale)
            car_width_px = int(self._vehicle.params.width * scale)
            hl, hw = car_length_px / 2, car_width_px / 2
            cos_t, sin_t = np.cos(vs.theta), np.sin(vs.theta)
            # pygame y 轴向下，所以 sin 取反
            corners_local = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
            car_corners = []
            for lx, ly in corners_local:
                sx = car_screen[0] + int(lx * cos_t - ly * (-sin_t))
                sy = car_screen[1] + int(lx * (-sin_t) + ly * cos_t)
                car_corners.append((sx, sy))
            pygame.draw.polygon(surf, (0, 200, 0), car_corners)
            pygame.draw.polygon(surf, (0, 255, 0), car_corners, 2)

            # 朝向线
            arrow_len = 18
            dx = int(arrow_len * cos_t)
            dy = int(-arrow_len * sin_t)
            pygame.draw.line(surf, (0, 255, 0), car_screen,
                             (car_screen[0] + dx, car_screen[1] + dy), 2)

            # 前轮转向角指示线
            steer_len = 12
            steer_angle = vs.theta + vs.steering
            sdx = int(steer_len * np.cos(steer_angle))
            sdy = int(-steer_len * np.sin(steer_angle))
            front_x = car_screen[0] + int(hl * cos_t)
            front_y = car_screen[1] + int(-hl * sin_t)
            pygame.draw.line(surf, (255, 100, 100), (front_x, front_y),
                             (front_x + sdx, front_y + sdy), 2)

            # ── 叠加轨迹（绘制在最上层，避免被车辆 box 遮挡）──
            if overlays:
                for ov in overlays:
                    pts_w = np.asarray(ov["points"])
                    if pts_w.ndim != 2 or pts_w.shape[0] < 2:
                        continue
                    color = ov.get("color", (0, 200, 255))
                    width = ov.get("width", 2)
                    style = ov.get("style", "solid")
                    s_pts = [to_screen(p[0], p[1]) for p in pts_w]
                    if style == "dots":
                        for sp in s_pts:
                            pygame.draw.circle(surf, color, sp, max(width, 2))
                    elif style == "dashed":
                        for k in range(0, len(s_pts) - 1, 2):
                            pygame.draw.line(surf, color, s_pts[k], s_pts[min(k+1, len(s_pts)-1)], width)
                    else:
                        pygame.draw.lines(surf, color, False, s_pts, width)

            # ── HUD: 车辆状态 + 进度 ──
            font = pygame.font.SysFont("monospace", max(14, H // 40))
            hud_lines = [
                f"x        = {vs.x:6.2f} m",
                f"y        = {vs.y:6.2f} m",
                f"heading  = {np.degrees(vs.theta):6.1f} deg",
                f"speed    = {vs.v:6.2f} m/s",
                f"steer    = {np.degrees(vs.steering):6.1f} deg",
                f"step_count = {self._step_count:d}/{self._cfg.max_steps:d}",
                
            ]
            if info_text:
                hud_lines = hud_lines + info_text
            if overlays:
                for ov in overlays:
                    lbl = ov.get("label")
                    if lbl:
                        hud_lines.append(f"-- {lbl}")
            for j, line in enumerate(hud_lines):
                txt_color = (0, 255, 255)
                if line.startswith("--") and overlays:
                    for ov in overlays:
                        if ov.get("label") and ov["label"] in line:
                            txt_color = ov.get("color", (0, 200, 255))
                            break
                txt = font.render(line, True, txt_color)
                surf.blit(txt, (10, 10 + j * (font.get_height() + 2)))

        self._screen.blit(surf, (0, 0))
        frame = np.transpose(
            np.array(pygame.surfarray.pixels3d(self._screen)), axes=(1, 0, 2),
        )
        if self._cfg.save_video:
            self._append_video_frame(frame)

        if self._cfg.render_mode == "human":
            pygame.event.pump()
            self._clock.tick(int(self._cfg.render_fps))
            pygame.display.flip()
            return None
        if self._cfg.render_mode == "rgb_array":
            return frame
        return None

    def close(self, wait: float = 0.0):
        """关闭环境和渲染窗口。

        Args:
            wait: 关闭前等待秒数（期间保持窗口响应）
        """
        self._finalize_video()

        if self._screen is not None:
            import pygame
            import time

            if wait > 0 and self._cfg.render_mode == "human":
                end_time = time.monotonic() + wait
                while time.monotonic() < end_time:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            break
                    if self._clock:
                        self._clock.tick(30)

            # 发送 QUIT 事件确保窗口关闭
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            pygame.event.pump()

            try:
                pygame.display.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass

            self._screen = None
            self._clock = None
