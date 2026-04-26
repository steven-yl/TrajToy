"""MPC 控制 + render 可视化示例。

运行: python example/render_toturial.py
需要: pip install casadi pygame matplotlib
"""

import time

import numpy as np
import matplotlib.pyplot as plt

from sim_env import (
    RoadVehicleEnv,
    EnvConfig,
    RoadGenerationConfig,
    RoadSegmentType,
    SegmentSpec,
)
from sim_env.vehicle_controller import VehicleMPC, MPCConfig

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 创建环境 + MPC ──

env = RoadVehicleEnv(
    EnvConfig(
        road_config=RoadGenerationConfig(
            num_lanes=3,
            loop_segments=True,
            fixed_segments=[
                SegmentSpec(RoadSegmentType.STRAIGHT, {"length": 50}),
                SegmentSpec(RoadSegmentType.CURVE, {"radius": 20, "angle_deg": 180}),
                SegmentSpec(RoadSegmentType.CURVE, {"radius": 10, "angle_deg": -180}),
            ],
        ),
        # road_config=RoadGenerationConfig(num_lanes=3, loop_segments=True),
        road_points_ahead=30,
        init_speed=0.0,
        dt=0.1,
        # render_width=1200,
        # render_height=1000,
        # render_scale=5.0,
        max_steps=1000,
        render_mode="human",
    ),
)

mpc = VehicleMPC(
    mpc_config=MPCConfig(vehicle_params=env._vehicle.params, horizon=20, target_speed=10.0, dt=0.1),
)

obs, info = env.reset(seed=1)
mpc.reset()

total_reward = 0

for step in range(8000):
    veh = obs["vehicle"]
    state = np.array([veh[0], veh[1], veh[2], veh[3], info.get("steering", 0.0)])
    ref_path = obs["centerline"]

    action, predicted_trajectory, ref_path_resample = mpc.compute(state, ref_path)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render(
        # info_text=[f"progress = {info.get("progress", 0.0):6.1%}"],
        overlays=[
            {"points": predicted_trajectory, "color": (255, 140, 0), "style": "solid", "width": 2, "label": "predicted_trajectory"},
            {"points": ref_path_resample,   "color": (70, 130, 200), "style": "solid", "width": 3, "label": "ref_path_resample"},
            {"points": obs["centerline"],   "color": (255, 69, 0), "style": "solid", "width": 2,   "label": "centerline"},
            {"points": obs["left_boundary"], "color": (255, 69, 0), "style": "solid",   "width": 2,   "label": "left_boundary"},
            {"points": obs["right_boundary"],"color": (255, 69, 0),   "style": "solid",   "width": 2,   "label": "right_boundary"},
        ]
    )
    if terminated or truncated:
        break
time.sleep(1)
env.close()

print(f"Episode 结束: steps={info['step']}, total_reward={total_reward:.1f}")
print(f"terminated={terminated}, truncated={truncated}")
print(f"最终位置: x={info['x']:.1f}, y={info['y']:.1f}, v={info['v']:.1f} m/s")