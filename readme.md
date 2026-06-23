# TrajToy

**TrajToy** 是一个面向自动驾驶轨迹规划的轻量研究平台：在程序化生成的多车道道路上仿真车辆行驶，用 MPC 专家轨迹构造模仿学习数据集，训练多种轨迹预测模型，并在闭环仿真中验证「预测 → MPC 跟踪 → 车辆执行」的完整链路。

### TrainFlow 训练框架

TrajToy 的训练与评估由 **TrainFlow** 框架驱动（`src/trainflow/`），专为轨迹模仿学习设计，在保持轻量的同时提供接近工业级训练栈的能力：

| 能力 | 说明 |
|------|------|
| **Hydra 原生** | 配置即代码：`Trainer` / `TrainableModel` / `DataModule` 全由 YAML `_target_` 组合实例化，训练、评估、闭环共用同一套配置体系 |
| **声明式指标** | 模型内 `self.log(...)` 自动聚合 epoch 指标、跨 rank 归约，无需手写训练循环 |
| **完整生命周期** | `fit` / `validate` / `test` / `predict` 四阶段统一接口，checkpoint 带版本 schema 与 `weights_only` 安全加载 |
| **Callback 生态** | ModelCheckpoint、EarlyStopping、EMA（指数滑动平均权重）、进度条、耗时监控、TensorBoard 轨迹可视化 |
| **分布式与精度** | torchrun + DDP 多卡、`precision=16/bf16` 混合精度、梯度累积与裁剪 |
| **模型解耦** | 算法逻辑封装在 `TrainableModel`，框架只负责优化器调度与数据流——同一 Trainer 可无缝切换 MLP / Diffusion / Flow |

相比直接堆叠脚本或重度依赖通用训练库，TrainFlow 把「轨迹 IL 的共性」（条件编码、归一化、扩散采样、闭环 checkpoint 加载）沉淀到 IL 层，框架层只保留可复用的训练原语，便于快速试验新 backbone 与新生成式范式。

### 实现的轨迹预测算法

| 算法 | 核心思路 | 条件编码 | 训练目标 |
|------|----------|----------|----------|
| **Transformer IL** | Encoder-Decoder + 可学习 future queries，一次前向回归未来 25 步 | 历史状态 + 道路中心线/边界/车道线 token | ADE / FDE（`TrajLoss`） |
| **Diffusion（Diffusers）** | DDIM 迭代去噪；默认 **Conditional DiT1D**（adaLN + 场景 cross-attention） | 同上，经 `StateEncoder` 注入去噪网络 | 噪声 / 样本 MSE |
| **Diffusion（自定义）** | DDPM / DDIM 采样循环，可选 **Conditional UNet1D** 1D 卷积去噪 | 同上 | 扩散 MSE |
| **Flow Matching** | Rectified Flow：线性插值 \(x_t=(1-\sigma)x_0+\sigma x_1\)，预测 velocity \(v=x_1-x_0\)，Euler 积分采样 | DiT1D + `FlowMatchEulerDiscreteScheduler` | velocity MSE（推理步数可调，通常少于 Diffusion） |

**轨迹表征工程**：heading 在扩散/flow 空间用 \((\sin\theta, \cos\theta)\) 编码以避免 \(\pm\pi\) 不连续；速度由 xy 位移后处理推出，减少冗余维度。训练数据来自 **CasADi MPC** 在仿真环境中的专家 rollout，闭环评估时模型输出参考路径、再由 MPC 低层跟踪，形成「学习规划 + 优化控制」分层架构。

```
仿真环境 (RoadVehicleEnv)  →  数据采集 (MPC rollout)  →  IL 训练 (TrainFlow)
        ↑                                                          ↓
        └──────────── 闭环评估 (预测 + MPC + 渲染/录屏) ←────────── checkpoint
```

---

## 演示视频

闭环评估效果：模型根据历史状态与前方道路几何预测参考路径（彩色轨迹），MPC 控制器跟踪该路径并驱动车辆；画面叠加预测轨迹、MPC 参考线与道路边界。

| MLP / Transformer | DDIM (Diffusers + DiT) | Flow Matching |
|:---:|:---:|:---:|
| ![MLP 闭环评估](videos/mlp.webp) | ![Diffusion 闭环评估](videos/DDIM.webp) | ![Flow 闭环评估](videos/flow.webp) |

本地复现录屏见下方 [闭环评估](#闭环评估) 一节（`save_video: true`）。

---

## 核心能力

| 模块 | 说明 |
|------|------|
| **仿真环境** | Gymnasium `RoadVehicleEnv`：运动学/动力学车辆、直线/弯道/路口/分合流道路、奖励与终止判据、Pygame 渲染与 MP4 录屏 |
| **MPC 控制器** | CasADi + IPOPT 路径跟踪，用于数据生成与闭环评估中的低层控制 |
| **数据采集** | 仿真 rollout → episode pkl → 预处理 → `TrainingSample` 训练集 |
| **模仿学习** | 基于 **TrainFlow** 训练框架，Hydra 配置驱动，支持 TensorBoard / checkpoint / DDP 多卡 |
| **轨迹模型** | MLP(Transformer)、自定义 Diffusion(UNet)、HuggingFace Diffusers(DiT/UNet)、Flow Matching |
| **闭环评估** | 模型预测参考路径 → MPC 执行 → 实时可视化与视频导出 |
| **强化学习** | 可选 PPO baseline（`src/rl/`） |

---

## 快速开始

### 安装

```bash
git clone <repo-url> TrajToy && cd TrajToy
pip install -e ".[dev]"   # 含 pytest；核心依赖见 pyproject.toml
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
```

主要依赖：`torch`、`diffusers`、`gymnasium`、`pygame`、`casadi`、`hydra-core`、`imageio`、`imageio-ffmpeg`（录屏需要）。

### 一键脚本

```bash
./script/il_run.sh              # 交互式：训练 / 闭环评估 / 查看配置
./script/data_process_run.sh    # 数据采集 + 预处理
```

### 数据采集

```bash
./script/data_process_run.sh           # 生成 + 预处理（默认）
./script/data_process_run.sh create    # 仅仿真 rollout
./script/data_process_run.sh preprocess
```

输出目录：`data/<timestamp>/`，预处理后数据供训练配置中的 `trainflow.data.data_set.data_dir` 引用。

### 训练

| 序号 | 模型 | 命令 |
|------|------|------|
| 1 | MLP (Transformer) | `./script/il_run.sh train 1` |
| 2 | Diffusion (Diffusers + DiT) | `./script/il_run.sh train 2` |
| 3 | Diffusion (自定义 UNet) | `./script/il_run.sh train 3` |
| 6 | Flow Matching | `./script/il_run.sh train 6` |

多卡 DDP 示例：

```bash
NPROC=4 RUN_ID=exp01 ./script/il_run.sh train 2 trainflow.trainer.max_epochs=50
```

日志与 checkpoint：`log/il_train_<config>_<run_id>/`

直接调用 Python：

```bash
python -m il.train training@_global_=train_traj_mlp
python -m il.train training@_global_=train_traj_flow
```

### 闭环评估

在 `src/il/conf/eval_config.yaml` 中设置 `resume_checkpoint`，然后：

```bash
./script/il_run.sh close_eval mlp
./script/il_run.sh close_eval diffusers
./script/il_run.sh close_eval flow
# 或交互菜单选 7–10
```

评估配置（`src/il/conf/eval/close_eval_traj_*.yaml`）中可开启：

```yaml
env:
  render_mode: human      # 实时窗口；headless 录屏可设为 null
  save_video: true
  video_dir: "videos"     # 输出 MP4，文件名 episode_seed<seed>_<ts>.mp4
```

---

## 闭环评估链路

```mermaid
flowchart LR
  A[历史轨迹 + 道路观测] --> B[轨迹预测模型]
  B --> C[参考路径 ref_path]
  C --> D[VehicleMPC]
  D --> E[加速度 / 转向率]
  E --> F[RoadVehicleEnv]
  F --> A
  F --> G[Pygame 渲染 / 录屏]
```

- **输入**：车辆历史状态（5 步 × 7 维）、前方道路中心线/边界/车道线（60 点）
- **输出**：未来轨迹（25 步 × `[x, y, heading, v]`），转世界坐标后作为 MPC 参考
- **控制**：MPC horizon=15，dt=0.1 s，CasADi 求解

---

## 支持的模型

| 模型 | 配置文件 | 骨干网络 | 采样 / 推理 |
|------|----------|----------|-------------|
| MLP | `traj_mlp_trainable_model.yaml` | Transformer Encoder-Decoder | 单步前向 |
| Diffusion (自定义) | `traj_diffusion_trainable_model.yaml` | Conditional UNet1D | DDPM / DDIM |
| Diffusion (Diffusers) | `traj_diffusers_trainable_model.yaml` | DiT1D（默认）/ UNet1D | diffusers DDIM |
| Flow Matching | `traj_flow_trainable_model.yaml` | DiT1D | Euler flow scheduler |

训练损失：MLP / Flow 使用 `TrajLoss`（ADE/FDE）；Diffusion 系列使用扩散 MSE 损失。

---

## 仿真环境

`RoadVehicleEnv`（[`src/sim_env/road_vehicle_env.py`](src/sim_env/road_vehicle_env.py)）

**观测**（Dict，绝对坐标）：

- `vehicle`: `[x, y, θ, v, steering]`
- `centerline` / `left_boundary` / `right_boundary`: `(N, 2)`
- `lane_dividers`: `(num_lanes-1, N, 2)`（多车道时）

**动作**：`[acceleration, steering_rate]`

**道路生成**：固定片段序列或随机加权采样（直线、弯道、路口、分合流）；`loop_segments=true` 时接近末端自动延长。

**渲染**：自车居中 Pygame 视图，支持轨迹 overlay（预测路径、MPC 参考线、道路几何）与 HUD。

示例脚本：

```bash
python example/render_toturial.py   # MPC + 实时渲染
```

---

## 项目结构

```
TrajToy/
├── src/
│   ├── sim_env/          # 仿真环境、道路/车辆/奖励模型、MPC 控制器
│   ├── trainflow/        # 通用训练框架（Trainer、Logger、Callback、DDP）
│   ├── il/               # 模仿学习：模型、数据、训练、闭环评估、Hydra 配置
│   ├── data_process/     # MPC rollout 数据采集与预处理
│   └── rl/               # PPO 强化学习 baseline
├── example/              # Jupyter 教程与 render 示例
├── script/               # il_run.sh、data_process_run.sh
├── tests/                # pytest 单元测试
├── videos/               # 闭环评估演示视频
├── data/                 # 生成的数据集
└── log/                  # 训练与评估日志
```

### 教程 Notebook（`example/`）

| Notebook | 内容 |
|----------|------|
| `road_vehicle_env_tutorial.ipynb` | 环境 reset/step、观测解析、渲染 |
| `road_model_tutorial.ipynb` | 道路片段生成 |
| `vehicle_model_tutorial.ipynb` | 车辆模型与积分器 |
| `vehicle_controller_tutorial.ipynb` | MPC 控制器 |
| `data_tutorial.ipynb` | 数据采集与预处理 |
| `il_mlp_train_toturial.ipynb` | MLP 训练 |
| `il_df_train_toturial.ipynb` | Diffusion 训练与可视化 |
| `rl_train_tutorial.ipynb` | PPO 训练 |

---

## 配置说明

项目使用 [Hydra](https://hydra.cc/) 管理配置：

| 用途 | 配置根目录 |
|------|-----------|
| IL 训练 | `src/il/conf/train_config.yaml` |
| IL 闭环评估 | `src/il/conf/eval_config.yaml` |
| 数据采集 | `src/data_process/conf/config.yaml` |
| RL 训练 | `src/rl/conf/config.yaml` |

**TrainFlow** 框架（`src/trainflow/`）提供通用的 `Trainer` / `TrainableModel` / `DataModule` 抽象；**IL** 层在其上实现轨迹预测模型、数据集与评估逻辑。

---

## 测试

```bash
pytest tests/
```

覆盖仿真环境、道路/车辆/奖励模型、数据处理与 RL 等模块。

---

## 许可

研究用途。使用前请根据实际情况补充 License 声明。
