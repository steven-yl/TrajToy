#!/usr/bin/env bash
# IL 训练 / 闭环评估启动脚本（单卡 python 或多卡 torchrun + DDP）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# GPU 进程数：1 = 单卡 python；>1 = torchrun + trainflow.trainer.strategy=ddp
NPROC="${NPROC:-${IL_NPROC:-1}}"

show_menu() {
  cat <<EOF
请选择运行模式 (当前 NPROC=${NPROC}):
  1) 训练 - MLP
  2) 训练 - Diffusion (traj_diffusers_trainable_model)
  3) 训练 - Diffusion (traj_diffusion_trainable_model)
  4) 训练 - Diffusion (traj_diffusers, num_inference_steps=50)
  5) 训练 - Diffusion (traj_diffusers, num_inference_steps=10)
  6) 闭环评估 - Diffusers (traj_diffusers_trainable_model)
  7) 闭环评估 - Diffusion (traj_diffusion_trainable_model)
  8) 闭环评估 - MLP
  9) 查看训练配置 (--cfg job)
  q) 退出

多卡示例: NPROC=8 ${0} train 2
EOF
}

show_help() {
  cat <<EOF
用法:
  ${0}                      交互式选择
  ${0} <序号>               直接运行 1-9
  ${0} train [1-5] [Hydra 覆盖...]   训练
  ${0} close_eval [mlp|diffusion|diffusers]
  ${0} showcfg                打印训练 Hydra 配置
  ${0} -h                    显示帮助

全局选项（放在子命令前）:
  --nproc <N>               等同 NPROC=<N>
  --run-id <ID>             等同 RUN_ID=<ID>（日志目录后缀，多卡强烈建议设置）

环境变量:
  NPROC / IL_NPROC          GPU 进程数，默认 1
  RUN_ID                    实验 ID；多卡时若未设置，脚本在 torchrun 前自动生成一次
                            （对应 train_config.yaml 中 app.run_id / oc.env:RUN_ID）

训练子模式:
  1  MLP
  2  Diffusion + traj_diffusers_trainable_model
  3  Diffusion + traj_diffusion_trainable_model
  4  Diffusion + traj_diffusers_trainable_model, num_inference_steps=50
  5  Diffusion + traj_diffusers_trainable_model, num_inference_steps=10

闭环评估:
  close_eval diffusers | diffusion | mlp

示例:
  ${0} train 1
  NPROC=4 ${0} train 2
  ${0} --nproc 8 train 3
  RUN_ID=exp01 ${0} --nproc 8 train 2 trainflow.trainer.max_epochs=50
  ${0} train 2 model@trainflow.model=traj_diffusers_trainable_model

多卡说明:
  NPROC>1 时使用: torchrun --nproc_per_node=N -m il.train ... trainflow.trainer.strategy=ddp
  并在启动前 export RUN_ID，避免各 rank 因 Hydra now 跨秒生成多个 log 目录。
EOF
}

run_cmd() {
  echo ">>> $*"
  "$@"
}

ensure_run_id() {
  if [[ -n "${RUN_ID:-}" ]]; then
    echo "RUN_ID=${RUN_ID} (preset)"
    return
  fi
  RUN_ID="$(date +%Y%m%d_%H%M%S)"
  export RUN_ID
  echo "RUN_ID=${RUN_ID} (auto, shared by all ranks)"
}

# 启动 il.train：单卡 python 或多卡 torchrun + DDP
launch_il_train() {
  local -a hydra_args=("$@")

  if [[ "${NPROC}" -gt 1 ]]; then
    ensure_run_id
    hydra_args+=(trainflow.trainer.strategy=ddp)
    run_cmd torchrun --nproc_per_node="${NPROC}" -m il.train "${hydra_args[@]}"
  else
    run_cmd python -m il.train "${hydra_args[@]}"
  fi
}

train_hydra_args_for_mode() {
  local mode="$1"
  case "${mode}" in
    1)
      echo "training@_global_=train_traj_mlp"
      ;;
    2)
      echo "training@_global_=train_traj_diffusion"
      echo "model@trainflow.model=traj_diffusers_trainable_model"
      ;;
    3)
      echo "training@_global_=train_traj_diffusion"
      echo "model@trainflow.model=traj_diffusion_trainable_model"
      ;;
    4)
      echo "training@_global_=train_traj_diffusion"
      echo "model@trainflow.model=traj_diffusers_trainable_model"
      echo "trainflow.model.predictor.num_inference_steps=50"
      ;;
    5)
      echo "training@_global_=train_traj_diffusion"
      echo "model@trainflow.model=traj_diffusers_trainable_model"
      echo "trainflow.model.predictor.num_inference_steps=10"
      ;;
    *)
      echo "无效训练模式: ${mode}（请输入 1-5）" >&2
      return 1
      ;;
  esac
}

run_train_mode() {
  local mode="$1"
  shift || true
  local -a hydra_args=()
  local line
  while IFS= read -r line; do
    hydra_args+=("${line}")
  done < <(train_hydra_args_for_mode "${mode}")
  hydra_args+=("$@")
  launch_il_train "${hydra_args[@]}"
}

run_close_eval() {
  case "${1:-diffusers}" in
    mlp)
      run_cmd python -m il.evaluation eval@_global_=close_eval_traj_mlp
      ;;
    diffusion|df)
      run_cmd python -m il.evaluation eval@_global_=close_eval_traj_diffusion
      ;;
    diffusers)
      run_cmd python -m il.evaluation eval@_global_=close_eval_traj_diffusers
      ;;
    *)
      echo "无效评估类型: $1（可用: mlp, diffusion, diffusers）" >&2
      return 1
      ;;
  esac
}

run_showcfg() {
  run_cmd python -m il.train --cfg job
}

run_choice() {
  case "$1" in
    1|2|3|4|5)
      run_train_mode "$1" "${@:2}"
      ;;
    6)
      run_close_eval diffusers
      ;;
    7)
      run_close_eval diffusion
      ;;
    8)
      run_close_eval mlp
      ;;
    9)
      run_showcfg
      ;;
    *)
      echo "无效序号: $1（请输入 1-9）" >&2
      return 1
      ;;
  esac
}

pick_train_mode_interactive() {
  cat <<EOF
请选择训练模式 (NPROC=${NPROC}):
  1) MLP
  2) Diffusion (traj_diffusers_trainable_model)
  3) Diffusion (traj_diffusion_trainable_model)
  4) Diffusion (traj_diffusers, num_inference_steps=50)
  5) Diffusion (traj_diffusers, num_inference_steps=10)
  q) 退出
EOF
  while true; do
    read -r -p "请输入序号 [1-5/q]: " choice
    case "${choice}" in
      q|Q)
        echo "已取消。"
        exit 0
        ;;
      1|2|3|4|5)
        run_train_mode "${choice}"
        return
        ;;
      *)
        echo "无效输入，请重新输入。"
        ;;
    esac
  done
}

pick_mode_interactive() {
  show_menu
  while true; do
    read -r -p "请输入序号 [1-9/q]: " choice
    case "${choice}" in
      q|Q)
        echo "已取消。"
        exit 0
        ;;
      1|2|3|4|5|6|7|8|9)
        run_choice "${choice}"
        return
        ;;
      *)
        echo "无效输入，请重新输入。"
        ;;
    esac
  done
}

# 解析 --nproc / --run-id 后的剩余参数（勿在子 shell 中调用，否则 exit 无法退出主脚本）
REST_ARGS=()

parse_global_opts() {
  REST_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --nproc)
        [[ $# -ge 2 ]] || { echo "缺少 --nproc 参数" >&2; exit 1; }
        NPROC="$2"
        shift 2
        ;;
      --run-id)
        [[ $# -ge 2 ]] || { echo "缺少 --run-id 参数" >&2; exit 1; }
        RUN_ID="$2"
        export RUN_ID
        shift 2
        ;;
      -h|--help|help)
        show_help
        exit 0
        ;;
      *)
        REST_ARGS+=("$1")
        shift
        ;;
    esac
  done
  if [[ "${NPROC}" -lt 1 ]]; then
    echo "NPROC 必须 >= 1，当前: ${NPROC}" >&2
    exit 1
  fi
}

main() {
  parse_global_opts "$@"
  set -- "${REST_ARGS[@]}"

  if [[ $# -eq 0 ]]; then
    pick_mode_interactive
    return
  fi

  case "$1" in
    train)
      shift
      if [[ $# -eq 0 ]]; then
        pick_train_mode_interactive
      elif [[ "$1" =~ ^[1-5]$ ]]; then
        local mode="$1"
        shift
        run_train_mode "${mode}" "$@"
      else
        echo "训练模式须为 1-5，收到: $1" >&2
        exit 1
      fi
      ;;
    close_eval|eval)
      run_close_eval "${2:-diffusers}"
      ;;
    showcfg|cfg)
      run_showcfg
      ;;
    1|2|3|4|5|6|7|8|9)
      run_choice "$@"
      ;;
    *)
      echo "未知参数: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
}

main "$@"
