#!/usr/bin/env bash
# IL 训练 / 闭环评估启动脚本（单卡 python 或多卡 torchrun + DDP）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# GPU 进程数：1 = 单卡 python；>1 = torchrun + trainflow.trainer.strategy=ddp
NPROC="${NPROC:-${IL_NPROC:-1}}"

# 训练模式：序号 -> Hydra 覆盖（空格分隔）；新增模式只需改此处与 TRAIN_LABELS
declare -A RUN_OPTIONS=(
  [1]="training@_global_=train_traj_mlp"
  [2]="training@_global_=train_traj_diffusion model@trainflow.model=traj_diffusers_trainable_model"
  [3]="training@_global_=train_traj_diffusion model@trainflow.model=traj_diffusion_trainable_model"
  [4]="training@_global_=train_traj_diffusion model@trainflow.model=traj_diffusers_trainable_model trainflow.model.predictor.num_inference_steps=50"
  [5]="training@_global_=train_traj_diffusion model@trainflow.model=traj_diffusers_trainable_model trainflow.model.predictor.num_inference_steps=10"
  [6]="training@_global_=train_traj_flow"
)
declare -A TRAIN_LABELS=(
  [1]="MLP"
  [2]="Diffusion (traj_diffusers_trainable_model)"
  [3]="Diffusion (traj_diffusion_trainable_model)"
  [4]="Diffusion (traj_diffusers, num_inference_steps=50)"
  [5]="Diffusion (traj_diffusers, num_inference_steps=10)"
  [6]="Flow (traj_flow_trainable_model)"
)
TRAIN_MODES=(1 2 3 4 5 6)
CLOSE_EVAL_BASE=7
SHOWCFG_CHOICE=11

_train_mode_valid() {
  [[ -n "${RUN_OPTIONS[$1]:-}" ]]
}

_train_modes_pattern() {
  local IFS='|'
  echo "${TRAIN_MODES[*]}"
}

_print_train_menu() {
  local prefix="${1:-训练 - }"
  local m
  for m in "${TRAIN_MODES[@]}"; do
    echo "  ${m}) ${prefix}${TRAIN_LABELS[$m]}"
  done
}

show_menu() {
  cat <<EOF
请选择运行模式 (当前 NPROC=${NPROC}):
$(_print_train_menu "训练 - ")
  ${CLOSE_EVAL_BASE}) 闭环评估 - Diffusers (traj_diffusers_trainable_model)
  $((CLOSE_EVAL_BASE + 1))) 闭环评估 - Diffusion (traj_diffusion_trainable_model)
  $((CLOSE_EVAL_BASE + 2))) 闭环评估 - MLP
  $((CLOSE_EVAL_BASE + 3))) 闭环评估 - Flow (traj_flow_trainable_model)
  ${SHOWCFG_CHOICE}) 查看训练配置 (--cfg job)
  q) 退出

多卡示例: NPROC=8 ${0} train 2
EOF
}

show_help() {
  local m help_train=""
  for m in "${TRAIN_MODES[@]}"; do
    help_train+="  ${m}  ${TRAIN_LABELS[$m]}"$'\n'
  done
  cat <<EOF
用法:
  ${0}                      交互式选择
  ${0} <序号>               直接运行 1-${SHOWCFG_CHOICE}
  ${0} train [$(_train_modes_pattern)] [Hydra 覆盖...]   训练
  ${0} close_eval [mlp|diffusion|diffusers|flow]
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
${help_train}
闭环评估:
  close_eval diffusers | diffusion | mlp | flow

示例:
  ${0} train 1
  NPROC=4 ${0} train 2
  ${0} --nproc 8 train 3
  RUN_ID=exp01 ${0} --nproc 8 train 2 trainflow.trainer.max_epochs=50
  ${0} train 2 model@trainflow.model=traj_diffusers_trainable_model

训练时会在启动前提示输入 task_msg（任务描述，可回车跳过），以 Hydra 覆盖 task_msg=... 写入配置。

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

_hydra_has_task_msg() {
  local arg
  for arg in "$@"; do
    [[ "${arg}" == task_msg=* ]] && return 0
  done
  return 1
}

# 训练前交互输入 task_msg，追加为 Hydra 覆盖 task_msg="..."
ensure_task_msg() {
  TASK_MSG=""
  if _hydra_has_task_msg "$@"; then
    return
  fi
  if [[ -t 0 ]]; then
    read -r -p "任务描述 task_msg（可选，直接回车跳过）: " TASK_MSG || true
    if [[ -n "${TASK_MSG}" ]]; then
      echo "task_msg=${TASK_MSG}"
    fi
  fi
}

# 将 task_msg 追加为 Hydra 参数覆盖（task_msg="..."）
append_task_msg_override() {
  local -n _args=$1
  if _hydra_has_task_msg "${_args[@]}"; then
    return
  fi
  if [[ -z "${TASK_MSG:-}" ]]; then
    return
  fi
  local safe="${TASK_MSG//\"/\\\"}"
  _args+=("task_msg=\"${safe}\"")
}

# 启动 il.train：单卡 python 或多卡 torchrun + DDP
launch_il_train() {
  local -a hydra_args=("$@")

  ensure_task_msg "${hydra_args[@]}"
  append_task_msg_override hydra_args

  if [[ "${NPROC}" -gt 1 ]]; then
    ensure_run_id
    hydra_args+=(trainflow.trainer.strategy=ddp)
    run_cmd torchrun --nproc_per_node="${NPROC}" -m il.train "${hydra_args[@]}"
  else
    run_cmd python -m il.train "${hydra_args[@]}"
  fi
}

run_train_mode() {
  local mode="$1"
  shift || true
  local opts="${RUN_OPTIONS[$mode]:-}"
  if [[ -z "${opts}" ]]; then
    echo "无效训练模式: ${mode}（请输入 $(_train_modes_pattern)）" >&2
    return 1
  fi
  local -a hydra_args
  read -ra hydra_args <<< "${opts}"
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
    flow)
      run_cmd python -m il.evaluation eval@_global_=close_eval_traj_flow
      ;;
    *)
      echo "无效评估类型: $1（可用: mlp, diffusion, diffusers, flow）" >&2
      return 1
      ;;
  esac
}

run_showcfg() {
  run_cmd python -m il.train --cfg job
}

run_choice() {
  local choice="$1"
  if _train_mode_valid "${choice}"; then
    run_train_mode "${choice}" "${@:2}"
    return
  fi
  case "${choice}" in
    "${CLOSE_EVAL_BASE}")
      run_close_eval diffusers
      ;;
    $((CLOSE_EVAL_BASE + 1)))
      run_close_eval diffusion
      ;;
    $((CLOSE_EVAL_BASE + 2)))
      run_close_eval mlp
      ;;
    $((CLOSE_EVAL_BASE + 3)))
      run_close_eval flow
      ;;
    "${SHOWCFG_CHOICE}")
      run_showcfg
      ;;
    *)
      echo "无效序号: ${choice}（请输入 1-${SHOWCFG_CHOICE}）" >&2
      return 1
      ;;
  esac
}

pick_train_mode_interactive() {
  echo "请选择训练模式 (NPROC=${NPROC}):"
  _print_train_menu ""
  echo "  q) 退出"
  while true; do
    read -r -p "请输入序号 [$(_train_modes_pattern)/q]: " choice
    case "${choice}" in
      q|Q)
        echo "已取消。"
        exit 0
        ;;
      *)
        if _train_mode_valid "${choice}"; then
          run_train_mode "${choice}"
          return
        fi
        echo "无效输入，请重新输入。"
        ;;
    esac
  done
}

pick_mode_interactive() {
  show_menu
  while true; do
    read -r -p "请输入序号 [1-${SHOWCFG_CHOICE}/q]: " choice
    case "${choice}" in
      q|Q)
        echo "已取消。"
        exit 0
        ;;
      *)
        if _train_mode_valid "${choice}" || [[ "${choice}" =~ ^(${CLOSE_EVAL_BASE}|$((CLOSE_EVAL_BASE + 1))|$((CLOSE_EVAL_BASE + 2))|$((CLOSE_EVAL_BASE + 3))|${SHOWCFG_CHOICE})$ ]]; then
          run_choice "${choice}"
          return
        fi
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
      elif _train_mode_valid "$1"; then
        local mode="$1"
        shift
        run_train_mode "${mode}" "$@"
      else
        echo "训练模式须为 $(_train_modes_pattern)，收到: $1" >&2
        exit 1
      fi
      ;;
    close_eval|eval)
      run_close_eval "${2:-diffusers}"
      ;;
    showcfg|cfg)
      run_showcfg
      ;;
    *)
      if _train_mode_valid "$1" || [[ "$1" =~ ^[0-9]+$ && "$1" -le "${SHOWCFG_CHOICE}" && "$1" -ge 1 ]]; then
        run_choice "$@"
      else
        echo "未知参数: $1" >&2
        show_help >&2
        exit 1
      fi
      ;;
  esac
}

main "$@"
