#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

show_menu() {
  cat <<'EOF'
请选择运行模式:
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
EOF
}

show_help() {
  cat <<EOF
用法:
  ${0}                 交互式选择
  ${0} <序号>          直接运行 1-9
  ${0} train [1-5]     训练（可选训练子模式）
  ${0} close_eval [mlp|diffusion|diffusers]
  ${0} showcfg         打印训练 Hydra 配置
  ${0} -h              显示帮助

训练子模式:
  1  MLP
  2  Diffusion + traj_diffusers_trainable_model
  3  Diffusion + traj_diffusion_trainable_model
  4  Diffusion + traj_diffusers_trainable_model, num_inference_steps=50
  5  Diffusion + traj_diffusers_trainable_model, num_inference_steps=10

闭环评估:
  6  close_eval_traj_diffusers  (DiffusionPipeline.sample)
  7  close_eval_traj_diffusion  (自研 traj_diffusion, sample_trajectory)
  8  close_eval_traj_mlp
EOF
}

run_cmd() {
  echo ">>> $*"
  "$@"
}

run_train_mode() {
  case "$1" in
    1)
      run_cmd python -m il.train training@_global_=train_traj_mlp
      ;;
    2)
      run_cmd python -m il.train \
        training@_global_=train_traj_diffusion \
        model@trainflow.model=traj_diffusers_trainable_model
      ;;
    3)
      run_cmd python -m il.train \
        training@_global_=train_traj_diffusion \
        model@trainflow.model=traj_diffusion_trainable_model
      ;;
    4)
      run_cmd python -m il.train \
        training@_global_=train_traj_diffusion \
        model@trainflow.model=traj_diffusers_trainable_model \
        trainflow.model.predictor.num_inference_steps=50
      ;;
    5)
      run_cmd python -m il.train \
        training@_global_=train_traj_diffusion \
        model@trainflow.model=traj_diffusers_trainable_model \
        trainflow.model.predictor.num_inference_steps=10
      ;;
    *)
      echo "无效训练模式: $1（请输入 1-5）" >&2
      return 1
      ;;
  esac
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
      run_train_mode "$1"
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
  cat <<'EOF'
请选择训练模式:
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

main() {
  if [[ $# -eq 0 ]]; then
    pick_mode_interactive
    return
  fi

  case "$1" in
    -h|--help|help)
      show_help
      ;;
    train)
      if [[ $# -ge 2 ]]; then
        run_train_mode "$2"
      else
        pick_train_mode_interactive
      fi
      ;;
    close_eval|eval)
      run_close_eval "${2:-diffusers}"
      ;;
    showcfg|cfg)
      run_showcfg
      ;;
    1|2|3|4|5|6|7|8|9)
      run_choice "$1"
      ;;
    *)
      echo "未知参数: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
}

main "$@"
