#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./script/data_process_run.sh [all|create|preprocess|--cfg job]
# Examples:
#   ./script/data_process_run.sh
#   ./script/data_process_run.sh create
#   ./script/data_process_run.sh preprocess
#   ./script/data_process_run.sh --cfg job

MODE="${1:-all}"

case "${MODE}" in
  all|create|preprocess)
    echo "Running data process with run_mode=${MODE}"
    python -m data_process.data_process_main "run_mode=${MODE}"
    ;;
  --cfg)
    if [[ "${2:-}" == "job" ]]; then
      echo "Displaying data process job configuration:"
      python -m data_process.data_process_main "run_mode=show_cfg"
    else
      echo "Usage: ./script/data_process_run.sh --cfg job" >&2
      exit 1
    fi
    ;;
  -h|--help|help)
    echo "Usage: ./script/data_process_run.sh [all|create|preprocess|--cfg job]"
    echo "  all         generate + preprocess (default)"
    echo "  create      generate only"
    echo "  preprocess  preprocess only"
    echo "  --cfg job   display job configuration"
    ;;
  *)
    echo "Invalid mode: ${MODE}" >&2
    echo "Usage: ./script/data_process_run.sh [all|create|preprocess|--cfg job]" >&2
    exit 1
    ;;
esac
