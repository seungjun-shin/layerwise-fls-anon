#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: scripts/launch_practical_rescue_queue.sh GPU_INDEX}"
cd "$(dirname "$0")/.."

run_sweep() {
  local device="$1"
  shift
  CUDA_VISIBLE_DEVICES="$device" .venv/bin/python scripts/sweep.py "$@"
}

case "$gpu" in
  0)
    run_sweep 0 --config configs/experiment_79_practical_boundary_location_c_lr_rescue.yaml \
      sweep.varied_boundaries='[after_late]'
    ;;
  1)
    run_sweep 1 --config configs/experiment_79_practical_boundary_location_c_lr_rescue.yaml \
      sweep.varied_boundaries='[after_middle]'
    ;;
  2)
    run_sweep 2 --config configs/experiment_79_practical_boundary_location_c_lr_rescue.yaml \
      sweep.varied_boundaries='[after_early]'
    ;;
  3)
    run_sweep 3 --config configs/experiment_81_practical_global_c_lr_reference.yaml
    while pgrep -af 'scripts/sweep.py.*experiment_79_practical_boundary_location_c_lr_rescue|scripts/sweep.py.*experiment_81_practical_global_c_lr_reference|scripts/train.py.*experiment_79|scripts/train.py.*experiment_81' >/dev/null; do
      sleep 60
    done
    .venv/bin/python scripts/summarize_practical_rescue.py
    ;;
  *)
    echo "Unknown GPU index: $gpu" >&2
    exit 2
    ;;
esac
