#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: scripts/launch_practical_narrow_queue.sh GPU_INDEX}"
cd "$(dirname "$0")/.."

run_sweep() {
  local device="$1"
  shift
  CUDA_VISIBLE_DEVICES="$device" .venv/bin/python scripts/sweep.py "$@"
}

case "$gpu" in
  0)
    run_sweep 0 --config configs/experiment_82_practical_early_narrow_c_lr.yaml \
      sweep.output_multipliers='[0.375,0.5]'
    ;;
  1)
    run_sweep 1 --config configs/experiment_82_practical_early_narrow_c_lr.yaml \
      sweep.output_multipliers='[0.625,0.75]'
    ;;
  2)
    run_sweep 2 --config configs/experiment_83_practical_global_narrow_c_lr.yaml
    ;;
  3)
    run_sweep 3 --config configs/experiment_84_practical_middle_late_identity_neighborhood.yaml
    while pgrep -af 'scripts/sweep.py.*experiment_82_practical_early_narrow_c_lr|scripts/sweep.py.*experiment_83_practical_global_narrow_c_lr|scripts/sweep.py.*experiment_84_practical_middle_late_identity_neighborhood|scripts/train.py.*experiment_82|scripts/train.py.*experiment_83|scripts/train.py.*experiment_84' >/dev/null; do
      sleep 60
    done
    .venv/bin/python scripts/summarize_practical_narrow.py
    ;;
  *)
    echo "Unknown GPU index: $gpu" >&2
    exit 2
    ;;
esac
