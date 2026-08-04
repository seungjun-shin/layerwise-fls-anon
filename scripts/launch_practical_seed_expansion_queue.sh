#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: scripts/launch_practical_seed_expansion_queue.sh GPU_INDEX}"
cd "$(dirname "$0")/.."

run_sweep() {
  local device="$1"
  shift
  CUDA_VISIBLE_DEVICES="$device" .venv/bin/python scripts/sweep.py "$@"
}

case "$gpu" in
  0)
    run_sweep 0 --config configs/experiment_85_practical_global_seed_expansion.yaml \
      experiment.name=experiment_85_practical_global_seed_expansion_c_1 \
      sweep.global_output_multipliers='[1.0]'
    ;;
  1)
    run_sweep 1 --config configs/experiment_85_practical_global_seed_expansion.yaml \
      experiment.name=experiment_85_practical_global_seed_expansion_c_1p25 \
      sweep.global_output_multipliers='[1.25]'
    ;;
  2)
    run_sweep 2 --config configs/experiment_86_practical_late_seed_expansion.yaml \
      sweep.output_multipliers='[1.25]'
    ;;
  3)
    run_sweep 3 --config configs/experiment_86_practical_late_seed_expansion.yaml \
      sweep.output_multipliers='[1.5]'
    run_sweep 3 --config configs/experiment_87_practical_early_seed_expansion.yaml
    while pgrep -af 'scripts/sweep.py.*experiment_85_practical_global_seed_expansion|scripts/sweep.py.*experiment_86_practical_late_seed_expansion|scripts/sweep.py.*experiment_87_practical_early_seed_expansion|scripts/train.py.*experiment_85|scripts/train.py.*experiment_86|scripts/train.py.*experiment_87' >/dev/null; do
      sleep 60
    done
    .venv/bin/python scripts/summarize_practical_seed_expansion.py
    ;;
  *)
    echo "Unknown GPU index: $gpu" >&2
    exit 2
    ;;
esac
