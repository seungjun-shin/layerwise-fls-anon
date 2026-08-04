#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?usage: scripts/launch_confound4way_queue.sh GPU_INDEX}"
cd "$(dirname "$0")/.."

run_sweep() {
  local device="$1"
  shift
  CUDA_VISIBLE_DEVICES="$device" .venv/bin/python scripts/sweep.py "$@"
}

case "$gpu" in
  0)
    run_sweep 0 --config configs/experiment_75_boundary_late_lr_only_ablation.yaml sweep.seeds='[0,1]'
    ;;
  1)
    run_sweep 1 --config configs/experiment_75_boundary_late_lr_only_ablation.yaml sweep.seeds='[2,3]'
    ;;
  2)
    run_sweep 2 --config configs/experiment_75_boundary_late_lr_only_ablation.yaml sweep.seeds='[4]'
    run_sweep 2 --config configs/experiment_63_boundary_late_no_comp_ablation.yaml sweep.seeds='[3]'
    ;;
  3)
    run_sweep 3 --config configs/experiment_63_boundary_late_no_comp_ablation.yaml sweep.seeds='[4]'
    while pgrep -af 'scripts/sweep.py.*experiment_75_boundary_late_lr_only_ablation|scripts/sweep.py.*experiment_63_boundary_late_no_comp_ablation|scripts/train.py.*experiment_75|scripts/train.py.*experiment_63' >/dev/null; do
      sleep 60
    done
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run_boundary_protocol_diagnostics.py \
      --preset confound4way \
      --checkpoint checkpoint_best.pt \
      --device cuda \
      --output-suffix confound4way \
      --force \
      --outputs-root outputs \
      --max-samples 256 \
      --sample-seed 2001 \
      --batch-size 128
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/compute_update_pressure_diagnostics.py \
      --preset confound4way \
      --outputs-root outputs \
      --checkpoint checkpoint_best.pt \
      --split train \
      --max-samples 256 \
      --sample-seed 2001 \
      --batch-size 128 \
      --device cuda \
      --output-dir outputs/update_pressure_confound4way
    .venv/bin/python scripts/summarize_confound4way.py \
      --outputs-root outputs \
      --diagnostics-file diagnostics_confound4way.csv \
      --output-dir outputs/confound4way_summary
    ;;
  *)
    echo "Unknown GPU index: $gpu" >&2
    exit 2
    ;;
esac
