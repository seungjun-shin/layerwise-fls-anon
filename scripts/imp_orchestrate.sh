#!/usr/bin/env bash
# Experiment orchestrator. Chains: wait(wave1) -> wave2 -> wave3 ->
# NC diagnostics -> OOD eval -> aggregate. Uses GPUs 0,1 only.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY=./.venv/bin/python
LOGDIR=outputs/_pool_logs_imp
mkdir -p "$LOGDIR" outputs/imp_analysis
GPUS=0,1

echo "[orch] $(date) waiting for wave1 to finish..."
while true; do
  if grep -q "\[pool\] complete" outputs/_pool_logs_imp_wave1.log 2>/dev/null; then break; fi
  sleep 60
done
echo "[orch] $(date) wave1 done."

echo "[orch] $(date) launching wave2 (causal)..."
$PY scripts/gpu_pool_run.py --jobs configs/jobs_wave2_causal.txt --gpus $GPUS --logdir "$LOGDIR" \
  > outputs/_pool_logs_imp_wave2.log 2>&1
echo "[orch] $(date) wave2 done."

echo "[orch] $(date) launching wave3 (vgg n=5)..."
$PY scripts/gpu_pool_run.py --jobs configs/jobs_wave3_vgg.txt --gpus $GPUS --logdir "$LOGDIR" \
  > outputs/_pool_logs_imp_wave3.log 2>&1
echo "[orch] $(date) wave3 done."

echo "[orch] $(date) computing NC diagnostics on dose + causal checkpoints..."
for ckpt in $(find outputs/imp_dose_* outputs/imp_causal_* -name checkpoint_best.pt 2>/dev/null); do
  rundir=$(dirname "$ckpt")
  if [ -f "$rundir/diagnostics.csv" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 $PY scripts/compute_diagnostics.py --run-dir "$rundir" \
    --checkpoint checkpoint_best.pt --split test --max-samples 256 --sample-seed 2001 \
    >> "$LOGDIR/diagnostics.log" 2>&1 || echo "[orch] diag failed: $rundir"
done
echo "[orch] $(date) diagnostics done."

echo "[orch] $(date) OOD eval (dose-late curve + matched + practical)..."
for c in c1p0 c0p5 c0p25 c0p125 c0p0625 c0p03125 c0p015625; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py --experiment imp_dose_after_late_$c \
    --skip-corruption --device cuda:0 > "outputs/imp_analysis/ood_dose_$c.txt" 2>&1 \
    || echo "[orch] ood failed: $c"
done
# matched pair (paper) and practical pair, global-vs-late
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py \
  --experiment experiment_49_global_matched_seed_expansion \
  --experiment experiment_40_boundary_late_seed_expansion_after_late \
  --skip-corruption --device cuda:0 > outputs/imp_analysis/ood_matched.txt 2>&1 || echo "[orch] ood matched failed"
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py \
  --experiment experiment_52_practical_global_matched \
  --experiment experiment_53_practical_boundary_late_matched_after_late \
  --skip-corruption --device cuda:0 > outputs/imp_analysis/ood_practical.txt 2>&1 || echo "[orch] ood practical failed"
echo "[orch] $(date) OOD done."

echo "[orch] $(date) aggregating..."
$PY scripts/imp_analyze.py > outputs/imp_analysis/aggregate.log 2>&1 || echo "[orch] aggregate failed"
echo "[orch] $(date) ALL DONE."
