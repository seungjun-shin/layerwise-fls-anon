#!/usr/bin/env bash
# Resume orchestrator (no inter-wave signal waiting). One pool for all remaining
# jobs (causal leftover + vgg + env + pos; already-done runs are skipped), then
# NC diagnostics -> OOD eval -> aggregate -> position profile -> envelope plot.
# GPUs 0,1 only.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY=./.venv/bin/python
LOGDIR=outputs/_pool_logs_imp
mkdir -p "$LOGDIR" outputs/imp_analysis
GPUS=0,1

echo "[resume] $(date) launching combined pool (causal+vgg+env+pos, done runs skipped)..."
$PY scripts/gpu_pool_run.py --jobs configs/jobs_resume_all.txt --gpus $GPUS --logdir "$LOGDIR" \
  > outputs/_pool_logs_imp_resume.log 2>&1
echo "[resume] $(date) combined pool done."

echo "[resume] $(date) computing NC diagnostics on dose + causal checkpoints..."
for ckpt in $(find outputs/imp_dose_* outputs/imp_causal_* -name checkpoint_best.pt 2>/dev/null); do
  rundir=$(dirname "$ckpt")
  if [ -f "$rundir/diagnostics.csv" ]; then continue; fi
  CUDA_VISIBLE_DEVICES=0 $PY scripts/compute_diagnostics.py --run-dir "$rundir" \
    --checkpoint checkpoint_best.pt --split test --max-samples 256 --sample-seed 2001 \
    >> "$LOGDIR/diagnostics.log" 2>&1 || echo "[resume] diag failed: $rundir"
done
echo "[resume] $(date) diagnostics done."

echo "[resume] $(date) OOD eval (dose-late curve + matched + practical)..."
for c in c1p0 c0p5 c0p25 c0p125 c0p0625 c0p03125 c0p015625; do
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py --experiment imp_dose_after_late_$c \
    --skip-corruption --device cuda:0 > "outputs/imp_analysis/ood_dose_$c.txt" 2>&1 \
    || echo "[resume] ood failed: $c"
done
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py \
  --experiment experiment_49_global_matched_seed_expansion \
  --experiment experiment_40_boundary_late_seed_expansion_after_late \
  --skip-corruption --device cuda:0 > outputs/imp_analysis/ood_matched.txt 2>&1 || echo "[resume] ood matched failed"
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_robustness.py \
  --experiment experiment_52_practical_global_matched \
  --experiment experiment_53_practical_boundary_late_matched_after_late \
  --skip-corruption --device cuda:0 > outputs/imp_analysis/ood_practical.txt 2>&1 || echo "[resume] ood practical failed"
echo "[resume] $(date) OOD done."

echo "[resume] $(date) aggregating accuracy + NC..."
$PY scripts/imp_analyze.py > outputs/imp_analysis/aggregate.log 2>&1 || echo "[resume] aggregate failed"

echo "[resume] $(date) building depth-position profile..."
$PY - <<'PY' > outputs/imp_analysis/position_profile.csv 2>&1
import csv,glob,statistics as st
def bestfinal(name):
    best=[]
    for s in range(3):
        ms=sorted(glob.glob(f"outputs/{name}/**/seed_{s}/metrics.csv",recursive=True))
        if not ms: continue
        r=list(csv.DictReader(open(ms[-1])))
        if not r or max(int(float(x['epoch'])) for x in r)<79: continue
        a=[float(x['test_accuracy'])*100 for x in r]
        best.append(max(a))
    return best
print("cut_position,coarse,n,best_mean,best_std")
coarse={2:'after_early',4:'after_middle',8:'after_late'}
b=bestfinal("imp_pos_baseline")
if b: print(f"baseline(c=1),-,{len(b)},{round(st.mean(b),2)},{round(st.stdev(b),2) if len(b)>1 else 0}")
for cut in [2,3,4,5,6,7,8]:
    b=bestfinal(f"imp_pos_cut{cut}")
    if not b: print(f"{cut},{coarse.get(cut,'')},0,,"); continue
    print(f"{cut},{coarse.get(cut,'')},{len(b)},{round(st.mean(b),2)},{round(st.stdev(b),2) if len(b)>1 else 0}")
PY
echo "[resume] $(date) position profile:" && cat outputs/imp_analysis/position_profile.csv

echo "[resume] $(date) rendering LR-envelope figure/table..."
$PY scripts/plot_lr_envelope.py > "$LOGDIR/envelope.log" 2>&1 || echo "[resume] envelope plot failed"

echo "[resume] $(date) ALL DONE."
