#!/usr/bin/env bash
# Depth-position profile: runs after the env follow-up. Sweeps the scaling cut
# position at fixed c=0.0625, then summarizes accuracy vs depth position.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY=./.venv/bin/python
LOGDIR=outputs/_pool_logs_imp

echo "[pos] $(date) waiting for env follow-up ALL ENV DONE..."
while true; do
  if grep -q "ALL ENV DONE" outputs/_imp_env_followup.log 2>/dev/null; then break; fi
  sleep 120
done
echo "[pos] $(date) launching depth-position profile."

$PY scripts/gpu_pool_run.py --jobs configs/jobs_wave_pos.txt --gpus 0,1 --logdir "$LOGDIR" \
  > outputs/_pool_logs_imp_pos.log 2>&1
echo "[pos] $(date) position runs done; summarizing."

$PY - <<'PY' > outputs/imp_analysis/position_profile.csv 2>&1
import csv,glob,statistics as st
def bestfinal(name):
    best=[];final=[]
    for s in range(3):
        ms=sorted(glob.glob(f"outputs/{name}/**/seed_{s}/metrics.csv",recursive=True))
        if not ms: continue
        r=list(csv.DictReader(open(ms[-1])))
        if not r or max(int(float(x['epoch'])) for x in r)<79: continue
        a=[float(x['test_accuracy'])*100 for x in r]; e=[int(float(x['epoch'])) for x in r]
        best.append(max(a)); final.append(a[e.index(max(e))])
    return best,final
print("cut_position,coarse,n,best_mean,best_std")
coarse={2:'after_early',4:'after_middle',8:'after_late'}
b,_=bestfinal("imp_pos_baseline")
if b: print(f"baseline(c=1),-,{len(b)},{round(st.mean(b),2)},{round(st.stdev(b),2) if len(b)>1 else 0}")
for cut in [2,3,4,5,6,7,8]:
    b,_=bestfinal(f"imp_pos_cut{cut}")
    if not b: print(f"{cut},{coarse.get(cut,'')},0,,"); continue
    print(f"{cut},{coarse.get(cut,'')},{len(b)},{round(st.mean(b),2)},{round(st.stdev(b),2) if len(b)>1 else 0}")
PY
echo "[pos] $(date) ALL POS DONE."
cat outputs/imp_analysis/position_profile.csv
