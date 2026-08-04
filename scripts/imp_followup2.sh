#!/usr/bin/env bash
# Follow-up experiment wave. Runs the causal 3x3 factorial and the
# depth-profile n=5 extension, then materializes a factorial summary and an
# isotonic/ordered-trend analysis of the depth profile. GPUs taken from $1 (default 0).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
PY=./.venv/bin/python
GPUS="${1:-0}"
LOGDIR=outputs/_pool_logs_imp2
mkdir -p "$LOGDIR" outputs/imp_analysis

echo "[fu2] $(date) factorial + depth-n5 pool on GPUs $GPUS ..."
cat configs/jobs_wave_fact.txt configs/jobs_wave_posN5.txt > configs/jobs_followup2.txt
$PY scripts/gpu_pool_run.py --jobs configs/jobs_followup2.txt --gpus "$GPUS" --logdir "$LOGDIR" \
  > outputs/_pool_logs_imp2.log 2>&1
echo "[fu2] $(date) pool done; analyzing."

# Causal 3x3 factorial: upstream-LR(m) x activation(a) best-acc grid (reuses prior cells).
$PY - <<'PY' > outputs/imp_analysis/causal_factorial.csv 2>&1
import csv,glob,statistics as st
def best(name):
    a=[]
    for m in glob.glob(f"outputs/{name}/**/seed_*/metrics.csv",recursive=True):
        r=list(csv.DictReader(open(m)))
        if not r or max(int(float(x['epoch'])) for x in r)<79: continue
        a.append(max(float(x['test_accuracy']) for x in r)*100)
    return (round(st.mean(a),2),len(a)) if a else (None,0)
# map (m,a)->experiment name; reuse done cells
cells={
 (1.0,1.0):"imp_dose_after_late_c1p0",(1.0,0.25):"imp_fact_m1p0_a0p25",(1.0,0.0625):"imp_causal_actonly",
 (0.25,1.0):"imp_fact_m0p25_a1p0",(0.25,0.25):"imp_fact_m0p25_a0p25",(0.25,0.0625):"imp_fact_m0p25_a0p0625",
 (0.0625,1.0):"imp_causal_fixlr_act1p0",(0.0625,0.25):"imp_causal_fixlr_act0p25",(0.0625,0.0625):"imp_causal_fixlr_act0p0625"}
print("m_lrcomp(upstreamLR=base/m),a_activation,best_acc,n")
grid={}
for (m,a),nm in cells.items():
    v,n=best(nm); grid[(m,a)]=v
    print(f"{m},{a},{v},{n}")
# crude main effects (row/col means where available)
ms=sorted({m for m,_ in cells}); as_=sorted({a for _,a in cells})
print("# row means (fix m, avg over a):")
for m in ms:
    vs=[grid[(m,a)] for a in as_ if grid.get((m,a)) is not None]
    if vs: print(f"#  m={m}: {round(sum(vs)/len(vs),2)}")
print("# col means (fix a, avg over m):")
for a in as_:
    vs=[grid[(m,a)] for m in ms if grid.get((m,a)) is not None]
    if vs: print(f"#  a={a}: {round(sum(vs)/len(vs),2)}")
PY
echo "[fu2] factorial:"; cat outputs/imp_analysis/causal_factorial.csv

# Depth profile n=5 + ordered-trend stats
$PY scripts/plot_depth_profile.py > "$LOGDIR/depth_profile_n5.log" 2>&1 || echo "[fu2] depth plot failed"
$PY - <<'PY' >> outputs/imp_analysis/position_profile_n5.csv 2>&1
import csv,glob,statistics as st
def accs(name):
    out=[]
    for m in glob.glob(f"outputs/{name}/**/seed_*/metrics.csv",recursive=True):
        r=list(csv.DictReader(open(m)))
        if not r or max(int(float(x['epoch'])) for x in r)<79: continue
        out.append(max(float(x['test_accuracy']) for x in r)*100)
    return out
cuts=[2,3,4,5,6,7,8]; means=[]
print("cut,n,mean,std")
for c in cuts:
    a=accs(f"imp_pos_cut{c}");
    if a: means.append(st.mean(a)); print(f"{c},{len(a)},{round(st.mean(a),2)},{round(st.pstdev(a),2) if len(a)>1 else 0}")
# Spearman + adjacent violations (no scipy dependency)
def spearman(x,y):
    def rank(v):
        s=sorted(range(len(v)),key=lambda i:v[i]); r=[0]*len(v)
        for i,idx in enumerate(s): r[idx]=i+1
        return r
    rx,ry=rank(x),rank(y); n=len(x)
    d2=sum((a-b)**2 for a,b in zip(rx,ry)); return 1-6*d2/(n*(n*n-1))
if len(means)==len(cuts):
    viol=sum(1 for i in range(1,len(means)) if means[i]<means[i-1])
    print(f"# spearman_rho={round(spearman(cuts,means),3)}, adjacent_violations={viol}/{len(cuts)-1}, slope_endpoints={round((means[-1]-means[0])/(cuts[-1]-cuts[0]),3)}pp/cut")
PY
echo "[fu2] depth n5 trend:"; cat outputs/imp_analysis/position_profile_n5.csv
echo "[fu2] $(date) FU2 DONE."
