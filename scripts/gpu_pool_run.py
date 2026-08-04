#!/usr/bin/env python
"""Run a list of train.py jobs across a pool of GPUs.

Each line of the jobs file is one job: a config path followed by override tokens,
e.g.

    configs/experiment_75_boundary_late_lr_only_ablation.yaml training.seed=0 \
        training.location_lr_compensation_multipliers.after_late=0.25 \
        experiment.name=experiment_88_lr_only_decoupled_k4

Blank lines and lines starting with '#' are ignored. Jobs are distributed over
the GPUs listed in --gpus, one job per GPU at a time, refilling as jobs finish.
Per-job stdout/stderr is written to <logdir>/<name>.log. A job whose
experiment.name already has a seed_<seed> metrics.csv reaching max_epochs-1 is
skipped (resumable).
"""
from __future__ import annotations

import argparse
import csv
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, help="Path to jobs file (one job per line).")
    parser.add_argument("--gpus", default="0,1,2,3", help="Comma-separated GPU ids.")
    parser.add_argument("--logdir", default="outputs/_pool_logs", help="Directory for per-job logs.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip already-complete runs.")
    return parser.parse_args()


def tokens_to_field(tokens: list[str], key: str) -> str | None:
    for tok in tokens:
        if tok.startswith(f"{key}="):
            return tok.split("=", 1)[1]
    return None


def last_epoch(metrics: Path) -> int | None:
    if not metrics.exists():
        return None
    with metrics.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(int(float(r.get("epoch", -1) or -1)) for r in rows)


def already_done(tokens: list[str]) -> bool:
    """Best-effort completion check by experiment.name + seed + max_epochs."""
    name = tokens_to_field(tokens, "experiment.name")
    seed = tokens_to_field(tokens, "training.seed") or "0"
    max_epochs = tokens_to_field(tokens, "training.max_epochs")
    if name is None:
        return False
    target = int(max_epochs) if max_epochs is not None else None
    for metrics in Path("outputs").glob(f"{name}/**/seed_{seed}/metrics.csv"):
        le = last_epoch(metrics)
        if le is None:
            continue
        if target is None or le >= target - 1:
            resolved = metrics.parent / "resolved_config.yaml"
            if resolved.exists():
                config = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
                final_once = bool((config.get("training") or {}).get("evaluate_test_at_end", False))
                if final_once and not (metrics.parent / "final_evaluation.json").exists():
                    continue
            return True
    return False


def load_jobs(path: str) -> list[list[str]]:
    jobs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        jobs.append(line.split())
    return jobs


def worker(gpu: str, jobq: queue.Queue[list[str]], args, results: list, lock: threading.Lock) -> None:
    while True:
        try:
            tokens = jobq.get_nowait()
        except queue.Empty:
            return
        name = tokens_to_field(tokens, "experiment.name") or "job"
        seed = tokens_to_field(tokens, "training.seed") or "0"
        tag = f"{name}__seed{seed}"
        if not args.no_skip and already_done(tokens):
            with lock:
                print(f"[gpu{gpu}] SKIP {tag} (complete)", flush=True)
                results.append((tag, "skipped"))
            jobq.task_done()
            continue
        config = tokens[0]
        overrides = tokens[1:]
        log_path = Path(args.logdir) / f"{tag}.log"
        cmd = [args.python, "scripts/train.py", "--config", config, *overrides]
        env = {"CUDA_VISIBLE_DEVICES": gpu}
        import os

        full_env = {**os.environ, **env}
        t0 = time.time()
        with lock:
            print(f"[gpu{gpu}] START {tag}", flush=True)
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=full_env)
        dt = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        with lock:
            print(f"[gpu{gpu}] DONE  {tag}  {status}  {dt/60:.1f}min", flush=True)
            results.append((tag, status))
        jobq.task_done()


def main() -> None:
    args = parse_args()
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    jobs = load_jobs(args.jobs)
    print(f"[pool] {len(jobs)} jobs over GPUs {gpus}", flush=True)
    jobq: queue.Queue[list[str]] = queue.Queue()
    for j in jobs:
        jobq.put(j)
    results: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g, jobq, args, results, lock)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = sum(1 for _, s in results if s == "ok")
    skip = sum(1 for _, s in results if s == "skipped")
    fail = [t for t, s in results if s.startswith("FAIL")]
    print(f"[pool] complete: {ok} ok, {skip} skipped, {len(fail)} failed", flush=True)
    if fail:
        print("[pool] failed jobs:", *fail, sep="\n  ", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
