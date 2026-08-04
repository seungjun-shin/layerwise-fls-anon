#!/usr/bin/env python
"""Wait for WP2 reconstruction and launch confirmation only after a clean gate."""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
STATE = WORKSPACE / "manifests" / "queue_state.json"
MANIFEST = WORKSPACE / "manifests" / "practical_main_jobs.json"
EXPECTED_RECONSTRUCTION_RUNS = 15
POLL_SECONDS = 20


def timestamp() -> str:
    """Return a concise local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main() -> None:
    """Launch the confirmatory block only after a clean reconstruction gate."""
    while True:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        jobs = [job for job in state["jobs"] if job["block"] == "reconstruction"]
        counts = Counter(job["status"] for job in jobs)
        print(f"[{timestamp()}] reconstruction {dict(counts)}", flush=True)
        if sum(counts[status] for status in ("pending", "running", "orphan_running")) == 0:
            break
        time.sleep(POLL_SECONDS)

    if len(jobs) != EXPECTED_RECONSTRUCTION_RUNS:
        raise SystemExit(
            f"Gate failed: expected {EXPECTED_RECONSTRUCTION_RUNS} reconstruction runs, "
            f"found {len(jobs)}."
        )
    failures = [job["id"] for job in jobs if job["status"] != "completed"]
    if failures:
        raise SystemExit(f"Gate failed: reconstruction has non-completed runs: {failures}")
    if not state.get("reconstruction_finished_at"):
        raise SystemExit("Gate failed: reconstruction queue did not record its completion marker.")

    command = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(WORKSPACE / "scripts" / "local_queue.py"),
        "--manifest",
        str(MANIFEST),
        "--state",
        str(STATE),
        "--gpus",
        "0,1,2,3",
        "--only-block",
        "confirmatory",
    ]
    print(f"[{timestamp()}] reconstruction gate passed; launching confirmatory block", flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
