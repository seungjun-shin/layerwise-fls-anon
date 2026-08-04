#!/usr/bin/env python
"""Wait for four free GPUs, pass the rescue pilot, then run frozen full queues."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
MANIFESTS = WORKSPACE / "manifests"
QUEUE = WORKSPACE / "scripts" / "generic_queue.py"
PYTHON = ROOT / ".venv" / "bin" / "python"
GPUS = "0,1,2,3"
FREE_THRESHOLD_MIB = 500
POLL_SECONDS = 30

RESCUE = (
    "practical_transfer_pilot2_jobs.json",
    "practical_transfer_pilot2",
    "0,1,2",
)
FULL_QUEUES = [
    ("practical_transfer_jobs.json", "practical_transfer", GPUS),
    ("head_norm_jobs.json", "head_norm", GPUS),
    ("vgg_depth_extension_jobs.json", "vgg_depth_extension", GPUS),
    ("scope_preservation_jobs.json", "scope_preservation", GPUS),
]


def gpu_memory_mib() -> dict[int, int]:
    """Return used memory for every visible physical GPU."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    values: dict[int, int] = {}
    for line in output.splitlines():
        index, used = (part.strip() for part in line.split(",", maxsplit=1))
        values[int(index)] = int(used)
    return values


def wait_for_all_gpus() -> None:
    """Block without reserving resources until GPUs 0--3 are all free."""
    while True:
        memory = gpu_memory_mib()
        ready = all(memory.get(index, FREE_THRESHOLD_MIB + 1) < FREE_THRESHOLD_MIB for index in range(4))
        print(f"GPU memory MiB: {memory}; all_free={ready}", flush=True)
        if ready:
            return
        time.sleep(POLL_SECONDS)


def queue_paths(stem: str) -> tuple[Path, Path, Path]:
    """Return isolated state, run-manifest, and failure-manifest paths."""
    return (
        MANIFESTS / f"{stem}_queue_state.json",
        MANIFESTS / f"{stem}_run_manifest.csv",
        MANIFESTS / f"{stem}_failed_runs.csv",
    )


def run_queue(manifest_name: str, stem: str, gpus: str) -> None:
    """Run one immutable manifest and require every job to complete."""
    state, runs, failures = queue_paths(stem)
    command = [
        str(PYTHON),
        str(QUEUE),
        "--manifest",
        str(MANIFESTS / manifest_name),
        "--state",
        str(state),
        "--run-manifest",
        str(runs),
        "--failed-manifest",
        str(failures),
        "--gpus",
        gpus,
    ]
    print(f"Starting {stem}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    payload = json.loads(state.read_text(encoding="utf-8"))
    failed = [job["id"] for job in payload["jobs"] if job["status"] != "completed"]
    if failed:
        raise RuntimeError(f"{stem} did not pass its frozen gate: {failed}")
    print(f"Completed {stem}: {len(payload['jobs'])} jobs", flush=True)


def main() -> None:
    """Enforce the user's all-GPU-free hold and every pilot/full dependency."""
    print("Holding all experiment launches until GPUs 0--3 are simultaneously free.", flush=True)
    wait_for_all_gpus()
    run_queue(*RESCUE)
    for queue_spec in FULL_QUEUES:
        wait_for_all_gpus()
        run_queue(*queue_spec)
    print("All frozen experiment-family training queues are terminal and valid.", flush=True)


if __name__ == "__main__":
    main()
