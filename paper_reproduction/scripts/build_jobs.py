#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIGS = WORKSPACE / "configs" / "practical_main"
CONDITIONS = {
    "P-REF": CONFIGS / "p_ref.yaml",
    "P-ACT-FINAL": CONFIGS / "p_act_final.yaml",
    "P-LR-FINAL": CONFIGS / "p_lr_final.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen practical-run queue.")
    parser.add_argument("--output", default=str(WORKSPACE / "manifests" / "practical_main_jobs.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs: list[dict] = []
    for block, seeds in (("reconstruction", range(0, 5)), ("confirmatory", range(5, 15))):
        for seed in seeds:
            for condition, config in CONDITIONS.items():
                jobs.append(
                    {
                        "id": f"{block}__{condition.lower().replace('-', '_')}__seed{seed}",
                        "work_package": "WP2",
                        "block": block,
                        "condition": condition,
                        "seed": seed,
                        "config": str(config.relative_to(ROOT)),
                        "overrides": [f"training.seed={seed}"],
                        "status": "pending",
                        "attempts": 0,
                    }
                )
    payload = {
        "project": "paper_practical_main",
        "created_before_test_evaluation": True,
        "max_parallel": 4,
        "gpus": [0, 1, 2, 3],
        "oom_retry": {"max_attempts": 2, "delay_seconds": 30},
        "jobs": jobs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
