#!/usr/bin/env python
"""Generate the unified protocol table and pseudocode from frozen configs."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
CONFIG_ROOT = WORKSPACE / "configs" / "practical_main"
CONDITIONS = {
    "P-REF": CONFIG_ROOT / "p_ref.yaml",
    "P-ACT-FINAL": CONFIG_ROOT / "p_act_final.yaml",
    "P-LR-FINAL": CONFIG_ROOT / "p_lr_final.yaml",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lr_allocation(config: dict[str, Any]) -> tuple[float, float]:
    base_lr = float(config["training"]["lr"])
    fls = config["fls"]
    if fls["mode"] == "global":
        multiplier = float(fls["global"]["output_multiplier"])
        effective = base_lr / multiplier if fls["global"].get("lr_compensation") else base_lr
        return effective, effective
    override = config["training"].get("location_lr_compensation_multipliers")
    source = override or fls["boundaries"]
    value = source["after_late"]
    if isinstance(value, dict):
        value = value["output_multiplier"]
    return base_lr / float(value), base_lr


def activation_value(config: dict[str, Any]) -> float:
    fls = config["fls"]
    if fls["mode"] == "global":
        return float(fls["global"]["output_multiplier"])
    return float(fls["boundaries"]["after_late"]["output_multiplier"])


def protocol_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition, path in CONDITIONS.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        upstream, downstream = lr_allocation(config)
        rows.append(
            {
                "condition": condition,
                "architecture": config["model"]["name"],
                "dataset": config["data"]["name"],
                "recipe": "practical_cosine",
                "cut_definition": (
                    "global logits" if config["fls"]["mode"] == "global" else "after_late before pool/head"
                ),
                "activation_multiplier": activation_value(config),
                "upstream_lr_initial": upstream,
                "downstream_lr_initial": downstream,
                "augmentation": config["data"]["augment"],
                "momentum": config["training"]["momentum"],
                "weight_decay": config["training"]["weight_decay"],
                "scheduler": config["training"]["lr_scheduler"],
                "selection_split": "fixed 10% validation, seed 0",
                "checkpoint_rule": "maximum validation accuracy; earliest tie",
                "seed_set": "5--14 confirmatory; 0--4 separate reconstruction",
                "diagnostic_subset": "first 512 examples of ordered fixed validation subset",
                "evidence_tier": "Tier 1 H1/H2" if condition != "P-REF" else "Tier 2 reference",
                "config_path": str(path.relative_to(ROOT)),
            }
        )
    return rows


def csv_text(rows: list[dict[str, Any]]) -> str:
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def latex_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llllrrrl}",
        r"\toprule",
        r"Condition & Architecture & Cut & $c$ & Upstream LR & Head LR & Scheduler & Evidence \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['condition']} & ResNet18-CIFAR & {row['cut_definition']} & "
            f"{row['activation_multiplier']:g} & {row['upstream_lr_initial']:.8g} & "
            f"{row['downstream_lr_initial']:.8g} & cosine & {row['evidence_tier']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def pseudocode_text() -> str:
    return """# Paper Protocol Pseudocode

## Boundary activation scaling with upstream compensation

```text
input: model body B, classifier H, final cut k, activation multiplier c,
       base learning rate eta
U <- parameters at or upstream of k
D <- parameters downstream of k (the classifier for the final cut)
z_pre <- B(x)
z_post <- c * z_pre
logits <- H(z_post)
optimizer LR(U) <- eta / c
optimizer LR(D) <- eta
```

The multiplier is an operational intervention at a declared boundary; it is not
asserted to be an intrinsic layer-wise FLS.

## Exact same-cut learning-rate-only control

```text
input: the same B, H, k, c, eta, seed, split, and data order
U, D <- the identical parameter sets used by the activation condition
z_pre <- B(x)
z_post <- z_pre                 # identity activation; no scaling
logits <- H(z_post)
optimizer LR(U) <- eta / c      # copied exactly from activation condition
optimizer LR(D) <- eta
assert parameter_group_checksum(ACT) == parameter_group_checksum(LR_ONLY)
```

## Validation-only selection and one-time test reporting

```text
for epoch in 0..79:
    train on the training subset
    evaluate only the fixed deterministic validation subset
    if validation accuracy strictly exceeds the previous maximum:
        freeze checkpoint_best
assert no test dataset/loader was constructed during training
after all planned seeds are terminal:
    validate run stability, pairing, hashes, and checkpoint rule
    atomically record TEST_EVALUATION_STARTED(checkpoint_hash)
    evaluate checkpoint_best on test exactly once
    write immutable test_metrics_once.json
```
"""


def write_versioned(directory: Path, stem: str, suffix: str, content: str, timestamp: str) -> list[Path]:
    latest = directory / f"{stem}{suffix}"
    versioned = directory / f"{stem}_{timestamp}{suffix}"
    latest.write_text(content, encoding="utf-8")
    versioned.write_text(content, encoding="utf-8")
    return [latest, versioned]


def main() -> None:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    rows = protocol_rows()
    outputs: list[Path] = []
    outputs += write_versioned(WORKSPACE / "tables", "unified_protocol_table", ".csv", csv_text(rows), timestamp)
    outputs += write_versioned(WORKSPACE / "tables", "unified_protocol_table", ".tex", latex_text(rows), timestamp)
    outputs += write_versioned(WORKSPACE / "reports", "protocol_pseudocode", ".md", pseudocode_text(), timestamp)
    manifest = WORKSPACE / "manifests" / "artifact_manifest.csv"
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for path in outputs:
            writer.writerow(
                [
                    f"protocol_{path.stem}",
                    str(path.relative_to(ROOT)),
                    path.suffix.lstrip("."),
                    ";".join(str(path.relative_to(ROOT)) for path in CONDITIONS.values()),
                    sha256(path),
                    datetime.now().astimezone().isoformat(timespec="seconds"),
                    str(Path(__file__).relative_to(ROOT)),
                    "Generated directly from frozen YAML configs.",
                ]
            )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
