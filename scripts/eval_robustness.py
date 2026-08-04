#!/usr/bin/env python
"""Corruption-robustness (CIFAR-100-C) and OOD-detection evaluation.

This CLI consumes EXISTING trained checkpoints (it does not train anything) and
reports, for one or more experiment conditions:

1. **Corruption robustness (CIFAR-100-C).**  For each of the 15 standard
   corruptions x 5 severities it loads the official ``<corruption>.npy`` array
   ([50000, 32, 32, 3] = 5 severities x 10000 images) and ``labels.npy``,
   applies the model's training-time CIFAR-100 normalization, and computes test
   accuracy.  It reports mean corruption accuracy (mCA) and per-severity means.

2. **OOD detection.**  Using CIFAR-100 test as in-distribution and SVHN test as
   OOD, it computes MSP / energy / Mahalanobis AUROCs via
   ``fls.diagnostics.ood``.

Results are written per seed to ``outputs/robustness/<experiment>/seed_<seed>.json``
and aggregated (mean +/- std over seeds) into ``summary.csv``.  When two
experiments are supplied (``--experiment`` twice, or via ``--compare``), a
comparison of *the second minus the first* (intended as late-boundary minus
global) is printed for mCA and the OOD AUROCs.

Default device is CPU to avoid contention with running GPU jobs; pass
``--device cuda`` only if a GPU is idle.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from fls.data.datasets import build_dataset, dataset_stats
from fls.diagnostics.ood import compute_ood_auroc
from fls.models.registry import build_model
from fls.scaling.blockwise_fls import apply_blockwise_scaling
from fls.scaling.global_fls import GlobalOutputMultiplier
from fls.scaling.location_fls import apply_location_scaling
from fls.scaling.profiles import build_profile
from fls.training.seed import set_seed
from fls.utils.device import resolve_device

REGIONS = ["early", "middle", "late", "head"]

# The 15 standard CIFAR-C corruptions (excludes the 4 "extra"/validation ones:
# speckle_noise, gaussian_blur, spatter, saturate).
STANDARD_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]

CIFAR100C_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
NUM_SEVERITIES = 5
IMAGES_PER_SEVERITY = 10000


# ---------------------------------------------------------------------------
# Checkpoint / model loading (mirrors scripts/compute_diagnostics.py).
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_scaled_model(config: dict[str, Any]) -> torch.nn.Module:
    """Rebuild the scaled model exactly as during training.

    This is the canonical pattern from ``scripts/compute_diagnostics.py``:
    build the base model from the registry, then wrap/scale it according to the
    ``fls`` config block.
    """
    model = build_model(config)
    fls_cfg = config["fls"]
    mode = fls_cfg["mode"]
    if mode == "global":
        return GlobalOutputMultiplier(model, fls_cfg.get("global", {}).get("output_multiplier", 1.0))
    if mode == "location":
        return apply_location_scaling(model, fls_cfg.get("boundaries") or {})
    if mode == "position":
        from fls.scaling.position_fls import apply_position_scaling

        return apply_position_scaling(
            model,
            int(fls_cfg.get("position", 0)),
            float(fls_cfg.get("output_multiplier", 1.0)),
        )
    if mode == "blockwise":
        profile = fls_cfg.get("groups")
        if profile is None:
            profile = build_profile(
                fls_cfg.get("profile_name", "progressive_increasing"),
                REGIONS,
                fls_cfg.get("base_values"),
                seed=int(config["training"]["seed"]),
            )
            fls_cfg["groups"] = profile
        return apply_blockwise_scaling(model, profile)
    raise ValueError(f"Unknown FLS mode: {mode}")


def load_model_from_run(run_dir: Path, checkpoint_name: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the trained, scaled model for a single seed run."""
    config = load_yaml(run_dir / "resolved_config.yaml")
    set_seed(int(config["training"]["seed"]))
    model = build_scaled_model(config).to(device)
    checkpoint_path = run_dir / checkpoint_name
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def find_seed_runs(experiment: str, seeds: list[int] | None) -> dict[int, Path]:
    """Map seed -> run directory for an experiment.

    Experiment directories live at ``outputs/<experiment>*/<timestamp>/seed_<seed>``.
    The ``*`` allows for the recorded directory carrying a descriptive suffix
    (e.g. ``..._seed_expansion_after_late``).  When multiple timestamps exist
    for a seed the most recent (lexicographically largest) is used.
    """
    outputs = Path("outputs")
    candidates = sorted(outputs.glob(f"{experiment}*"))
    # Prefer an exact-name match if present; otherwise accept prefix matches.
    exact = [c for c in candidates if c.name == experiment]
    search_dirs = exact if exact else candidates
    found: dict[int, Path] = {}
    for exp_dir in search_dirs:
        if not exp_dir.is_dir():
            continue
        for seed_dir in sorted(exp_dir.glob("*/seed_*")):
            if not (seed_dir / "checkpoint_best.pt").exists():
                continue
            try:
                seed = int(seed_dir.name.split("_")[-1])
            except ValueError:
                continue
            if seeds is not None and seed not in seeds:
                continue
            # Keep the latest timestamp for each seed.
            if seed not in found or str(seed_dir) > str(found[seed]):
                found[seed] = seed_dir
    return dict(sorted(found.items()))


def find_seed_runs_from_table(
    seed_table: Path, condition: str, seeds: list[int] | None
) -> dict[int, Path]:
    """Resolve frozen checkpoint directories from a seed table.

    The table is the source of truth for confirmatory analyses whose checkpoints
    are archived outside ``outputs/``. Relative ``run_dir`` entries are resolved
    against the repository working directory.
    """
    with seed_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    requested = set(seeds) if seeds is not None else None
    found: dict[int, Path] = {}
    for row in rows:
        if row.get("condition") != condition:
            continue
        seed = int(row["seed"])
        if requested is not None and seed not in requested:
            continue
        run_dir = Path(row["run_dir"])
        if not run_dir.is_absolute():
            run_dir = Path.cwd() / run_dir
        if not (run_dir / "checkpoint_best.pt").exists():
            raise FileNotFoundError(f"Missing checkpoint for {condition}, seed {seed}: {run_dir}")
        if seed in found and found[seed] != run_dir:
            raise ValueError(f"Duplicate run directories for {condition}, seed {seed}")
        found[seed] = run_dir
    if requested is not None:
        missing = sorted(requested - set(found))
        if missing:
            raise ValueError(f"Missing {condition} rows for seeds: {missing}")
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------
# CIFAR-100-C.
# ---------------------------------------------------------------------------
def ensure_cifar100c(data_root: Path) -> Path:
    """Return the CIFAR-100-C directory, downloading + extracting if absent."""
    c_dir = data_root / "CIFAR-100-C"
    if (c_dir / "labels.npy").exists():
        return c_dir
    data_root.mkdir(parents=True, exist_ok=True)
    archive = data_root / "CIFAR-100-C.tar"
    if not archive.exists():
        print(f"Downloading CIFAR-100-C from {CIFAR100C_URL} (~2.9GB)...")
        urllib.request.urlretrieve(CIFAR100C_URL, archive)
    print(f"Extracting {archive}...")
    with tarfile.open(archive) as tar:
        tar.extractall(data_root)
    if not (c_dir / "labels.npy").exists():
        raise FileNotFoundError(f"CIFAR-100-C extraction did not produce {c_dir / 'labels.npy'}")
    return c_dir


def _normalize_uint8_images(images: np.ndarray, mean: tuple, std: tuple) -> torch.Tensor:
    """Convert HWC uint8 [0,255] images to a normalized CHW float tensor.

    This reproduces ``transforms.ToTensor() + Normalize(mean, std)``:
    ToTensor scales to [0,1] and reorders to CHW; Normalize subtracts the
    per-channel mean and divides by the per-channel std.
    """
    tensor = torch.from_numpy(images.astype(np.float32) / 255.0)  # (N, H, W, C)
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (N, C, H, W)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


@torch.no_grad()
def _accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        correct += int((preds == y).sum().item())
        total += y.numel()
    return correct / max(1, total)


def evaluate_cifar100c(
    model: torch.nn.Module,
    config: dict[str, Any],
    c_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Compute per-corruption, per-severity accuracy and aggregate mCA.

    The normalization statistics are taken from the dataset stats keyed by the
    model's training dataset name (CIFAR-100), and only applied when the run was
    trained with ``normalize: true`` (which the CIFAR runs are).
    """
    data_name = config["data"]["name"].lower()
    stats = dataset_stats(data_name)
    normalize = bool(config["data"].get("normalize", False))
    mean = stats["mean"] if normalize else (0.0, 0.0, 0.0)
    std = stats["std"] if normalize else (1.0, 1.0, 1.0)

    labels = np.load(c_dir / "labels.npy")  # (50000,)
    labels_t = torch.from_numpy(labels.astype(np.int64))

    per_corruption: dict[str, list[float]] = {}
    # severity index (1..5) -> list of accuracies across corruptions
    per_severity: dict[int, list[float]] = {s: [] for s in range(1, NUM_SEVERITIES + 1)}

    for corruption in STANDARD_CORRUPTIONS:
        npy_path = c_dir / f"{corruption}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing corruption file: {npy_path}")
        images = np.load(npy_path)  # (50000, 32, 32, 3) uint8
        sev_accs: list[float] = []
        for sev in range(1, NUM_SEVERITIES + 1):
            lo = (sev - 1) * IMAGES_PER_SEVERITY
            hi = sev * IMAGES_PER_SEVERITY
            x = _normalize_uint8_images(images[lo:hi], mean, std)
            y = labels_t[lo:hi]
            loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)
            acc = _accuracy(model, loader, device)
            sev_accs.append(acc)
            per_severity[sev].append(acc)
        per_corruption[corruption] = sev_accs

    # mCA = mean over all corruptions and severities.
    all_accs = [a for accs in per_corruption.values() for a in accs]
    mca = float(np.mean(all_accs)) if all_accs else 0.0
    per_severity_mean = {s: float(np.mean(v)) if v else 0.0 for s, v in per_severity.items()}

    return {
        "mCA": mca,
        "per_corruption": per_corruption,
        "per_corruption_mean": {c: float(np.mean(a)) for c, a in per_corruption.items()},
        "per_severity_mean": per_severity_mean,
    }


# ---------------------------------------------------------------------------
# OOD (CIFAR-100 ID vs SVHN OOD).
# ---------------------------------------------------------------------------
def _eval_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """A copy of the run config with augmentation disabled for eval loaders."""
    eval_cfg = json.loads(json.dumps(config))  # deep copy of plain dict
    eval_cfg["data"]["augment"] = False
    eval_cfg.setdefault("label_noise", {})["rate"] = 0.0
    return eval_cfg


def build_svhn_loader(config: dict[str, Any], batch_size: int) -> DataLoader:
    """Build the SVHN *test* loader using CIFAR-100's normalization?  No --

    SVHN is normalized with its own stats by the datasets module; the OOD score
    is computed on logits/features of whatever the model sees, and using each
    dataset's own normalization (matching how the in-distribution loader is
    built) is the standard OOD protocol here.  We override the dataset name and
    num_classes while keeping root/normalize flags.
    """
    eval_cfg = _eval_data_config(config)
    eval_cfg["data"]["name"] = "svhn"
    # DictDataset still needs a num_classes for its (unused at eval) noise path.
    eval_cfg["data"]["num_classes"] = config["data"]["num_classes"]
    dataset = build_dataset(eval_cfg, train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def build_id_loader(config: dict[str, Any], batch_size: int) -> DataLoader:
    eval_cfg = _eval_data_config(config)
    dataset = build_dataset(eval_cfg, train=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_ood(
    model: torch.nn.Module,
    config: dict[str, Any],
    device: torch.device,
    batch_size: int,
    include_mahalanobis: bool = True,
) -> dict[str, float]:
    id_loader = build_id_loader(config, batch_size)
    ood_loader = build_svhn_loader(config, batch_size)
    num_classes = int(config["data"]["num_classes"])
    return compute_ood_auroc(
        model,
        id_loader,
        ood_loader,
        device=device,
        num_classes=num_classes,
        include_mahalanobis=include_mahalanobis,
    )


# ---------------------------------------------------------------------------
# Per-experiment driver + aggregation.
# ---------------------------------------------------------------------------
def evaluate_experiment(
    experiment: str,
    seeds: list[int] | None,
    device: torch.device,
    batch_size: int,
    c_dir: Path | None,
    skip_corruption: bool,
    skip_ood: bool,
    include_mahalanobis: bool,
    seed_table: Path | None = None,
    table_condition: str | None = None,
) -> dict[str, Any]:
    """Evaluate all seed runs of an experiment and write per-seed JSON + summary."""
    if seed_table is not None:
        if table_condition is None:
            raise ValueError("table_condition is required when seed_table is provided")
        runs = find_seed_runs_from_table(seed_table, table_condition, seeds)
    else:
        runs = find_seed_runs(experiment, seeds)
    if not runs:
        raise SystemExit(f"No seed runs with checkpoint_best.pt found for experiment '{experiment}'.")

    out_dir = Path("outputs") / "robustness" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results: dict[int, dict[str, Any]] = {}
    for seed, run_dir in runs.items():
        print(f"[{experiment}] seed {seed}: {run_dir}")
        model, config = load_model_from_run(run_dir, "checkpoint_best.pt", device)
        result: dict[str, Any] = {"experiment": experiment, "seed": seed, "run_dir": str(run_dir)}

        if not skip_corruption and c_dir is not None:
            corr = evaluate_cifar100c(model, config, c_dir, device, batch_size)
            result["corruption"] = corr
            print(f"    mCA = {corr['mCA']:.4f}")

        if not skip_ood:
            ood = evaluate_ood(model, config, device, batch_size, include_mahalanobis)
            result["ood"] = ood
            print("    OOD AUROC: " + ", ".join(f"{k}={v:.4f}" for k, v in ood.items()))

        with (out_dir / f"seed_{seed}.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        seed_results[seed] = result

    summary = aggregate_summary(experiment, seed_results)
    write_summary_csv(out_dir / "summary.csv", summary)
    return {"experiment": experiment, "out_dir": str(out_dir), "seed_results": seed_results, "summary": summary}


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) < 2 else statistics.stdev(values)
    return mean, std


def aggregate_summary(experiment: str, seed_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate mean +/- std over seeds for mCA, per-severity means, and OOD AUROCs."""
    mca_vals = [r["corruption"]["mCA"] for r in seed_results.values() if "corruption" in r]
    summary: dict[str, Any] = {"experiment": experiment, "num_seeds": len(seed_results)}

    if mca_vals:
        m, s = _mean_std(mca_vals)
        summary["mCA_mean"], summary["mCA_std"] = m, s
        for sev in range(1, NUM_SEVERITIES + 1):
            sev_vals: list[float] = []
            for r in seed_results.values():
                if "corruption" not in r:
                    continue
                sev_map = r["corruption"]["per_severity_mean"]
                # per_severity_mean may be keyed by int (in-memory) or str (from JSON).
                sev_vals.append(sev_map.get(sev, sev_map.get(str(sev))))
            sm, ss = _mean_std(sev_vals)
            summary[f"severity{sev}_mean"], summary[f"severity{sev}_std"] = sm, ss

    for key in ("msp_auroc", "energy_auroc", "mahalanobis_auroc"):
        vals = [r["ood"][key] for r in seed_results.values() if "ood" in r and key in r["ood"]]
        if vals:
            m, s = _mean_std(vals)
            summary[f"{key}_mean"], summary[f"{key}_std"] = m, s
    return summary


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def print_comparison(global_summary: dict[str, Any], late_summary: dict[str, Any]) -> None:
    """Print late-boundary minus global differences for mCA and OOD AUROCs."""
    print("\n=== Comparison: late-boundary minus global ===")
    print(f"  global = {global_summary['experiment']}")
    print(f"  late   = {late_summary['experiment']}")
    metrics = [
        ("mCA", "mCA_mean"),
        ("MSP AUROC", "msp_auroc_mean"),
        ("Energy AUROC", "energy_auroc_mean"),
        ("Mahalanobis AUROC", "mahalanobis_auroc_mean"),
    ]
    for label, key in metrics:
        if key in global_summary and key in late_summary:
            g = global_summary[key]
            late = late_summary[key]
            print(
                f"  {label:>20}: global={g:.4f}  late={late:.4f}  "
                f"(late-global)={late - g:+.4f}"
            )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="Experiment name (prefix). Pass twice for a global-vs-late comparison "
        "(first = global, second = late).",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="Seeds to include (default: all found).")
    parser.add_argument("--device", default="cpu", help="torch device (default cpu to avoid GPU contention).")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", default="data", help="Root holding CIFAR-100-C/ and torchvision datasets.")
    parser.add_argument(
        "--seed-table",
        type=Path,
        help="Optional frozen seed-table CSV whose run_dir column supplies checkpoints.",
    )
    parser.add_argument(
        "--table-condition",
        action="append",
        help="Condition label in --seed-table, one per --experiment in matching order.",
    )
    parser.add_argument("--skip-corruption", action="store_true", help="Skip CIFAR-100-C evaluation.")
    parser.add_argument("--skip-ood", action="store_true", help="Skip OOD evaluation.")
    parser.add_argument("--no-mahalanobis", action="store_true", help="Skip the Mahalanobis OOD score.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download CIFAR-100-C; error out if it is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_table is not None:
        if not args.table_condition or len(args.table_condition) != len(args.experiment):
            raise SystemExit(
                "--seed-table requires one --table-condition for each --experiment"
            )
    elif args.table_condition:
        raise SystemExit("--table-condition requires --seed-table")
    device = resolve_device(args.device) if args.device == "auto" else torch.device(args.device)
    data_root = Path(args.data_root)

    c_dir: Path | None = None
    if not args.skip_corruption:
        c_marker = data_root / "CIFAR-100-C" / "labels.npy"
        if args.no_download and not c_marker.exists():
            raise SystemExit(
                f"CIFAR-100-C not found at {data_root / 'CIFAR-100-C'} and --no-download set. "
                "Download must finish before corruption eval."
            )
        c_dir = ensure_cifar100c(data_root)

    experiment_summaries: list[dict[str, Any]] = []
    table_conditions = args.table_condition or [None] * len(args.experiment)
    for experiment, table_condition in zip(
        args.experiment, table_conditions, strict=True
    ):
        outcome = evaluate_experiment(
            experiment,
            args.seeds,
            device,
            args.batch_size,
            c_dir,
            args.skip_corruption,
            args.skip_ood,
            include_mahalanobis=not args.no_mahalanobis,
            seed_table=args.seed_table,
            table_condition=table_condition,
        )
        experiment_summaries.append(outcome["summary"])
        print(f"\nSummary for {experiment}: {outcome['summary']}")

    if len(experiment_summaries) >= 2:
        # Convention: first experiment is the global reference, second is late boundary.
        print_comparison(experiment_summaries[0], experiment_summaries[1])


if __name__ == "__main__":
    main()
