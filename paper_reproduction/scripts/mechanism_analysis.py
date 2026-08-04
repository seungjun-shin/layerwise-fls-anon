#!/usr/bin/env python
"""Compute the frozen H2 validation diagnostics at selected checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from fls.data.datasets import build_train_val_datasets
from fls.scaling.location_fls import OutputScaledRegion
from fls.training.model_factory import build_scaled_model

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "paper_reproduction"
FAMILY_DESIGNS = {
    "practical_main_confirmatory": {
        "conditions": {"P-REF", "P-ACT-FINAL", "P-LR-FINAL"},
        "seeds": set(range(5, 15)),
    },
    "head_norm": {
        "conditions": {
            "HN-COSINE-ACT",
            "HN-COSINE-LR",
            "HN-PRELN-ACT",
            "HN-PRELN-LR",
        },
        "seeds": set(range(20, 25)),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-table",
        default=str(WORKSPACE / "seed_tables" / "practical_main_confirmatory_seed_level.csv"),
    )
    parser.add_argument(
        "--evaluation-family",
        default="practical_main_confirmatory",
        choices=sorted(FAMILY_DESIGNS),
    )
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--jacobian-max-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--conditions",
        default="P-REF,P-ACT-FINAL,P-LR-FINAL",
        help="Comma-separated frozen conditions to analyze.",
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE / "analysis" / "mechanism" / "mechanism_seed_level.csv"),
    )
    parser.add_argument(
        "--metadata-output",
        default=str(WORKSPACE / "analysis" / "mechanism" / "diagnostic_subset.json"),
    )
    parser.add_argument(
        "--bundle-complete-output",
        default=str(WORKSPACE / "analysis" / "mechanism" / "mechanism_bundle_complete.json"),
    )
    return parser.parse_args()


def base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", getattr(model, "base", model))


def effective_rank_from_singular_values(values: torch.Tensor) -> float:
    values = values.float().clamp_min(0)
    probabilities = values / values.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp().item())


def expected_calibration_error(confidence: torch.Tensor, correct: torch.Tensor) -> float:
    error = torch.tensor(0.0)
    edges = torch.linspace(0.0, 1.0, 16)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
    return float(error.item())


def class_geometry(
    features: torch.Tensor,
    labels: torch.Tensor,
    classifier_weight: torch.Tensor,
) -> dict[str, float]:
    """Compute trace-based NC1 and classifier/prototype alignment on a fixed subset."""
    features = features.float()
    labels = labels.long()
    global_mean = features.mean(dim=0)
    within_sum = torch.tensor(0.0)
    between_sum = torch.tensor(0.0)
    alignments: list[torch.Tensor] = []
    observed_classes = labels.unique(sorted=True)
    for class_index in observed_classes:
        mask = labels == class_index
        class_features = features[mask]
        prototype = class_features.mean(dim=0)
        within_sum += (class_features - prototype).square().sum()
        between_sum += float(mask.sum()) * (prototype - global_mean).square().sum()
        centered_prototype = prototype - global_mean
        classifier_vector = classifier_weight[int(class_index)]
        if centered_prototype.norm() > 0 and classifier_vector.norm() > 0:
            alignments.append(F.cosine_similarity(centered_prototype, classifier_vector, dim=0))
    within_trace = within_sum / max(int(features.shape[0]), 1)
    between_trace = between_sum / max(int(features.shape[0]), 1)
    alignment = torch.stack(alignments).mean() if alignments else torch.tensor(float("nan"))
    return {
        "within_class_scatter_trace": float(within_trace.item()),
        "between_class_scatter_trace": float(between_trace.item()),
        "nc1_within_between_trace_ratio": float(
            (within_trace / between_trace.clamp_min(1e-12)).item()
        ),
        "nc3_classifier_prototype_alignment": float(alignment.item()),
        "observed_class_count": float(observed_classes.numel()),
    }


def operational_classifier_weight(head: torch.nn.Module) -> tuple[torch.Tensor, str]:
    """Return the exact weight representation used by the classifier forward map."""
    weight = head.weight.detach().float().cpu()
    if hasattr(head, "logit_scale"):
        scale = float(head.logit_scale)
        return F.normalize(weight, dim=1) * scale, "row_normalized_fixed_scale"
    return weight, "raw_linear_weight"


def diagnostic_subset(config: dict[str, Any], max_samples: int) -> tuple[Subset, str, str]:
    _, validation, validation_indices_hash = build_train_val_datasets(
        config,
        float(config["data"]["val_fraction"]),
        int(config["data"]["val_seed"]),
    )
    if validation_indices_hash != config["data"].get("val_indices_sha256"):
        raise RuntimeError(
            "Regenerated validation split does not match the training-time frozen index hash."
        )
    selected = list(range(min(max_samples, len(validation))))
    underlying = [int(validation.indices[index]) for index in selected]
    digest = hashlib.sha256(",".join(map(str, underlying)).encode("utf-8")).hexdigest()
    return Subset(validation, selected), digest, validation_indices_hash


def display_path(path: Path) -> str:
    """Prefer repository-relative paths while preserving external run paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_run_dir(value: str) -> Path:
    """Resolve either an absolute external run directory or a repository-relative one."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_seed_table(
    path: Path,
    evaluation_family: str,
) -> list[dict[str, str]]:
    """Bind diagnostics to the immutable complete family table and checkpoints."""
    completion_path = (
        WORKSPACE / "test_once" / evaluation_family / "evaluation_complete.json"
    )
    if not completion_path.exists():
        raise RuntimeError(f"Family test evaluation is not complete: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("family") != evaluation_family:
        raise RuntimeError("Family completion ledger identity mismatch.")
    if sha256(path) != completion.get("seed_table_sha256"):
        raise RuntimeError("Seed table is not the immutable family-completion artifact.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    design = FAMILY_DESIGNS[evaluation_family]
    expected = {
        (condition, seed)
        for condition in design["conditions"]
        for seed in design["seeds"]
    }
    observed = [(row["condition"], int(row["seed"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise RuntimeError("Seed table contains duplicate condition/seed rows.")
    if set(observed) != expected:
        raise RuntimeError(
            "Seed table does not contain the complete frozen condition/seed design: "
            f"missing={sorted(expected - set(observed))}, extra={sorted(set(observed) - expected)}"
        )
    if len(rows) != int(completion.get("evaluated_jobs", -1)):
        raise RuntimeError("Seed table row count disagrees with its completion ledger.")
    run_ids: set[str] = set()
    for row in rows:
        run_id = row.get("run_id", "")
        if not run_id or run_id in run_ids:
            raise RuntimeError("Seed table has a missing or duplicate run identifier.")
        run_ids.add(run_id)
        if row.get("evaluation_family") != evaluation_family:
            raise RuntimeError(f"Evaluation-family mismatch for {run_id}.")
        if row.get("selection_split") != "validation":
            raise RuntimeError(f"Non-validation checkpoint selection for {run_id}.")
        if int(row.get("test_evaluation_count", 0)) != 1:
            raise RuntimeError(f"Unexpected test-evaluation count for {run_id}.")
        run_dir = resolve_run_dir(row["run_dir"])
        checkpoint_path = run_dir / "checkpoint_best.pt"
        if sha256(checkpoint_path) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"Checkpoint digest mismatch for {run_id}.")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if int(checkpoint["epoch"]) != int(row["checkpoint_epoch"]):
            raise RuntimeError(f"Checkpoint epoch mismatch for {run_id}.")
        config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
        if int(config["training"]["seed"]) != int(row["seed"]):
            raise RuntimeError(f"Resolved-config seed mismatch for {run_id}.")
    return rows


def compute_run(
    run_dir: Path,
    condition: str,
    device: torch.device,
    max_samples: int,
    jacobian_max_samples: int,
    batch_size: int,
) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset, subset_hash, validation_indices_hash = diagnostic_subset(config, max_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_scaled_model(config).to(device)
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    base = base_model(model)
    head = base.head
    captured: dict[str, torch.Tensor] = {}

    def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        feature = inputs[0]
        feature.retain_grad()
        captured["feature"] = feature

    late = base.late

    def capture_boundary_identity(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["boundary_pre"] = output
        captured["boundary_post"] = output

    if isinstance(late, OutputScaledRegion):
        boundary_pre_handle = late.region.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("boundary_pre", output)
        )
        boundary_post_handle = late.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("boundary_post", output)
        )
    else:
        boundary_pre_handle = late.register_forward_hook(capture_boundary_identity)
        boundary_post_handle = boundary_pre_handle

    handle = head.register_forward_pre_hook(capture_input)
    boundary_pre_norms: list[torch.Tensor] = []
    boundary_post_norms: list[torch.Tensor] = []
    feature_norms: list[torch.Tensor] = []
    feature_grad_norms: list[torch.Tensor] = []
    jacobian_norms: list[torch.Tensor] = []
    logit_norms: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    features_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    corrects: list[torch.Tensor] = []
    jacobian_generator = torch.Generator(device="cpu")
    jacobian_generator.manual_seed(310_000 + int(config["training"]["seed"]))
    jacobian_samples = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = torch.as_tensor(batch["y"], device=device, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        feature = captured["feature"]
        jacobian_batch = min(int(logits.shape[0]), jacobian_max_samples - jacobian_samples)
        if jacobian_batch > 0:
            probe = 2.0 * torch.randint(
                0,
                2,
                logits[:jacobian_batch].shape,
                generator=jacobian_generator,
                device="cpu",
                dtype=torch.int64,
            ).to(device=device, dtype=logits.dtype) - 1.0
            jacobian_vjp = torch.autograd.grad(
                (logits[:jacobian_batch] * probe).sum(),
                feature,
                retain_graph=True,
                create_graph=False,
            )[0]
            jacobian_norms.append(
                jacobian_vjp[:jacobian_batch].detach().float().norm(dim=1).square().cpu()
            )
            jacobian_samples += jacobian_batch
        loss = F.cross_entropy(logits, y, reduction="sum")
        loss.backward()
        probabilities = logits.detach().softmax(dim=1)
        top2 = logits.detach().topk(2, dim=1).values
        boundary_pre = captured["boundary_pre"].detach().float().flatten(1)
        boundary_post = captured["boundary_post"].detach().float().flatten(1)
        boundary_pre_norms.append(boundary_pre.norm(dim=1).cpu())
        boundary_post_norms.append(boundary_post.norm(dim=1).cpu())
        feature_norms.append(feature.detach().float().norm(dim=1).cpu())
        feature_grad_norms.append(feature.grad.detach().float().norm(dim=1).cpu())
        logit_norms.append(logits.detach().float().norm(dim=1).cpu())
        logits_all.append(logits.detach().float().cpu())
        features_all.append(feature.detach().float().cpu())
        labels_all.append(y.detach().cpu())
        margins.append((top2[:, 0] - top2[:, 1]).float().cpu())
        entropies.append((-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)).cpu())
        confidence, prediction = probabilities.max(dim=1)
        confidences.append(confidence.cpu())
        corrects.append(prediction.eq(y).cpu())
    handle.remove()
    boundary_pre_handle.remove()
    if boundary_post_handle is not boundary_pre_handle:
        boundary_post_handle.remove()
    confidence = torch.cat(confidences)
    correct = torch.cat(corrects)
    merged_logits = torch.cat(logits_all)
    merged_features = torch.cat(features_all)
    merged_labels = torch.cat(labels_all)
    classifier_weight, weight_representation = operational_classifier_weight(head)
    singular_values = torch.linalg.svdvals(classifier_weight)
    condition_number = float((singular_values.max() / singular_values.min().clamp_min(1e-12)).item())
    if weight_representation == "row_normalized_fixed_scale":
        geometry_features = F.normalize(merged_features, dim=1)
        feature_representation = "row_normalized_for_cosine_classifier"
    else:
        geometry_features = merged_features
        feature_representation = "classifier_input"
    geometry = class_geometry(geometry_features, merged_labels, classifier_weight)
    boundary_pre_mean = float(torch.cat(boundary_pre_norms).mean().item())
    boundary_post_mean = float(torch.cat(boundary_post_norms).mean().item())
    metrics = {
        "condition": condition,
        "seed": int(config["training"]["seed"]),
        "run_dir": display_path(run_dir),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": "validation",
        "diagnostic_subset_size": len(dataset),
        "diagnostic_subset_sha256": subset_hash,
        "validation_indices_sha256": validation_indices_hash,
        "jacobian_subset_size": jacobian_samples,
        "classifier_weight_representation": weight_representation,
        "class_geometry_feature_representation": feature_representation,
        "boundary_pre_activation_l2_norm": boundary_pre_mean,
        "boundary_post_activation_l2_norm": boundary_post_mean,
        "boundary_activation_norm_ratio": boundary_post_mean / max(boundary_pre_mean, 1e-12),
        "feature_l2_norm": float(torch.cat(feature_norms).mean().item()),
        "classifier_input_gradient_l2_norm": float(torch.cat(feature_grad_norms).mean().item()),
        "logit_jacobian_frobenius_squared_hutchinson": float(
            torch.cat(jacobian_norms).mean().item()
        ),
        "logit_jacobian_frobenius_hutchinson": float(
            torch.cat(jacobian_norms).mean().sqrt().item()
        ),
        "logit_l2_norm": float(torch.cat(logit_norms).mean().item()),
        "logit_mean": float(merged_logits.mean().item()),
        "logit_standard_deviation": float(merged_logits.std(unbiased=True).item()),
        "classifier_weight_effective_rank": effective_rank_from_singular_values(singular_values),
        "classifier_weight_spectral_norm": float(singular_values.max().item()),
        "classifier_weight_condition_number": condition_number,
        "classifier_weight_frobenius_norm": float(classifier_weight.norm().item()),
        "top1_top2_margin": float(torch.cat(margins).mean().item()),
        "predictive_entropy": float(torch.cat(entropies).mean().item()),
        "expected_confidence": float(confidence.mean().item()),
        "validation_subset_accuracy": float(correct.float().mean().item()),
        "ece_15bin": expected_calibration_error(confidence, correct),
    }
    metrics.update(geometry)
    metrics["finite"] = all(
        math.isfinite(float(value))
        for key, value in metrics.items()
        if key
        not in {
            "condition",
            "run_dir",
            "split",
            "diagnostic_subset_sha256",
            "validation_indices_sha256",
            "classifier_weight_representation",
            "class_geometry_feature_representation",
            "finite",
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or not 0 < args.jacobian_max_samples <= args.max_samples:
        raise ValueError("Require 0 < jacobian-max-samples <= max-samples.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    bundle_complete_output = Path(args.bundle_complete_output)
    output_tmp = output.with_name(f".{output.name}.partial")
    metadata_tmp = metadata_output.with_name(f".{metadata_output.name}.partial")
    existing = [
        str(path)
        for path in (
            output,
            metadata_output,
            bundle_complete_output,
            output_tmp,
            metadata_tmp,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Mechanism outputs already exist: {existing}")
    seed_rows = validate_seed_table(Path(args.seed_table), args.evaluation_family)
    conditions = {condition.strip() for condition in args.conditions.split(",") if condition.strip()}
    frozen_conditions = FAMILY_DESIGNS[args.evaluation_family]["conditions"]
    if conditions != frozen_conditions:
        raise RuntimeError(
            f"Mechanism conditions must equal the complete frozen family: {sorted(frozen_conditions)}."
        )
    selected = [row for row in seed_rows if row["condition"] in conditions]
    observed = {row["condition"] for row in selected}
    if observed != conditions:
        raise RuntimeError(
            f"Requested conditions {sorted(conditions)} do not match available rows {sorted(observed)}."
        )
    for path in (output, metadata_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    rows = [
        compute_run(
            resolve_run_dir(row["run_dir"]),
            row["condition"],
            device,
            args.max_samples,
            args.jacobian_max_samples,
            args.batch_size,
        )
        for row in selected
    ]
    hashes = {row["diagnostic_subset_sha256"] for row in rows}
    if len(hashes) != 1:
        raise RuntimeError(f"Diagnostic subsets differ across paired runs: {sorted(hashes)}")
    validation_hashes = {row["validation_indices_sha256"] for row in rows}
    if len(validation_hashes) != 1:
        raise RuntimeError(f"Validation splits differ across paired runs: {sorted(validation_hashes)}")
    with output_tmp.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    metadata = {
        "split": "validation",
        "size": int(rows[0]["diagnostic_subset_size"]),
        "jacobian_size": int(rows[0]["jacobian_subset_size"]),
        "sha256": rows[0]["diagnostic_subset_sha256"],
        "validation_indices_sha256": rows[0]["validation_indices_sha256"],
        "selection_policy": (
            f"first {args.max_samples} examples in the ordered fixed validation subset"
        ),
        "confirmatory_metrics": [
            "feature_l2_norm",
            "classifier_input_gradient_l2_norm",
            "logit_l2_norm",
            "classifier_weight_effective_rank",
        ],
        "descriptive_interface_metrics": [
            "boundary_pre_activation_l2_norm",
            "boundary_post_activation_l2_norm",
            "boundary_activation_norm_ratio",
            "logit_jacobian_frobenius_hutchinson",
            "logit_jacobian_frobenius_squared_hutchinson",
            "classifier_weight_spectral_norm",
            "classifier_weight_condition_number",
            "top1_top2_margin",
            "predictive_entropy",
            "expected_confidence",
            "ece_15bin",
            "within_class_scatter_trace",
            "between_class_scatter_trace",
            "nc1_within_between_trace_ratio",
            "nc3_classifier_prototype_alignment",
        ],
        "jacobian_policy": (
            "one fixed Rademacher probe per example over the first "
            f"{args.jacobian_max_samples} diagnostic examples; mean ||J^T v||^2 "
            "estimates squared Frobenius norm and its square root is reported as the "
            "Frobenius estimate"
        ),
        "conditions": sorted(conditions),
        "evaluation_family": args.evaluation_family,
        "seed_table": str(Path(args.seed_table)),
        "seed_table_sha256": sha256(Path(args.seed_table)),
    }
    with metadata_tmp.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(output_tmp, output)
    os.replace(metadata_tmp, metadata_output)
    bundle_complete_output.parent.mkdir(parents=True, exist_ok=True)
    with bundle_complete_output.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "completed",
                "evaluation_family": args.evaluation_family,
                "output": str(output),
                "output_sha256": sha256(output),
                "metadata_output": str(metadata_output),
                "metadata_sha256": sha256(metadata_output),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(output)


if __name__ == "__main__":
    main()
