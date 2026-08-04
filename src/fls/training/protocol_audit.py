from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch

from fls.scaling.location_fls import OutputScaledRegion
from fls.scaling.global_fls import GlobalOutputMultiplier


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "model", getattr(model, "base", model))


def _parameter_group_records(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[list[dict[str, Any]], str]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    records: list[dict[str, Any]] = []
    checksum_items: list[str] = []
    for index, group in enumerate(optimizer.param_groups):
        names = sorted(names_by_id[id(parameter)] for parameter in group["params"])
        name = str(group.get("name", index))
        digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        records.append(
            {
                "name": name,
                "learning_rate": float(group["lr"]),
                "parameter_count": sum(int(parameter.numel()) for parameter in group["params"]),
                "parameter_name_sha256": digest,
                "parameter_names": names,
            }
        )
        checksum_items.append(f"{name}:{digest}:{float(group['lr']):.17g}")
    combined = hashlib.sha256("\n".join(checksum_items).encode("utf-8")).hexdigest()
    return records, combined


def _model_parameter_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batch_checksum(batch: dict[str, Any], key: str) -> str | None:
    value = batch.get(key)
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _boundary_observation(
    model: torch.nn.Module,
    x: torch.Tensor,
) -> dict[str, Any]:
    if isinstance(model, GlobalOutputMultiplier):
        captured: dict[str, torch.Tensor] = {}
        pre_handle = model.model.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("pre", output.detach())
        )
        post_handle = model.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("post", output.detach())
        )
        expected = float(model.multiplier)
        boundary = "after_classifier"
    elif all(hasattr(model, field) for field in ("base", "cut", "scale")):
        cut = int(model.cut)
        captured: dict[str, torch.Tensor] = {}
        pre_handle = model.scale.register_forward_pre_hook(
            lambda _module, inputs: captured.__setitem__("pre", inputs[0].detach())
        )
        post_handle = model.scale.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("post", output.detach())
        )
        expected = float(model.scale.multiplier)
        boundary = f"position_{cut}"
    else:
        base = _base_model(model)
        late = base.late
        captured = {}
        boundary = "after_late"
        if isinstance(late, OutputScaledRegion):
            pre_handle = late.region.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("pre", output.detach())
            )
            post_handle = late.register_forward_hook(
                lambda _module, _inputs, output: captured.__setitem__("post", output.detach())
            )
            expected = float(late.multiplier)
        else:
            def capture_identity(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                captured["pre"] = output.detach()
                captured["post"] = output.detach()

            handle = late.register_forward_hook(capture_identity)
            pre_handle = post_handle = handle
            expected = 1.0
    was_training = model.training
    model.eval()
    model(x)
    model.train(was_training)
    pre_handle.remove()
    if post_handle is not pre_handle:
        post_handle.remove()
    pre = captured["pre"]
    post = captured["post"]
    pre_norm = float(pre.float().norm().item())
    post_norm = float(post.float().norm().item())
    return {
        "boundary": boundary,
        "tensor_shape": list(post.shape),
        "pre_scale_l2_norm": pre_norm,
        "post_scale_l2_norm": post_norm,
        "expected_scale_ratio": expected,
        "observed_scale_ratio": post_norm / pre_norm if pre_norm else None,
    }


def write_protocol_sanity(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    batch: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> Path:
    """Record one-batch scale and optimizer-group evidence for a training run."""
    groups, group_checksum = _parameter_group_records(model, optimizer)
    observation = _boundary_observation(model, batch["x"].to(device))
    metadata = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "experiment_name": config["experiment"]["name"],
        "seed": int(config["training"]["seed"]),
        "test_evaluated_during_training": bool(config["training"].get("evaluate_test_each_epoch", True)),
        "validation_indices_sha256": config["data"].get("val_indices_sha256"),
        "initial_parameter_sha256": _model_parameter_checksum(model),
        "audit_batch_indices_sha256": _batch_checksum(batch, "index"),
        "audit_batch_inputs_sha256": _batch_checksum(batch, "x"),
        "boundary_observation": observation,
        "optimizer_parameter_groups": groups,
        "optimizer_group_checksum": group_checksum,
    }
    output = run_dir / "protocol_sanity.json"
    output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
