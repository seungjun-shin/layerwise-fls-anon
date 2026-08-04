from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fls.training.checkpointing import save_checkpoint
from fls.utils.logging import CSVLogger


def _batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    label_key: str = "y",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move inputs and the requested label field to ``device``."""
    return batch["x"].to(device), torch.as_tensor(batch[label_key], device=device, dtype=torch.long)


def apply_bn_controls(model: torch.nn.Module, config: dict) -> None:
    """Apply optional BN freezing controls after train/eval mode changes."""
    model_cfg = config.get("model", {})
    freeze_affine = bool(model_cfg.get("freeze_bn_affine", False))
    freeze_stats = bool(model_cfg.get("freeze_bn_stats", False))
    if not freeze_affine and not freeze_stats:
        return
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if freeze_stats:
                module.eval()
            if freeze_affine:
                if module.weight is not None:
                    module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)


class Trainer:
    """Basic supervised trainer with CSV metrics and checkpoints."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader | None,
        config: dict,
        run_dir: Path,
        device: torch.device,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.config = config
        self.run_dir = run_dir
        self.device = device
        self.logger = CSVLogger(run_dir / "metrics.csv")
        self.step = 0
        # Selection metric: validation accuracy when a val split is present, else test
        # accuracy. checkpoint_best and checkpoint_at_accs use this, so the test set
        # stays untouched for final evaluation when val_loader is set.
        self.best_sel_acc = -1.0
        self.start_time = time.time()
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Optional per-epoch LR scheduler (training.lr_scheduler: cosine|none)."""
        name = str(self.config["training"].get("lr_scheduler", "none")).lower()
        if name in {"none", "", "constant"}:
            return None
        max_epochs = int(self.config["training"]["max_epochs"])
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        raise ValueError(f"Unknown lr_scheduler: {name}")

    def fit(self) -> None:
        max_epochs = int(self.config["training"]["max_epochs"])
        max_steps = self.config["training"].get("max_steps")
        save_checkpoints = bool(self.config["training"].get("save_checkpoints", True))
        # Iso-accuracy control: save the first checkpoint whose selection metric
        # crosses each target. With a validation loader this is validation accuracy;
        # otherwise it falls back to test accuracy for legacy runs.
        checkpoint_at_accs = [float(a) for a in (self.config["training"].get("checkpoint_at_accs") or [])]
        saved_acc_targets: set[float] = set()
        evaluate_test_each_epoch = bool(self.config["training"].get("evaluate_test_each_epoch", True))
        evaluate_test_at_end = bool(self.config["training"].get("evaluate_test_at_end", False))
        if not evaluate_test_each_epoch and self.val_loader is None:
            raise ValueError("Disabling per-epoch test evaluation requires a validation loader.")
        if evaluate_test_at_end and (not save_checkpoints or self.test_loader is None):
            raise ValueError("Final test evaluation requires checkpoints and a test loader.")
        for epoch in range(max_epochs):
            train_loss, train_acc = self.train_epoch(max_steps)
            if evaluate_test_each_epoch:
                if self.test_loader is None:
                    raise ValueError("Per-epoch test evaluation requested without a test loader.")
                test_loss, test_acc = self.evaluate()
            else:
                test_loss, test_acc = None, None
            if self.val_loader is not None:
                val_label_key = "y_clean" if self.config["data"].get("val_use_clean_labels", False) else "y"
                val_loss, val_acc = self.evaluate(self.val_loader, label_key=val_label_key)
            else:
                val_loss, val_acc = None, None
            # selection metric: validation when available, else test
            sel_acc = val_acc if val_acc is not None else test_acc
            if sel_acc is None:
                raise RuntimeError("No checkpoint-selection metric is available.")
            row = self._metrics_row(epoch, train_loss, train_acc, test_loss, test_acc,
                                    val_loss=val_loss, val_acc=val_acc)
            self.logger.log(row)
            for target in checkpoint_at_accs:
                if target not in saved_acc_targets and sel_acc * 100.0 >= target:
                    saved_acc_targets.add(target)
                    save_checkpoint(self.run_dir / f"checkpoint_atacc_{target:g}.pt",
                                    self.model, self.optimizer, epoch, self.step, row)
            if save_checkpoints:
                save_checkpoint(self.run_dir / "checkpoint_last.pt", self.model, self.optimizer, epoch, self.step, row)
            if sel_acc > self.best_sel_acc:
                self.best_sel_acc = sel_acc
                if save_checkpoints:
                    save_checkpoint(self.run_dir / "checkpoint_best.pt", self.model, self.optimizer, epoch, self.step, row)
            if self.scheduler is not None:
                self.scheduler.step()
            if self._should_stop(train_loss, train_acc, max_steps):
                break
        if evaluate_test_at_end:
            self._evaluate_selected_checkpoint_once()

    def train_epoch(self, max_steps: int | None) -> tuple[float, float]:
        self.model.train()
        apply_bn_controls(self.model, self.config)
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in tqdm(self.train_loader, desc="train", leave=False, disable=not sys.stderr.isatty()):
            x, y = _batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()
            batch_size = y.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size
            self.step += 1
            if max_steps is not None and self.step >= int(max_steps):
                break
        return total_loss / max(total, 1), total_correct / max(total, 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader | None = None,
        label_key: str = "y",
    ) -> tuple[float, float]:
        """Evaluate a loader using ``label_key`` as its target field."""
        self.model.eval()
        loader = loader if loader is not None else self.test_loader
        if loader is None:
            raise ValueError("No evaluation loader was provided.")
        total_loss = 0.0
        total_correct = 0
        total = 0
        for batch in loader:
            x, y = _batch_to_device(batch, self.device, label_key=label_key)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            batch_size = y.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == y).sum().item())
            total += batch_size
        return total_loss / max(total, 1), total_correct / max(total, 1)

    @staticmethod
    def _base_model(model: torch.nn.Module) -> torch.nn.Module:
        """Unwrap repository scaling wrappers to expose the classifier head."""
        current = model
        while True:
            if hasattr(current, "model"):
                current = current.model
                continue
            if hasattr(current, "base"):
                current = current.base
                continue
            return current

    @torch.no_grad()
    def _evaluate_with_artifacts(
        self,
        loader: DataLoader,
        label_key: str = "y",
    ) -> tuple[dict[str, float | int], dict[str, torch.Tensor]]:
        """Evaluate once and retain logits plus exact classifier-input features."""
        self.model.eval()
        base = self._base_model(self.model)
        head = getattr(base, "head", None)
        if head is None:
            raise ValueError("Evaluation artifacts require a model exposing a head module.")
        captured: dict[str, torch.Tensor] = {}

        def capture_input(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured["pre_head"] = inputs[0].detach()

        handle = head.register_forward_pre_hook(capture_input)
        logits_all: list[torch.Tensor] = []
        labels_all: list[torch.Tensor] = []
        features_all: list[torch.Tensor] = []
        total_loss = 0.0
        total_correct = 0
        total = 0
        try:
            for batch in loader:
                x, y = _batch_to_device(batch, self.device, label_key=label_key)
                logits = self.model(x)
                if "pre_head" not in captured:
                    raise RuntimeError("Classifier-input capture hook did not run.")
                batch_size = y.numel()
                total_loss += float(self.loss_fn(logits, y).item()) * batch_size
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total += batch_size
                logits_all.append(logits.float().cpu())
                labels_all.append(y.cpu())
                features_all.append(captured["pre_head"].float().cpu())
        finally:
            handle.remove()
        metrics: dict[str, float | int] = {
            "loss": total_loss / max(total, 1),
            "accuracy": total_correct / max(total, 1),
            "num_examples": total,
        }
        artifacts = {
            "logits": torch.cat(logits_all),
            "labels": torch.cat(labels_all),
            "pre_head_features": torch.cat(features_all),
        }
        return metrics, artifacts

    def _evaluate_selected_checkpoint_once(self) -> None:
        """Load the validation-selected checkpoint and evaluate the test set once."""
        result_path = self.run_dir / "final_evaluation.json"
        artifact_path = self.run_dir / "final_eval_artifacts.pt"
        if result_path.exists() or artifact_path.exists():
            raise FileExistsError("Final evaluation artifacts already exist; refusing a second test pass.")
        if self.val_loader is None or self.test_loader is None:
            raise ValueError("Final evaluation requires validation and test loaders.")
        checkpoint_path = self.run_dir / "checkpoint_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        val_label_key = "y_clean" if self.config["data"].get("val_use_clean_labels", False) else "y"
        validation, validation_artifacts = self._evaluate_with_artifacts(
            self.val_loader, label_key=val_label_key
        )
        test, test_artifacts = self._evaluate_with_artifacts(self.test_loader)
        result = {
            "selection_split": "validation",
            "checkpoint": "checkpoint_best.pt",
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "validation": validation,
            "test": test,
            "test_evaluation_count": 1,
            "wall_clock_time": time.time() - self.start_time,
        }
        if bool(self.config["training"].get("save_eval_artifacts", False)):
            base = self._base_model(self.model)
            head = base.head
            payload: dict[str, Any] = {
                "validation": validation_artifacts,
                "test": test_artifacts,
                "checkpoint_epoch": int(checkpoint["epoch"]),
            }
            if hasattr(head, "weight"):
                payload["classifier_weight"] = head.weight.detach().float().cpu()
            if getattr(head, "bias", None) is not None:
                payload["classifier_bias"] = head.bias.detach().float().cpu()
            torch.save(payload, artifact_path)
            result["artifact_file"] = artifact_path.name
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _metrics_row(self, epoch: int, train_loss: float, train_acc: float,
                     test_loss: float | None, test_acc: float | None,
                     val_loss: float | None = None, val_acc: float | None = None) -> dict[str, Any]:
        lr = self.optimizer.param_groups[0]["lr"]
        lr_groups = {
            str(group.get("name", index)): group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        }
        fls_cfg = self.config["fls"]
        blockwise_groups = fls_cfg.get("groups") if fls_cfg.get("mode") == "blockwise" else None
        location_boundaries = fls_cfg.get("boundaries") if fls_cfg.get("mode") == "location" else None
        return {
            "experiment_name": self.config["experiment"]["name"],
            "seed": self.config["training"]["seed"],
            "epoch": epoch,
            "step": self.step,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": lr,
            "learning_rates": json.dumps(lr_groups, sort_keys=True),
            "global_output_multiplier": fls_cfg.get("global", {}).get("output_multiplier"),
            "location_boundaries": json.dumps(location_boundaries, sort_keys=True) if location_boundaries is not None else None,
            "blockwise_profile_name": fls_cfg.get("profile_name"),
            "blockwise_groups": json.dumps(blockwise_groups, sort_keys=True) if blockwise_groups is not None else None,
            "wall_clock_time": time.time() - self.start_time,
        }

    def _should_stop(self, train_loss: float, train_acc: float, max_steps: int | None) -> bool:
        cfg = self.config["training"]
        if max_steps is not None and self.step >= int(max_steps):
            return True
        target_acc = cfg.get("target_train_accuracy")
        if target_acc is not None and train_acc >= float(target_acc):
            return True
        target_loss = cfg.get("target_train_loss")
        return target_loss is not None and train_loss <= float(target_loss)
