from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fls.data.datasets import build_train_val_datasets
from fls.training.trainer import Trainer


def test_deterministic_validation_copy_and_index_hash() -> None:
    config = {
        "data": {
            "name": "fake",
            "root": "data",
            "num_classes": 10,
            "image_size": 32,
            "train_size": 40,
            "test_size": 10,
            "batch_size": 8,
            "augment": True,
            "normalize": False,
            "return_clean_labels": True,
        },
        "label_noise": {"rate": 0.0, "seed": 0},
    }
    train_a, val_a, digest_a = build_train_val_datasets(config, 0.2, seed=7)
    train_b, val_b, digest_b = build_train_val_datasets(config, 0.2, seed=7)

    assert len(train_a) == len(train_b) == 32
    assert len(val_a) == len(val_b) == 8
    assert digest_a == digest_b
    assert torch.equal(val_a[0]["x"], val_a[0]["x"])
    assert torch.equal(val_a[0]["x"], val_b[0]["x"])


def test_training_can_select_on_validation_without_a_test_loader(tmp_path: Path) -> None:
    samples = [
        {"x": torch.tensor([2.0, 0.0]), "y": 0, "y_clean": 0},
        {"x": torch.tensor([0.0, 2.0]), "y": 1, "y_clean": 1},
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    config = {
        "experiment": {"name": "no_test_training"},
        "training": {
            "seed": 0,
            "max_epochs": 1,
            "max_steps": 1,
            "evaluate_test_each_epoch": False,
            "save_checkpoints": False,
        },
        "data": {},
        "fls": {"mode": "global", "global": {}},
    }
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=loader,
        test_loader=None,
        config=config,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        val_loader=loader,
    )

    trainer.fit()

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["test_loss"] == ""
    assert row["test_accuracy"] == ""
    assert row["val_accuracy"] != ""
