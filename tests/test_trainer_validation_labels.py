from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fls.training.trainer import Trainer


def test_evaluate_can_use_clean_validation_labels(tmp_path: Path) -> None:
    """A noisy-label validation batch can be scored against its clean labels."""
    samples = [
        {"x": torch.tensor([5.0, 0.0]), "y": 1, "y_clean": 0},
        {"x": torch.tensor([0.0, 5.0]), "y": 0, "y_clean": 1},
    ]
    loader = DataLoader(samples, batch_size=2)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        train_loader=loader,
        test_loader=loader,
        config={"training": {}, "data": {}, "fls": {"mode": "global", "global": {}}},
        run_dir=tmp_path,
        device=torch.device("cpu"),
        val_loader=loader,
    )

    _, noisy_accuracy = trainer.evaluate(loader)
    _, clean_accuracy = trainer.evaluate(loader, label_key="y_clean")

    assert noisy_accuracy == 0.0
    assert clean_accuracy == 1.0
