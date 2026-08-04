#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fls.data.datasets import (
    build_dataset,
    build_official_validation_dataset,
    build_train_val_datasets,
)
from fls.training.losses import build_loss
from fls.training.model_factory import build_scaled_model
from fls.training.optim import build_optimizer
from fls.training.protocol_audit import write_protocol_sanity
from fls.training.seed import set_seed
from fls.training.trainer import Trainer
from fls.utils.config import load_config, save_config
from fls.utils.device import resolve_device
from fls.utils.paths import make_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> Path:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    set_seed(int(config["training"]["seed"]))
    run_dir = make_run_dir(config)
    device = resolve_device(config["training"].get("device", "auto"))
    bs = int(config["data"]["batch_size"])
    nw = int(config["data"].get("num_workers", 0))
    val_loader = None
    val_source = str(config["data"].get("val_source", "split"))
    val_fraction = config["data"].get("val_fraction")
    if val_source == "official":
        train_dataset = build_dataset(config, train=True)
        val_dataset = build_official_validation_dataset(config)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw)
    elif val_fraction:
        if bool(config["data"].get("deterministic_validation", False)):
            train_dataset, val_dataset, val_indices_sha256 = build_train_val_datasets(
                config, float(val_fraction), int(config["data"].get("val_seed", 0))
            )
            config["data"]["val_indices_sha256"] = val_indices_sha256
        else:
            from fls.data.datasets import split_train_val
            train_dataset = build_dataset(config, train=True)
            train_dataset, val_dataset = split_train_val(
                train_dataset, float(val_fraction), int(config["data"].get("val_seed", 0)))
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=nw)
    else:
        train_dataset = build_dataset(config, train=True)
    loader_generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        generator=loader_generator,
    )
    evaluate_test_each_epoch = bool(config["training"].get("evaluate_test_each_epoch", True))
    evaluate_test_at_end = bool(config["training"].get("evaluate_test_at_end", False))
    test_loader = None
    if evaluate_test_each_epoch or evaluate_test_at_end:
        test_dataset = build_dataset(config, train=False)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=nw)
    model = build_scaled_model(config).to(device)
    save_config(config, run_dir / "resolved_config.yaml")
    optimizer = build_optimizer(model, config)
    if bool((config["training"].get("protocol_sanity") or {}).get("enabled", False)):
        generator_state = loader_generator.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        first_batch = next(iter(train_loader))
        write_protocol_sanity(model, optimizer, config, first_batch, run_dir, device)
        # The audit batch must not consume the seeded training permutation.
        loader_generator.set_state(generator_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
    trainer = Trainer(model, optimizer, build_loss(), train_loader, test_loader, config, run_dir, device,
                      val_loader=val_loader)
    trainer.fit()
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
