"""Tests for the extended dataset builder (SVHN, STL-10, Tiny-ImageNet).

These tests validate the transform/normalization branching and dispatch logic
WITHOUT downloading multi-GB datasets.  The one optionally-live test (SVHN) is
guarded so it only runs when the data already exists locally.
"""

from __future__ import annotations

import os

import pytest
import torch

from fls.data.datasets import (
    DATASET_STATS,
    DictDataset,
    _transforms,
    build_dataset,
    dataset_stats,
)


def _cfg(name: str, **over):
    base = {"name": name, "root": "data", "num_classes": 10, "augment": False, "normalize": False}
    base.update(over)
    return {"data": base, "label_noise": {"rate": 0.0, "seed": 0}}


# --- dataset_stats mapping -------------------------------------------------


@pytest.mark.parametrize(
    "name,working,native",
    [
        ("cifar10", 32, 32),
        ("cifar100", 32, 32),
        ("svhn", 32, 32),
        ("stl10", 32, 96),
        ("tiny_imagenet", 32, 64),
    ],
)
def test_stats_working_and_native_sizes(name, working, native):
    stats = dataset_stats(name)
    assert stats["working_size"] == working
    assert stats["native_size"] == native
    assert len(stats["mean"]) == 3
    assert len(stats["std"]) == 3


def test_stats_case_insensitive():
    assert dataset_stats("SVHN") == dataset_stats("svhn")


def test_unknown_name_falls_back_to_cifar_stats():
    # Unknown -> cifar-like defaults (no resize, cifar mean/std).
    assert dataset_stats("does_not_exist") == DATASET_STATS["cifar10"]


# --- transform branching ---------------------------------------------------


def test_cifar_transforms_unchanged():
    """CIFAR transform repr must be byte-for-byte the original pipeline."""
    import torchvision.transforms as T

    def original(cfg, train):
        ops = []
        if train and cfg.get("augment", False):
            ops.extend([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
        ops.append(T.ToTensor())
        if cfg.get("normalize", False):
            ops.append(T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)))
        return T.Compose(ops)

    for train in (True, False):
        for aug in (True, False):
            for norm in (True, False):
                cfg = {"augment": aug, "normalize": norm}
                for name in ("cifar10", "cifar100"):
                    assert repr(_transforms(cfg, train, name)) == repr(original(cfg, train))


def test_svhn_uses_padded_crop_and_svhn_norm():
    t = _transforms({"augment": True, "normalize": True}, True, "svhn")
    rep = repr(t)
    assert "RandomCrop" in rep and "padding=4" in rep
    assert "RandomResizedCrop" not in rep
    assert "0.4377" in rep  # SVHN mean


def test_stl10_train_uses_resized_crop():
    t = _transforms({"augment": True, "normalize": True}, True, "stl10")
    rep = repr(t)
    assert "RandomResizedCrop" in rep
    assert "size=(32, 32)" in rep
    assert "RandomCrop(size" not in rep  # not the padded crop path


def test_stl10_eval_uses_plain_resize():
    t = _transforms({"augment": True, "normalize": True}, False, "stl10")
    rep = repr(t)
    assert "Resize(size=(32, 32)" in rep
    assert "RandomResizedCrop" not in rep


def test_tiny_imagenet_resizes_to_32():
    t = _transforms({"augment": True, "normalize": True}, True, "tiny_imagenet")
    assert "size=(32, 32)" in repr(t)
    t_eval = _transforms({"augment": False, "normalize": True}, False, "tiny_imagenet")
    assert "Resize(size=(32, 32)" in repr(t_eval)


def test_normalize_flag_gates_normalization():
    on = _transforms({"augment": False, "normalize": True}, False, "svhn")
    off = _transforms({"augment": False, "normalize": False}, False, "svhn")
    assert "Normalize" in repr(on)
    assert "Normalize" not in repr(off)


# --- dispatch --------------------------------------------------------------


def test_build_dataset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        build_dataset(_cfg("totally_unknown"), train=True)


def test_fake_dispatch_still_works():
    cfg = _cfg("fake", train_size=8, test_size=4, num_classes=10, image_size=32)
    ds = build_dataset(cfg, train=True)
    assert len(ds) == 8
    sample = ds[0]
    assert sample["x"].shape == (3, 32, 32)


# --- DictDataset label extraction (mocks, no download) ---------------------


class _LabelsDataset(torch.utils.data.Dataset):
    """Mimics SVHN/STL-10 which expose ``.labels`` (not ``.targets``)."""

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), self.labels[i]


class _TargetsDataset(torch.utils.data.Dataset):
    """Mimics CIFAR/ImageFolder which expose ``.targets``."""

    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), self.targets[i]


def test_dictdataset_reads_labels_attribute():
    base = _LabelsDataset([3, 1, 4, 1, 5])
    dd = DictDataset(base, num_classes=10, noise_rate=0.0)
    assert dd.clean_labels.tolist() == [3, 1, 4, 1, 5]
    assert dd[2]["y_clean"] == 4


def test_dictdataset_reads_targets_attribute():
    base = _TargetsDataset([0, 9, 2])
    dd = DictDataset(base, num_classes=10, noise_rate=0.0)
    assert dd.clean_labels.tolist() == [0, 9, 2]


# --- optional live SVHN test (skips unless already downloaded) --------------


def _svhn_available() -> bool:
    return os.path.exists(os.path.join("data", "train_32x32.mat"))


@pytest.mark.skipif(not _svhn_available(), reason="SVHN not downloaded")
def test_svhn_build_live():
    cfg = _cfg("svhn", num_classes=10, normalize=True, augment=True)
    ds = build_dataset(cfg, train=True)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["x"].shape == (3, 32, 32)
    assert 0 <= sample["y"] < 10
