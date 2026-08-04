from __future__ import annotations

import os
import urllib.request
import zipfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from fls.data.label_noise import symmetric_label_noise

# Per-dataset normalization statistics and image geometry.
#
# ``mean``/``std`` are the channel-wise normalization constants (gated by the
# ``normalize`` flag).  ``native_size`` is the source image resolution and
# ``working_size`` is the resolution the models actually consume.  The repo's
# models are CIFAR-sized (32x32), so STL-10 (96x96) and Tiny-ImageNet (64x64)
# are resized down to 32x32 so existing models work unchanged.
#
# CIFAR mean/std are kept byte-for-byte identical to the original
# implementation so CIFAR results do not change.
DATASET_STATS: dict[str, dict[str, Any]] = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "native_size": 32,
        "working_size": 32,
    },
    "cifar100": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "native_size": 32,
        "working_size": 32,
    },
    "svhn": {
        "mean": (0.4377, 0.4438, 0.4728),
        "std": (0.1980, 0.2010, 0.1970),
        "native_size": 32,
        "working_size": 32,
    },
    "stl10": {
        "mean": (0.4467, 0.4398, 0.4066),
        "std": (0.2603, 0.2566, 0.2713),
        "native_size": 96,
        "working_size": 32,
    },
    "tiny_imagenet": {
        "mean": (0.4802, 0.4481, 0.3975),
        "std": (0.2770, 0.2691, 0.2821),
        "native_size": 64,
        "working_size": 32,
    },
    "oxford_iiit_pet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "native_size": 224,
        "working_size": 224,
    },
    "flowers102": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "native_size": 224,
        "working_size": 224,
    },
    "fake": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "native_size": 32,
        "working_size": 32,
    },
}

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def dataset_stats(name: str) -> dict[str, Any]:
    """Return the normalization/geometry mapping for ``name``.

    Falls back to CIFAR-like defaults (32x32, no resize) for unknown names so
    callers (e.g. ``fake``) still get a sensible transform.
    """
    return DATASET_STATS.get(name.lower(), DATASET_STATS["cifar10"])


class DictDataset(Dataset):
    """Wrap classification datasets to return dictionaries with clean/noisy labels."""

    def __init__(
        self,
        base: Dataset,
        num_classes: int,
        noise_rate: float = 0.0,
        noise_seed: int = 0,
        return_clean_labels: bool = True,
    ) -> None:
        self.base = base
        self.return_clean_labels = return_clean_labels
        labels = np.asarray(_extract_labels(base), dtype=np.int64)
        self.clean_labels = labels
        self.noisy_labels, self.is_corrupted = symmetric_label_noise(labels, num_classes, noise_rate, noise_seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        x, _ = self.base[index]
        sample = {
            "x": x,
            "y": int(self.noisy_labels[index]),
            "y_clean": int(self.clean_labels[index]),
            "is_corrupted": bool(self.is_corrupted[index]),
            "index": index,
        }
        if not self.return_clean_labels:
            sample.pop("y_clean")
        return sample


def _extract_labels(base: Dataset) -> Any:
    """Read integer labels from a torchvision dataset efficiently.

    Different torchvision datasets expose labels under different attributes:
    CIFAR/ImageFolder use ``.targets``, while SVHN and STL-10 use ``.labels``.
    Reading the native attribute avoids materializing every image (which the
    ``base[i][1]`` fallback would do, decoding/transforming each sample).
    """
    if hasattr(base, "targets"):
        return base.targets
    if hasattr(base, "labels"):
        return base.labels
    if hasattr(base, "_labels"):
        return base._labels
    return [base[i][1] for i in range(len(base))]


def _transforms(config: dict[str, Any], train: bool, name: str = "cifar10") -> transforms.Compose:
    """Build the transform pipeline for ``name``.

    The working image size and normalization statistics are looked up per
    dataset.  CIFAR/SVHN are native 32x32: train augmentation is
    ``RandomCrop(working_size, padding=4) + HFlip`` exactly as before.  STL-10
    and Tiny-ImageNet are larger natively and are resized to the working size;
    train augmentation uses ``RandomResizedCrop`` (crop-then-resize) while eval
    uses a plain ``Resize``.
    """
    stats = dataset_stats(name)
    native_size = int(stats["native_size"])
    working_size = int(stats["working_size"])
    needs_resize = native_size != working_size

    ops: list[Any] = []
    augment = bool(train and config.get("augment", False))

    if name in {"oxford_iiit_pet", "flowers102"}:
        if augment:
            ops.extend(
                [
                    transforms.RandomResizedCrop(working_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            resize_size = int(round(working_size * 256 / 224))
            ops.extend([transforms.Resize(resize_size), transforms.CenterCrop(working_size)])
        ops.append(transforms.ToTensor())
        if config.get("normalize", False):
            ops.append(transforms.Normalize(stats["mean"], stats["std"]))
        return transforms.Compose(ops)

    if needs_resize:
        # Larger source images (STL-10, Tiny-ImageNet): resize to the model's
        # working size.  Use a random-resized crop for augmented training and a
        # deterministic resize otherwise.
        if augment:
            ops.append(transforms.RandomResizedCrop(working_size, scale=(0.6, 1.0)))
            ops.append(transforms.RandomHorizontalFlip())
        else:
            ops.append(transforms.Resize((working_size, working_size)))
    else:
        # Native 32x32 sources (CIFAR, SVHN): keep the original padded-crop aug.
        if augment:
            ops.extend(
                [transforms.RandomCrop(working_size, padding=4), transforms.RandomHorizontalFlip()]
            )

    ops.append(transforms.ToTensor())
    if config.get("normalize", False):
        ops.append(transforms.Normalize(stats["mean"], stats["std"]))
    return transforms.Compose(ops)


def _tiny_imagenet_root(root: str) -> str:
    """Download (if missing) and return the extracted tiny-imagenet-200 dir."""
    extracted = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(os.path.join(extracted, "train")):
        return extracted
    os.makedirs(root, exist_ok=True)
    archive = os.path.join(root, "tiny-imagenet-200.zip")
    if not os.path.exists(archive):
        urllib.request.urlretrieve(TINY_IMAGENET_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return extracted


def _build_tiny_imagenet(root: str, train: bool, transform: Any) -> Dataset:
    """Build a Tiny-ImageNet split as an ImageFolder.

    The train split has a standard ``train/<wnid>/images/*.JPEG`` layout.  The
    val split ships as a flat ``val/images`` dir plus ``val_annotations.txt``;
    we materialize a class-subdir layout once so ``ImageFolder`` can read it and
    its ``.targets`` are consistent with the train split's class ordering.
    """
    base_dir = _tiny_imagenet_root(root)
    train_dir = os.path.join(base_dir, "train")
    if train:
        return datasets.ImageFolder(train_dir, transform=transform)

    val_dir = os.path.join(base_dir, "val")
    structured = os.path.join(val_dir, "structured")
    if not os.path.isdir(structured):
        # Reorganize val/images into per-class subdirectories.
        annotations = os.path.join(val_dir, "val_annotations.txt")
        images_dir = os.path.join(val_dir, "images")
        os.makedirs(structured, exist_ok=True)
        with open(annotations) as fh:
            for line in fh:
                fields = line.split("\t")
                fname, wnid = fields[0], fields[1]
                cls_dir = os.path.join(structured, wnid)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(images_dir, fname)
                dst = os.path.join(cls_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)
    # Use the train split's class ordering so labels are consistent.
    train_classes = sorted(os.listdir(train_dir))
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    dataset = datasets.ImageFolder(structured, transform=transform)
    # Remap to the train ordering (ImageFolder sorts the val subdirs the same
    # way, but be explicit so labels are guaranteed consistent).
    dataset.classes = train_classes
    dataset.class_to_idx = class_to_idx
    remapped = [class_to_idx[os.path.basename(os.path.dirname(path))] for path, _ in dataset.samples]
    dataset.samples = [(path, remapped[i]) for i, (path, _) in enumerate(dataset.samples)]
    dataset.imgs = dataset.samples
    dataset.targets = remapped
    return dataset


def build_dataset(
    config: dict[str, Any],
    train: bool,
    *,
    augment_override: bool | None = None,
    split: str | None = None,
) -> Dataset:
    """Build a configured dataset.

    ``train`` selects the native dataset split. ``augment_override`` controls
    only whether training-time stochastic transforms are enabled.  The latter
    is useful for constructing an evaluation-transform copy of the training
    split for deterministic validation without changing its examples or
    labels.
    """
    data_cfg = config["data"]
    noise_cfg = config.get("label_noise", {})
    name = data_cfg["name"].lower()
    transform_train = train if augment_override is None else bool(augment_override)
    transform = _transforms(data_cfg, transform_train, name)
    if name == "fake":
        size = data_cfg["train_size"] if train else data_cfg["test_size"]
        base = datasets.FakeData(
            size=size,
            image_size=(data_cfg.get("in_channels", 3), data_cfg.get("image_size", 32), data_cfg.get("image_size", 32)),
            num_classes=data_cfg["num_classes"],
            transform=transform,
            random_offset=0 if train else 10_000,
        )
    elif name == "cifar10":
        base = datasets.CIFAR10(data_cfg["root"], train=train, transform=transform, download=True)
    elif name == "cifar100":
        base = datasets.CIFAR100(data_cfg["root"], train=train, transform=transform, download=True)
    elif name == "svhn":
        split = "train" if train else "test"
        base = datasets.SVHN(data_cfg["root"], split=split, transform=transform, download=True)
    elif name == "stl10":
        split = "train" if train else "test"
        base = datasets.STL10(data_cfg["root"], split=split, transform=transform, download=True)
    elif name == "tiny_imagenet":
        base = _build_tiny_imagenet(data_cfg["root"], train=train, transform=transform)
    elif name == "oxford_iiit_pet":
        dataset_split = split or ("trainval" if train else "test")
        base = datasets.OxfordIIITPet(
            data_cfg["root"],
            split=dataset_split,
            target_types="category",
            transform=transform,
            download=True,
        )
    elif name == "flowers102":
        dataset_split = split or ("train" if train else "test")
        base = datasets.Flowers102(
            data_cfg["root"],
            split=dataset_split,
            transform=transform,
            download=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {data_cfg['name']}")
    dataset = DictDataset(
        base,
        num_classes=data_cfg["num_classes"],
        noise_rate=float(noise_cfg.get("rate", 0.0)) if train else 0.0,
        noise_seed=int(noise_cfg.get("seed", 0)),
        return_clean_labels=bool(data_cfg.get("return_clean_labels", True)),
    )
    subset_fraction = data_cfg.get("train_subset_fraction" if train else "test_subset_fraction")
    if subset_fraction is not None:
        fraction = float(subset_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Subset fraction must be in (0, 1], got {fraction}")
        size = max(1, int(len(dataset) * fraction))
        generator = torch.Generator().manual_seed(int(data_cfg.get("subset_seed", 0)))
        indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
        dataset = Subset(dataset, indices)
    return dataset


def build_official_validation_dataset(config: dict[str, Any]) -> Dataset:
    """Build a dataset's official validation split with evaluation transforms."""
    name = str(config["data"]["name"]).lower()
    if name != "flowers102":
        raise ValueError(f"Dataset {name!r} does not expose an official validation split.")
    return build_dataset(config, train=False, augment_override=False, split="val")


def split_train_val(dataset: Dataset, val_fraction: float, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Disjoint, fixed train/validation split of a training dataset.

    Returns (train_subset, val_subset). The split is deterministic given ``seed``
    so that selection (best checkpoint, iso-accuracy matching, hyperparameters)
    can use validation accuracy while the test set stays untouched.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n = len(dataset)
    k = max(1, int(n * val_fraction))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:k], perm[k:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_train_val_datasets(
    config: dict[str, Any],
    val_fraction: float,
    seed: int = 0,
) -> tuple[Dataset, Dataset, str]:
    """Build paired train/validation subsets with deterministic validation transforms.

    The two full datasets refer to the same native training examples and use
    the same label-noise realization.  Only their transforms differ: training
    retains the configured augmentation, whereas validation always uses the
    deterministic evaluation transform.  The returned SHA-256 digest records
    the ordered validation indices for protocol manifests.
    """
    import hashlib

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    train_full = build_dataset(config, train=True)
    val_full = build_dataset(config, train=True, augment_override=False)
    if len(train_full) != len(val_full):
        raise RuntimeError("Train and validation-transform dataset copies differ in length.")
    n = len(train_full)
    k = max(1, int(n * val_fraction))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator).tolist()
    val_idx, train_idx = perm[:k], perm[k:]
    digest = hashlib.sha256(",".join(str(index) for index in val_idx).encode("utf-8")).hexdigest()
    return Subset(train_full, train_idx), Subset(val_full, val_idx), digest
