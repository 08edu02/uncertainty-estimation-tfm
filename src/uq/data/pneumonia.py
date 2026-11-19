"""Loader for PneumoniaMNIST, part of the MedMNIST v2 collection.

Source (official): https://medmnist.com/  (DOI: 10.1038/s41597-022-01721-8)
Licence of the MedMNIST datasets: CC BY 4.0; the ``medmnist`` code is Apache-2.0.

PneumoniaMNIST is a binary classification task on paediatric chest X-rays
(normal vs. pneumonia). MedMNIST ships official train/validation/test splits;
the thesis keeps the official **test** split untouched and re-splits the
official train+validation pool later on. Nothing is downloaded at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from uq.settings import data_dir

DATASET_NAME = "pneumonia_mnist"
SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT = Literal["train", "val", "test"]

#: Sizes offered by MedMNIST+ for 2D datasets.
SUPPORTED_SIZES: tuple[int, ...] = (28, 64, 128, 224)
DEFAULT_SIZE = 28

CLASS_NAMES: tuple[str, ...] = ("normal", "pneumonia")


def dataset_dir(root: Path | None = None) -> Path:
    """Folder where MedMNIST stores its ``.npz`` files."""
    return (root or data_dir()) / DATASET_NAME


def _dataset(split: SPLIT, root: Path | None, size: int, *, download: bool):
    """Build the ``medmnist`` dataset object for one split."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}, got {size!r}")

    from medmnist import PneumoniaMNIST  # imported lazily: pulls in torch

    target_dir = dataset_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    return PneumoniaMNIST(split=split, root=str(target_dir), download=download, size=size)


def download(root: Path | None = None, *, size: int = DEFAULT_SIZE) -> dict[str, tuple[int, ...]]:
    """Download every split and return ``{split: images shape}``."""
    shapes: dict[str, tuple[int, ...]] = {}
    for split in SPLITS:
        dataset = _dataset(split, root, size, download=True)
        shapes[split] = tuple(int(dimension) for dimension in dataset.imgs.shape)
    return shapes


def load_split(
    split: SPLIT,
    root: Path | None = None,
    *,
    size: int = DEFAULT_SIZE,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images, labels)`` for one split.

    ``images`` keeps the MedMNIST layout ``(N, size, size, C)`` with ``uint8``
    values; ``labels`` is a 1-D ``int64`` array of 0/1 for convenience.
    """
    dataset = _dataset(split, root, size, download=download_if_missing)
    images = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    return images, labels
