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

#: Order in which the official splits are concatenated by :func:`load_pooled`.
#: The indices produced by :func:`uq.data.splits.make_image_split` refer to that
#: concatenation, so the two must always be used together.
POOL_ORDER: tuple[str, ...] = ("train", "val", "test")


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


def load_pooled(
    root: Path | None = None,
    *,
    size: int = DEFAULT_SIZE,
    download_if_missing: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    """Concatenate the official splits in :data:`POOL_ORDER`.

    A single array with one index space is what makes the partition auditable:
    the training code and :func:`uq.data.splits.make_image_split` share the same
    positions, so a test index can never be confused with a development one.

    Parameters
    ----------
    root:
        Data folder; defaults to :func:`uq.settings.data_dir`.
    size:
        Resolution passed to MedMNIST; the project uses 28.
    download_if_missing:
        Whether to download the ``.npz`` files when they are absent.

    Returns
    -------
    images, labels, bounds
        Images of shape ``(N, size, size, 1)``, labels of shape ``(N,)`` and the
        slice each official split occupies in the concatenation.
    """
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    bounds: dict[str, slice] = {}

    offset = 0
    for split in POOL_ORDER:
        split_images, split_labels = load_split(
            split, root, size=size, download_if_missing=download_if_missing
        )
        bounds[split] = slice(offset, offset + split_labels.size)
        offset += split_labels.size
        images.append(np.asarray(split_images))
        labels.append(split_labels)

    return np.concatenate(images, axis=0), np.concatenate(labels), bounds
