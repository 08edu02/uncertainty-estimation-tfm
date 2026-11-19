"""Figures produced during the exploratory analysis.

The functions only draw and save: they never print nor read files, so the
scripts and the notebooks stay in control of what is produced where.
All figures are saved at 200 dpi as PNG (the thesis will export vector copies
from the final result scripts).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no interactive backend needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 200


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return path


def plot_credit_class_balance(frame: pd.DataFrame, target: str, path: Path) -> Path:
    """Bar chart with the size of each class of the target."""
    counts = frame[target].value_counts().sort_index()
    figure, axes = plt.subplots(figsize=(4, 3))
    axes.bar([str(index) for index in counts.index], counts.to_numpy(), color="#4c72b0")
    for position, value in enumerate(counts.to_numpy()):
        axes.text(position, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    axes.set_xlabel("default (0 = no, 1 = yes)")
    axes.set_ylabel("clients")
    axes.set_title("Class balance")
    return _save(figure, path)


def plot_credit_distributions(frame: pd.DataFrame, path: Path, *, bins: int = 40) -> Path:
    """Histograms of credit limit, age and latest repayment status."""
    figure, axes = plt.subplots(1, 3, figsize=(10, 3))

    axes[0].hist(frame["limit_bal"].to_numpy(), bins=bins, color="#4c72b0")
    axes[0].set_title("Credit limit (NT$)")
    axes[0].set_xlabel("limit_bal")

    axes[1].hist(frame["age"].to_numpy(), bins=bins, color="#55a868")
    axes[1].set_title("Age (years)")
    axes[1].set_xlabel("age")

    status = frame["pay_0"].value_counts().sort_index()
    axes[2].bar(status.index.astype(str), status.to_numpy(), color="#c44e52")
    axes[2].set_title("Repayment status, September")
    axes[2].set_xlabel("pay_0")

    return _save(figure, path)


def plot_split_balance(
    counts: dict[str, dict[str, int]],
    path: Path,
    *,
    title: str,
    class_names: tuple[str, ...] = ("0", "1"),
) -> Path:
    """Grouped bar chart comparing class counts across splits."""
    splits = list(counts)
    positions = np.arange(len(splits))
    width = 0.8 / max(len(class_names), 1)

    figure, axes = plt.subplots(figsize=(6, 3))
    for offset, name in enumerate(class_names):
        values = [counts[split].get(str(index), 0) for split in splits for index in [offset]]
        axes.bar(positions + offset * width, values, width=width, label=f"class {name}")

    axes.set_xticks(positions + width * (len(class_names) - 1) / 2)
    axes.set_xticklabels(splits)
    axes.set_ylabel("images")
    axes.set_title(title)
    axes.legend()
    return _save(figure, path)


def plot_image_samples(
    images: np.ndarray,
    labels: np.ndarray,
    path: Path,
    *,
    class_names: tuple[str, ...],
    n: int = 8,
    seed: int = 0,
) -> Path:
    """Grid of random samples, two per class."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).reshape(-1)
    per_class = max(n // len(class_names), 1)

    picks: list[int] = []
    for index in range(len(class_names)):
        candidates = np.flatnonzero(labels == index)
        if len(candidates):
            picks.extend(
                rng.choice(candidates, size=min(per_class, len(candidates)), replace=False)
            )
    picks = [int(index) for index in picks]

    columns = per_class
    rows = max(len(picks) // columns, 1)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.8))
    axes_array = np.atleast_1d(axes).reshape(-1)

    for axis, index in zip(axes_array, picks, strict=False):
        image = images[index]
        axis.imshow(image.squeeze(), cmap="gray")
        axis.set_title(class_names[int(labels[index])], fontsize=8)
        axis.axis("off")
    for axis in axes_array[len(picks) :]:
        axis.axis("off")

    return _save(figure, path)
