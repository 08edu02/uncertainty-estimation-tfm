"""JSON-serialisable summaries used by the EDA scripts and notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any) -> float | int | None:
    """Convert numpy scalars to plain Python numbers (``NaN`` -> ``None``)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    number = float(value)
    return None if np.isnan(number) else round(number, 6)


def tabular_summary(
    frame: pd.DataFrame,
    target: str,
    *,
    sensitive: tuple[str, ...] = (),
    categorical: tuple[str, ...] = (),
    top_k: int = 10,
) -> dict[str, Any]:
    """Summarise a tabular dataset: size, duplicates, target balance and groups.

    Parameters
    ----------
    frame:
        Cleaned dataset, including the ``target`` column.
    target:
        Name of the binary target column.
    sensitive:
        Columns whose positive rate is reported separately (bias check).
    categorical:
        Columns treated as categories when counting values.
    top_k:
        Maximum number of categories reported per column.
    """
    if target not in frame.columns:
        raise KeyError(f"target column {target!r} not found")
    if not set(sensitive) <= set(frame.columns):
        raise KeyError("some sensitive columns are missing from the frame")

    features = frame.drop(columns=[target])
    counts = frame[target].value_counts().sort_index()

    columns: dict[str, dict[str, Any]] = {}
    for name in features.columns:
        series = features[name]
        entry: dict[str, Any] = {
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique()),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry |= {
                "min": _number(series.min()),
                "max": _number(series.max()),
                "mean": _number(series.mean()),
                "std": _number(series.std()),
            }
        columns[name] = entry

    categories: dict[str, dict[str, int]] = {}
    for name in categorical:
        if name in features.columns:
            value_counts = features[name].value_counts().head(top_k)
            categories[name] = {str(key): int(value) for key, value in value_counts.items()}

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for name in sensitive:
        frame_by_group = frame.groupby(name, observed=True)
        groups[name] = {
            str(key): {
                "n": len(block),
                "positive_rate": _number(block[target].mean()),
            }
            for key, block in frame_by_group
        }

    return {
        "n_rows": len(frame),
        "n_columns": len(frame.columns),
        "n_features": len(features.columns),
        "n_duplicate_rows": int(features.duplicated().sum()),
        "n_missing_values": int(frame.isna().sum().sum()),
        "target": {
            "name": target,
            "counts": {str(key): int(value) for key, value in counts.items()},
            "positive_rate": _number(frame[target].mean()),
        },
        "columns": columns,
        "categories": categories,
        "groups": groups,
    }


def image_summary(images: np.ndarray, labels: np.ndarray, *, split: str) -> dict[str, Any]:
    """Summarise one split of an image dataset."""
    flat = images.astype(np.float32)
    counts = np.bincount(np.asarray(labels).reshape(-1), minlength=2)
    total = int(counts.sum())

    return {
        "split": split,
        "n_samples": total,
        "shape": [int(dimension) for dimension in images.shape],
        "dtype": str(images.dtype),
        "pixel_min": _number(flat.min()),
        "pixel_max": _number(flat.max()),
        "pixel_mean": _number(flat.mean()),
        "pixel_std": _number(flat.std()),
        "class_counts": {str(index): int(count) for index, count in enumerate(counts)},
        "class_rates": {
            str(index): _number(count / total if total else 0.0)
            for index, count in enumerate(counts)
        },
    }
