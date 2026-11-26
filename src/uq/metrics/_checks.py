"""Input validation shared by the metric modules.

Every public metric ends up here before doing arithmetic. The checks are
deliberately strict: a metric that silently accepts the wrong shape returns a
plausible number, and a plausible wrong number is the most expensive bug in a
thesis.
"""

from __future__ import annotations

import numpy as np


def check_probabilities(probs: np.ndarray, *, n_classes: int | None = None) -> np.ndarray:
    """Validate a matrix of predicted probabilities.

    Parameters
    ----------
    probs:
        Array of shape ``(n_samples, n_classes)``.
    n_classes:
        Expected number of classes, if the caller has one.

    Returns
    -------
    np.ndarray
        The same values as ``float``, after validation.
    """
    values = np.asarray(probs, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_samples, n_classes), got shape {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("probs must contain at least one sample")
    if values.shape[1] < 2:
        raise ValueError(f"probs must have at least two classes, got {values.shape[1]}")
    if n_classes is not None and values.shape[1] != n_classes:
        raise ValueError(f"probs must have {n_classes} classes, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("probs contains non-finite values")
    if np.any(values < 0.0):
        raise ValueError("probs contains negative values")

    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("rows of probs must sum to 1")

    return values


def check_labels(labels: np.ndarray, *, n_samples: int, n_classes: int) -> np.ndarray:
    """Validate integer class labels against the expected sample count."""
    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError(f"labels must be 1-D (n_samples,), got shape {values.shape}")
    if values.shape[0] != n_samples:
        raise ValueError(f"labels must have {n_samples} elements, got {values.shape[0]}")
    if values.dtype.kind not in "iu":
        raise ValueError(f"labels must be integers, got dtype {values.dtype}")

    integer_values = values.astype(np.int64)
    if np.any(integer_values < 0) or np.any(integer_values >= n_classes):
        raise ValueError(f"labels must be in [0, {n_classes - 1}]")
    return integer_values


def check_predictions(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a pair of probability matrix and labels, returning both cast."""
    checked_probs = check_probabilities(probs)
    checked_labels = check_labels(
        labels, n_samples=checked_probs.shape[0], n_classes=checked_probs.shape[1]
    )
    return checked_probs, checked_labels


def check_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a 1-D array of finite numbers (confidences, uncertainties, scores)."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one element")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def check_correctness(correct: np.ndarray, *, n_samples: int) -> np.ndarray:
    """Validate a boolean-able vector saying whether each prediction was right."""
    values = np.asarray(correct)

    if values.ndim != 1:
        raise ValueError(f"correct must be 1-D, got shape {values.shape}")
    if values.shape[0] != n_samples:
        raise ValueError(f"correct must have {n_samples} elements, got {values.shape[0]}")
    if values.dtype.kind not in "bi":
        raise ValueError(f"correct must be boolean or 0/1 integers, got dtype {values.dtype}")
    return values.astype(bool)


def check_binary_sets(sets: np.ndarray, *, n_samples: int, n_classes: int) -> np.ndarray:
    """Validate a boolean matrix of prediction sets."""
    values = np.asarray(sets)
    expected = (n_samples, n_classes)

    if values.shape != expected:
        raise ValueError(f"sets must have shape {expected}, got {values.shape}")
    if values.dtype.kind != "b":
        raise ValueError(f"sets must be boolean, got dtype {values.dtype}")
    if np.any(values.sum(axis=1) == 0):
        raise ValueError("every prediction set must contain at least one class")
    return values
