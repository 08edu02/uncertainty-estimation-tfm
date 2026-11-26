"""Split conformal prediction: prediction sets with a coverage guarantee.

The recipe, for a calibration set of ``n`` samples and a target error rate
``alpha``:

1. score each calibration sample with ``s(x, y) = 1 - p_y(x)``;
2. take the ``ceil((n + 1) * (1 - alpha))``-th smallest score as the threshold
   (the finite-sample correction is what makes the guarantee hold);
3. include in the prediction set every class whose score is at or below it.

The guarantee is **marginal**: averaged over the data, the true label is
covered with probability at least ``1 - alpha``. It is not conditional on the
class, the subgroup or the individual sample, and it assumes exchangeability
between calibration and test data.
"""

from __future__ import annotations

import math

import numpy as np

from uq.metrics._checks import (
    check_binary_sets,
    check_labels,
    check_predictions,
    check_probabilities,
    check_vector,
)


def nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``1 - p_true`` for each sample, shape ``(n,)``.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape ``(n, n_classes)``.
    labels:
        True integer class labels, shape ``(n,)``.
    """
    checked_probs, checked_labels = check_predictions(probs, labels)
    true_probabilities = checked_probs[np.arange(checked_labels.size), checked_labels]
    return 1.0 - true_probabilities


def conformal_quantile(scores: np.ndarray, *, alpha: float) -> float:
    """Finite-sample corrected quantile of the calibration scores.

    The position is ``ceil((n + 1) * (1 - alpha))`` in 0-based indexing, i.e. the
    ``ceil((n + 1) * (1 - alpha))``-th smallest score. With ``n = 100`` and
    ``alpha = 0.1`` that is the 91st smallest score.

    If there are too few calibration samples for the requested ``alpha``
    (``alpha < 1 / (n + 1)``), the largest score is returned: the widest sets
    the data can support, which is the conservative choice.

    Parameters
    ----------
    scores:
        Calibration nonconformity scores, shape ``(n,)``.
    alpha:
        Target error rate in ``(0, 1)``; ``0.1`` targets 90 % coverage.
    """
    values = check_vector(scores, name="scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    position = math.ceil((values.size + 1) * (1.0 - alpha))
    index = min(max(position - 1, 0), values.size - 1)
    return float(np.sort(values)[index])


def prediction_sets(probs: np.ndarray, quantile: float) -> np.ndarray:
    """Boolean matrix marking, for each sample, which classes enter the set.

    A class enters the set when its nonconformity score ``1 - p`` is at most
    ``quantile``. A fully confident prediction can produce a single-class set;
    a binomial-style set is not enforced.
    """
    checked_probs = check_probabilities(probs)
    if not np.isfinite(quantile):
        raise ValueError(f"quantile must be finite, got {quantile}")

    sets = (1.0 - checked_probs) <= quantile
    if np.any(sets.sum(axis=1) == 0):
        raise ValueError("quantile excludes every class for some sample; check its scale")
    return sets


def empirical_coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of samples whose true label is inside the prediction set."""
    checked_sets, checked_labels = _check_sets_and_labels(sets, labels)
    covered = checked_sets[np.arange(checked_labels.size), checked_labels]
    return float(np.mean(covered))


def average_set_size(sets: np.ndarray) -> float:
    """Mean number of classes per prediction set; ``1`` is the sharpest possible."""
    values = np.asarray(sets)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"sets must be 2-D with at least one row, got shape {values.shape}")

    checked = check_binary_sets(values, n_samples=values.shape[0], n_classes=values.shape[1])
    return float(checked.sum(axis=1).mean())


def coverage_by_class(sets: np.ndarray, labels: np.ndarray) -> dict[int, float]:
    """Empirical coverage separately for each class present in ``labels``.

    The marginal guarantee says nothing about these numbers: reporting them is
    how a class that is badly served becomes visible.
    """
    checked_sets, checked_labels = _check_sets_and_labels(sets, labels)

    coverage: dict[int, float] = {}
    for class_index in np.unique(checked_labels):
        rows = checked_labels == class_index
        covered = checked_sets[rows, class_index]
        coverage[int(class_index)] = float(np.mean(covered))
    return coverage


def _check_sets_and_labels(sets: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate a prediction-set matrix together with its labels."""
    values = np.asarray(sets)
    if values.ndim != 2:
        raise ValueError(f"sets must be 2-D (n_samples, n_classes), got shape {values.shape}")

    checked_sets = check_binary_sets(values, n_samples=values.shape[0], n_classes=values.shape[1])
    checked_labels = check_labels(
        labels, n_samples=checked_sets.shape[0], n_classes=checked_sets.shape[1]
    )
    return checked_sets, checked_labels
