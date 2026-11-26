"""Derived views of a prediction: the chosen class, its confidence and correctness.

These three functions exist so that calibration, error detection and selective
classification all use the same definition of "confidence" (the probability of
the predicted class) and of "correct" (the prediction matches the label).
Having two subtly different versions of either one is a classic source of
results that look right and are not.
"""

from __future__ import annotations

import numpy as np

from uq.metrics._checks import check_predictions, check_probabilities


def predicted_class(probs: np.ndarray) -> np.ndarray:
    """Index of the most probable class for each sample, shape ``(n,)``."""
    return check_probabilities(probs).argmax(axis=1)


def predicted_confidence(probs: np.ndarray) -> np.ndarray:
    """Probability assigned to the predicted class, shape ``(n,)``."""
    return check_probabilities(probs).max(axis=1)


def is_correct(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Whether the predicted class equals the true label, shape ``(n,)``."""
    checked_probs, checked_labels = check_predictions(probs, labels)
    predictions = checked_probs.argmax(axis=1)
    return predictions == checked_labels
