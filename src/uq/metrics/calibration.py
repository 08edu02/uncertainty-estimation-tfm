"""Calibration metrics: ECE, Brier score, negative log-likelihood and reliability.

All of them describe the same thing from different angles: whether the
probability a model reports matches the frequency with which it is right.
Lower is better in the three scalar scores.

Definitions used throughout the thesis (declare them in the text):

* ``ECE = sum_m (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|``, where
  ``confidence`` is the probability of the predicted class and ``B_m`` are
  bins over that confidence.
* ``Brier = mean_i sum_k (p_ik - y_ik)^2``.
* ``NLL = -mean_i log p_i,y_i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from uq.metrics._checks import check_predictions, check_vector
from uq.metrics.predictions import is_correct, predicted_confidence

DEFAULT_BINS = 15
BinningStrategy = Literal["uniform", "quantile"]

#: Probabilities are clipped here before the logarithm, so that ``p = 0`` yields
#: a large finite NLL instead of an infinity.
MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per-bin summary used to draw a reliability diagram.

    Attributes
    ----------
    lower, upper:
        Bin edges over the confidence of the predicted class.
    counts:
        Number of samples in each bin; empty bins keep their count and have
        ``nan`` in the two means below.
    mean_confidence:
        Average confidence of the samples in the bin.
    accuracy:
        Fraction of correct predictions in the bin.
    """

    lower: np.ndarray
    upper: np.ndarray
    counts: np.ndarray
    mean_confidence: np.ndarray
    accuracy: np.ndarray

    @property
    def gap(self) -> np.ndarray:
        """Signed difference ``accuracy - confidence``; positive means underconfident."""
        return self.accuracy - self.mean_confidence


def bin_indices(
    confidence: np.ndarray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: BinningStrategy = "uniform",
) -> np.ndarray:
    """Assign each confidence value to a bin index in ``[0, n_bins - 1]``.

    ``uniform`` uses equal-width bins over ``[0, 1]`` (the ECE definition
    declared in the thesis). ``quantile`` puts the same number of samples in
    each bin and is reported only as a complement, because it makes the ECE
    depend on the sample.
    """
    values = check_vector(confidence, name="confidence")
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("confidence must be in [0, 1]")

    if strategy == "uniform":
        # ``confidence == 1`` would land on index ``n_bins``, hence the clip.
        scaled = np.floor(values * n_bins).astype(int)
        return np.clip(scaled, 0, n_bins - 1)

    if strategy == "quantile":
        n_samples = values.size
        if n_bins > n_samples:
            raise ValueError(
                f"quantile binning needs n_bins <= n_samples, got {n_bins} and {n_samples}"
            )
        order = np.argsort(values, kind="stable")
        ranks = np.empty(n_samples, dtype=int)
        ranks[order] = np.arange(n_samples)
        return np.minimum((ranks * n_bins) // n_samples, n_bins - 1)

    raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: BinningStrategy = "uniform",
) -> ReliabilityCurve:
    """Binned confidence and accuracy for a reliability diagram.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape ``(n, n_classes)``.
    labels:
        Integer class labels, shape ``(n,)``.
    n_bins:
        Number of bins.
    strategy:
        ``uniform`` (equal width) or ``quantile`` (equal mass).
    """
    checked_probs, checked_labels = check_predictions(probs, labels)
    confidence = predicted_confidence(checked_probs)
    correct = is_correct(checked_probs, checked_labels).astype(float)

    index = bin_indices(confidence, n_bins=n_bins, strategy=strategy)

    counts = np.bincount(index, minlength=n_bins)
    confidence_sum = np.bincount(index, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(index, weights=correct, minlength=n_bins)

    empty = np.full(n_bins, np.nan)
    mean_confidence = np.divide(confidence_sum, counts, out=empty.copy(), where=counts > 0)
    accuracy = np.divide(correct_sum, counts, out=empty.copy(), where=counts > 0)

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        lower = edges[:-1]
        upper = edges[1:]
    else:
        ordered = np.sort(confidence)
        boundaries = np.array_split(ordered, n_bins)
        lower = np.array([block[0] for block in boundaries])
        upper = np.array([block[-1] for block in boundaries])

    return ReliabilityCurve(
        lower=lower,
        upper=upper,
        counts=counts,
        mean_confidence=mean_confidence,
        accuracy=accuracy,
    )


def ece(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: BinningStrategy = "uniform",
) -> float:
    """Expected calibration error of the predicted class.

    Bins the confidence of the predicted class into ``n_bins`` intervals and
    averages ``|accuracy - confidence|`` weighted by bin size. Empty bins are
    ignored. Lower is better; ``0`` means perfectly calibrated.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape ``(n, n_classes)``.
    labels:
        Integer class labels, shape ``(n,)``.
    n_bins:
        Number of bins.
    strategy:
        ``uniform`` (equal width) or ``quantile`` (equal mass).

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """
    curve = reliability_curve(probs, labels, n_bins=n_bins, strategy=strategy)

    populated = curve.counts > 0
    gaps = np.abs(curve.accuracy[populated] - curve.mean_confidence[populated])
    weights = curve.counts[populated] / curve.counts.sum()
    return float(np.sum(weights * gaps))


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Brier score: mean squared error between probabilities and one-hot labels.

    ``0`` is a perfect and fully confident model; lower is better.
    """
    checked_probs, checked_labels = check_predictions(probs, labels)
    one_hot = np.eye(checked_probs.shape[1])[checked_labels]
    squared_errors = (checked_probs - one_hot) ** 2
    return float(squared_errors.sum(axis=1).mean())


def nll(
    probs: np.ndarray, labels: np.ndarray, *, min_probability: float = MIN_PROBABILITY
) -> float:
    """Negative log-likelihood of the true class, in nats.

    Probabilities are clipped at ``min_probability`` so that a ``0`` yields a
    large finite value instead of ``inf``.
    """
    checked_probs, checked_labels = check_predictions(probs, labels)
    if min_probability <= 0.0:
        raise ValueError(f"min_probability must be positive, got {min_probability}")

    true_probabilities = checked_probs[np.arange(checked_labels.size), checked_labels]
    clipped = np.clip(true_probabilities, min_probability, 1.0)
    return float(-np.log(clipped).mean())
