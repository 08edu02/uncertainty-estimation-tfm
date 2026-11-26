"""Tests for the calibration metrics, with answers that can be checked by hand."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from uq.metrics.calibration import (
    BinningStrategy,
    bin_indices,
    brier,
    ece,
    nll,
    reliability_curve,
)

LN_2 = float(np.log(2.0))


def _binary_constant(confidence: float, n_samples: int) -> np.ndarray:
    """Probabilities where every sample puts ``confidence`` on class 1."""
    return np.array([[1.0 - confidence, confidence]] * n_samples)


def test_ece_is_zero_for_a_perfectly_calibrated_model() -> None:
    probs = _binary_constant(confidence=0.5, n_samples=100)
    labels = np.array([0, 1] * 50)

    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-12)


def test_ece_is_the_confidence_when_the_model_is_always_wrong() -> None:
    probs = _binary_constant(confidence=0.9, n_samples=100)
    labels = np.zeros(100, dtype=int)

    assert ece(probs, labels) == pytest.approx(0.9, abs=1e-12)


def test_ece_with_a_single_bin_is_the_overall_gap() -> None:
    probs = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
        ]
    )
    labels = np.array([1, 0, 0, 0])

    mean_confidence = float(probs.max(axis=1).mean())
    accuracy = 0.75

    assert ece(probs, labels, n_bins=1) == pytest.approx(abs(accuracy - mean_confidence))


def test_ece_ignores_empty_bins_instead_of_dividing_by_zero() -> None:
    probs = _binary_constant(confidence=0.999, n_samples=10)
    labels = np.ones(10, dtype=int)

    assert ece(probs, labels, n_bins=15) == pytest.approx(1.0 - 0.999, abs=1e-12)


def test_ece_does_not_depend_on_the_order_of_the_rows() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 3))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 3, size=200)
    permutation = rng.permutation(200)

    shuffled = ece(probs[permutation], labels[permutation])

    assert ece(probs, labels) == pytest.approx(shuffled, abs=1e-12)


def test_brier_is_zero_for_a_confident_and_correct_model() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    assert brier(probs, labels) == pytest.approx(0.0, abs=1e-12)


def test_brier_of_a_coin_flip_on_the_true_class_is_one_half() -> None:
    probs = np.array([[0.5, 0.5]])
    labels = np.array([1])

    assert brier(probs, labels) == pytest.approx(0.5, abs=1e-12)


def test_brier_multiclass_sums_over_every_class() -> None:
    probs = np.array([[0.7, 0.2, 0.1]])
    labels = np.array([0])
    expected = (0.3**2) + (0.2**2) + (0.1**2)

    assert brier(probs, labels) == pytest.approx(expected, abs=1e-12)


def test_nll_of_a_confident_correct_prediction_is_zero() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])

    assert nll(probs, labels) == pytest.approx(0.0, abs=1e-12)


def test_nll_of_a_coin_flip_is_the_logarithm_of_two() -> None:
    probs = np.array([[0.5, 0.5]])
    labels = np.array([1])

    assert nll(probs, labels) == pytest.approx(LN_2, abs=1e-12)


def test_nll_stays_finite_when_the_true_class_has_zero_probability() -> None:
    probs = np.array([[1.0, 0.0]])
    labels = np.array([1])

    value = nll(probs, labels)

    assert np.isfinite(value)
    assert value == pytest.approx(-np.log(1e-12), rel=1e-12)


def test_reliability_curve_counts_every_sample_once() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(150, 2))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = rng.integers(0, 2, size=150)

    curve = reliability_curve(probs, labels, n_bins=15)

    assert curve.counts.sum() == 150
    assert np.all(curve.counts >= 0)


def test_empty_bins_are_reported_as_nan() -> None:
    probs = np.array([[0.99, 0.01]] * 20)
    labels = np.zeros(20, dtype=int)

    curve = reliability_curve(probs, labels, n_bins=15)

    empty = curve.counts == 0
    assert empty.any()
    assert np.isnan(curve.mean_confidence[empty]).all()
    assert np.isnan(curve.accuracy[empty]).all()


def test_quantile_binning_gives_each_bin_the_same_number_of_samples() -> None:
    confidence = np.linspace(0.0, 1.0, 100)

    index = bin_indices(confidence, n_bins=5, strategy="quantile")
    counts = np.bincount(index, minlength=5)

    assert np.array_equal(counts, np.array([20, 20, 20, 20, 20]))


def test_uniform_binning_puts_confidence_one_in_the_last_bin() -> None:
    confidence = np.array([0.0, 0.5, 1.0])

    index = bin_indices(confidence, n_bins=15)

    assert index[0] == 0
    assert index[2] == 14


def test_quantile_binning_rejects_more_bins_than_samples() -> None:
    confidence = np.linspace(0.0, 1.0, 5)

    with pytest.raises(ValueError, match="n_bins <= n_samples"):
        bin_indices(confidence, n_bins=10, strategy="quantile")


def test_probabilities_that_do_not_sum_to_one_are_rejected() -> None:
    probs = np.array([[0.7, 0.7]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="must sum to 1"):
        ece(probs, labels)


def test_a_probability_matrix_with_one_class_is_rejected() -> None:
    probs = np.ones((4, 1))
    labels = np.zeros(4, dtype=int)

    with pytest.raises(ValueError, match="at least two classes"):
        brier(probs, labels)


def test_labels_of_the_wrong_length_are_rejected() -> None:
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="labels must have 2 elements"):
        nll(probs, labels)


def test_labels_outside_the_class_range_are_rejected() -> None:
    probs = np.array([[0.5, 0.5]])
    labels = np.array([2])

    with pytest.raises(ValueError, match=r"labels must be in \[0, 1\]"):
        nll(probs, labels)


def test_unknown_binning_strategy_is_rejected() -> None:
    probs = np.array([[0.5, 0.5]])
    labels = np.array([0])
    unknown_strategy = cast(BinningStrategy, "equal-mass")

    with pytest.raises(ValueError, match="strategy must be"):
        ece(probs, labels, strategy=unknown_strategy)
