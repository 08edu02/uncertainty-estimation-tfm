"""Tests for error detection and predictive entropy."""

from __future__ import annotations

import numpy as np
import pytest

from uq.metrics.error_detection import (
    error_auprc,
    error_auroc,
    predictive_entropy,
)

LN_2 = float(np.log(2.0))


def test_entropy_of_a_confident_prediction_is_zero() -> None:
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert predictive_entropy(probs) == pytest.approx(np.zeros(2), abs=1e-12)


def test_entropy_of_a_coin_flip_is_one_bit() -> None:
    probs = np.array([[0.5, 0.5]])

    assert predictive_entropy(probs) == pytest.approx(np.array([LN_2]), abs=1e-12)
    assert predictive_entropy(probs, unit="bits") == pytest.approx(np.array([1.0]), abs=1e-12)


def test_entropy_is_zero_for_a_probability_that_is_exactly_zero() -> None:
    probs = np.array([[0.0, 1.0]])

    assert predictive_entropy(probs) == pytest.approx(np.zeros(1), abs=1e-12)


def test_auroc_is_one_when_uncertainty_ranks_every_error_above_every_hit() -> None:
    uncertainty = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    correct = np.array([False, False, False, True, True])

    assert error_auroc(uncertainty, correct) == pytest.approx(1.0)


def test_auroc_is_zero_when_the_ranking_is_inverted() -> None:
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    correct = np.array([False, False, True, True])

    assert error_auroc(uncertainty, correct) == pytest.approx(0.0)


def test_auroc_is_one_half_when_the_score_carries_no_information() -> None:
    uncertainty = np.zeros(6)
    correct = np.array([True, False, True, False, True, False])

    assert error_auroc(uncertainty, correct) == pytest.approx(0.5)


def test_auroc_is_none_when_there_are_no_errors() -> None:
    uncertainty = np.array([0.1, 0.2, 0.3])
    correct = np.array([True, True, True])

    assert error_auroc(uncertainty, correct) is None


def test_auroc_is_none_when_every_prediction_is_wrong() -> None:
    uncertainty = np.array([0.1, 0.2, 0.3])
    correct = np.array([False, False, False])

    assert error_auroc(uncertainty, correct) is None


def test_average_precision_is_one_for_a_perfect_detector() -> None:
    uncertainty = np.array([0.9, 0.8, 0.1, 0.05])
    correct = np.array([False, False, True, True])

    assert error_auprc(uncertainty, correct) == pytest.approx(1.0)


def test_average_precision_is_none_when_there_are_no_errors() -> None:
    uncertainty = np.array([0.1, 0.2])
    correct = np.array([True, True])

    assert error_auprc(uncertainty, correct) is None


def test_scores_and_correctness_must_have_the_same_length() -> None:
    uncertainty = np.array([0.1, 0.2, 0.3])
    correct = np.array([True, False])

    with pytest.raises(ValueError, match="correct must have 3 elements"):
        error_auroc(uncertainty, correct)


def test_a_two_dimensional_uncertainty_is_rejected() -> None:
    uncertainty = np.array([[0.1], [0.2]])
    correct = np.array([True, False])

    with pytest.raises(ValueError, match="uncertainty must be 1-D"):
        error_auroc(uncertainty, correct)


def test_non_binary_correctness_is_rejected() -> None:
    uncertainty = np.array([0.1, 0.2])
    correct = np.array([0.5, 1.0])

    with pytest.raises(ValueError, match="correct must be boolean"):
        error_auroc(uncertainty, correct)
