"""Tests for the shared views of a prediction: class, confidence and correctness."""

from __future__ import annotations

import numpy as np
import pytest

from uq.metrics.predictions import is_correct, predicted_class, predicted_confidence

PROBS = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
    ]
)


def test_predicted_class_is_the_highest_probability() -> None:
    assert np.array_equal(predicted_class(PROBS), np.array([0, 1, 2]))


def test_predicted_confidence_is_the_probability_of_that_class() -> None:
    assert predicted_confidence(PROBS) == pytest.approx(np.array([0.7, 0.6, 0.6]))


def test_correctness_compares_the_predicted_class_with_the_label() -> None:
    labels = np.array([0, 2, 2])

    assert np.array_equal(is_correct(PROBS, labels), np.array([True, False, True]))


def test_a_tie_between_classes_resolves_to_the_lowest_index() -> None:
    probs = np.array([[0.5, 0.5]])

    assert predicted_class(probs)[0] == 0


def test_confidence_and_correctness_use_the_predicted_class_not_the_positive_one() -> None:
    probs = np.array([[0.4, 0.6]])
    labels = np.array([0])

    assert predicted_confidence(probs)[0] == pytest.approx(0.6)
    assert not is_correct(probs, labels)[0]


def test_probabilities_with_a_negative_value_are_rejected() -> None:
    probs = np.array([[1.2, -0.2]])
    labels = np.array([0])

    with pytest.raises(ValueError, match="negative values"):
        is_correct(probs, labels)


def test_non_finite_probabilities_are_rejected() -> None:
    probs = np.array([[np.nan, 1.0]])

    with pytest.raises(ValueError, match="non-finite"):
        predicted_confidence(probs)


def test_an_empty_probability_matrix_is_rejected() -> None:
    probs = np.empty((0, 2))

    with pytest.raises(ValueError, match="at least one sample"):
        predicted_class(probs)


def test_a_matrix_of_zero_dimensions_is_rejected() -> None:
    probs = np.array([0.5, 0.5])

    with pytest.raises(ValueError, match="probs must be 2-D"):
        predicted_class(probs)
