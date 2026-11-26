"""Tests for split conformal prediction, including its finite-sample correction."""

from __future__ import annotations

import numpy as np
import pytest

from uq.metrics.conformal import (
    average_set_size,
    conformal_quantile,
    coverage_by_class,
    empirical_coverage,
    nonconformity_scores,
    prediction_sets,
)


def test_nonconformity_score_is_one_minus_the_true_class_probability() -> None:
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 1])

    assert nonconformity_scores(probs, labels) == pytest.approx(np.array([0.3, 0.2]), abs=1e-12)


def test_quantile_with_one_hundred_scores_takes_the_ninety_first() -> None:
    scores = np.arange(1.0, 101.0)

    assert conformal_quantile(scores, alpha=0.1) == pytest.approx(91.0)


def test_quantile_with_nine_scores_takes_the_ninth() -> None:
    scores = np.arange(1.0, 10.0)

    assert conformal_quantile(scores, alpha=0.1) == pytest.approx(9.0)


def test_quantile_at_ninety_five_percent_with_fifty_scores() -> None:
    # ceil(51 * 0.95) = 49, so the 49th smallest score of the 50.
    scores = np.arange(1.0, 51.0)

    assert conformal_quantile(scores, alpha=0.05) == pytest.approx(49.0)


def test_quantile_falls_back_to_the_largest_score_when_calibration_is_too_small() -> None:
    scores = np.array([0.1, 0.2, 0.3])

    assert conformal_quantile(scores, alpha=0.01) == pytest.approx(0.3)


def test_quantile_does_not_depend_on_the_order_of_the_input() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(200)

    assert conformal_quantile(scores, alpha=0.1) == pytest.approx(
        conformal_quantile(rng.permutation(scores), alpha=0.1)
    )


def test_alpha_outside_the_open_unit_interval_is_rejected() -> None:
    scores = np.array([0.1, 0.2])

    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_quantile(scores, alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_quantile(scores, alpha=1.0)


def test_prediction_sets_include_the_classes_below_the_quantile() -> None:
    probs = np.array([[0.7, 0.3], [0.4, 0.6]])

    sets = prediction_sets(probs, quantile=0.5)

    assert np.array_equal(sets, np.array([[True, False], [False, True]]))


def test_a_large_quantile_includes_every_class() -> None:
    probs = np.array([[0.9, 0.1], [0.6, 0.4]])

    sets = prediction_sets(probs, quantile=0.99)

    assert np.array_equal(sets, np.ones((2, 2), dtype=bool))


def test_a_quantile_that_excludes_every_class_is_rejected() -> None:
    probs = np.array([[0.9, 0.1]])

    with pytest.raises(ValueError, match="excludes every class"):
        prediction_sets(probs, quantile=-0.1)


def test_empirical_coverage_counts_the_true_label_inside_the_set() -> None:
    sets = np.array([[True, False], [False, True], [True, True], [True, False]])
    labels = np.array([0, 0, 0, 1])

    assert empirical_coverage(sets, labels) == pytest.approx(0.5)


def test_average_set_size_is_one_for_singletons() -> None:
    sets = np.array([[True, False], [False, True], [True, False]])

    assert average_set_size(sets) == pytest.approx(1.0)


def test_coverage_by_class_exposes_a_badly_served_class() -> None:
    sets = np.array([[True, False], [True, False], [True, False]])
    labels = np.array([0, 1, 1])

    coverage = coverage_by_class(sets, labels)

    assert coverage[0] == pytest.approx(1.0)
    assert coverage[1] == pytest.approx(0.0)


def test_calibration_on_a_calibrated_model_reaches_the_nominal_coverage() -> None:
    # A perfectly calibrated binary model: p1 is drawn, then y follows it.
    rng = np.random.default_rng(0)
    true_probabilities = rng.uniform(0.05, 0.95, size=4000)
    labels = (rng.random(4000) < true_probabilities).astype(int)
    probs = np.column_stack([1.0 - true_probabilities, true_probabilities])

    calibration_scores = nonconformity_scores(probs[:2000], labels[:2000])
    quantile = conformal_quantile(calibration_scores, alpha=0.1)
    test_sets = prediction_sets(probs[2000:], quantile)

    assert empirical_coverage(test_sets, labels[2000:]) == pytest.approx(0.9, abs=0.03)


def test_a_prediction_set_with_no_class_is_rejected() -> None:
    sets = np.array([[False, False], [True, False]])
    labels = np.array([0, 0])

    with pytest.raises(ValueError, match="at least one class"):
        empirical_coverage(sets, labels)


def test_a_two_dimensional_label_vector_is_rejected() -> None:
    sets = np.array([[True, False]])

    with pytest.raises(ValueError, match="labels must be 1-D"):
        empirical_coverage(sets, np.array([[0]]))
