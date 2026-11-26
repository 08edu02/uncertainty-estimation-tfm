"""Tests for the risk-coverage curve and the metrics derived from it."""

from __future__ import annotations

import numpy as np
import pytest

from uq.metrics.selective import (
    aurc,
    error_at_coverage,
    risk_coverage_curve,
    threshold_for_coverage,
)

CONFIDENCE = np.array([0.9, 0.8, 0.7, 0.6])
CORRECT = np.array([True, True, False, False])


def test_risk_is_zero_while_only_correct_predictions_are_accepted() -> None:
    curve = risk_coverage_curve(CONFIDENCE, CORRECT)

    assert curve.coverage == pytest.approx(np.array([0.25, 0.5, 0.75, 1.0]))
    assert curve.risk[:2] == pytest.approx(np.array([0.0, 0.0]))


def test_risk_counts_the_errors_in_the_accepted_prefix() -> None:
    curve = risk_coverage_curve(CONFIDENCE, CORRECT)

    assert curve.risk[2] == pytest.approx(1.0 / 3.0)
    assert curve.risk[3] == pytest.approx(2.0 / 4.0)


def test_aurc_of_a_perfect_ranking_matches_its_hand_computed_value() -> None:
    # Prefixes of size 1, 2, 3 and 4 have risk 0, 0, 1/3 and 1/2.
    expected = (0.0 + 0.0 + (1.0 / 3.0) + 0.5) / 4.0

    assert aurc(CONFIDENCE, CORRECT) == pytest.approx(expected)


def test_a_perfect_ranking_has_a_lower_aurc_than_a_random_one() -> None:
    rng = np.random.default_rng(0)
    n_samples = 500
    correct = rng.random(n_samples) < 0.8
    confidence = np.where(correct, 0.9, 0.1)
    random_confidence = rng.random(n_samples)

    assert aurc(confidence, correct) < aurc(random_confidence, correct)


def test_aurc_is_the_error_rate_when_the_ranking_is_uninformative() -> None:
    rng = np.random.default_rng(3)
    n_samples = 2000
    correct = rng.random(n_samples) < 0.9

    assert aurc(np.zeros(n_samples), correct) == pytest.approx(0.1, abs=0.02)


def test_error_at_ninety_percent_coverage_answers_the_most_confident_ninety() -> None:
    rng = np.random.default_rng(1)
    n_samples = 100
    correct = rng.random(n_samples) < 0.9
    confidence = np.where(correct, 0.9, 0.1)

    answerable = error_at_coverage(confidence, correct, coverage=0.9)

    assert answerable < 0.05


def test_error_at_full_coverage_is_one_minus_accuracy() -> None:
    assert error_at_coverage(CONFIDENCE, CORRECT, coverage=1.0) == pytest.approx(0.5)


def test_error_at_coverage_rounds_up_so_the_requested_share_is_reached() -> None:
    # ceil(0.9 * 4) = 4 samples, so the fourth prefix is used and not the third.
    assert error_at_coverage(CONFIDENCE, CORRECT, coverage=0.9) == pytest.approx(0.5)


def test_threshold_for_coverage_is_the_last_accepted_confidence() -> None:
    assert threshold_for_coverage(CONFIDENCE, coverage=0.5) == pytest.approx(0.8)
    assert threshold_for_coverage(CONFIDENCE, coverage=1.0) == pytest.approx(0.6)


def test_accepting_everything_above_the_threshold_reaches_the_coverage() -> None:
    rng = np.random.default_rng(2)
    confidence = rng.random(200)
    requested = 0.8

    threshold = threshold_for_coverage(confidence, coverage=requested)
    accepted_fraction = float((confidence >= threshold).mean())

    assert accepted_fraction >= requested


def test_coverage_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="coverage must be in"):
        error_at_coverage(CONFIDENCE, CORRECT, coverage=0.0)

    with pytest.raises(ValueError, match="coverage must be in"):
        threshold_for_coverage(CONFIDENCE, coverage=1.5)


def test_confidence_and_correctness_must_have_the_same_length() -> None:
    with pytest.raises(ValueError, match="correct must have 4 elements"):
        risk_coverage_curve(CONFIDENCE, np.array([True, False]))
