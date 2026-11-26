"""Selective classification: what happens when the model is allowed to abstain.

Samples are sorted by confidence, the most confident ones are answered and the
rest are rejected (sent to a person). The risk-coverage curve records the error
rate among the answered samples as the accepted fraction grows.

``risk`` is an error rate over the accepted subset, not a count: accepting the
40 % most confident samples and getting one of them wrong is a risk of
``1 / (0.4 * n)``, not ``1``. This choice matters, and it is the one used in
the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uq.metrics._checks import check_correctness, check_vector


@dataclass(frozen=True)
class RiskCoverageCurve:
    """Risk and coverage evaluated at every accepted prefix of the ranking.

    Attributes
    ----------
    coverage:
        Fraction of samples answered, from ``1 / n`` to ``1``.
    risk:
        Error rate among the answered samples at that coverage.
    """

    coverage: np.ndarray
    risk: np.ndarray


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> RiskCoverageCurve:
    """Risk-coverage curve obtained by accepting the most confident samples first.

    Parameters
    ----------
    confidence:
        Confidence per sample, shape ``(n,)``; higher is more confident.
    correct:
        Whether each prediction was right, shape ``(n,)``.
    """
    scores = check_vector(confidence, name="confidence")
    correctness = check_correctness(correct, n_samples=scores.size)

    order = np.argsort(-scores, kind="stable")
    ordered_correct = correctness[order]

    accepted = np.arange(1, scores.size + 1)
    cumulative_correct = np.cumsum(ordered_correct)
    risk = 1.0 - cumulative_correct / accepted
    coverage = accepted / scores.size

    return RiskCoverageCurve(coverage=coverage, risk=risk)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the risk-coverage curve, in its discrete form.

    Implemented as the mean risk over the ``n`` prefixes of the ranking, which
    is the usual discrete approximation of the area. Lower is better.

    Only comparable between runs with a similar accuracy: because the risk is
    an error rate, the curve of a good ranker starts below the flat line of an
    uninformative one and ends above it when accuracy is low, so the areas can
    cross. Report the accuracy of the same run next to it, always.
    """
    curve = risk_coverage_curve(confidence, correct)
    return float(np.mean(curve.risk))


def error_at_coverage(confidence: np.ndarray, correct: np.ndarray, *, coverage: float) -> float:
    """Error rate among the most confident ``coverage`` fraction of samples.

    ``coverage = 0.9`` answers the 90 % most confident samples and reports the
    error rate on them. The number of accepted samples is rounded up, so the
    requested coverage is always reached.
    """
    curve = risk_coverage_curve(confidence, correct)
    accepted = _accepted_count(curve.coverage.size, coverage)
    return float(curve.risk[accepted - 1])


def threshold_for_coverage(confidence: np.ndarray, *, coverage: float) -> float:
    """Lowest confidence that still answers the requested fraction of samples.

    Returns the ``k``-th largest confidence, with ``k = ceil(coverage * n)``.
    Accepting every sample with ``confidence >= threshold`` therefore accepts
    at least ``coverage`` of them (more if there are ties).
    """
    scores = check_vector(confidence, name="confidence")
    accepted = _accepted_count(scores.size, coverage)
    descending = np.sort(scores)[::-1]
    return float(descending[accepted - 1])


def _accepted_count(n_samples: int, coverage: float) -> int:
    """Number of samples to accept for a given coverage, always in ``[1, n]``."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    return min(max(int(np.ceil(coverage * n_samples)), 1), n_samples)
