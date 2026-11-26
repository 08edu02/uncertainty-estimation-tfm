"""Metrics used to compare uncertainty methods.

Every function is pure: arrays in, numbers or dataclasses out. No file access,
no printing, no model knowledge. That keeps them testable with hand-made cases
whose answer can be computed on paper.

The four families mirror the four questions of the thesis: calibration
(:mod:`uq.metrics.calibration`), error detection
(:mod:`uq.metrics.error_detection`), selective classification
(:mod:`uq.metrics.selective`) and conformal prediction
(:mod:`uq.metrics.conformal`).
"""

from __future__ import annotations

from uq.metrics.calibration import (
    ReliabilityCurve,
    brier,
    ece,
    nll,
    reliability_curve,
)
from uq.metrics.conformal import (
    average_set_size,
    conformal_quantile,
    coverage_by_class,
    empirical_coverage,
    nonconformity_scores,
    prediction_sets,
)
from uq.metrics.error_detection import (
    error_auprc,
    error_auroc,
    predictive_entropy,
)
from uq.metrics.predictions import is_correct, predicted_class, predicted_confidence
from uq.metrics.selective import (
    RiskCoverageCurve,
    aurc,
    error_at_coverage,
    risk_coverage_curve,
    threshold_for_coverage,
)

__all__ = [
    "ReliabilityCurve",
    "RiskCoverageCurve",
    "aurc",
    "average_set_size",
    "brier",
    "conformal_quantile",
    "coverage_by_class",
    "ece",
    "empirical_coverage",
    "error_at_coverage",
    "error_auprc",
    "error_auroc",
    "is_correct",
    "nll",
    "nonconformity_scores",
    "predicted_class",
    "predicted_confidence",
    "prediction_sets",
    "predictive_entropy",
    "reliability_curve",
    "risk_coverage_curve",
    "threshold_for_coverage",
]
