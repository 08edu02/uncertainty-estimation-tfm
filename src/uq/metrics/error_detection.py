"""Error detection: does the uncertainty separate wrong predictions from right ones?

The question is not how uncertain the model is, but whether the samples it
flags as uncertain are the ones it gets wrong. The standard summary is the
AUROC between an uncertainty score and the label "this prediction was wrong":
``0.5`` means the score is useless, ``1.0`` means every error is ranked above
every correct prediction.

AUROC and AUPRC depend on how many errors there are, so a value without the
accuracy of the same run next to it is not interpretable. Report both.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from uq.metrics._checks import check_correctness, check_probabilities, check_vector

EntropyUnit = Literal["nats", "bits"]

_LN_2 = float(np.log(2.0))


def predictive_entropy(probs: np.ndarray, *, unit: EntropyUnit = "nats") -> np.ndarray:
    """Shannon entropy of each predicted distribution, shape ``(n,)``.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape ``(n, n_classes)``.
    unit:
        ``nats`` (default) or ``bits``. The two differ by a factor ``ln 2``;
        the unit must be declared wherever the number is reported.

    Returns
    -------
    np.ndarray
        Entropy per sample; ``0`` for a fully confident prediction.
    """
    checked_probs = check_probabilities(probs)

    # ``log(0)`` would warn; feeding 1.0 instead keeps the term at 0 * 0 = 0.
    logarithms = np.log(np.where(checked_probs > 0.0, checked_probs, 1.0))
    entropy = -np.sum(checked_probs * logarithms, axis=1)

    if unit == "nats":
        return entropy
    if unit == "bits":
        return entropy / _LN_2
    raise ValueError(f"unit must be 'nats' or 'bits', got {unit!r}")


def error_auroc(uncertainty: np.ndarray, correct: np.ndarray) -> float | None:
    """AUROC of an uncertainty score as a detector of wrong predictions.

    Returns ``None`` when every prediction is correct or every one is wrong:
    the AUROC is undefined there, and returning ``0.5`` would hide a degenerate
    run behind a plausible number.

    Parameters
    ----------
    uncertainty:
        Higher means more uncertain, shape ``(n,)``.
    correct:
        Whether each prediction was right, shape ``(n,)``.
    """
    prepared = _prepare_detection(uncertainty, correct)
    if prepared is None:
        return None
    scores, errors = prepared
    return float(roc_auc_score(errors, scores))


def error_auprc(uncertainty: np.ndarray, correct: np.ndarray) -> float | None:
    """Average precision of an uncertainty score as a detector of errors.

    Preferred over the AUROC under strong class imbalance (few errors): the
    baseline is the error rate, not ``0.5``.
    """
    prepared = _prepare_detection(uncertainty, correct)
    if prepared is None:
        return None
    scores, errors = prepared
    return float(average_precision_score(errors, scores))


def _prepare_detection(
    uncertainty: np.ndarray, correct: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(uncertainty, is_error)`` or ``None`` if only one class is present."""
    scores = check_vector(uncertainty, name="uncertainty")
    correctness = check_correctness(correct, n_samples=scores.size)

    errors = ~correctness
    if errors.all() or not errors.any():
        return None
    return scores, errors.astype(int)
