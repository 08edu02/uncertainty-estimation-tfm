"""Exploratory data analysis: summaries and figures.

Everything here is plain functions over ``pandas``/``numpy`` objects so that
the notebooks and the scripts share exactly the same code. Nothing critical
should live in a notebook.
"""

from __future__ import annotations

from uq.eda.figures import (
    plot_credit_class_balance,
    plot_credit_distributions,
    plot_image_samples,
    plot_split_balance,
)
from uq.eda.summary import image_summary, tabular_summary

__all__ = [
    "image_summary",
    "plot_credit_class_balance",
    "plot_credit_distributions",
    "plot_image_samples",
    "plot_split_balance",
    "tabular_summary",
]
