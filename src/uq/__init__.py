"""Uncertainty estimation benchmark for classification.

The package is organised around the two scenarios of the thesis:

* ``uq.data``     -- download, loading and splits of the public datasets.
* ``uq.eda``      -- exploratory summaries and figures.
* ``uq.models``   -- reference models (added in later milestones).
* ``uq.methods``  -- the five uncertainty methods (added in later milestones).
* ``uq.metrics``  -- calibration, error detection, selective and conformal metrics.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
