"""Project paths, resolved once and overridable through environment variables.

The repository never stores datasets or generated figures: both are rebuilt by
the scripts. Keeping every path in one place means the same code runs on the
author's machine, on a clean clone and inside CI.

Environment variables
---------------------
``UQ_DATA_DIR``     where the downloaded datasets live (default: ``./data``).
``UQ_RESULTS_DIR``  where JSON/CSV summaries are written (default: ``./results``).
``UQ_FIGURES_DIR``  where generated figures are written
                    (default: ``./results/figures``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ``src/uq/settings.py`` -> parents[2] is the repository root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _resolve(env_var: str, default: Path) -> Path:
    """Return the path declared in ``env_var`` or ``default``, both absolute."""
    raw = os.environ.get(env_var)
    path = Path(raw).expanduser() if raw else default
    return path.resolve()


def data_dir() -> Path:
    """Directory holding the raw datasets (never committed)."""
    return _resolve("UQ_DATA_DIR", PROJECT_ROOT / "data")


def results_dir() -> Path:
    """Directory holding the small, versioned result summaries."""
    return _resolve("UQ_RESULTS_DIR", PROJECT_ROOT / "results")


def figures_dir() -> Path:
    """Directory holding generated figures (never committed)."""
    return _resolve("UQ_FIGURES_DIR", PROJECT_ROOT / "results" / "figures")


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
