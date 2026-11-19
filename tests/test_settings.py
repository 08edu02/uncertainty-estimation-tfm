"""Tests for path resolution and the MedMNIST loader's input validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from uq import settings
from uq.data import pneumonia


def test_paths_follow_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UQ_DATA_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("UQ_RESULTS_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("UQ_FIGURES_DIR", str(tmp_path / "out" / "figures"))

    assert settings.data_dir() == (tmp_path / "datasets").resolve()
    assert settings.results_dir() == (tmp_path / "out").resolve()
    assert settings.figures_dir() == (tmp_path / "out" / "figures").resolve()


def test_defaults_point_inside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in ("UQ_DATA_DIR", "UQ_RESULTS_DIR", "UQ_FIGURES_DIR"):
        monkeypatch.delenv(variable, raising=False)

    assert settings.data_dir() == settings.PROJECT_ROOT / "data"
    assert settings.results_dir() == settings.PROJECT_ROOT / "results"
    assert settings.figures_dir() == settings.PROJECT_ROOT / "results" / "figures"


def test_ensure_dir_creates_the_folder(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    assert settings.ensure_dir(nested) == nested
    assert nested.is_dir()


def test_unknown_split_and_size_are_rejected() -> None:
    with pytest.raises(ValueError, match="split must be one of"):
        pneumonia._dataset("training", None, 28, download=False)
    with pytest.raises(ValueError, match="size must be one of"):
        pneumonia._dataset("train", None, 32, download=False)
