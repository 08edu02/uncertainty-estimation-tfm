"""Tests for the EDA summaries, using small hand-made datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uq.eda import summary


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 10, 20, 20],
            "group": ["x", "x", "y", "y"],
            "target": [0, 0, 1, 1],
        }
    )


def test_tabular_summary_reports_balance_and_groups(frame: pd.DataFrame) -> None:
    result = summary.tabular_summary(frame, "target", sensitive=("group",), categorical=("group",))

    assert result["n_rows"] == 4
    assert result["n_features"] == 3
    assert result["n_missing_values"] == 0
    assert result["target"]["counts"] == {"0": 2, "1": 2}
    assert result["target"]["positive_rate"] == pytest.approx(0.5)
    assert result["categories"]["group"] == {"x": 2, "y": 2}
    assert result["groups"]["group"]["y"]["positive_rate"] == pytest.approx(1.0)
    assert result["columns"]["feature_a"]["mean"] == pytest.approx(2.5)


def test_tabular_summary_counts_duplicate_feature_rows(frame: pd.DataFrame) -> None:
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    result = summary.tabular_summary(duplicated, "target")

    assert result["n_duplicate_rows"] == 1
    assert result["n_rows"] == 5


def test_tabular_summary_validates_inputs(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        summary.tabular_summary(frame, "missing_target")
    with pytest.raises(KeyError):
        summary.tabular_summary(frame, "target", sensitive=("missing_group",))


def test_image_summary_reports_shapes_and_rates() -> None:
    images = np.zeros((4, 8, 8), dtype=np.uint8)
    images[2:] = 255
    labels = np.array([0, 0, 1, 1])

    result = summary.image_summary(images, labels, split="test")

    assert result["split"] == "test"
    assert result["n_samples"] == 4
    assert result["shape"] == [4, 8, 8]
    assert result["class_counts"] == {"0": 2, "1": 2}
    assert result["class_rates"]["1"] == pytest.approx(0.5)
    assert result["pixel_max"] == 255
    assert result["pixel_min"] == 0
