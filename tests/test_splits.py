"""Tests for the partition logic, with counts that can be verified by hand."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uq.data import credit, pneumonia, splits
from uq.data.splits import (
    CREDIT_FRACTIONS,
    IMAGE_DEV_FRACTIONS,
    Split,
    allocation_by_rounding,
    make_credit_split,
    make_image_split,
    stratified_partition,
)


def balanced_labels(n_positives: int, n_negatives: int) -> np.ndarray:
    """Labels with the positives first, so ordering mistakes become visible."""
    return np.concatenate(
        [np.ones(n_positives, dtype=np.int64), np.zeros(n_negatives, dtype=np.int64)]
    )


def test_allocation_adds_up_to_the_total() -> None:
    counts = allocation_by_rounding(29944, CREDIT_FRACTIONS)

    assert sum(counts.values()) == 29944


def test_allocation_follows_the_fractions() -> None:
    counts = allocation_by_rounding(1000, CREDIT_FRACTIONS)

    assert counts == {"train": 600, "val": 100, "calibration": 100, "test": 200}


def test_allocation_gives_the_rounding_remainder_to_the_last_set() -> None:
    counts = allocation_by_rounding(9, {"train": 0.6, "val": 0.1, "calibration": 0.1, "test": 0.2})

    assert sum(counts.values()) == 9


def test_partition_covers_every_sample_exactly_once() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    partition = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=0)
    combined = np.sort(np.concatenate(list(partition.values())))

    assert np.array_equal(combined, np.arange(1000))


def test_partition_sizes_match_the_hand_computed_counts() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    partition = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=0)

    assert {name: indices.size for name, indices in partition.items()} == {
        "train": 600,
        "val": 100,
        "calibration": 100,
        "test": 200,
    }


def test_every_set_keeps_the_class_balance() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    partition = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=0)

    for indices in partition.values():
        share_of_positives = labels[indices].mean()
        assert share_of_positives == pytest.approx(0.3, abs=0.02)


def test_indices_are_sorted_and_unique() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    partition = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=0)

    for indices in partition.values():
        assert np.array_equal(indices, np.sort(indices))
        assert np.unique(indices).size == indices.size


def test_the_same_seed_gives_the_same_partition() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    first = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=7)
    second = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=7)

    for name in first:
        assert np.array_equal(first[name], second[name])


def test_a_different_seed_gives_a_different_partition() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)

    first = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=7)
    second = stratified_partition(labels, fractions=CREDIT_FRACTIONS, seed=8)

    assert not np.array_equal(first["train"], second["train"])


def test_candidates_are_the_only_positions_considered() -> None:
    labels = balanced_labels(n_positives=300, n_negatives=700)
    candidates = np.arange(0, 1000, 2)

    partition = stratified_partition(
        labels, fractions=CREDIT_FRACTIONS, seed=0, candidates=candidates
    )
    combined = np.concatenate(list(partition.values()))

    assert np.all(combined % 2 == 0)
    assert combined.size == 500


def test_fractions_that_do_not_sum_to_one_are_rejected() -> None:
    labels = balanced_labels(n_positives=100, n_negatives=100)

    with pytest.raises(ValueError, match="fractions must sum to 1"):
        stratified_partition(labels, fractions={"train": 0.5, "test": 0.4}, seed=0)


def test_unknown_split_names_are_rejected() -> None:
    labels = balanced_labels(n_positives=100, n_negatives=100)

    with pytest.raises(ValueError, match="unknown split names"):
        stratified_partition(labels, fractions={"train": 0.5, "holdout": 0.5}, seed=0)


def test_a_pool_with_a_single_class_is_rejected() -> None:
    labels = np.zeros(100, dtype=np.int64)

    with pytest.raises(ValueError, match="at least two classes"):
        stratified_partition(labels, fractions={"train": 0.5, "test": 0.5}, seed=0)


def test_a_class_missing_from_one_set_is_rejected() -> None:
    labels = balanced_labels(n_positives=1, n_negatives=100)

    with pytest.raises(ValueError, match="has no samples of classes"):
        stratified_partition(labels, fractions={"train": 0.2, "test": 0.8}, seed=0)


def test_split_rejects_two_sets_sharing_an_index() -> None:
    with pytest.raises(ValueError, match="share 1 indices"):
        Split(
            train=np.array([0, 1, 2]),
            val=np.array([2, 3]),
            calibration=np.array([4]),
            test=np.array([5]),
        )


def test_split_rejects_repeated_indices_inside_one_set() -> None:
    with pytest.raises(ValueError, match="repeated indices"):
        Split(
            train=np.array([0, 0, 1]),
            val=np.array([2]),
            calibration=np.array([3]),
            test=np.array([4]),
        )


def test_split_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError, match="is empty"):
        Split(
            train=np.array([0]),
            val=np.array([], dtype=np.int64),
            calibration=np.array([2]),
            test=np.array([3]),
        )


def test_split_rejects_float_indices() -> None:
    with pytest.raises(ValueError, match="integer indices"):
        Split(
            train=np.array([0.0, 1.0]),
            val=np.array([2]),
            calibration=np.array([3]),
            test=np.array([4]),
        )


def test_split_sizes_are_reported_for_every_set() -> None:
    split = Split(
        train=np.array([0, 1]),
        val=np.array([2]),
        calibration=np.array([3]),
        test=np.array([4, 5, 6]),
    )

    assert split.sizes() == {"train": 2, "val": 1, "calibration": 1, "test": 3}


def test_two_splits_are_equal_when_their_indices_are() -> None:
    first = Split(
        train=np.array([0, 1]),
        val=np.array([2]),
        calibration=np.array([3]),
        test=np.array([4]),
    )
    second = Split(
        train=np.array([0, 1]),
        val=np.array([2]),
        calibration=np.array([3]),
        test=np.array([4]),
    )

    assert first == second


def credit_frame(n_rows: int, n_positives: int) -> pd.DataFrame:
    """Minimal credit-shaped frame: one informative feature plus the target."""
    labels = balanced_labels(n_positives=n_positives, n_negatives=n_rows - n_positives)
    return pd.DataFrame({credit.TARGET: labels, "limit_bal": np.arange(n_rows) * 10.0})


def test_credit_split_leaves_duplicated_rows_out_of_every_set() -> None:
    frame = credit_frame(n_rows=1000, n_positives=300)
    repeated = frame.iloc[[0, 1, 2]].copy()
    frame = pd.concat([frame, repeated], ignore_index=True)
    duplicated = set(np.flatnonzero(credit.duplicate_mask(frame)).tolist())

    split = make_credit_split(frame, seed=0)
    selected = set(np.concatenate([split.train, split.val, split.calibration, split.test]).tolist())

    assert duplicated
    assert selected.isdisjoint(duplicated)


def test_credit_split_works_on_the_rows_that_remain() -> None:
    frame = credit_frame(n_rows=1000, n_positives=300)

    split = make_credit_split(frame, seed=0)

    assert split.sizes() == {"train": 600, "val": 100, "calibration": 100, "test": 200}


def test_credit_split_indices_point_into_the_given_frame() -> None:
    frame = credit_frame(n_rows=1000, n_positives=300)
    frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)

    split = make_credit_split(frame, seed=0)
    largest = max(int(getattr(split, name).max()) for name in splits.SPLIT_NAMES)

    assert largest < len(frame)


def test_credit_split_keeps_the_class_balance_of_the_whole_frame() -> None:
    frame = credit_frame(n_rows=1000, n_positives=300)

    split = make_credit_split(frame, seed=0)
    labels = frame[credit.TARGET].to_numpy()

    for indices in split.as_dict().values():
        assert labels[indices].mean() == pytest.approx(0.3, abs=0.02)


def fake_pool(
    monkeypatch: pytest.MonkeyPatch, sizes: dict[str, int], positives: dict[str, int]
) -> None:
    """Replace the MedMNIST loader with arrays built from the given counts."""
    pieces = []
    bounds: dict[str, slice] = {}
    offset = 0
    for split in pneumonia.POOL_ORDER:
        labels = balanced_labels(
            n_positives=positives[split], n_negatives=sizes[split] - positives[split]
        )
        pieces.append(labels)
        bounds[split] = slice(offset, offset + labels.size)
        offset += labels.size

    all_labels = np.concatenate(pieces)

    def fake_load_pooled(root: Path | None = None, **_: object):
        images = np.zeros((all_labels.size, 28, 28, 1), dtype=np.uint8)
        return images, all_labels, bounds

    monkeypatch.setattr(splits.pneumonia, "load_pooled", fake_load_pooled)


def test_image_split_keeps_the_official_test_set_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool(
        monkeypatch,
        sizes={"train": 1000, "val": 100, "test": 100},
        positives={"train": 700, "val": 70, "test": 60},
    )

    split = make_image_split(seed=0)

    assert np.array_equal(split.test, np.arange(1100, 1200))
    assert split.sizes() == {"train": 880, "val": 110, "calibration": 110, "test": 100}


def test_image_split_partitions_only_the_pooled_development_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_pool(
        monkeypatch,
        sizes={"train": 1000, "val": 100, "test": 100},
        positives={"train": 700, "val": 70, "test": 60},
    )

    split = make_image_split(seed=0)
    development = np.sort(np.concatenate([split.train, split.val, split.calibration]))

    assert np.array_equal(development, np.arange(1100))
    assert IMAGE_DEV_FRACTIONS == {"train": 0.8, "val": 0.1, "calibration": 0.1}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    split = make_credit_split(credit_frame(n_rows=1000, n_positives=300), seed=0)
    path = tmp_path / "nested" / "credit_seed0.json"

    written = splits.save(path, split, dataset="credit_default", seed=0)
    restored = splits.load(written)

    assert written == path
    assert restored == split


def test_saved_file_describes_itself(tmp_path: Path) -> None:
    split = make_credit_split(credit_frame(n_rows=1000, n_positives=300), seed=3)
    path = splits.save(tmp_path / "credit_seed3.json", split, dataset="credit_default", seed=3)

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset"] == "credit_default"
    assert payload["seed"] == 3
    assert payload["sizes"] == {"train": 600, "val": 100, "calibration": 100, "test": 200}
    assert payload["proportions"]["train"] == pytest.approx(0.6)


def test_loading_a_file_with_overlapping_sets_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [0, 1, 2],
                    "val": [2, 3],
                    "calibration": [4],
                    "test": [5],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="share 1 indices"):
        splits.load(path)


def test_loading_a_file_without_one_set_raises(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.json"
    path.write_text(
        json.dumps({"splits": {"train": [0], "val": [1], "test": [2]}}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="missing the sets"):
        splits.load(path)


def test_split_path_lives_under_the_data_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UQ_DATA_DIR", str(tmp_path))

    path = splits.split_path("credit_default", seed=4)

    assert path == (tmp_path).resolve() / "splits" / "credit_default_seed4.json"
