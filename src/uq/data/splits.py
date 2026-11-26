"""Train, validation, calibration and test partitions, stored as indices.

A split is never a copy of the data: it is a set of **positional indices** into
the arrays returned by :mod:`uq.data.credit` and :mod:`uq.data.pneumonia`. That
keeps a partition small enough to live in a JSON, auditable by eye and identical
between runs, and it removes the classic source of silent errors, which is two
scripts disagreeing about which rows are in training.

The two scenarios follow different rules, because the datasets deserve it:

* **Credit:** exact duplicated rows are excluded, then the remaining rows are
  split 60/10/10/20, stratified by the target. The duplicates are dropped
  before the split so that the same client cannot appear on both sides.
* **Images:** the official test split is kept untouched. The official train and
  validation splits are pooled and re-split 80/10/10, stratified by the label.

Which set is used for what (this is the rule the whole comparison rests on):

* ``train`` fits the model and any preprocessing scaler.
* ``val`` stops training early, fits the softmax temperature and fixes the
  abstention threshold.
* ``calibration`` is used only by split conformal prediction, which needs a set
  disjoint from validation to keep the exchangeability assumption.
* ``test`` is read once, in the final evaluation. Nothing is decided with it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from uq.data import credit, pneumonia
from uq.settings import data_dir

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "calibration", "test")

#: Credit: 60 % train, 10 % validation, 10 % calibration, 20 % test.
CREDIT_FRACTIONS: dict[str, float] = {
    "train": 0.6,
    "val": 0.1,
    "calibration": 0.1,
    "test": 0.2,
}

#: Images: the pooled train+validation split, 80/10/10. The official test split
#: is not part of these fractions.
IMAGE_DEV_FRACTIONS: dict[str, float] = {
    "train": 0.8,
    "val": 0.1,
    "calibration": 0.1,
}

FRACTIONS_SUM_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class Split:
    """Four disjoint sets of positional indices.

    The sets are validated on construction: every set must be non-empty, must
    not repeat an index and must not share one with another set. A split with
    an overlap is a leak, and a leak found here costs minutes; found in the
    results, it costs the thesis.

    Attributes
    ----------
    train, val, calibration, test:
        Sorted ``int64`` arrays of positional indices.
    """

    train: np.ndarray
    val: np.ndarray
    calibration: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            values = getattr(self, name)
            if not isinstance(values, np.ndarray) or values.ndim != 1:
                raise ValueError(f"{name} must be a 1-D numpy array of indices")
            if values.dtype.kind not in "iu":
                raise ValueError(f"{name} must hold integer indices, got dtype {values.dtype}")
            if values.size == 0:
                raise ValueError(f"{name} is empty")
            if np.unique(values).size != values.size:
                raise ValueError(f"{name} contains repeated indices")
            if np.any(values < 0):
                raise ValueError(f"{name} contains negative indices")

        for first, second in combinations(SPLIT_NAMES, 2):
            shared = np.intersect1d(getattr(self, first), getattr(self, second))
            if shared.size:
                raise ValueError(f"{first} and {second} share {shared.size} indices")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Split):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in SPLIT_NAMES
        )

    def sizes(self) -> dict[str, int]:
        """Number of samples in each set, in ``SPLIT_NAMES`` order."""
        return {name: int(getattr(self, name).size) for name in SPLIT_NAMES}

    def as_dict(self) -> dict[str, list[int]]:
        """Plain integers, ready for ``json.dumps``."""
        return {name: [int(index) for index in getattr(self, name)] for name in SPLIT_NAMES}


def stratified_partition(
    labels: np.ndarray,
    *,
    fractions: Mapping[str, float],
    seed: int,
    candidates: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Split a pool of samples into named sets, keeping the class balance.

    Each class is shuffled on its own and divided according to ``fractions``, so
    every set ends up with roughly the same proportion of positives as the whole
    pool. The last set named in ``fractions`` takes the rounding remainder of
    each class, which makes the counts add up exactly.

    Parameters
    ----------
    labels:
        One integer label per sample in the dataset, shape ``(n,)``.
    fractions:
        Ordered mapping ``{set_name: fraction}``; the fractions must sum to 1.
    seed:
        Seed of the generator. The same seed and the same inputs always give
        the same partition.
    candidates:
        Positions that may enter the split. Defaults to every sample. The credit
        scenario uses it to leave duplicated rows out of the pool.

    Returns
    -------
    dict[str, np.ndarray]
        One sorted array of indices per name in ``fractions``.

    Raises
    ------
    ValueError
        If the fractions do not sum to 1, if a name is unknown, or if some
        class ends up missing from one of the sets (which would make per-class
        coverage impossible to compute).
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if labels.dtype.kind not in "iu":
        raise ValueError(f"labels must be integers, got dtype {labels.dtype}")

    names = tuple(fractions)
    if not names:
        raise ValueError("fractions must name at least one set")
    unknown = set(names) - set(SPLIT_NAMES)
    if unknown:
        raise ValueError(f"unknown split names {sorted(unknown)}; expected {SPLIT_NAMES}")
    if any(fraction <= 0.0 or fraction >= 1.0 for fraction in fractions.values()):
        raise ValueError("every fraction must be strictly between 0 and 1")
    total_fraction = sum(fractions.values())
    if abs(total_fraction - 1.0) > FRACTIONS_SUM_TOLERANCE:
        raise ValueError(f"fractions must sum to 1, got {total_fraction}")

    pool = np.arange(labels.size) if candidates is None else np.asarray(candidates)
    if pool.ndim != 1 or pool.dtype.kind not in "iu":
        raise ValueError("candidates must be a 1-D array of integer positions")
    if pool.size == 0:
        raise ValueError("candidates is empty")
    if np.unique(labels[pool]).size < 2:
        raise ValueError("the pool must contain at least two classes to stratify")

    generator = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for class_label in np.unique(labels[pool]):
        class_positions = pool[labels[pool] == class_label]
        shuffled = generator.permutation(np.sort(class_positions))
        counts = allocation_by_rounding(shuffled.size, fractions)

        offset = 0
        for name in names:
            size = counts[name]
            chunks[name].append(shuffled[offset : offset + size])
            offset += size

    partition: dict[str, np.ndarray] = {}
    for name in names:
        pieces = [piece for piece in chunks[name] if piece.size]
        indices = np.sort(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"split {name!r} ended up empty; adjust the fractions")
        partition[name] = indices.astype(np.int64)

    _check_every_set_has_every_class(partition, labels)

    return partition


def allocation_by_rounding(total: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Turn fractions into integer counts that add up to ``total``.

    Every set except the last is rounded to the nearest integer; the last one
    takes whatever is left, so the counts always add up exactly.

    Parameters
    ----------
    total:
        Number of samples to distribute.
    fractions:
        Ordered mapping ``{set_name: fraction}`` summing to 1.

    Returns
    -------
    dict[str, int]
        A count per set name.
    """
    names = tuple(fractions)
    counts: dict[str, int] = {}
    assigned = 0

    for name in names[:-1]:
        size = round(total * fractions[name])
        counts[name] = size
        assigned += size

    remainder = total - assigned
    if remainder < 0:
        raise ValueError(f"rounding overshoots: {assigned} of {total} samples already assigned")
    counts[names[-1]] = remainder
    return counts


def _check_every_set_has_every_class(
    partition: Mapping[str, np.ndarray], labels: np.ndarray
) -> None:
    """Fail if a set is missing a class, which breaks per-class coverage."""
    classes = np.unique(labels)
    for name, indices in partition.items():
        present = np.unique(labels[indices])
        missing = sorted(set(classes.tolist()) - set(present.tolist()))
        if missing:
            raise ValueError(f"split {name!r} has no samples of classes {missing}")


def make_credit_split(frame: pd.DataFrame, *, seed: int) -> Split:
    """Build the credit partition, leaving the duplicated rows out.

    Indices point into ``frame`` **as given**: the caller keeps using its own
    frame and the duplicated rows simply never appear in any set. That is why
    the split does not depend on the order in which the caller reindexes it.

    Parameters
    ----------
    frame:
        Cleaned credit frame, as returned by :func:`uq.data.credit.load`.
    seed:
        Seed of the partition generator.
    """
    labels = frame[credit.TARGET].to_numpy(dtype=np.int64)
    unique_rows = np.flatnonzero(~credit.duplicate_mask(frame))

    partition = stratified_partition(
        labels, fractions=CREDIT_FRACTIONS, seed=seed, candidates=unique_rows
    )
    return Split(
        train=partition["train"],
        val=partition["val"],
        calibration=partition["calibration"],
        test=partition["test"],
    )


def make_image_split(
    *,
    seed: int,
    root: Path | None = None,
    size: int = pneumonia.DEFAULT_SIZE,
    download_if_missing: bool = True,
) -> Split:
    """Build the image partition, keeping the official test split intact.

    All four sets index the concatenation built by
    :func:`uq.data.pneumonia.load_pooled`, whose order is
    ``train + val + test``. The official test split keeps its positions, so the
    partition can be validated for overlap like any other, and a test index
    cannot be confused with a development one.

    Parameters
    ----------
    seed:
        Seed of the partition generator.
    root:
        Data folder; defaults to :func:`uq.settings.data_dir`.
    size:
        Resolution passed to MedMNIST; the project uses 28.
    download_if_missing:
        Whether to download the ``.npz`` files when they are absent.
    """
    _, labels, bounds = pneumonia.load_pooled(
        root, size=size, download_if_missing=download_if_missing
    )

    development_pool = np.arange(bounds["train"].start, bounds["val"].stop)
    partition = stratified_partition(
        labels, fractions=IMAGE_DEV_FRACTIONS, seed=seed, candidates=development_pool
    )

    return Split(
        train=partition["train"],
        val=partition["val"],
        calibration=partition["calibration"],
        test=np.arange(bounds["test"].start, bounds["test"].stop, dtype=np.int64),
    )


def split_path(dataset: str, seed: int) -> Path:
    """Where the partition of ``dataset`` with ``seed`` is stored."""
    return data_dir() / "splits" / f"{dataset}_seed{seed}.json"


def save(path: Path, split: Split, *, dataset: str, seed: int) -> Path:
    """Write a split as JSON and return the path.

    The file records the dataset, the seed and the sizes next to the indices,
    so a JSON found on disk explains itself without consulting this module.
    ``proportions`` are the fractions observed in the resulting split, which for
    the images differ from the 80/10/10 protocol because the test set is the
    official one and is not re-split.
    """
    payload = {
        "dataset": dataset,
        "seed": seed,
        "index_base": "positional",
        "proportions": _observed_proportions(split),
        "sizes": split.sizes(),
        "splits": split.as_dict(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load(path: Path) -> Split:
    """Read a split written by :func:`save`, validating it on the way in.

    A file whose sets overlap raises here instead of producing a model that has
    already seen the test set.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "splits" not in payload:
        raise ValueError(f"{path} has no 'splits' key; is it a split file?")

    stored = payload["splits"]
    missing = set(SPLIT_NAMES) - set(stored)
    if missing:
        raise ValueError(f"{path} is missing the sets {sorted(missing)}")

    return Split(**{name: np.asarray(stored[name], dtype=np.int64) for name in SPLIT_NAMES})


def _observed_proportions(split: Split) -> dict[str, float]:
    """Share of the whole partition held by each set, rounded for JSON."""
    sizes = split.sizes()
    total = sum(sizes.values())
    return {name: round(size / total, 6) for name, size in sizes.items()}
