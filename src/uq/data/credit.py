"""Loader for the UCI *Default of Credit Card Clients* dataset.

Source (official): https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Licence: CC BY 4.0. The file is **not** redistributed here: it is downloaded
from UCI by ``scripts/download_data.py``.

The workbook published by UCI has two quirks that this module handles:

* the real column names live in the *first data row*, so pandas alone reads
  ``Unnamed: 0`` and ``X1..X23`` as headers;
* there are no missing values, but there are exact duplicate rows, which must
  be removed *before* splitting to avoid train/test leakage;
* the repayment-history block jumps from ``PAY_0`` to ``PAY_2``: there is no
  ``PAY_1`` column in the published file. The names are kept as published
  (``pay_0``, ``pay_2`` ... ``pay_6``) instead of renumbering them.

Column names are normalised to ``snake_case`` and the ``id`` column is dropped
(it carries no signal and would leak the original ordering).
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from uq.settings import data_dir

DATASET_NAME = "credit_default"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DOI = "10.24432/C55S3H"
LICENCE = "CC BY 4.0"

ARCHIVE_NAME = "credit_default.zip"
WORKBOOK_NAME = "default of credit card clients.xls"

TARGET = "default"
ID_COLUMN = "id"

#: Column names exactly as published in the UCI workbook, in order.
RAW_COLUMNS: tuple[str, ...] = (
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *(f"BILL_AMT{i}" for i in range(1, 7)),
    *(f"PAY_AMT{i}" for i in range(1, 7)),
    "default payment next month",
)

# Raw header -> snake_case name.
_RENAMES: dict[str, str] = {
    "ID": ID_COLUMN,
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "default payment next month": TARGET,
}
_RENAMES |= {f"PAY_{i}": f"pay_{i}" for i in (0, 2, 3, 4, 5, 6)}
_RENAMES |= {f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)}
_RENAMES |= {f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)}

#: Columns that describe the client and may induce bias (documented, not skipped).
SENSITIVE_COLUMNS: tuple[str, ...] = ("sex", "education", "marriage", "age")

#: Columns whose values are labels encoded as integers, not quantities.
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "sex",
    "education",
    "marriage",
    *(f"pay_{i}" for i in (0, 2, 3, 4, 5, 6)),
)


def dataset_dir(root: Path | None = None) -> Path:
    """Folder where the credit dataset is stored."""
    return (root or data_dir()) / DATASET_NAME


def sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, computed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: Path | None = None, *, force: bool = False) -> Path:
    """Download and unpack the workbook, returning the path to the ``.xls`` file.

    The archive is fetched from UCI and kept next to the workbook so the
    checksum can be recorded in the data manifest.
    """
    target_dir = dataset_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    workbook = target_dir / WORKBOOK_NAME

    if workbook.exists() and not force:
        return workbook

    archive = target_dir / ARCHIVE_NAME
    if force or not archive.exists():
        with urllib.request.urlopen(SOURCE_URL) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    with zipfile.ZipFile(archive) as bundle:
        member = next((n for n in bundle.namelist() if n.lower().endswith(".xls")), None)
        if member is None:
            raise FileNotFoundError(f"no .xls member found in {archive}")
        with bundle.open(member) as source, workbook.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    return workbook


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw workbook: header row, column names, dtypes and ID.

    Parameters
    ----------
    frame:
        Output of ``pandas.read_excel`` on the UCI workbook, as read
        (``Unnamed: 0``/``X1..X23``) or with the published header already in
        place (``ID``, ``LIMIT_BAL`` ...).

    Returns
    -------
    A numeric frame with ``snake_case`` columns, without the ``id`` column and
    with the binary target ``default`` as the last column.
    """
    frame = frame.copy()

    # The real header lives in the first row of the sheet.
    first_cell = str(frame.iloc[0, 0]).strip()
    if first_cell.upper() == "ID":
        frame.columns = [str(value).strip() for value in frame.iloc[0]]
        frame = frame.iloc[1:].reset_index(drop=True)

    frame.columns = [str(column).strip() for column in frame.columns]
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"unexpected workbook columns, missing: {sorted(missing)[:5]}")

    frame = frame.rename(columns=_RENAMES)
    frame = frame.apply(pd.to_numeric, errors="coerce")

    if frame.isna().any().any():  # pragma: no cover - guarded by a test
        raise ValueError("the credit dataset contains unparsable or missing values")

    frame[TARGET] = frame[TARGET].astype("int8")
    frame = frame.drop(columns=[ID_COLUMN])
    return frame


def load(root: Path | None = None, *, force_download: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset (30 000 rows x 23 features + target)."""
    workbook = download(root, force=force_download)
    raw = pd.read_excel(workbook, engine="calamine")
    return clean(raw)


def duplicates(frame: pd.DataFrame) -> int:
    """Number of exact duplicate rows, ignoring the target column."""
    features = frame.drop(columns=[TARGET])
    return int(features.duplicated().sum())
