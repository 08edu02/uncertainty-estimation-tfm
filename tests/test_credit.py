"""Tests for the credit-card dataset loader (no network access needed)."""

from __future__ import annotations

import pandas as pd
import pytest

from uq.data import credit

N_ROWS = 4


def _values(row: int) -> list[object]:
    values: dict[str, object] = dict.fromkeys(credit.RAW_COLUMNS, -1)
    values["ID"] = row + 1
    values["LIMIT_BAL"] = 20_000 + row
    values["SEX"] = 2
    values["EDUCATION"] = 2
    values["MARRIAGE"] = 1
    values["AGE"] = 24 + row
    values["default payment next month"] = row % 2
    return [values[name] for name in credit.RAW_COLUMNS]


def _published_frame() -> pd.DataFrame:
    """Frame whose header row is already the published one."""
    return pd.DataFrame([_values(row) for row in range(N_ROWS)], columns=credit.RAW_COLUMNS)


def _as_read_by_pandas() -> pd.DataFrame:
    """Mimic ``read_excel``: generic headers, real names in the first row."""
    generic = ["Unnamed: 0", *(f"X{i}" for i in range(1, len(credit.RAW_COLUMNS)))]
    body = pd.DataFrame([_values(row) for row in range(N_ROWS)], columns=generic)
    header = pd.DataFrame([list(credit.RAW_COLUMNS)], columns=generic)
    return pd.concat([header, body], ignore_index=True)


@pytest.mark.parametrize("frame", [_published_frame(), _as_read_by_pandas()])
def test_clean_normalises_names_and_drops_id(frame: pd.DataFrame) -> None:
    cleaned = credit.clean(frame)

    assert len(cleaned) == N_ROWS
    assert credit.ID_COLUMN not in cleaned.columns
    assert cleaned.columns[-1] == credit.TARGET
    assert list(cleaned.columns[:5]) == ["limit_bal", "sex", "education", "marriage", "age"]
    assert "pay_1" not in cleaned.columns  # the published file has no PAY_1
    assert cleaned[credit.TARGET].dtype == "int8"
    assert cleaned["limit_bal"].dtype == "int64"


def test_clean_rejects_unexpected_columns() -> None:
    frame = _published_frame().drop(columns=["PAY_3"])
    with pytest.raises(ValueError, match="unexpected workbook columns"):
        credit.clean(frame)


def test_duplicates_ignore_the_target() -> None:
    frame = _published_frame()
    features = list(credit.RAW_COLUMNS[:-1])
    frame.loc[1, features] = frame.loc[0, features]
    frame.loc[1, "default payment next month"] = 1 - frame.loc[0, "default payment next month"]

    cleaned = credit.clean(frame)
    assert credit.duplicates(cleaned) == 1
