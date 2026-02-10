"""Unit tests for dataio.schema."""

from __future__ import annotations

import pandas as pd
import pytest

from ugradio_lab1.dataio.schema import (
    build_status_table,
    empty_table,
    get_table_schema,
    validate_table,
)


def test_get_table_schema_t2_contains_manifest_fields() -> None:
    schema = get_table_schema("T2")
    assert "run_id" in schema
    assert "sample_rate_hz" in schema


def test_empty_table_has_expected_columns() -> None:
    table = empty_table("T1")
    assert list(table.columns) == list(get_table_schema("T1"))


def test_validate_table_rejects_missing_columns() -> None:
    rows = [{"run_id": "r1"}]
    with pytest.raises(ValueError, match="missing required columns"):
        validate_table("T2", rows)


def test_build_status_table_shape() -> None:
    status = build_status_table(["F1", "F2"], id_column="figure_id")
    assert isinstance(status, pd.DataFrame)
    assert status.shape[0] == 2
    assert "status" in status.columns
