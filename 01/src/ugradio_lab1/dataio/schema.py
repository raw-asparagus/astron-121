"""Schemas and metadata conventions for Lab 1 datasets/tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "T1": (
        "timestamp",
        "instrument",
        "check",
        "measured_value",
        "tolerance",
        "pass_fail",
        "notes",
    ),
    "T2": (
        "run_id",
        "experiment",
        "sample_rate_hz",
        "center_frequency_hz",
        "tones_hz",
        "source_power_dbm",
        "mixer_config",
        "cable_config",
        "n_samples",
    ),
    "T3": (
        "run_id",
        "f_true_hz",
        "sample_rate_hz",
        "predicted_alias_hz",
        "measured_alias_hz",
        "residual_hz",
        "uncertainty_hz",
    ),
    "T4": (
        "mode",
        "passband_estimate_hz",
        "rolloff_metric_db_per_hz",
        "ripple_db",
        "fit_residuals_db",
    ),
    "T5": (
        "run_id",
        "n_samples",
        "delta_f_bin_hz",
        "leakage_metric",
        "min_resolvable_delta_f_hz",
    ),
    "T6": (
        "block_size",
        "n_avg",
        "sigma_power",
        "fitted_slope",
        "expected_slope",
        "chi2_dof",
    ),
    "T7": (
        "config",
        "f_lo_hz",
        "f_rf_hz",
        "expected_line_hz",
        "observed_line_hz",
        "level_db",
        "interpretation",
    ),
    "T8": (
        "goal_id",
        "evidence_items",
        "status",
        "limitations",
    ),
}


def get_table_schema(table_id: str) -> tuple[str, ...]:
    """Return required columns for a given blueprint table ID (T1..T8)."""

    normalized = table_id.upper()
    if normalized not in TABLE_SCHEMAS:
        raise ValueError(f"Unsupported table_id: {table_id!r}.")
    return TABLE_SCHEMAS[normalized]


def empty_table(table_id: str) -> pd.DataFrame:
    """Create an empty DataFrame with columns for ``table_id``."""

    return pd.DataFrame(columns=list(get_table_schema(table_id)))


def validate_table(
    table_id: str,
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
    *,
    allow_extra: bool = True,
    fill_missing: bool = False,
) -> pd.DataFrame:
    """Validate/coerce records against one blueprint table schema."""

    schema = get_table_schema(table_id)
    frame = _to_frame(records)
    required = set(schema)
    present = set(frame.columns)

    missing = sorted(required - present)
    if missing:
        if fill_missing:
            for column in missing:
                frame[column] = pd.NA
        else:
            raise ValueError(
                f"Table {table_id} is missing required columns: {missing}."
            )

    if not allow_extra:
        frame = frame.loc[:, list(schema)]
    else:
        # Required columns first for predictable ordering.
        extras = [column for column in frame.columns if column not in required]
        frame = frame.loc[:, list(schema) + extras]

    return frame.reset_index(drop=True)


def build_status_table(
    ids: Sequence[str],
    *,
    id_column: str,
    status_column: str = "status",
    default_status: str = "not started",
) -> pd.DataFrame:
    """Build a small status tracker table (used by notebook tracking cells)."""

    if len(ids) == 0:
        raise ValueError("ids cannot be empty.")
    return pd.DataFrame(
        {
            id_column: list(ids),
            status_column: [default_status] * len(ids),
            "path": [""] * len(ids),
            "notes": [""] * len(ids),
        }
    )


def _to_frame(
    records: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    elif isinstance(records, Mapping):
        frame = pd.DataFrame(records)
    else:
        frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("records cannot be empty.")
    return frame


__all__ = [
    "TABLE_SCHEMAS",
    "build_status_table",
    "empty_table",
    "get_table_schema",
    "validate_table",
]
