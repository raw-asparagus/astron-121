"""Dataset catalog utilities and manifest indexing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ugradio_lab1.dataio.schema import empty_table, validate_table


def build_manifest(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    table_id: str = "T2",
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Build a validated manifest table for one blueprint table type."""

    manifest = validate_table(table_id, records, allow_extra=allow_extra, fill_missing=False)
    if table_id.upper() == "T2" and "run_id" in manifest.columns:
        manifest = manifest.sort_values("run_id", kind="stable").reset_index(drop=True)
    return manifest


def read_manifest_csv(
    path: str | Path,
    *,
    table_id: str | None = None,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Read a CSV manifest and optionally validate against a table schema."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    if table_id is None:
        return frame
    if frame.empty:
        return empty_table(table_id)
    return validate_table(table_id, frame, allow_extra=allow_extra, fill_missing=False)


def write_manifest_csv(
    manifest: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write a manifest DataFrame to CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination, index=False)
    return destination


def append_manifest_rows(
    path: str | Path,
    rows: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    table_id: str = "T2",
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Append rows into a CSV manifest; create it if missing."""

    destination = Path(path)
    new_rows = validate_table(table_id, rows, allow_extra=allow_extra, fill_missing=False)

    if destination.exists():
        existing = read_manifest_csv(destination, table_id=table_id, allow_extra=allow_extra)
    else:
        existing = empty_table(table_id)

    combined = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    combined = validate_table(table_id, combined, allow_extra=allow_extra, fill_missing=False)
    write_manifest_csv(combined, destination)
    return combined


def filter_manifest(
    manifest: pd.DataFrame,
    **criteria: Any,
) -> pd.DataFrame:
    """Filter a manifest by equality or membership criteria."""

    if manifest.empty:
        return manifest.copy()

    mask = pd.Series(True, index=manifest.index)
    for column, expected in criteria.items():
        if column not in manifest.columns:
            raise ValueError(f"Unknown manifest column: {column!r}")
        if isinstance(expected, (list, tuple, set, frozenset)):
            mask &= manifest[column].isin(expected)
        else:
            mask &= manifest[column] == expected
    return manifest.loc[mask].reset_index(drop=True)


def build_goal_coverage_table(
    goal_ids: Sequence[str],
    *,
    evidence_items: Mapping[str, Sequence[str] | str],
    status: Mapping[str, str] | None = None,
    limitations: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build a T8-style goal coverage and evidence matrix."""

    if len(goal_ids) == 0:
        raise ValueError("goal_ids cannot be empty.")

    rows: list[dict[str, str]] = []
    for goal_id in goal_ids:
        evidence = evidence_items.get(goal_id, [])
        if isinstance(evidence, str):
            evidence_str = evidence
        else:
            evidence_str = "; ".join(str(item) for item in evidence)
        rows.append(
            {
                "goal_id": goal_id,
                "evidence_items": evidence_str,
                "status": (status or {}).get(goal_id, "partial"),
                "limitations": (limitations or {}).get(goal_id, ""),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "append_manifest_rows",
    "build_goal_coverage_table",
    "build_manifest",
    "filter_manifest",
    "read_manifest_csv",
    "write_manifest_csv",
]
