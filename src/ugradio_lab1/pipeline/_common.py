"""Shared pipeline infrastructure for Lab 1 experiment modules."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any, Callable

import numpy as np
import pandas as pd

from ugradio_lab1.dataio.catalog import write_manifest_csv

_NPZ_METADATA_KEY = "__metadata_json__"


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def is_tar_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar")


def load_npz_from_path(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return unpack_npz_payload(payload)


def load_npz_from_bytes(raw: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        return unpack_npz_payload(payload)


def load_npz_from_tar_member(
    archive: tarfile.TarFile,
    *,
    members: dict[str, tarfile.TarInfo],
    member_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if member_name not in members:
        raise FileNotFoundError(f"Tar member not found: {member_name}")
    extracted = archive.extractfile(members[member_name])
    if extracted is None:
        raise FileNotFoundError(f"Unable to extract tar member: {member_name}")
    return load_npz_from_bytes(extracted.read())


def load_npz_from_directory(source_dir: Path, *, member_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = source_dir / member_name
    if not path.exists():
        raise FileNotFoundError(path)
    return load_npz_from_path(path)


def unpack_npz_payload(payload: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for key in payload.files:
        if key == _NPZ_METADATA_KEY:
            metadata = json.loads(str(payload[key].item()))
        else:
            arrays[key] = np.asarray(payload[key])
    return arrays, metadata


def load_arrays_for_catalog_row(row: pd.Series) -> dict[str, np.ndarray]:
    source_kind = str(row.get("source_kind"))
    source_path = Path(str(row.get("source_path")))
    source_member = str(row.get("source_member"))

    if source_kind == "tar":
        with tarfile.open(source_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            arrays, _ = load_npz_from_tar_member(
                archive,
                members=members,
                member_name=source_member,
            )
    elif source_kind == "directory":
        arrays, _ = load_npz_from_directory(source_path, member_name=source_member)
    else:
        raise ValueError(f"Unsupported source_kind: {source_kind!r}")
    return arrays


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return int(-1)


def coerce_bool(value: Any) -> bool | Any:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return pd.NA


# ---------------------------------------------------------------------------
# Series helpers
# ---------------------------------------------------------------------------

def series_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(False, index=frame.index)
    return values.fillna(False).astype(bool)


def series_numeric(frame: pd.DataFrame | pd.Series, column: str | None = None) -> pd.Series:
    if isinstance(frame, pd.Series):
        values = frame
    else:
        if column is None:
            raise ValueError("column is required when frame is a DataFrame.")
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


# ---------------------------------------------------------------------------
# Record collection
# ---------------------------------------------------------------------------

RecordFn = Callable[
    ...,  # keyword arguments: npz_name, source_kind, source_path, source_member, arrays, metadata
    dict[str, Any],
]


def collect_records_from_tar(source: Path, record_fn: RecordFn) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with tarfile.open(source, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            arrays, metadata = load_npz_from_bytes(extracted.read())
            records.append(
                record_fn(
                    npz_name=Path(member.name).name,
                    source_kind="tar",
                    source_path=str(source),
                    source_member=member.name,
                    arrays=arrays,
                    metadata=metadata,
                )
            )
    return records


def collect_records_from_directory(source: Path, record_fn: RecordFn) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for npz_path in sorted(source.rglob("*.npz")):
        arrays, metadata = load_npz_from_path(npz_path)
        records.append(
            record_fn(
                npz_name=npz_path.name,
                source_kind="directory",
                source_path=str(source),
                source_member=str(npz_path.relative_to(source)),
                arrays=arrays,
                metadata=metadata,
            )
        )
    return records


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_dataframe_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def write_table_manifest_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a table/manifest DataFrame with standard CSV behavior."""

    return write_manifest_csv(frame, path)
