"""Read/write helpers for NPZ-based data products."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_METADATA_KEY = "__metadata_json__"


def save_npz_dataset(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any] | None = None,
    compressed: bool = True,
    overwrite: bool = True,
) -> Path:
    """Write arrays + JSON metadata to one NPZ file."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("path must end with .npz")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

    payload: dict[str, np.ndarray] = {}
    if len(arrays) == 0:
        raise ValueError("arrays cannot be empty.")
    for key, value in arrays.items():
        if not key:
            raise ValueError("Array keys cannot be empty.")
        if key == _METADATA_KEY:
            raise ValueError(f"{_METADATA_KEY!r} is reserved for internal metadata storage.")
        payload[key] = np.asarray(value)

    metadata_json = json.dumps(dict(metadata or {}), sort_keys=True)
    payload[_METADATA_KEY] = np.array(metadata_json)

    destination.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(destination, **payload)
    else:
        np.savez(destination, **payload)
    return destination


def load_npz_dataset(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load arrays + metadata from an NPZ dataset created by save_npz_dataset."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    with np.load(source, allow_pickle=False) as payload:
        for key in payload.files:
            if key == _METADATA_KEY:
                raw = payload[key]
                metadata = json.loads(str(raw.item()))
            else:
                arrays[key] = np.asarray(payload[key])
    return arrays, metadata


__all__ = ["load_npz_dataset", "save_npz_dataset"]
