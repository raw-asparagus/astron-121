"""Frequency-resolution estimation and validation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def resolution_vs_n(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build resolution metrics table as a function of sample count ``N``.

    Required columns in each record:
    - ``n_samples``
    - ``sample_rate_hz``

    Optional columns:
    - ``run_id``
    - ``true_delta_f_hz``
    - ``measured_delta_f_hz``
    """

    frame = _to_frame(records)
    required = {"n_samples", "sample_rate_hz"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"records is missing required columns: {sorted(missing)}")

    n_samples = frame["n_samples"].to_numpy(dtype=float)
    sample_rate_hz = frame["sample_rate_hz"].to_numpy(dtype=float)
    if np.any(n_samples <= 0):
        raise ValueError("n_samples values must be positive.")
    if np.any(sample_rate_hz <= 0):
        raise ValueError("sample_rate_hz values must be positive.")

    delta_f_bin_hz = sample_rate_hz / n_samples
    if "measured_delta_f_hz" in frame.columns:
        measured_delta = frame["measured_delta_f_hz"].to_numpy(dtype=float)
    elif "true_delta_f_hz" in frame.columns:
        measured_delta = frame["true_delta_f_hz"].to_numpy(dtype=float)
    else:
        measured_delta = np.full(frame.shape[0], np.nan, dtype=float)

    if "true_delta_f_hz" in frame.columns:
        true_delta = frame["true_delta_f_hz"].to_numpy(dtype=float)
    else:
        true_delta = np.full(frame.shape[0], np.nan, dtype=float)

    summary = pd.DataFrame(
        {
            "run_id": frame["run_id"] if "run_id" in frame.columns else pd.Series(frame.index, dtype=str),
            "n_samples": n_samples.astype(int),
            "sample_rate_hz": sample_rate_hz,
            "delta_f_bin_hz": delta_f_bin_hz,
            "true_delta_f_hz": true_delta,
            "measured_delta_f_hz": measured_delta,
            "resolution_ratio": measured_delta / delta_f_bin_hz,
        }
    )
    summary = summary.sort_values("n_samples", kind="stable").reset_index(drop=True)
    return summary


def _to_frame(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("records cannot be empty.")
    return frame


__all__ = ["resolution_vs_n"]
