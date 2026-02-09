"""Spectral-leakage analysis and Nyquist-window diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ugradio_lab1.utils.validation import as_1d_array


def leakage_metric(
    frequency_hz: np.ndarray | Sequence[float],
    power_v2: np.ndarray | Sequence[float],
    *,
    tone_frequency_hz: float,
    main_lobe_half_width_bins: int = 1,
) -> dict[str, float]:
    """Compute a scalar leakage metric for one tone measurement.

    The metric is defined as:

    ``leakage_fraction = (total_power - main_lobe_power) / total_power``

    where the main lobe is centered on the bin nearest ``tone_frequency_hz`` and
    includes ``main_lobe_half_width_bins`` bins on each side.
    """

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power = as_1d_array(power_v2, "power_v2", dtype=float)
    _require_same_length(frequency, power, "frequency_hz", "power_v2")

    if np.any(power < 0.0):
        raise ValueError("power_v2 must be non-negative.")
    if main_lobe_half_width_bins < 0:
        raise ValueError("main_lobe_half_width_bins must be >= 0.")

    total_power = float(np.sum(power))
    if total_power <= 0.0:
        raise ValueError("Total power must be positive to compute leakage_metric.")

    center_idx = int(np.argmin(np.abs(frequency - float(tone_frequency_hz))))
    lo = max(0, center_idx - main_lobe_half_width_bins)
    hi = min(power.size, center_idx + main_lobe_half_width_bins + 1)

    main_lobe_power = float(np.sum(power[lo:hi]))
    leakage_power = total_power - main_lobe_power
    leakage_fraction = leakage_power / total_power

    return {
        "tone_frequency_hz": float(tone_frequency_hz),
        "main_bin_index": float(center_idx),
        "main_lobe_power_v2": main_lobe_power,
        "leakage_power_v2": leakage_power,
        "total_power_v2": total_power,
        "leakage_fraction": float(leakage_fraction),
        "leakage_db": float(10.0 * np.log10(np.maximum(leakage_fraction, np.finfo(float).tiny))),
    }


def leakage_resolution_table(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build the T5-style leakage/resolution summary table.

    Required columns:
    - ``run_id``
    - ``n_samples``
    - ``sample_rate_hz``
    - ``leakage_metric``

    Optional columns:
    - ``min_resolvable_delta_f_hz``
    - ``measured_delta_f_hz`` (used as fallback for min resolvable delta-f)
    """

    frame = _to_frame(records)
    required = {"run_id", "n_samples", "sample_rate_hz", "leakage_metric"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"records is missing required columns: {sorted(missing)}")

    n_samples = frame["n_samples"].to_numpy(dtype=float)
    sample_rate_hz = frame["sample_rate_hz"].to_numpy(dtype=float)
    leakage = frame["leakage_metric"].to_numpy(dtype=float)

    if np.any(n_samples <= 0.0):
        raise ValueError("n_samples values must be positive.")
    if np.any(sample_rate_hz <= 0.0):
        raise ValueError("sample_rate_hz values must be positive.")
    if np.any(leakage < 0.0):
        raise ValueError("leakage_metric values must be non-negative.")

    if "min_resolvable_delta_f_hz" in frame.columns:
        min_resolvable = frame["min_resolvable_delta_f_hz"].to_numpy(dtype=float)
    elif "measured_delta_f_hz" in frame.columns:
        min_resolvable = frame["measured_delta_f_hz"].to_numpy(dtype=float)
    else:
        min_resolvable = np.full(frame.shape[0], np.nan, dtype=float)

    summary = pd.DataFrame(
        {
            "run_id": frame["run_id"].astype(str).to_numpy(),
            "n_samples": n_samples.astype(int),
            "delta_f_bin_hz": sample_rate_hz / n_samples,
            "leakage_metric": leakage,
            "min_resolvable_delta_f_hz": min_resolvable,
        }
    )
    summary = summary.sort_values("n_samples", kind="stable").reset_index(drop=True)
    return summary


def nyquist_window_extension(
    frequency_hz: np.ndarray | Sequence[float],
    power_v2: np.ndarray | Sequence[float],
    *,
    sample_rate_hz: float | None = None,
    window_indices: Sequence[int] = (-1, 0, 1),
) -> pd.DataFrame:
    """Replicate one Nyquist-zone spectrum across integer zone indices.

    Returns a long-form table with columns:
    ``window_index``, ``frequency_hz``, ``power_v2``.
    """

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power = as_1d_array(power_v2, "power_v2", dtype=float)
    _require_same_length(frequency, power, "frequency_hz", "power_v2")

    if sample_rate_hz is None:
        if frequency.size < 2:
            raise ValueError(
                "sample_rate_hz must be provided when frequency_hz has fewer than 2 points."
            )
        sorted_frequency = np.sort(frequency)
        delta_f = float(np.median(np.diff(sorted_frequency)))
        inferred_fs = abs(delta_f) * frequency.size
        if inferred_fs <= 0.0:
            raise ValueError("Could not infer positive sample_rate_hz from frequency_hz.")
        fs = inferred_fs
    else:
        fs = float(sample_rate_hz)
    if fs <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")

    if len(window_indices) == 0:
        raise ValueError("window_indices cannot be empty.")

    rows: list[pd.DataFrame] = []
    for idx in window_indices:
        shifted_frequency = frequency + int(idx) * fs
        rows.append(
            pd.DataFrame(
                {
                    "window_index": np.full(frequency.size, int(idx), dtype=int),
                    "frequency_hz": shifted_frequency,
                    "power_v2": power,
                }
            )
        )

    extended = pd.concat(rows, ignore_index=True)
    extended = extended.sort_values(["window_index", "frequency_hz"], kind="stable")
    return extended.reset_index(drop=True)


def _to_frame(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("records cannot be empty.")
    return frame


def _require_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if left.size != right.size:
        raise ValueError(f"{left_name} and {right_name} must have the same length.")


__all__ = ["leakage_metric", "leakage_resolution_table", "nyquist_window_extension"]
