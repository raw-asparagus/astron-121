"""Bandpass characterization routines for SDR filtering behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ugradio_lab1.utils.validation import as_1d_array


def bandpass_curve(
    frequency_hz: np.ndarray | Sequence[float],
    amplitude_v: np.ndarray | Sequence[float],
    *,
    reference_amplitude_v: float | None = None,
    mode: str | None = None,
) -> pd.DataFrame:
    """Compute gain curve quantities from a measured amplitude sweep."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    amplitude = np.abs(as_1d_array(amplitude_v, "amplitude_v"))
    _require_same_length(frequency, amplitude, "frequency_hz", "amplitude_v")

    if reference_amplitude_v is None:
        reference = float(np.nanmax(amplitude))
    else:
        reference = float(reference_amplitude_v)
    if reference <= 0.0:
        raise ValueError("reference_amplitude_v must be positive.")

    gain_linear = amplitude / reference
    gain_db = 20.0 * np.log10(np.maximum(gain_linear, np.finfo(float).tiny))

    curve = pd.DataFrame(
        {
            "frequency_hz": frequency,
            "amplitude_v": amplitude,
            "reference_amplitude_v": np.full(frequency.size, reference, dtype=float),
            "gain_linear": gain_linear,
            "gain_db": gain_db,
        }
    )
    if mode is not None:
        curve["mode"] = mode
    return curve


def bandpass_summary_metrics(
    frequency_hz: np.ndarray | Sequence[float],
    gain_db: np.ndarray | Sequence[float],
) -> dict[str, float]:
    """Compute summary metrics for a gain-vs-frequency bandpass curve."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    gain = as_1d_array(gain_db, "gain_db", dtype=float)
    _require_same_length(frequency, gain, "frequency_hz", "gain_db")

    order = np.argsort(frequency)
    frequency = frequency[order]
    gain = gain[order]

    peak_idx = int(np.nanargmax(gain))
    peak_gain_db = float(gain[peak_idx])
    peak_frequency_hz = float(frequency[peak_idx])

    passband_mask = gain >= peak_gain_db - 3.0
    if np.any(passband_mask):
        passband_frequency = frequency[passband_mask]
        passband_gain = gain[passband_mask]
        passband_low_hz = float(passband_frequency[0])
        passband_high_hz = float(passband_frequency[-1])
        passband_width_hz = float(passband_high_hz - passband_low_hz)
        passband_ripple_db = float(np.nanmax(passband_gain) - np.nanmin(passband_gain))
    else:
        passband_low_hz = np.nan
        passband_high_hz = np.nan
        passband_width_hz = np.nan
        passband_ripple_db = np.nan

    left_rolloff = _rolloff_slope(
        frequency_hz=frequency,
        gain_db=gain,
        peak_gain_db=peak_gain_db,
        peak_idx=peak_idx,
        side="left",
    )
    right_rolloff = _rolloff_slope(
        frequency_hz=frequency,
        gain_db=gain,
        peak_gain_db=peak_gain_db,
        peak_idx=peak_idx,
        side="right",
    )

    fit_coeff = np.polyfit(frequency, gain, deg=2)
    fit_gain = np.polyval(fit_coeff, frequency)
    fit_residual_rms_db = float(np.sqrt(np.nanmean((gain - fit_gain) ** 2)))

    return {
        "peak_frequency_hz": peak_frequency_hz,
        "peak_gain_db": peak_gain_db,
        "passband_low_hz": passband_low_hz,
        "passband_high_hz": passband_high_hz,
        "passband_width_hz": passband_width_hz,
        "passband_ripple_db": passband_ripple_db,
        "left_rolloff_db_per_hz": left_rolloff,
        "right_rolloff_db_per_hz": right_rolloff,
        "fit_residual_rms_db": fit_residual_rms_db,
    }


def bandpass_summary_table(
    frequency_hz: np.ndarray | Sequence[float],
    gain_db_by_mode: Mapping[str, np.ndarray | Sequence[float]],
) -> pd.DataFrame:
    """Build a T4-style summary table from mode-keyed bandpass curves."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    if not gain_db_by_mode:
        raise ValueError("gain_db_by_mode cannot be empty.")

    rows: list[dict[str, float | str]] = []
    for mode, gain_db in gain_db_by_mode.items():
        gain = as_1d_array(gain_db, f"gain_db_by_mode[{mode!r}]", dtype=float)
        _require_same_length(frequency, gain, "frequency_hz", f"gain_db_by_mode[{mode!r}]")
        metrics = bandpass_summary_metrics(frequency, gain)
        rolloff_values = np.array(
            [metrics["left_rolloff_db_per_hz"], metrics["right_rolloff_db_per_hz"]],
            dtype=float,
        )
        rolloff_metric = float(np.nanmean(np.abs(rolloff_values)))
        rows.append(
            {
                "mode": mode,
                "passband_estimate_hz": float(metrics["passband_width_hz"]),
                "rolloff_metric_db_per_hz": rolloff_metric,
                "ripple_db": float(metrics["passband_ripple_db"]),
                "fit_residuals_db": float(metrics["fit_residual_rms_db"]),
                "peak_frequency_hz": float(metrics["peak_frequency_hz"]),
                "peak_gain_db": float(metrics["peak_gain_db"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values("mode", kind="stable").reset_index(drop=True)
    return summary


def _rolloff_slope(
    *,
    frequency_hz: np.ndarray,
    gain_db: np.ndarray,
    peak_gain_db: float,
    peak_idx: int,
    side: str,
) -> float:
    target_3db = peak_gain_db - 3.0
    target_20db = peak_gain_db - 20.0

    if side == "left":
        search_idx = np.arange(0, peak_idx + 1)
    elif side == "right":
        search_idx = np.arange(peak_idx, frequency_hz.size)
    else:
        raise ValueError("side must be either 'left' or 'right'.")

    if search_idx.size < 2:
        return np.nan

    search_gain = gain_db[search_idx]
    search_freq = frequency_hz[search_idx]

    idx_3 = int(np.nanargmin(np.abs(search_gain - target_3db)))
    idx_20 = int(np.nanargmin(np.abs(search_gain - target_20db)))
    f3 = float(search_freq[idx_3])
    f20 = float(search_freq[idx_20])
    g3 = float(search_gain[idx_3])
    g20 = float(search_gain[idx_20])

    if np.isclose(f20, f3):
        return np.nan
    return (g20 - g3) / (f20 - f3)


def _require_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if left.size != right.size:
        raise ValueError(f"{left_name} and {right_name} must have the same length.")


__all__ = ["bandpass_curve", "bandpass_summary_metrics", "bandpass_summary_table"]
