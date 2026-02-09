"""Nyquist-zone and aliasing analysis routines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ugradio_lab1.utils.validation import as_1d_array


def predict_alias_frequency(
    f_true_hz: np.ndarray | Sequence[float] | float,
    sample_rate_hz: np.ndarray | Sequence[float] | float,
) -> np.ndarray:
    """Project true frequencies into the principal Nyquist zone.

    The returned aliases are in the interval ``[-fs/2, fs/2)`` using broadcast
    rules across ``f_true_hz`` and ``sample_rate_hz``.
    """

    f_true = np.asarray(f_true_hz, dtype=float)
    fs = np.asarray(sample_rate_hz, dtype=float)
    if np.any(fs <= 0.0):
        raise ValueError("sample_rate_hz must be strictly positive.")
    alias = np.mod(f_true + fs / 2.0, fs) - fs / 2.0
    return np.asarray(alias, dtype=float)


def alias_residual_table(
    f_true_hz: np.ndarray | Sequence[float],
    sample_rate_hz: np.ndarray | Sequence[float],
    measured_alias_hz: np.ndarray | Sequence[float],
    *,
    predicted_alias_hz: np.ndarray | Sequence[float] | None = None,
    run_id: Sequence[str] | None = None,
    uncertainty_hz: np.ndarray | Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build a residual table suitable for aliasing deliverables (e.g., T3)."""

    f_true = as_1d_array(f_true_hz, "f_true_hz", dtype=float)
    fs = as_1d_array(sample_rate_hz, "sample_rate_hz", dtype=float)
    measured = as_1d_array(measured_alias_hz, "measured_alias_hz", dtype=float)
    _require_same_length(f_true, fs, "f_true_hz", "sample_rate_hz")
    _require_same_length(f_true, measured, "f_true_hz", "measured_alias_hz")

    if np.any(fs <= 0.0):
        raise ValueError("sample_rate_hz must be strictly positive.")

    if predicted_alias_hz is None:
        predicted = predict_alias_frequency(f_true, fs)
    else:
        predicted = as_1d_array(predicted_alias_hz, "predicted_alias_hz", dtype=float)
        _require_same_length(f_true, predicted, "f_true_hz", "predicted_alias_hz")

    if run_id is None:
        run_values = np.array([f"run_{idx:03d}" for idx in range(f_true.size)], dtype=object)
    else:
        run_values = np.asarray(list(run_id), dtype=object)
        if run_values.ndim != 1 or run_values.size != f_true.size:
            raise ValueError("run_id must be a 1D sequence with the same length as f_true_hz.")

    if uncertainty_hz is None:
        uncertainty = np.full(f_true.size, np.nan, dtype=float)
    else:
        uncertainty = as_1d_array(uncertainty_hz, "uncertainty_hz", dtype=float)
        _require_same_length(f_true, uncertainty, "f_true_hz", "uncertainty_hz")

    residual = measured - predicted
    table = pd.DataFrame(
        {
            "run_id": run_values,
            "f_true_hz": f_true,
            "sample_rate_hz": fs,
            "predicted_alias_hz": predicted,
            "measured_alias_hz": measured,
            "residual_hz": residual,
            "abs_residual_hz": np.abs(residual),
            "uncertainty_hz": uncertainty,
        }
    )
    return table


def _require_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if left.size != right.size:
        raise ValueError(f"{left_name} and {right_name} must have the same length.")


__all__ = ["alias_residual_table", "predict_alias_frequency"]
