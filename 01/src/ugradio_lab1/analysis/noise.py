"""Noise statistics and averaging/radiometer-equation analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ugradio_lab1.utils.validation import as_1d_array


def radiometer_fit(
    n_avg: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Fit ``sigma`` vs averaging count with log-log regression.

    Model:
    ``log10(sigma) = intercept + slope * log10(n_avg)``
    """

    n = as_1d_array(n_avg, "n_avg", dtype=float)
    s = as_1d_array(sigma, "sigma", dtype=float)
    if n.size != s.size:
        raise ValueError("n_avg and sigma must have the same length.")
    if np.any(n <= 0.0):
        raise ValueError("n_avg values must be positive.")
    if np.any(s <= 0.0):
        raise ValueError("sigma values must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in the open interval (0, 1).")
    if n.size < 2:
        raise ValueError("At least two points are required for radiometer fitting.")

    x = np.log10(n)
    y = np.log10(s)
    fit = stats.linregress(x, y)
    slope = float(fit.slope)
    intercept = float(fit.intercept)

    y_hat = intercept + slope * x
    residual = y - y_hat
    r_squared = 1.0 - (np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2))

    dof = n.size - 2
    alpha = 1.0 - confidence_level
    if dof > 0:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
        slope_ci_low = slope - t_crit * float(fit.stderr)
        slope_ci_high = slope + t_crit * float(fit.stderr)
        intercept_stderr = float(fit.intercept_stderr) if fit.intercept_stderr is not None else np.nan
        intercept_ci_low = intercept - t_crit * intercept_stderr
        intercept_ci_high = intercept + t_crit * intercept_stderr
    else:
        slope_ci_low = np.nan
        slope_ci_high = np.nan
        intercept_ci_low = np.nan
        intercept_ci_high = np.nan

    expected_slope = -0.5
    return {
        "n_points": float(n.size),
        "confidence_level": confidence_level,
        "slope": slope,
        "intercept": intercept,
        "slope_ci_low": float(slope_ci_low),
        "slope_ci_high": float(slope_ci_high),
        "intercept_ci_low": float(intercept_ci_low),
        "intercept_ci_high": float(intercept_ci_high),
        "expected_slope": expected_slope,
        "slope_minus_expected": slope - expected_slope,
        "r_value": float(fit.rvalue),
        "r_squared": float(r_squared),
        "p_value": float(fit.pvalue),
        "slope_stderr": float(fit.stderr),
        "intercept_stderr": float(fit.intercept_stderr) if fit.intercept_stderr is not None else np.nan,
        "residual_rms_log10": float(np.sqrt(np.mean(residual**2))),
    }


def radiometer_summary_table(
    n_avg: np.ndarray | Sequence[float],
    sigma_power: np.ndarray | Sequence[float],
    *,
    block_size: int,
    fit_result: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Build a T6-style radiometer summary table.

    Columns match the notebook blueprint:
    ``block_size, n_avg, sigma_power, fitted_slope, expected_slope, chi2_dof``.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    n = as_1d_array(n_avg, "n_avg", dtype=float)
    sigma = as_1d_array(sigma_power, "sigma_power", dtype=float)
    if n.size != sigma.size:
        raise ValueError("n_avg and sigma_power must have the same length.")
    if np.any(n <= 0.0):
        raise ValueError("n_avg values must be positive.")
    if np.any(sigma <= 0.0):
        raise ValueError("sigma_power values must be positive.")

    if fit_result is None:
        fit = radiometer_fit(n, sigma)
    else:
        fit = {key: float(value) for key, value in fit_result.items()}
        for key in ("slope", "intercept"):
            if key not in fit:
                raise ValueError(f"fit_result must include {key!r}.")
        fit.setdefault("expected_slope", -0.5)

    x = np.log10(n)
    y = np.log10(sigma)
    y_hat = fit["intercept"] + fit["slope"] * x
    residual = y - y_hat
    dof = max(n.size - 2, 1)
    chi2_dof = float(np.sum(residual**2) / dof)

    table = pd.DataFrame(
        {
            "block_size": np.full(n.size, int(block_size), dtype=int),
            "n_avg": n.astype(int),
            "sigma_power": sigma,
            "fitted_slope": np.full(n.size, fit["slope"], dtype=float),
            "expected_slope": np.full(n.size, fit["expected_slope"], dtype=float),
            "chi2_dof": np.full(n.size, chi2_dof, dtype=float),
        }
    )
    table = table.sort_values("n_avg", kind="stable").reset_index(drop=True)
    return table


__all__ = ["radiometer_fit", "radiometer_summary_table"]
