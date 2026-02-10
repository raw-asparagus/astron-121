from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ugradio_lab1.utils.validation import as_1d_array, require_same_length


# ---------------------------------------------------------------------------
# Bandpass
# ---------------------------------------------------------------------------


def bandpass_curve(
        frequency_hz: np.ndarray | Sequence[float],
        amplitude_v: np.ndarray | Sequence[float],
        *,
        reference_amplitude_v: float | None = None,
        mode: str | None = None,
) -> pd.DataFrame:
    """Compute gain curve quantities from a measured amplitude sweep.

    Converts raw amplitude measurements into gain (linear and dB) relative to
    a reference amplitude.  If no reference is given, the maximum observed
    amplitude is used.

    Parameters
    ----------
    frequency_hz : array_like
        Frequency points of the sweep (Hz).
    amplitude_v : array_like
        Measured voltage amplitudes at each frequency (absolute values used).
    reference_amplitude_v : float or None
        Reference voltage for 0 dB gain.  Defaults to ``max(amplitude_v)``.
    mode : str or None
        Optional label (e.g. ``"default"`` or ``"aliased"``) added as a column.

    Returns
    -------
    pandas.DataFrame
        Columns: ``frequency_hz``, ``amplitude_v``, ``reference_amplitude_v``,
        ``gain_linear``, ``gain_db``, and optionally ``mode``.
    """

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    amplitude = np.abs(as_1d_array(amplitude_v, "amplitude_v"))
    require_same_length(frequency, amplitude, "frequency_hz", "amplitude_v")

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


# ---------------------------------------------------------------------------
# DSB/SSB mixers
# ---------------------------------------------------------------------------


def expected_dsb_lines(
    f_lo_hz: float,
    f_rf_hz: float,
    *,
    orders: int = 3,
) -> pd.DataFrame:
    """Generate expected DSB and harmonic product frequencies.

    Computes all combinations ``|m * f_lo +/- n * f_rf|`` with harmonic
    indices ``0 <= m, n <= orders`` (at least one must be nonzero).
    Includes LO harmonics (n=0), RF harmonics (m=0), and mixer products.
    The two primary DSB products (``|f_rf - f_lo|`` and ``f_rf + f_lo``)
    are tagged with family ``"dsb_primary"``, LO/RF harmonics are
    ``"lo_harmonic"``/``"rf_harmonic"``, and all others are ``"mixing_harmonic"``.

    Parameters
    ----------
    f_lo_hz : float
        Local-oscillator frequency in Hz.
    f_rf_hz : float
        RF signal frequency in Hz.
    orders : int
        Maximum harmonic order for mixing harmonic combinations.

    Returns
    -------
    pandas.DataFrame
        Columns: ``line_id``, ``family``, ``m_lo``, ``n_rf``, ``sign``,
        ``formula``, ``expected_hz``.  Sorted by ``expected_hz``,
        duplicates removed.

    Raises
    ------
    ValueError
        If frequencies are non-positive or *orders* < 1.
    """

    f_lo = float(f_lo_hz)
    f_rf = float(f_rf_hz)
    if f_lo <= 0.0 or f_rf <= 0.0:
        raise ValueError("f_lo_hz and f_rf_hz must be positive.")
    if orders < 1:
        raise ValueError("orders must be >= 1.")

    rows: list[dict[str, float | int | str]] = []

    # Primary DSB products (tagged specially)
    rows.extend(
        [
            {
                "family": "dsb_primary",
                "m_lo": 1,
                "n_rf": 1,
                "sign": "-",
                "expected_hz": abs(f_rf - f_lo),
                "formula": "|f_rf - f_lo|",
            },
            {
                "family": "dsb_primary",
                "m_lo": 1,
                "n_rf": 1,
                "sign": "+",
                "expected_hz": f_rf + f_lo,
                "formula": "f_rf + f_lo",
            },
        ]
    )

    # LO harmonics (leakage): m*f_lo for m=1..orders
    for m in range(1, orders + 1):
        rows.append(
            {
                "family": "lo_harmonic",
                "m_lo": m,
                "n_rf": 0,
                "sign": "",
                "expected_hz": m * f_lo,
                "formula": f"{m}*f_lo" if m > 1 else "f_lo",
            }
        )

    # RF harmonics (feedthrough): n*f_rf for n=1..orders
    for n in range(1, orders + 1):
        rows.append(
            {
                "family": "rf_harmonic",
                "m_lo": 0,
                "n_rf": n,
                "sign": "",
                "expected_hz": n * f_rf,
                "formula": f"{n}*f_rf" if n > 1 else "f_rf",
            }
        )

    # Mixing harmonics: |m*f_lo ± n*f_rf| for m,n >= 1
    for m in range(1, orders + 1):
        for n in range(1, orders + 1):
            plus = abs(m * f_lo + n * f_rf)
            minus = abs(m * f_lo - n * f_rf)
            rows.append(
                {
                    "family": "mixing_harmonic",
                    "m_lo": m,
                    "n_rf": n,
                    "sign": "+",
                    "expected_hz": plus,
                    "formula": f"|{m}*f_lo + {n}*f_rf|",
                }
            )
            rows.append(
                {
                    "family": "mixing_harmonic",
                    "m_lo": m,
                    "n_rf": n,
                    "sign": "-",
                    "expected_hz": minus,
                    "formula": f"|{m}*f_lo - {n}*f_rf|",
                }
            )

    frame = pd.DataFrame(rows)
    frame = frame[frame["expected_hz"] > 0.0].copy()
    frame = frame.sort_values("expected_hz", kind="stable").drop_duplicates(
        subset=["expected_hz", "formula"], keep="first"
    )
    frame = frame.reset_index(drop=True)
    frame["line_id"] = [f"L{idx:03d}" for idx in range(frame.shape[0])]
    return frame[
        ["line_id", "family", "m_lo", "n_rf", "sign", "formula", "expected_hz"]
    ]


def match_observed_lines(
    expected_hz: pd.DataFrame | np.ndarray | Sequence[float],
    observed_hz: np.ndarray | Sequence[float],
    *,
    tolerance_hz: float,
) -> pd.DataFrame:
    """Match expected spectral lines to nearest observed peaks within a tolerance.

    Each expected line is matched to the closest unmatched observed peak
    (greedy, first-come-first-served).  Unmatched observed peaks are appended
    with family ``"unmatched_observed"`` for spur surveys.

    Parameters
    ----------
    expected_hz : DataFrame, array_like
        Expected line frequencies.  If a DataFrame, must contain an
        ``expected_hz`` column (and optionally ``line_id``, ``family``, etc.).
    observed_hz : array_like
        Observed peak frequencies (Hz).
    tolerance_hz : float
        Maximum allowed distance for a match (Hz).

    Returns
    -------
    pandas.DataFrame
        One row per expected line plus one per unmatched observed peak.
        Columns include ``expected_hz``, ``observed_hz``, ``delta_hz``,
        ``abs_delta_hz``, ``matched`` (bool), ``match_tolerance_hz``.
    """

    if tolerance_hz <= 0.0:
        raise ValueError("tolerance_hz must be positive.")

    expected_table, expected_values = _normalize_expected(expected_hz)
    observed = as_1d_array(observed_hz, "observed_hz", dtype=float)

    if observed.size == 0:
        raise ValueError("observed_hz cannot be empty.")

    used_observed: set[int] = set()
    matches: list[dict[str, float | int | bool | str]] = []

    for idx, row in expected_table.iterrows():
        expected_value = float(row["expected_hz"])
        deltas = np.abs(observed - expected_value)
        nearest_idx = int(np.argmin(deltas))
        nearest_delta = float(deltas[nearest_idx])
        within = nearest_delta <= tolerance_hz and nearest_idx not in used_observed
        if within:
            used_observed.add(nearest_idx)
            observed_value = float(observed[nearest_idx])
            observed_index = float(nearest_idx)
        else:
            observed_value = np.nan
            observed_index = np.nan

        matches.append(
            {
                **row.to_dict(),
                "observed_hz": observed_value,
                "observed_index": observed_index,
                "delta_hz": observed_value - expected_value if within else np.nan,
                "abs_delta_hz": nearest_delta if within else np.nan,
                "matched": bool(within),
                "match_tolerance_hz": float(tolerance_hz),
            }
        )

    # Include unmatched observed lines for completeness in spur surveys.
    for idx, observed_value in enumerate(observed):
        if idx in used_observed:
            continue
        matches.append(
            {
                "line_id": f"OBS{idx:03d}",
                "family": "unmatched_observed",
                "m_lo": np.nan,
                "n_rf": np.nan,
                "sign": "",
                "formula": "",
                "expected_hz": np.nan,
                "observed_hz": float(observed_value),
                "observed_index": float(idx),
                "delta_hz": np.nan,
                "abs_delta_hz": np.nan,
                "matched": False,
                "match_tolerance_hz": float(tolerance_hz),
            }
        )

    result = pd.DataFrame(matches)
    return result


def line_spur_catalog(
    *,
    config: str,
    f_lo_hz: float,
    f_rf_hz: float,
    observed_hz: np.ndarray | Sequence[float],
    observed_level_db: np.ndarray | Sequence[float] | None = None,
    tolerance_hz: float,
    orders: int = 3,
) -> pd.DataFrame:
    """Build an expected/observed line and spur catalog for one mixer configuration.

    Combines :func:`expected_dsb_lines` and :func:`match_observed_lines`, then
    annotates each row with an interpretation label (``"expected_dsb_product"``,
    ``"lo_leakage"``, ``"rf_feedthrough"``, ``"mixing_harmonic"``,
    ``"missing_expected_line"``, or ``"unmatched_observed_spur"``).

    Parameters
    ----------
    config : str
        Human-readable mixer configuration label (e.g. ``"DSB-1"``).
    f_lo_hz : float
        Local-oscillator frequency in Hz.
    f_rf_hz : float
        RF signal frequency in Hz.
    observed_hz : array_like
        Observed peak frequencies (Hz).
    observed_level_db : array_like or None
        Observed peak levels in dB.  If None, filled with NaN.
    tolerance_hz : float
        Match tolerance passed to :func:`match_observed_lines`.
    orders : int
        Maximum harmonic order passed to :func:`expected_dsb_lines`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``config``, ``f_lo_hz``, ``f_rf_hz``, ``expected_line_hz``,
        ``observed_line_hz``, ``level_db``, ``interpretation``, ``family``,
        ``formula``, ``line_id``.
    """

    observed = as_1d_array(observed_hz, "observed_hz", dtype=float)
    if observed_level_db is not None:
        levels = as_1d_array(observed_level_db, "observed_level_db", dtype=float)
        if levels.size != observed.size:
            raise ValueError("observed_level_db must have the same length as observed_hz.")
    else:
        levels = np.full(observed.size, np.nan, dtype=float)

    expected = expected_dsb_lines(f_lo_hz=f_lo_hz, f_rf_hz=f_rf_hz, orders=orders)
    matched = match_observed_lines(expected, observed, tolerance_hz=tolerance_hz)

    level_by_index = {idx: float(level) for idx, level in enumerate(levels)}
    level_values: list[float] = []
    interpretation: list[str] = []
    for _, row in matched.iterrows():
        observed_idx = row.get("observed_index", np.nan)
        if np.isfinite(observed_idx):
            level_values.append(level_by_index.get(int(observed_idx), np.nan))
        else:
            level_values.append(np.nan)

        family = str(row["family"])
        is_matched = bool(row["matched"])
        if family == "dsb_primary" and is_matched:
            interpretation.append("expected_dsb_product")
        elif family == "lo_harmonic" and is_matched:
            interpretation.append("lo_leakage")
        elif family == "rf_harmonic" and is_matched:
            interpretation.append("rf_feedthrough")
        elif family == "mixing_harmonic" and is_matched:
            interpretation.append("mixing_harmonic")
        elif family in {"dsb_primary", "lo_harmonic", "rf_harmonic", "mixing_harmonic"} and not is_matched:
            interpretation.append("missing_expected_line")
        else:
            interpretation.append("unmatched_observed_spur")

    catalog = pd.DataFrame(
        {
            "config": np.full(matched.shape[0], config, dtype=object),
            "f_lo_hz": np.full(matched.shape[0], float(f_lo_hz), dtype=float),
            "f_rf_hz": np.full(matched.shape[0], float(f_rf_hz), dtype=float),
            "expected_line_hz": matched["expected_hz"].to_numpy(dtype=float),
            "observed_line_hz": matched["observed_hz"].to_numpy(dtype=float),
            "level_db": np.asarray(level_values, dtype=float),
            "interpretation": np.asarray(interpretation, dtype=object),
            "family": matched["family"].to_numpy(dtype=object),
            "formula": matched["formula"].to_numpy(dtype=object),
            "line_id": matched["line_id"].to_numpy(dtype=object),
        }
    )
    return catalog


# ---------------------------------------------------------------------------
# Noise / radiometer
# ---------------------------------------------------------------------------


def radiometer_fit(
    n_avg: np.ndarray | Sequence[float],
    sigma: np.ndarray | Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Fit the radiometer equation via log-log linear regression.

    Model: ``log10(sigma) = intercept + slope * log10(n_avg)``.

    For an ideal radiometer the slope is -0.5, meaning noise decreases as
    ``1 / sqrt(N)``.  The fit result includes confidence intervals on both
    slope and intercept, goodness-of-fit metrics, and the deviation from the
    expected slope.

    Parameters
    ----------
    n_avg : array_like
        Number of averages (positive integers or floats).
    sigma : array_like
        Measured standard deviation at each averaging count (positive).
    confidence_level : float
        Confidence level for slope/intercept intervals (0 < CL < 1).

    Returns
    -------
    dict
        Keys: ``n_points``, ``confidence_level``, ``slope``, ``intercept``,
        ``slope_ci_low``, ``slope_ci_high``, ``intercept_ci_low``,
        ``intercept_ci_high``, ``expected_slope`` (-0.5),
        ``slope_minus_expected``, ``r_value``, ``r_squared``, ``p_value``,
        ``slope_stderr``, ``intercept_stderr``, ``residual_rms_log10``.

    Raises
    ------
    ValueError
        If arrays differ in length, contain non-positive values, or have
        fewer than 2 points.
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


# ---------------------------------------------------------------------------
# Nyquist / aliasing
# ---------------------------------------------------------------------------


def predict_alias_frequency(
    f_true_hz: np.ndarray | Sequence[float] | float,
    sample_rate_hz: np.ndarray | Sequence[float] | float,
) -> np.ndarray:
    """Project true frequencies into the principal Nyquist zone ``[-fs/2, fs/2)``.

    Uses ``mod(f + fs/2, fs) - fs/2`` with NumPy broadcasting, so both
    arguments can be scalars, 1-D arrays, or broadcastable shapes.

    Parameters
    ----------
    f_true_hz : float or array_like
        True signal frequency/frequencies in Hz.
    sample_rate_hz : float or array_like
        Sampling rate(s) in Hz.  Must be strictly positive.

    Returns
    -------
    ndarray
        Predicted alias frequency/frequencies in Hz.
    """

    f_true = np.asarray(f_true_hz, dtype=float)
    fs = np.asarray(sample_rate_hz, dtype=float)
    if np.any(fs <= 0.0):
        raise ValueError("sample_rate_hz must be strictly positive.")
    alias = np.mod(f_true + fs / 2.0, fs) - fs / 2.0
    return np.asarray(alias, dtype=float)


# ---------------------------------------------------------------------------
# Spectral leakage
# ---------------------------------------------------------------------------


def leakage_metric(
    frequency_hz: np.ndarray | Sequence[float],
    power_v2: np.ndarray | Sequence[float],
    *,
    tone_frequency_hz: float,
    main_lobe_half_width_bins: int = 1,
) -> dict[str, float]:
    """Compute a scalar spectral-leakage metric for a single-tone measurement.

    Leakage fraction is the ratio of power outside the main lobe to total
    power:

    ``leakage_fraction = (total_power - main_lobe_power) / total_power``

    The main lobe is centred on the bin nearest *tone_frequency_hz* and
    extends *main_lobe_half_width_bins* bins on each side.

    Parameters
    ----------
    frequency_hz : array_like
        Frequency axis in Hz.
    power_v2 : array_like
        Power spectrum in V^2 (non-negative).
    tone_frequency_hz : float
        Nominal tone frequency in Hz.
    main_lobe_half_width_bins : int
        Number of bins on each side of the peak to include in the main lobe.

    Returns
    -------
    dict
        Keys: ``tone_frequency_hz``, ``main_bin_index``,
        ``main_lobe_power_v2``, ``leakage_power_v2``, ``total_power_v2``,
        ``leakage_fraction``, ``leakage_db``.
    """

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power = as_1d_array(power_v2, "power_v2", dtype=float)
    require_same_length(frequency, power, "frequency_hz", "power_v2")

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


def nyquist_window_extension(
    frequency_hz: np.ndarray | Sequence[float],
    power_v2: np.ndarray | Sequence[float],
    *,
    sample_rate_hz: float | None = None,
    window_indices: Sequence[int] = (-1, 0, 1),
) -> pd.DataFrame:
    """Replicate a single Nyquist-zone spectrum across adjacent zones.

    Shifts the frequency axis by ``window_index * fs`` for each zone index,
    producing a long-form table useful for visualising aliased copies of a
    spectrum.

    Parameters
    ----------
    frequency_hz : array_like
        Frequency axis of the base spectrum (Hz).
    power_v2 : array_like
        Power values (same length as *frequency_hz*).
    sample_rate_hz : float or None
        Sampling rate in Hz.  If None, inferred from the median spacing of
        *frequency_hz*.
    window_indices : sequence of int
        Nyquist-zone indices to produce (e.g. ``(-1, 0, 1)``).

    Returns
    -------
    pandas.DataFrame
        Columns: ``window_index``, ``frequency_hz``, ``power_v2``.
        Sorted by ``(window_index, frequency_hz)``.
    """

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power = as_1d_array(power_v2, "power_v2", dtype=float)
    require_same_length(frequency, power, "frequency_hz", "power_v2")

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


# ---------------------------------------------------------------------------
# Frequency resolution
# ---------------------------------------------------------------------------


def resolution_vs_n(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a frequency-resolution metrics table as a function of sample count N.

    For each record, computes the bin width ``delta_f = fs / N`` and the ratio
    of the measured (or true) frequency separation to the bin width.

    Parameters
    ----------
    records : DataFrame or sequence of dicts
        Each record must contain ``n_samples`` and ``sample_rate_hz``.
        Optionally ``run_id``, ``true_delta_f_hz``, ``measured_delta_f_hz``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``run_id``, ``n_samples``, ``sample_rate_hz``,
        ``delta_f_bin_hz``, ``true_delta_f_hz``, ``measured_delta_f_hz``,
        ``resolution_ratio``.  Sorted by ``n_samples``.
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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_expected(
    expected_hz: pd.DataFrame | np.ndarray | Sequence[float],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Coerce *expected_hz* into a canonical ``(DataFrame, ndarray)`` pair."""
    if isinstance(expected_hz, pd.DataFrame):
        if "expected_hz" not in expected_hz.columns:
            raise ValueError("expected_hz DataFrame must include an 'expected_hz' column.")
        frame = expected_hz.copy()
        values = as_1d_array(frame["expected_hz"].to_numpy(dtype=float), "expected_hz", dtype=float)
        if "line_id" not in frame.columns:
            frame["line_id"] = [f"L{idx:03d}" for idx in range(frame.shape[0])]
        for col, default in [
            ("family", "expected"),
            ("m_lo", np.nan),
            ("n_rf", np.nan),
            ("sign", ""),
            ("formula", ""),
        ]:
            if col not in frame.columns:
                frame[col] = default
        frame = frame[["line_id", "family", "m_lo", "n_rf", "sign", "formula", "expected_hz"]]
        return frame, values

    values = as_1d_array(expected_hz, "expected_hz", dtype=float)
    frame = pd.DataFrame(
        {
            "line_id": [f"L{idx:03d}" for idx in range(values.size)],
            "family": ["expected"] * values.size,
            "m_lo": np.full(values.size, np.nan),
            "n_rf": np.full(values.size, np.nan),
            "sign": [""] * values.size,
            "formula": [""] * values.size,
            "expected_hz": values,
        }
    )
    return frame, values


def _to_frame(records: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Coerce *records* to a DataFrame, raising if empty."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(list(records))
    if frame.empty:
        raise ValueError("records cannot be empty.")
    return frame


__all__ = [
    "bandpass_curve",
    "expected_dsb_lines",
    "leakage_metric",
    "line_spur_catalog",
    "match_observed_lines",
    "nyquist_window_extension",
    "predict_alias_frequency",
    "radiometer_fit",
    "resolution_vs_n",
]
