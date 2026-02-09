"""Axes-level plotting functions that receive a Matplotlib Axes object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.text import Text

from ugradio_lab1.utils.validation import as_1d_array

VoltageComponent = Literal["real", "imag", "magnitude", "phase"]
CorrelationComponent = Literal["real", "imag", "magnitude"]
SampleSlice = slice | tuple[int, int] | None


def plot_time_series(
    ax: Axes,
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    *,
    sample_slice: SampleSlice = slice(0, 300),
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Voltage (V)",
    grid: bool = True,
) -> Line2D:
    """Plot voltage vs time on a provided axis."""

    time = as_1d_array(time_s, "time_s", dtype=float)
    voltage = as_1d_array(voltage_v, "voltage_v")
    _require_same_length(time, voltage, "time_s", "voltage_v")
    time, voltage = _apply_sample_slice_pair(time, voltage, sample_slice)

    line, = ax.plot(time, voltage, label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return line


def plot_voltage_spectrum(
    ax: Axes,
    frequency_hz: np.ndarray,
    spectrum_v: np.ndarray,
    *,
    component: VoltageComponent = "magnitude",
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str | None = None,
    grid: bool = True,
) -> Line2D:
    """Plot one component of a complex voltage spectrum."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    spectrum = as_1d_array(spectrum_v, "spectrum_v")
    _require_same_length(frequency, spectrum, "frequency_hz", "spectrum_v")

    if component == "real":
        y = np.real(spectrum)
        resolved_ylabel = "Re[Voltage Spectrum] (V)"
    elif component == "imag":
        y = np.imag(spectrum)
        resolved_ylabel = "Im[Voltage Spectrum] (V)"
    elif component == "magnitude":
        y = np.abs(spectrum)
        resolved_ylabel = "|Voltage Spectrum| (V)"
    elif component == "phase":
        y = np.angle(spectrum)
        resolved_ylabel = "Phase (rad)"
    else:
        raise ValueError(f"Unsupported component: {component}")

    line, = ax.plot(frequency, y, label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else resolved_ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return line


def plot_power_spectrum(
    ax: Axes,
    frequency_hz: np.ndarray,
    power: np.ndarray,
    *,
    db: bool = False,
    floor_db: float = -240.0,
    log_y: bool = False,
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str | None = None,
    grid: bool = True,
) -> Line2D:
    """Plot a power spectrum, optionally in dB."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power_values = as_1d_array(power, "power", dtype=float)
    _require_same_length(frequency, power_values, "frequency_hz", "power")

    if np.any(power_values < 0.0):
        raise ValueError("power values must be non-negative.")

    if db:
        floor_linear = 10.0 ** (floor_db / 10.0)
        y = 10.0 * np.log10(np.maximum(power_values, floor_linear))
        resolved_ylabel = "Power (dB re 1 V^2)"
    else:
        y = power_values
        resolved_ylabel = "Power (V^2)"

    line, = ax.plot(frequency, y, label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else resolved_ylabel)
    if log_y and not db:
        if np.any(power_values <= 0.0):
            raise ValueError("log_y requires strictly positive power values.")
        ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, which="both")
    return line


def plot_autocorrelation(
    ax: Axes,
    lag_s: np.ndarray,
    autocorrelation_values: np.ndarray,
    *,
    component: CorrelationComponent = "real",
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 1.0,
    title: str | None = None,
    xlabel: str = "Lag (s)",
    ylabel: str | None = None,
    grid: bool = True,
) -> Line2D:
    """Plot autocorrelation against lag."""

    lag = as_1d_array(lag_s, "lag_s", dtype=float)
    correlation = as_1d_array(autocorrelation_values, "autocorrelation_values")
    _require_same_length(lag, correlation, "lag_s", "autocorrelation_values")

    if component == "real":
        y = np.real(correlation)
        resolved_ylabel = "Autocorrelation (real)"
    elif component == "imag":
        y = np.imag(correlation)
        resolved_ylabel = "Autocorrelation (imag)"
    elif component == "magnitude":
        y = np.abs(correlation)
        resolved_ylabel = "|Autocorrelation|"
    else:
        raise ValueError(f"Unsupported component: {component}")

    line, = ax.plot(lag, y, label=label, color=color, linewidth=linewidth, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else resolved_ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return line


def plot_histogram_with_gaussian(
    ax: Axes,
    samples: np.ndarray,
    *,
    bins: int = 60,
    density: bool = True,
    hist_label: str | None = None,
    gaussian_label: str = "Gaussian fit",
    hist_color: str | None = None,
    gaussian_color: str | None = "C3",
    alpha: float = 0.5,
    linewidth: float = 2.0,
    annotate_stats: bool = True,
    title: str | None = None,
    xlabel: str = "Sample Value",
    ylabel: str | None = None,
    grid: bool = True,
) -> dict[str, BarContainer | Line2D | Text | None]:
    """Plot a histogram and overlay a Gaussian using sample mean/std."""

    values = as_1d_array(samples, "samples", dtype=float)
    if bins <= 0:
        raise ValueError("bins must be a positive integer.")

    _, edges, bar_container = ax.hist(
        values,
        bins=bins,
        density=density,
        color=hist_color,
        alpha=alpha,
        label=hist_label,
    )

    mean = float(np.mean(values))
    ddof = 1 if values.size > 1 else 0
    std = float(np.std(values, ddof=ddof))
    x_grid = np.linspace(float(edges[0]), float(edges[-1]), 512)

    if std > 0.0:
        gaussian = np.exp(-0.5 * ((x_grid - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
    else:
        gaussian = np.zeros_like(x_grid)
        gaussian[np.argmin(np.abs(x_grid - mean))] = 1.0

    if not density:
        bin_width = float(np.mean(np.diff(edges)))
        gaussian = gaussian * values.size * bin_width

    gaussian_line, = ax.plot(
        x_grid,
        gaussian,
        color=gaussian_color,
        linewidth=linewidth,
        label=gaussian_label,
    )

    stats_text: Text | None = None
    if annotate_stats:
        stats_text = ax.text(
            0.02,
            0.98,
            f"mu={mean:.4g}\nsigma={std:.4g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

    ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel("Density" if density else "Count")
    else:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)

    return {
        "hist": bar_container,
        "gaussian": gaussian_line,
        "stats_text": stats_text,
    }


def plot_iq_scatter(
    ax: Axes,
    i_samples_v: np.ndarray,
    q_samples_v: np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    marker_size: float = 14.0,
    alpha: float = 0.8,
    equal_aspect: bool = True,
    center_lines: bool = True,
    title: str | None = None,
    xlabel: str = "In-phase I (V)",
    ylabel: str = "Quadrature Q (V)",
    grid: bool = True,
) -> PathCollection:
    """Plot I/Q samples as a scatter plot."""

    i_values = as_1d_array(i_samples_v, "i_samples_v", dtype=float)
    q_values = as_1d_array(q_samples_v, "q_samples_v", dtype=float)
    _require_same_length(i_values, q_values, "i_samples_v", "q_samples_v")

    scatter = ax.scatter(i_values, q_values, s=marker_size, alpha=alpha, label=label, color=color)
    if center_lines:
        ax.axhline(0.0, color="0.65", linewidth=1.0, linestyle="--")
        ax.axvline(0.0, color="0.65", linewidth=1.0, linestyle="--")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return scatter


def plot_alias_map(
    ax: Axes,
    true_frequency_hz: np.ndarray,
    measured_alias_hz: np.ndarray,
    *,
    predicted_alias_hz: np.ndarray | None = None,
    predicted_as_scatter: bool = False,
    residual_ax: Axes | None = None,
    residual_hz: np.ndarray | None = None,
    include_identity: bool = False,
    label_measured: str = "Measured alias",
    label_predicted: str = "Predicted alias",
    title: str | None = None,
    xlabel: str = "True Frequency (Hz)",
    ylabel: str = "Alias Frequency (Hz)",
    grid: bool = True,
) -> dict[str, object]:
    """Plot alias-map data with optional prediction and residual inset."""

    true_frequency = as_1d_array(true_frequency_hz, "true_frequency_hz", dtype=float)
    measured_alias = as_1d_array(measured_alias_hz, "measured_alias_hz", dtype=float)
    _require_same_length(true_frequency, measured_alias, "true_frequency_hz", "measured_alias_hz")

    measured_artist = ax.scatter(true_frequency, measured_alias, s=24.0, alpha=0.9, label=label_measured)
    predicted_artist: Line2D | PathCollection | None = None
    if predicted_alias_hz is not None:
        predicted_alias = as_1d_array(predicted_alias_hz, "predicted_alias_hz", dtype=float)
        _require_same_length(true_frequency, predicted_alias, "true_frequency_hz", "predicted_alias_hz")
        if predicted_as_scatter:
            predicted_artist = ax.scatter(
                true_frequency,
                predicted_alias,
                s=18.0,
                alpha=0.8,
                marker="x",
                label=label_predicted,
            )
        else:
            predicted_artist, = ax.plot(
                true_frequency,
                predicted_alias,
                linewidth=1.5,
                linestyle="--",
                label=label_predicted,
            )
    else:
        predicted_alias = None

    identity_artist: Line2D | None = None
    if include_identity:
        x_min = float(np.min(true_frequency))
        x_max = float(np.max(true_frequency))
        identity_artist, = ax.plot([x_min, x_max], [x_min, x_max], linewidth=1.0, linestyle=":", color="0.5")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)

    residual_artist: Line2D | PathCollection | None = None
    if residual_ax is not None:
        if residual_hz is None:
            if predicted_alias is None:
                raise ValueError(
                    "Provide residual_hz or predicted_alias_hz when residual_ax is supplied."
                )
            residual = measured_alias - predicted_alias
        else:
            residual = as_1d_array(residual_hz, "residual_hz", dtype=float)
            _require_same_length(true_frequency, residual, "true_frequency_hz", "residual_hz")
        residual_artist = residual_ax.scatter(true_frequency, residual, s=18.0, alpha=0.9)
        residual_ax.axhline(0.0, linestyle="--", linewidth=1.0, color="0.5")
        residual_ax.set_xlabel(xlabel)
        residual_ax.set_ylabel("Residual (Hz)")
        if grid:
            residual_ax.grid(True, alpha=0.25)

    return {
        "measured": measured_artist,
        "predicted": predicted_artist,
        "identity": identity_artist,
        "residual": residual_artist,
    }


def plot_bandpass_curves(
    ax: Axes,
    frequency_hz: np.ndarray,
    gain_db_by_mode: Mapping[str, np.ndarray],
    *,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str = "Gain (dB)",
    linewidth: float = 1.6,
    grid: bool = True,
) -> list[Line2D]:
    """Plot multiple bandpass curves keyed by mode name."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    if not gain_db_by_mode:
        raise ValueError("gain_db_by_mode cannot be empty.")

    lines: list[Line2D] = []
    for mode, gain_db in gain_db_by_mode.items():
        gain_values = as_1d_array(gain_db, f"gain_db_by_mode[{mode!r}]", dtype=float)
        _require_same_length(frequency, gain_values, "frequency_hz", f"gain_db_by_mode[{mode!r}]")
        line, = ax.plot(frequency, gain_values, linewidth=linewidth, label=mode)
        lines.append(line)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return lines


def plot_resolution_vs_n(
    ax: Axes,
    n_samples: np.ndarray,
    measured_delta_f_hz: np.ndarray,
    *,
    sample_rate_hz: float | np.ndarray | None = None,
    log_x: bool = True,
    log_y: bool = True,
    title: str | None = None,
    xlabel: str = "N Samples",
    ylabel: str = "Minimum Resolvable Delta f (Hz)",
    grid: bool = True,
) -> dict[str, Line2D | None]:
    """Plot frequency-resolution measurements versus sample count."""

    n_values = as_1d_array(n_samples, "n_samples", dtype=float)
    measured = as_1d_array(measured_delta_f_hz, "measured_delta_f_hz", dtype=float)
    _require_same_length(n_values, measured, "n_samples", "measured_delta_f_hz")
    if np.any(n_values <= 0.0):
        raise ValueError("n_samples must be positive.")
    if np.any(measured <= 0.0):
        raise ValueError("measured_delta_f_hz must be positive.")

    measured_line, = ax.plot(n_values, measured, marker="o", linewidth=1.5, label="Measured")

    theory_line: Line2D | None = None
    if sample_rate_hz is not None:
        fs = np.asarray(sample_rate_hz, dtype=float)
        if fs.ndim == 0:
            if fs <= 0.0:
                raise ValueError("sample_rate_hz must be positive.")
            theory = float(fs) / n_values
        else:
            fs_values = as_1d_array(fs, "sample_rate_hz", dtype=float)
            _require_same_length(n_values, fs_values, "n_samples", "sample_rate_hz")
            if np.any(fs_values <= 0.0):
                raise ValueError("sample_rate_hz must be positive.")
            theory = fs_values / n_values
        theory_line, = ax.plot(n_values, theory, linestyle="--", linewidth=1.4, label="fs/N")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, which="both")
    return {"measured": measured_line, "theory": theory_line}


def plot_radiometer_fit(
    ax: Axes,
    n_avg: np.ndarray,
    sigma: np.ndarray,
    *,
    fit_result: Mapping[str, float] | None = None,
    show_ci: bool = True,
    show_expected: bool = True,
    title: str | None = None,
    xlabel: str = "Averaging Count",
    ylabel: str = "Sigma",
    grid: bool = True,
) -> dict[str, object]:
    """Plot radiometer scaling data with fitted and optional CI curves."""

    n_values = as_1d_array(n_avg, "n_avg", dtype=float)
    sigma_values = as_1d_array(sigma, "sigma", dtype=float)
    _require_same_length(n_values, sigma_values, "n_avg", "sigma")
    if np.any(n_values <= 0.0):
        raise ValueError("n_avg must be positive.")
    if np.any(sigma_values <= 0.0):
        raise ValueError("sigma must be positive.")

    x_log = np.log10(n_values)
    y_log = np.log10(sigma_values)

    if fit_result is None:
        slope, intercept = np.polyfit(x_log, y_log, deg=1)
        slope_ci_low = np.nan
        slope_ci_high = np.nan
        expected_slope = -0.5
    else:
        slope = float(fit_result["slope"])
        intercept = float(fit_result["intercept"])
        slope_ci_low = float(fit_result.get("slope_ci_low", np.nan))
        slope_ci_high = float(fit_result.get("slope_ci_high", np.nan))
        expected_slope = float(fit_result.get("expected_slope", -0.5))

    order = np.argsort(n_values)
    n_sorted = n_values[order]
    sigma_sorted = sigma_values[order]
    fit_sorted = 10.0 ** (intercept + slope * np.log10(n_sorted))

    data_artist = ax.scatter(n_sorted, sigma_sorted, s=24.0, alpha=0.9, label="Measured")
    fit_artist, = ax.plot(n_sorted, fit_sorted, linewidth=1.6, label=f"Fit slope={slope:.3f}")

    expected_artist: Line2D | None = None
    if show_expected:
        anchor = sigma_sorted[0] * (n_sorted[0] ** (-expected_slope))
        expected_curve = anchor * (n_sorted**expected_slope)
        expected_artist, = ax.plot(n_sorted, expected_curve, linestyle="--", linewidth=1.3, label="Expected")

    ci_artist = None
    if show_ci and np.isfinite(slope_ci_low) and np.isfinite(slope_ci_high):
        lower = 10.0 ** (intercept + slope_ci_low * np.log10(n_sorted))
        upper = 10.0 ** (intercept + slope_ci_high * np.log10(n_sorted))
        ci_artist = ax.fill_between(n_sorted, lower, upper, alpha=0.15, label="Slope CI")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, which="both")

    return {"data": data_artist, "fit": fit_artist, "expected": expected_artist, "ci": ci_artist}


def plot_spur_survey(
    ax: Axes,
    frequency_hz: np.ndarray,
    power: np.ndarray,
    *,
    expected_lines_hz: np.ndarray | Sequence[float] | None = None,
    db: bool = True,
    annotate_top_n: int = 0,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str | None = None,
    grid: bool = True,
) -> dict[str, object]:
    """Plot wide dynamic-range spectrum with optional expected-line markers."""

    freq = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power_values = as_1d_array(power, "power", dtype=float)
    _require_same_length(freq, power_values, "frequency_hz", "power")

    spectrum_line = plot_power_spectrum(
        ax,
        freq,
        power_values,
        db=db,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
    )

    expected_artists: list[Line2D] = []
    if expected_lines_hz is not None:
        expected = as_1d_array(expected_lines_hz, "expected_lines_hz", dtype=float)
        for idx, line_hz in enumerate(expected):
            artist = ax.axvline(
                float(line_hz),
                linestyle=":",
                linewidth=1.0,
                color="C3",
                alpha=0.8,
                label="Expected lines" if idx == 0 else None,
            )
            expected_artists.append(artist)

    annotation_artists: list[Text] = []
    if annotate_top_n > 0:
        y_values = spectrum_line.get_ydata()
        top_idx = np.argsort(y_values)[-annotate_top_n:]
        for idx in top_idx:
            txt = ax.text(freq[idx], y_values[idx], f"{freq[idx]:.3g}", fontsize=8, ha="left", va="bottom")
            annotation_artists.append(txt)

    return {"spectrum": spectrum_line, "expected_lines": expected_artists, "annotations": annotation_artists}


def plot_iq_phase_trajectory(
    ax: Axes,
    i_samples_v: np.ndarray,
    q_samples_v: np.ndarray,
    *,
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 1.5,
    alpha: float = 0.9,
    mark_start_end: bool = True,
    equal_aspect: bool = True,
    title: str | None = None,
    xlabel: str = "In-phase I (V)",
    ylabel: str = "Quadrature Q (V)",
    grid: bool = True,
) -> dict[str, object]:
    """Plot IQ trajectory to visualize sideband phase behavior."""

    i_values = as_1d_array(i_samples_v, "i_samples_v", dtype=float)
    q_values = as_1d_array(q_samples_v, "q_samples_v", dtype=float)
    _require_same_length(i_values, q_values, "i_samples_v", "q_samples_v")

    trajectory, = ax.plot(i_values, q_values, linewidth=linewidth, alpha=alpha, label=label, color=color)
    start_artist = None
    end_artist = None
    if mark_start_end:
        start_artist = ax.scatter([i_values[0]], [q_values[0]], marker="o", s=36, color="C2", label="Start")
        end_artist = ax.scatter([i_values[-1]], [q_values[-1]], marker="s", s=36, color="C1", label="End")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)
    return {"trajectory": trajectory, "start": start_artist, "end": end_artist}


def plot_time_series_comparison(
    ax: Axes,
    time_s: np.ndarray,
    reference_voltage_v: np.ndarray,
    comparison_voltage_v: np.ndarray,
    *,
    sample_slice: SampleSlice = slice(0, 300),
    reference_label: str = "Reference",
    comparison_label: str = "Comparison",
    clip_limits_v: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Voltage (V)",
    grid: bool = True,
) -> dict[str, object]:
    """Overlay two time-domain traces (used for F3/F14-style comparisons)."""

    time = as_1d_array(time_s, "time_s", dtype=float)
    reference = as_1d_array(reference_voltage_v, "reference_voltage_v")
    comparison = as_1d_array(comparison_voltage_v, "comparison_voltage_v")
    _require_same_length(time, reference, "time_s", "reference_voltage_v")
    _require_same_length(time, comparison, "time_s", "comparison_voltage_v")
    time, reference, comparison = _apply_sample_slice_triplet(
        time, reference, comparison, sample_slice
    )

    reference_line, = ax.plot(time, reference, linewidth=1.5, label=reference_label)
    comparison_line, = ax.plot(time, comparison, linewidth=1.2, alpha=0.9, label=comparison_label)

    clip_artists: list[Line2D] = []
    if clip_limits_v is not None:
        clip_low, clip_high = float(clip_limits_v[0]), float(clip_limits_v[1])
        clip_artists.append(
            ax.axhline(clip_low, linestyle="--", linewidth=1.0, color="C3", label="Clip bounds")
        )
        clip_artists.append(ax.axhline(clip_high, linestyle="--", linewidth=1.0, color="C3"))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25)

    return {
        "reference": reference_line,
        "comparison": comparison_line,
        "clip_bounds": clip_artists,
    }


def plot_windowed_spectra(
    ax: Axes,
    frequency_hz: np.ndarray,
    power_by_window: Mapping[str, np.ndarray],
    *,
    db: bool = True,
    floor_db: float = -240.0,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str | None = None,
    grid: bool = True,
) -> list[Line2D]:
    """Plot multiple spectra keyed by window name on one axis."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    if not power_by_window:
        raise ValueError("power_by_window cannot be empty.")

    artists: list[Line2D] = []
    for window_name, power in power_by_window.items():
        power_values = as_1d_array(power, f"power_by_window[{window_name!r}]", dtype=float)
        _require_same_length(
            frequency, power_values, "frequency_hz", f"power_by_window[{window_name!r}]"
        )
        line = plot_power_spectrum(
            ax,
            frequency,
            power_values,
            db=db,
            floor_db=floor_db,
            label=window_name,
            grid=False,
        )
        artists.append(line)

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, which="both")
    return artists


def plot_waveform_comparison(
    ax: Axes,
    time_s: np.ndarray,
    original_voltage_v: np.ndarray,
    reconstructed_voltage_v: np.ndarray,
    *,
    sample_slice: SampleSlice = slice(0, 300),
    title: str | None = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Voltage (V)",
    grid: bool = True,
) -> dict[str, Line2D]:
    """Compare original and reconstructed waveform traces."""

    artists = plot_time_series_comparison(
        ax,
        time_s,
        original_voltage_v,
        reconstructed_voltage_v,
        sample_slice=sample_slice,
        reference_label="Original",
        comparison_label="Reconstructed",
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
    )
    return {
        "original": artists["reference"],  # type: ignore[return-value]
        "reconstructed": artists["comparison"],  # type: ignore[return-value]
    }


def plot_power_spectrum_comparison(
    ax: Axes,
    frequency_hz: np.ndarray,
    power_a_v2: np.ndarray,
    power_b_v2: np.ndarray,
    *,
    label_a: str = "A",
    label_b: str = "B",
    db: bool = True,
    floor_db: float = -240.0,
    title: str | None = None,
    xlabel: str = "Frequency (Hz)",
    ylabel: str | None = None,
    grid: bool = True,
) -> dict[str, Line2D]:
    """Overlay two power spectra on one axis."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power_a = as_1d_array(power_a_v2, "power_a_v2", dtype=float)
    power_b = as_1d_array(power_b_v2, "power_b_v2", dtype=float)
    _require_same_length(frequency, power_a, "frequency_hz", "power_a_v2")
    _require_same_length(frequency, power_b, "frequency_hz", "power_b_v2")

    line_a = plot_power_spectrum(
        ax,
        frequency,
        power_a,
        db=db,
        floor_db=floor_db,
        label=label_a,
        grid=False,
    )
    line_b = plot_power_spectrum(
        ax,
        frequency,
        power_b,
        db=db,
        floor_db=floor_db,
        label=label_b,
        grid=False,
    )

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.25, which="both")
    return {"a": line_a, "b": line_b}


def _normalize_sample_slice(sample_slice: SampleSlice, n_samples: int) -> slice:
    if sample_slice is None:
        return slice(0, n_samples)
    if isinstance(sample_slice, tuple):
        if len(sample_slice) != 2:
            raise ValueError("sample_slice tuple must be (start, stop).")
        start, stop = int(sample_slice[0]), int(sample_slice[1])
        return slice(start, stop)
    return sample_slice


def _apply_sample_slice_pair(
    time: np.ndarray,
    values: np.ndarray,
    sample_slice: SampleSlice,
) -> tuple[np.ndarray, np.ndarray]:
    selection = _normalize_sample_slice(sample_slice, time.size)
    sliced_time = time[selection]
    sliced_values = values[selection]
    if sliced_time.size == 0:
        raise ValueError("sample_slice selects zero samples.")
    return sliced_time, sliced_values


def _apply_sample_slice_triplet(
    time: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sample_slice: SampleSlice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selection = _normalize_sample_slice(sample_slice, time.size)
    sliced_time = time[selection]
    sliced_first = first[selection]
    sliced_second = second[selection]
    if sliced_time.size == 0:
        raise ValueError("sample_slice selects zero samples.")
    return sliced_time, sliced_first, sliced_second


def _require_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    if left.size != right.size:
        raise ValueError(f"{left_name} and {right_name} must have the same length.")


__all__ = [
    "plot_alias_map",
    "plot_bandpass_curves",
    "plot_autocorrelation",
    "plot_histogram_with_gaussian",
    "plot_iq_scatter",
    "plot_iq_phase_trajectory",
    "plot_power_spectrum",
    "plot_power_spectrum_comparison",
    "plot_radiometer_fit",
    "plot_resolution_vs_n",
    "plot_spur_survey",
    "plot_time_series_comparison",
    "plot_time_series",
    "plot_voltage_spectrum",
    "plot_waveform_comparison",
    "plot_windowed_spectra",
]
