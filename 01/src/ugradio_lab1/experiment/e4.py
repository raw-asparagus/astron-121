"""E4 leakage/resolution/windows figures for §5.4 (cells 39, 42)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.figure_builders import (
    ACFSpectrumConsistencyFigureBuilder,
    LeakageAndResolutionFigureBuilder,
    LeakageComparisonFigureBuilder,
    MultiWindowSpectrumFigureBuilder,
    ResolutionFigureBuilder,
)


# -- Simulation wrappers (cell 39) --


def plot_e4_leakage_sim(
    frequency_hz: np.ndarray,
    bin_centered_power_v2: np.ndarray,
    off_bin_power_v2: np.ndarray,
    *,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps LeakageComparisonFigureBuilder for sim data."""
    return LeakageComparisonFigureBuilder().build(
        frequency_hz, bin_centered_power_v2, off_bin_power_v2, db=db,
    )


def plot_e4_resolution_sim(
    n_samples: np.ndarray,
    measured_delta_f_hz: np.ndarray,
    *,
    sample_rate_hz: float | np.ndarray | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps ResolutionFigureBuilder for sim data."""
    return ResolutionFigureBuilder().build(
        n_samples, measured_delta_f_hz, sample_rate_hz=sample_rate_hz,
    )


def plot_e4_windows_sim(
    window_spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    ncols: int = 2,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps MultiWindowSpectrumFigureBuilder for sim data."""
    return MultiWindowSpectrumFigureBuilder().build(
        window_spectra, ncols=ncols, db=db,
    )


def plot_e4_leakage_and_resolution_sim(
    leakage_frequency_hz: np.ndarray,
    bin_centered_power_v2: np.ndarray,
    off_bin_power_v2: np.ndarray,
    n_samples: np.ndarray,
    measured_delta_f_hz: np.ndarray,
    *,
    db: bool = True,
    sample_rate_hz: float | np.ndarray | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Combined leakage and resolution figure for sim data."""
    return LeakageAndResolutionFigureBuilder().build(
        leakage_frequency_hz,
        bin_centered_power_v2,
        off_bin_power_v2,
        n_samples,
        measured_delta_f_hz,
        db=db,
        sample_rate_hz=sample_rate_hz,
    )


def plot_e4_acf_sim(
    lag_s_direct: np.ndarray,
    acf_direct: np.ndarray,
    lag_s_ifft: np.ndarray,
    acf_ifft: np.ndarray,
) -> tuple[Figure, dict[str, Axes]]:
    """Wiener-Khinchin theorem: overlay direct ACF and IFFT(PSD) to show they match."""
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(10.0, 6.0))
    ax = figure.add_subplot(1, 1, 1)

    # Plot both methods overlaid
    ax.plot(lag_s_direct * 1e6, np.real(acf_direct),
            label="Direct ACF (time domain)", linewidth=2, alpha=0.8)
    ax.plot(lag_s_ifft * 1e6, acf_ifft,
            label="IFFT of PSD (Wiener-Khinchin)", linewidth=2,
            linestyle='--', alpha=0.8)

    ax.set_xlabel("Lag (μs)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Wiener–Khinchin Theorem: ACF = IFFT(PSD)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Limit x-axis to show meaningful structure
    max_lag_us = min(50, lag_s_direct[-1] * 1e6)
    ax.set_xlim(0, max_lag_us)

    return figure, {"main": ax}


# -- Physical wrappers (cell 42) --


def plot_e4_leakage_physical(
    frequency_hz: np.ndarray,
    bin_centered_power_v2: np.ndarray,
    off_bin_power_v2: np.ndarray,
    *,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E4 leakage (replaces PNG loading)."""
    return LeakageComparisonFigureBuilder().build(
        frequency_hz, bin_centered_power_v2, off_bin_power_v2, db=db,
    )


def plot_e4_resolution_physical(
    n_samples: np.ndarray,
    measured_delta_f_hz: np.ndarray,
    *,
    sample_rate_hz: float | np.ndarray | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E4 resolution (replaces PNG loading)."""
    return ResolutionFigureBuilder().build(
        n_samples, measured_delta_f_hz, sample_rate_hz=sample_rate_hz,
    )


def plot_e4_windows_physical(
    window_spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    ncols: int = 2,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E4 multi-window (replaces PNG loading)."""
    return MultiWindowSpectrumFigureBuilder().build(
        window_spectra, ncols=ncols, db=db,
    )


def plot_e4_leakage_and_resolution_physical(
    leakage_frequency_hz: np.ndarray,
    bin_centered_power_v2: np.ndarray,
    off_bin_power_v2: np.ndarray,
    n_samples: np.ndarray,
    measured_delta_f_hz: np.ndarray,
    *,
    db: bool = True,
    sample_rate_hz: float | np.ndarray | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Combined leakage and resolution figure for physical data."""
    return LeakageAndResolutionFigureBuilder().build(
        leakage_frequency_hz,
        bin_centered_power_v2,
        off_bin_power_v2,
        n_samples,
        measured_delta_f_hz,
        db=db,
        sample_rate_hz=sample_rate_hz,
    )


def plot_e4_acf_physical(
    lag_s_direct: np.ndarray,
    acf_direct: np.ndarray,
    lag_s_ifft: np.ndarray,
    acf_ifft: np.ndarray,
) -> tuple[Figure, dict[str, Axes]]:
    """Wiener-Khinchin theorem: overlay direct ACF and IFFT(PSD) to show they match."""
    from matplotlib import pyplot as plt

    figure = plt.figure(figsize=(10.0, 6.0))
    ax = figure.add_subplot(1, 1, 1)

    # Plot both methods overlaid
    ax.plot(lag_s_direct * 1e6, np.real(acf_direct),
            label="Direct ACF (time domain)", linewidth=2, alpha=0.8)
    ax.plot(lag_s_ifft * 1e6, acf_ifft,
            label="IFFT of PSD (Wiener-Khinchin)", linewidth=2,
            linestyle='--', alpha=0.8)

    ax.set_xlabel("Lag (μs)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Wiener–Khinchin Theorem: ACF = IFFT(PSD)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Limit x-axis to show meaningful structure
    max_lag_us = min(50, lag_s_direct[-1] * 1e6)
    ax.set_xlim(0, max_lag_us)

    return figure, {"main": ax}


__all__ = [
    "plot_e4_leakage_sim",
    "plot_e4_resolution_sim",
    "plot_e4_windows_sim",
    "plot_e4_leakage_and_resolution_sim",
    "plot_e4_acf_sim",
    "plot_e4_leakage_physical",
    "plot_e4_resolution_physical",
    "plot_e4_windows_physical",
    "plot_e4_leakage_and_resolution_physical",
    "plot_e4_acf_physical",
]
