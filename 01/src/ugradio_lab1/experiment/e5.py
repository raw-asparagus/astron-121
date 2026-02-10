"""E5 noise/radiometer/ACF figures for §5.5 (cells 45, 48)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.figure_builders import (
    ACFSpectrumConsistencyFigureBuilder,
    NoiseHistogramAndRadiometerFigureBuilder,
    NoiseHistogramFigureBuilder,
    RadiometerFigureBuilder,
)


# -- Simulation wrappers (cell 45) --


def plot_e5_histogram_sim(
    samples: np.ndarray,
    *,
    bins: int = 70,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps NoiseHistogramFigureBuilder for sim data."""
    return NoiseHistogramFigureBuilder().build(samples, bins=bins)


def plot_e5_radiometer_sim(
    n_avg: np.ndarray,
    sigma: np.ndarray,
    *,
    fit_result: dict[str, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps RadiometerFigureBuilder for sim data."""
    return RadiometerFigureBuilder().build(n_avg, sigma, fit_result=fit_result)


def plot_e5_acf_sim(
    lag_s: np.ndarray,
    autocorrelation_values: np.ndarray,
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
) -> tuple[Figure, dict[str, Axes]]:
    """Wraps ACFSpectrumConsistencyFigureBuilder for sim data."""
    return ACFSpectrumConsistencyFigureBuilder().build(
        lag_s, autocorrelation_values, frequency_hz, power_v2,
    )


def plot_e5_histogram_and_radiometer_sim(
    samples: np.ndarray,
    n_avg: np.ndarray,
    sigma: np.ndarray,
    *,
    bins: int = 70,
    fit_result: dict[str, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Combined histogram and radiometer figure for sim data."""
    return NoiseHistogramAndRadiometerFigureBuilder().build(
        samples,
        n_avg,
        sigma,
        bins=bins,
        fit_result=fit_result,
    )


# -- Physical wrappers (cell 48) --


def plot_e5_histogram_physical(
    samples: np.ndarray,
    *,
    bins: int = 70,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E5 histogram (replaces PNG loading)."""
    return NoiseHistogramFigureBuilder().build(samples, bins=bins)


def plot_e5_radiometer_physical(
    n_avg: np.ndarray,
    sigma: np.ndarray,
    *,
    fit_result: dict[str, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E5 radiometer (replaces PNG loading)."""
    return RadiometerFigureBuilder().build(n_avg, sigma, fit_result=fit_result)


def plot_e5_acf_physical(
    lag_s: np.ndarray,
    autocorrelation_values: np.ndarray,
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
) -> tuple[Figure, dict[str, Axes]]:
    """Inline figure for physical E5 ACF (replaces PNG loading)."""
    return ACFSpectrumConsistencyFigureBuilder().build(
        lag_s, autocorrelation_values, frequency_hz, power_v2,
    )


def plot_e5_histogram_and_radiometer_physical(
    samples: np.ndarray,
    n_avg: np.ndarray,
    sigma: np.ndarray,
    *,
    bins: int = 70,
    fit_result: dict[str, float] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Combined histogram and radiometer figure for physical data."""
    return NoiseHistogramAndRadiometerFigureBuilder().build(
        samples,
        n_avg,
        sigma,
        bins=bins,
        fit_result=fit_result,
    )


__all__ = [
    "plot_e5_histogram_sim",
    "plot_e5_radiometer_sim",
    "plot_e5_acf_sim",
    "plot_e5_histogram_and_radiometer_sim",
    "plot_e5_histogram_physical",
    "plot_e5_radiometer_physical",
    "plot_e5_acf_physical",
    "plot_e5_histogram_and_radiometer_physical",
]
