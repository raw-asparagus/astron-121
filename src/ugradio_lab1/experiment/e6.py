"""E6 DSB mixer figures for §5.6 (cells 52, 54)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.figure_builders import (
    DSBOutputSpectrumFigureBuilder,
    FilteredWaveformFigureBuilder,
    SpurSurveyFigureBuilder,
)


def plot_e6_dsb_spectrum(
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
    *,
    expected_lines_hz: np.ndarray | None = None,
    annotate_top_n: int = 8,
    f_rf_hz: float | None = None,
    f_lo_hz: float | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced DSB spectrum with frequency markers and product labels."""
    figure, axes = DSBOutputSpectrumFigureBuilder().build(
        frequency_hz,
        power_v2,
        expected_lines_hz=expected_lines_hz,
        annotate_top_n=annotate_top_n,
    )

    ax = axes["main"]

    # Add RF and LO markers if provided
    if f_rf_hz is not None and f_lo_hz is not None:
        ymax = ax.get_ylim()[1]
        ymin = ax.get_ylim()[0]

        # Mark key frequencies
        f_sum = f_rf_hz + f_lo_hz
        f_diff = abs(f_rf_hz - f_lo_hz)

        # Add vertical lines for sum and difference
        ax.axvline(f_sum / 1e6, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Sum: {f_sum/1e6:.1f} MHz')
        ax.axvline(f_diff / 1e6, color='blue', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Diff: {f_diff/1e6:.1f} MHz')

        # Add text annotations
        ax.text(f_sum / 1e6, ymax - 5, 'RF+LO', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))
        ax.text(f_diff / 1e6, ymax - 5, 'RF-LO', ha='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.2))

    ax.set_ylim(-100 * 1.05, 10 * 1.05 - 10)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    return figure, axes


def plot_e6_filtered_waveform(
    time_s: np.ndarray,
    original_voltage_v: np.ndarray,
    reconstructed_voltage_v: np.ndarray,
    *,
    sample_slice: slice | tuple[int, int] | None = slice(0, 300),
    if_frequency_hz: float | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced filtered waveform with period markers and frequency annotation."""
    figure, axes = FilteredWaveformFigureBuilder().build(
        time_s,
        original_voltage_v,
        reconstructed_voltage_v,
        sample_slice=sample_slice,
    )

    ax = axes["main"]

    # Add expected IF frequency annotation if provided
    if if_frequency_hz is not None:
        expected_period_us = 1e6 / if_frequency_hz
        ax.text(0.02, 0.98, f'Expected IF: {if_frequency_hz/1e3:.1f} kHz\nPeriod: {expected_period_us:.2f} μs',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    return figure, axes


def plot_e6_spur_survey(
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
    *,
    expected_lines_hz: np.ndarray | None = None,
    annotate_top_n: int = 8,
    spur_labels: dict[float, str] | None = None,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced spur survey with harmonic order labels."""
    figure, axes = SpurSurveyFigureBuilder().build(
        frequency_hz,
        power_v2,
        expected_lines_hz=expected_lines_hz,
        annotate_top_n=annotate_top_n,
    )

    ax = axes["main"]

    # Add spur order labels if provided
    if spur_labels is not None:
        ymax = ax.get_ylim()[1]
        for freq_mhz, label in spur_labels.items():
            # Add vertical line and label
            ax.axvline(freq_mhz, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax.text(freq_mhz, ymax - 10, label, rotation=90, va='top', ha='right',
                    fontsize=8, alpha=0.7)

    # Add dynamic range annotation
    power_db = 10 * np.log10(np.maximum(power_v2, 1e-20))
    peak_db = np.max(power_db)
    noise_floor_db = np.percentile(power_db, 5)
    dynamic_range_db = peak_db - noise_floor_db

    ax.text(0.02, 0.02, f'Dynamic Range: {dynamic_range_db:.1f} dB\nNoise Floor: {noise_floor_db:.1f} dB',
            transform=ax.transAxes, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            fontsize=9)

    ax.grid(True, alpha=0.3, which='both')

    return figure, axes


__all__ = [
    "plot_e6_dsb_spectrum",
    "plot_e6_filtered_waveform",
    "plot_e6_spur_survey",
]
