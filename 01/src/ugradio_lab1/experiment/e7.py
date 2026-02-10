"""E7 SSB mixer figures for §5.7 (cells 57, 59)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.figure_builders import (
    R820TComparisonFigureBuilder,
    RevertedDSBComparisonFigureBuilder,
    SSBIQBehaviorFigureBuilder,
)


def plot_e7_ssb_iq(
    i_usb: np.ndarray,
    q_usb: np.ndarray,
    i_lsb: np.ndarray,
    q_lsb: np.ndarray,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced SSB IQ plot with rotation direction indicators and phase markers."""
    figure, axes = SSBIQBehaviorFigureBuilder().build(i_usb, q_usb, i_lsb, q_lsb)

    # Add rotation direction arrows
    ax_usb = axes["upper"]
    ax_lsb = axes["lower"]

    # USB rotates counterclockwise (positive frequency)
    ax_usb.annotate('', xy=(0.5, 0.9), xytext=(0.7, 0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6),
                    annotation_clip=False)
    ax_usb.text(0.85, 0.85, 'CCW\n(+f)', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2),
                fontsize=9)

    # LSB rotates clockwise (negative frequency)
    ax_lsb.annotate('', xy=(0.9, 0.5), xytext=(0.7, 0.7),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.6),
                    annotation_clip=False)
    ax_lsb.text(0.85, 0.85, 'CW\n(-f)', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.2),
                fontsize=9)

    # Add phase angle markers (0°, 90°, 180°, 270°)
    for ax in [ax_usb, ax_lsb]:
        lim = max(abs(ax.get_xlim()[1]), abs(ax.get_ylim()[1]))
        r = lim * 0.95
        angles = [0, 90, 180, 270]
        labels = ['0°', '90°', '180°', '270°']
        for angle, label in zip(angles, labels):
            rad = np.deg2rad(angle)
            ax.plot([0, r * np.cos(rad)], [0, r * np.sin(rad)],
                   'k--', alpha=0.2, linewidth=0.8)
            ax.text(r * 1.15 * np.cos(rad), r * 1.15 * np.sin(rad), label,
                   ha='center', va='center', fontsize=8, alpha=0.6)

    return figure, axes


def plot_e7_reverted_dsb(
    frequency_hz: np.ndarray,
    ssb_power_v2: np.ndarray,
    reverted_dsb_power_v2: np.ndarray,
    *,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced reverted-DSB comparison with image suppression measurement."""
    figure, axes = RevertedDSBComparisonFigureBuilder().build(
        frequency_hz, ssb_power_v2, reverted_dsb_power_v2, db=db,
    )

    ax = axes["main"]

    # Calculate and display image suppression
    if db:
        ssb_db = 10 * np.log10(np.maximum(ssb_power_v2, 1e-20))
        dsb_db = 10 * np.log10(np.maximum(reverted_dsb_power_v2, 1e-20))

        # Find peak in SSB (desired sideband)
        ssb_peak_db = np.max(ssb_db)
        # Find peak in DSB (image sideband reappears)
        dsb_peak_db = np.max(dsb_db)

        # Estimate suppression loss
        suppression_loss_db = dsb_peak_db - ssb_peak_db

        ax.text(0.02, 0.98,
                f'Image Suppression Loss: {abs(suppression_loss_db):.1f} dB\n' +
                f'SSB peak: {ssb_peak_db:.1f} dB\n' +
                f'Reverted DSB peak: {dsb_peak_db:.1f} dB',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    return figure, axes


def plot_e7_r820t_comparison(
    frequency_hz: np.ndarray,
    external_power_v2: np.ndarray,
    r820t_power_v2: np.ndarray,
    *,
    db: bool = True,
) -> tuple[Figure, dict[str, Axes]]:
    """Enhanced R820T comparison with SFDR and noise floor analysis."""
    figure, axes = R820TComparisonFigureBuilder().build(
        frequency_hz, external_power_v2, r820t_power_v2, db=db,
    )

    ax = axes["main"]

    if db:
        ext_db = 10 * np.log10(np.maximum(external_power_v2, 1e-20))
        r820t_db = 10 * np.log10(np.maximum(r820t_power_v2, 1e-20))

        # Calculate SFDR (Spurious-Free Dynamic Range)
        ext_peak = np.max(ext_db)
        ext_noise = np.percentile(ext_db, 10)
        ext_sfdr = ext_peak - ext_noise

        r820t_peak = np.max(r820t_db)
        r820t_noise = np.percentile(r820t_db, 10)
        r820t_sfdr = r820t_peak - r820t_noise

        # Calculate relative noise floor advantage
        noise_advantage_db = ext_noise - r820t_noise

        ax.text(0.02, 0.02,
                f'SFDR Comparison:\n' +
                f'External: {ext_sfdr:.1f} dB\n' +
                f'R820T: {r820t_sfdr:.1f} dB\n' +
                f'\n' +
                f'R820T Noise Advantage: {noise_advantage_db:.1f} dB',
                transform=ax.transAxes, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.4),
                fontsize=9)

        # Add horizontal lines for noise floors
        xlim = ax.get_xlim()
        ax.hlines(ext_noise, xlim[0], xlim[1], colors='orange', linestyles=':', alpha=0.5, linewidth=1.5, label='External noise floor')
        ax.hlines(r820t_noise, xlim[0], xlim[1], colors='green', linestyles=':', alpha=0.5, linewidth=1.5, label='R820T noise floor')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

    return figure, axes


__all__ = [
    "plot_e7_ssb_iq",
    "plot_e7_reverted_dsb",
    "plot_e7_r820t_comparison",
]
