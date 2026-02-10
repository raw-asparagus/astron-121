"""Theory demo figures for §3 (cells 7, 11, 14, 18)."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch


def _format_freq(freq_hz: float) -> str:
    abs_nu = abs(freq_hz)
    if abs_nu >= 1e6:
        return f"{freq_hz/1e6:.3g} MHz"
    elif abs_nu >= 1e3:
        return f"{freq_hz/1e3:.3g} kHz"
    return f"{freq_hz:.3g} Hz"


def plot_theory_beats(
    t_s: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x_sum: np.ndarray,
    *,
    f1_hz: float = 1200.0,
    f2_hz: float = 1400.0,
    figsize: tuple[float, float] = (10, 3),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 7: 1x2 layout showing two sinusoids and their beat sum."""

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300)
    axes[0].plot(t_s * 1e3, x1, label=r"$x_1(t)$")
    axes[0].plot(t_s * 1e3, x2, label=r"$x_2(t)$", alpha=0.85)
    axes[0].set_ylabel("Voltage (arb)")
    axes[0].set_title(
        rf"$x_1$ at {f1_hz:.0f} Hz and $x_2$ at {f2_hz:.0f} Hz"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_s * 1e3, x_sum, color="C3")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Voltage (arb)")
    axes[1].set_title(
        rf"Superposition $x_1+x_2$: Beat at $|f_1-f_2|={abs(f1_hz - f2_hz):.0f}$ Hz"
    )
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    return fig, {"left": axes[0], "right": axes[1]}


def plot_theory_fourier(
    t_s: np.ndarray,
    x: np.ndarray,
    f_hz: np.ndarray,
    voltage_real: np.ndarray,
    voltage_imag: np.ndarray,
    power_bin: np.ndarray,
    *,
    fs_hz: float = 3200.0,
    f1: float = 300.0,
    f2: float = 700.0,
    a1: float = 1.20,
    a2: float = 0.70,
    phase1: float = 0.2,
    phase2: float = -0.6,
    n_show: int = 30,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 11: Fourier decomposition with time-domain + spectra layout."""

    n_samples = len(t_s)
    center = n_samples // 2
    i0, i1 = center - n_show // 2, center + n_show // 2
    t_show_us = t_s[i0:i1] * 1e6
    x_show = x[i0:i1]

    t_fine = np.linspace(t_s[i0], t_s[i1 - 1], 3000)
    x_fine = (
        a1 * np.cos(2.0 * np.pi * f1 * t_fine + phase1)
        + a2 * np.cos(2.0 * np.pi * f2 * t_fine + phase2)
    )
    x1_fine = a1 * np.cos(2.0 * np.pi * f1 * t_fine + phase1)
    x2_fine = a2 * np.cos(2.0 * np.pi * f2 * t_fine + phase2)

    fig = plt.figure(figsize=figsize, dpi=300)
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

    # Left: time domain
    gs_left = outer[0, 0].subgridspec(2, 1, hspace=0.0)
    ax_tl = fig.add_subplot(gs_left[0, 0])
    ax_bl = fig.add_subplot(gs_left[1, 0], sharex=ax_tl)

    ax_tl.plot(
        t_fine * 1e6, x1_fine, "--", color="C1", alpha=0.6,
        label=f"{_format_freq(f1)} @ {a1:.2f} V",
    )
    ax_tl.plot(
        t_fine * 1e6, x2_fine, "--", color="C2", alpha=0.6,
        label=f"{_format_freq(f2)} @ {a2:.2f} V",
    )
    ax_tl.plot(t_fine * 1e6, x_fine, color="C0", alpha=1.0, label="Sum")
    ax_tl.plot(
        t_show_us, x_show, "xk", markersize=5,
        label=f"Samples @ {_format_freq(fs_hz)}",
    )
    ax_tl.set_ylabel("Voltage (V)")
    ax_tl.set_title("Injected signals (continuous) + sampled points")
    ax_tl.legend(loc="upper right", fontsize=7)
    ax_tl.grid(True, alpha=0.2)
    ax_tl.tick_params(labelbottom=False)

    ax_bl.plot(t_show_us, x_show, "xk", markersize=5)
    ax_bl.plot(t_show_us, x_show, color="C1")
    ax_bl.set_xlabel("Time (\u00b5s)")
    ax_bl.set_ylabel("Voltage (V)")
    ax_bl.grid(True, alpha=0.2)

    for ti, yi in zip(t_show_us, x_show):
        con = ConnectionPatch(
            xyA=(ti, yi), coordsA=ax_tl.transData,
            xyB=(ti, yi), coordsB=ax_bl.transData,
            linestyle=":", color="gray", linewidth=0.6, alpha=0.7,
        )
        fig.add_artist(con)

    # Right: spectra
    gs_right = outer[0, 1].subgridspec(2, 1, height_ratios=[2, 3], hspace=0.25)
    gs_right_bot = gs_right[1, 0].subgridspec(2, 1, hspace=0.0)

    ax_power = fig.add_subplot(gs_right[0, 0])
    ax_vr = fig.add_subplot(gs_right_bot[0, 0], sharex=ax_power)
    ax_vi = fig.add_subplot(gs_right_bot[1, 0], sharex=ax_power)

    f_khz = f_hz / 1e3

    ax_power.plot(f_khz, power_bin, color="C2")
    ax_power.set_title(
        f"Power spectrum"
    )
    ax_power.set_ylabel("Power (arb)")
    ax_power.grid(True, alpha=0.2)
    ax_power.tick_params(labelbottom=False)

    vmin = min(np.min(voltage_real), np.min(voltage_imag))
    vmax = max(np.max(voltage_real), np.max(voltage_imag))
    vpad = 0.05 * (vmax - vmin) if vmax != vmin else 0.01

    ax_vr.plot(f_khz, voltage_real, color="C3")
    ax_vr.set_title("Voltage spectrum")
    ax_vr.set_ylabel("Re V (arb)")
    ax_vr.set_ylim(vmin - vpad, vmax + vpad)
    ax_vr.grid(True, alpha=0.2)
    ax_vr.tick_params(labelbottom=False)

    ax_vi.plot(f_khz, voltage_imag, color="C4")
    ax_vi.set_xlabel("Frequency (kHz)")
    ax_vi.set_ylabel("Im V (arb)")
    ax_vi.set_ylim(vmin - vpad, vmax + vpad)
    ax_vi.grid(True, alpha=0.2)

    fig.suptitle(
        f"Sampling {_format_freq(f1)} + {_format_freq(f2)} at {_format_freq(fs_hz)}",
        fontsize=12, y=1.01,
    )
    return fig, {
        "time_top": ax_tl,
        "time_bottom": ax_bl,
        "power": ax_power,
        "voltage_real": ax_vr,
        "voltage_imag": ax_vi,
    }


def plot_theory_aliasing(
    f_true_sweep: np.ndarray,
    f_alias_sweep: np.ndarray,
    fs_hz: float,
    f_low_hz: float,
    f_high_hz: float,
    t_s: np.ndarray,
    x_low: np.ndarray,
    x_high: np.ndarray,
    f_hz: np.ndarray,
    X_low: np.ndarray,
    X_high: np.ndarray,
    *,
    n_show: int = 10,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 14: 2x2 aliasing demo (alias map, time, power, voltage)."""

    P_low = np.real(X_low) ** 2 + np.imag(X_low) ** 2
    P_high = np.real(X_high) ** 2 + np.imag(X_high) ** 2

    t_show = t_s[:n_show]
    x_low_show = x_low[:n_show]
    t_fine = np.linspace(t_s[0], t_s[n_show - 1], 5000)
    x_low_fine = np.cos(2.0 * np.pi * f_low_hz * t_fine)
    x_high_fine = np.cos(2.0 * np.pi * f_high_hz * t_fine)

    fig = plt.figure(figsize=figsize, dpi=300)
    outer = fig.add_gridspec(
        2, 2, width_ratios=[1, 1], height_ratios=[1, 1],
        hspace=0.35, wspace=0.3,
    )

    # Top left: time domain
    ax_time = fig.add_subplot(outer[0, 0])
    ax_time.plot(
        t_fine * 1e3, x_low_fine, color="C1", alpha=0.5,
        label=f"{_format_freq(f_low_hz)} (continuous)",
    )
    ax_time.plot(
        t_fine * 1e3, x_high_fine, color="C2", alpha=0.5,
        label=f"{_format_freq(f_high_hz)} (continuous)",
    )
    ax_time.plot(
        t_show * 1e3, x_low_show, "xk", markersize=6,
        label=f"Samples @ {_format_freq(fs_hz)}",
    )
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Voltage (V)")
    ax_time.set_title(
        f"Aliased pair: {_format_freq(f_low_hz)} and "
        f"{_format_freq(f_high_hz)} at $f_s = {_format_freq(fs_hz)}$"
    )
    ax_time.legend(loc="upper right", fontsize=7)
    ax_time.grid(True, alpha=0.2)

    # Bottom left: alias map
    ax_alias = fig.add_subplot(outer[1, 0])
    ax_alias.plot(f_true_sweep, f_alias_sweep, alpha=0.8, color="C0")
    for boundary in np.arange(0.0, 4500.0, fs_hz / 2.0):
        ax_alias.axvline(boundary, color="0.85", linewidth=0.8)
    ax_alias.axhline(fs_hz / 2.0, color="0.7", linestyle="--", linewidth=1.0)
    ax_alias.axhline(f_low_hz, color="C3", linestyle="--", linewidth=1.0, label="Aliased signal")
    ax_alias.axvline(
        f_low_hz, color="C1", ls="--", lw=1.0, alpha=0.7,
        label=f"{_format_freq(f_low_hz)}",
    )
    ax_alias.axvline(
        f_high_hz, color="C2", ls="--", lw=1.0, alpha=0.7,
        label=f"{_format_freq(f_high_hz)}",
    )
    ax_alias.set_xlabel(r"$f_{\rm true}$ (Hz)")
    ax_alias.set_ylabel(r"$|f_{\rm alias}|$ (Hz)")
    ax_alias.set_title(f"Alias map for $f_s = {_format_freq(fs_hz)}$")
    ax_alias.legend(loc="upper right", fontsize=8)
    ax_alias.grid(True, alpha=0.2)

    # Top right: power spectra
    ax_pow = fig.add_subplot(outer[0, 1])
    ax_pow.plot(
        f_hz, 10.0 * np.log10(np.maximum(P_low, 1e-18)),
        color="C1", label=f"{_format_freq(f_low_hz)} tone",
    )
    ax_pow.plot(
        f_hz, 10.0 * np.log10(np.maximum(P_high, 1e-18)),
        color="C2", ls="--", label=f"{_format_freq(f_high_hz)} tone",
    )
    ax_pow.set_xlabel("Frequency (Hz)")
    ax_pow.set_ylabel("Power (dB, arb)")
    ax_pow.set_title("Power spectra overlap \u2014 alias ambiguity")
    ax_pow.legend(loc="upper right", fontsize=8)
    ax_pow.grid(True, alpha=0.2)

    # Bottom right: voltage spectrum Re/Im stacked
    gs_volt = outer[1, 1].subgridspec(2, 1, hspace=0.0)
    ax_vr = fig.add_subplot(gs_volt[0, 0])
    ax_vi = fig.add_subplot(gs_volt[1, 0], sharex=ax_vr)

    ax_vr.plot(f_hz, np.real(X_low), color="C1", label=f"{_format_freq(f_low_hz)}")
    ax_vr.plot(
        f_hz, np.real(X_high), color="C2", ls="--",
        label=f"{_format_freq(f_high_hz)}",
    )
    ax_vr.set_ylabel("Re V (arb)")
    ax_vr.set_title("Voltage spectrum: Re and Im components")
    ax_vr.legend(loc="upper right", fontsize=7)
    ax_vr.grid(True, alpha=0.2)
    ax_vr.tick_params(labelbottom=False)

    ax_vi.plot(f_hz, np.imag(X_low), color="C1", label=f"{_format_freq(f_low_hz)}")
    ax_vi.plot(
        f_hz, np.imag(X_high), color="C2", ls="--",
        label=f"{_format_freq(f_high_hz)}",
    )
    ax_vi.set_xlabel("Frequency (Hz)")
    ax_vi.set_ylabel("Im V (arb)")
    ax_vi.legend(loc="upper right", fontsize=7)
    ax_vi.grid(True, alpha=0.2)

    fig.suptitle(
        f"Sampling at {_format_freq(fs_hz)}: {_format_freq(f_low_hz)} and "
        f"{_format_freq(f_high_hz)} are indistinguishable",
        fontsize=12, y=1.01,
    )
    return fig, {
        "alias_map": ax_alias,
        "time": ax_time,
        "power": ax_pow,
        "voltage_real": ax_vr,
        "voltage_imag": ax_vi,
    }


def plot_theory_mixer(
    t_s: np.ndarray,
    mixed: np.ndarray,
    filtered: np.ndarray,
    f_spec: np.ndarray,
    X_raw: np.ndarray,
    X_filt: np.ndarray,
    P_raw: np.ndarray,
    P_filt: np.ndarray,
    w_hz: np.ndarray,
    h: np.ndarray,
    conv_direct: np.ndarray,
    conv_freq: np.ndarray,
    *,
    f_lo_hz: float = 2500.0,
    f_rf_hz: float = 3200.0,
    fs_hz: float = 20_000.0,
    cutoff_hz: float = 1200.0,
    num_taps: int = 129,
    conv_rel_err: float = 0.0,
    figsize: tuple[float, float] = (14, 8),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 18: 2x2 mixer demo (time, spectra, FIR, convolution check)."""

    fig = plt.figure(figsize=figsize, dpi=300)
    outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Top left: time-domain zoom
    beat_period_ms = 1e3 / abs(f_rf_hz - f_lo_hz)
    t_show_ms = 3.5 * beat_period_ms
    n_show = int(t_show_ms * 1e-3 * fs_hz)
    ax_time = fig.add_subplot(outer[0, 0])
    ax_time.plot(t_s[:n_show] * 1e3, mixed[:n_show], alpha=0.7, label="Mixer output")
    ax_time.plot(
        t_s[:n_show] * 1e3, filtered[:n_show], alpha=0.9, label="After FIR LP",
    )
    ax_time.set_xlabel("Time (ms)")
    ax_time.set_ylabel("Voltage (arb)")
    ax_time.set_title(
        f"DSB mixer: $f_{{LO}}$={f_lo_hz:.0f} Hz, "
        f"$f_{{RF}}$={f_rf_hz:.0f} Hz, $f_s$={fs_hz/1e3:.0f} kHz"
    )
    ax_time.legend(loc="upper right", fontsize=8)
    ax_time.grid(True, alpha=0.2)

    # Top right: power spectrum + Re/Im voltage stacked
    gs_spec = outer[0, 1].subgridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.0)
    ax_pow = fig.add_subplot(gs_spec[0, 0])
    ax_vr = fig.add_subplot(gs_spec[1, 0], sharex=ax_pow)
    ax_vi = fig.add_subplot(gs_spec[2, 0], sharex=ax_pow)

    ax_pow.plot(
        f_spec / 1e3, 10.0 * np.log10(np.maximum(P_raw, 1e-18)),
        label="Before FIR",
    )
    ax_pow.plot(
        f_spec / 1e3, 10.0 * np.log10(np.maximum(P_filt, 1e-18)),
        label="After FIR",
    )
    ax_pow.set_ylabel("Power (dB)")
    ax_pow.set_title(
        f"Spectrum: diff at {abs(f_rf_hz - f_lo_hz):.0f} Hz, "
        f"sum at {f_rf_hz + f_lo_hz:.0f} Hz"
    )
    ax_pow.legend(loc="upper right", fontsize=7)
    ax_pow.grid(True, alpha=0.2)
    ax_pow.tick_params(labelbottom=False)

    ax_vr.plot(f_spec / 1e3, np.real(X_raw), alpha=0.7, label="Before FIR")
    ax_vr.plot(f_spec / 1e3, np.real(X_filt), alpha=0.7, label="After FIR")
    ax_vr.set_ylabel("Re V")
    ax_vr.grid(True, alpha=0.2)
    ax_vr.tick_params(labelbottom=False)

    ax_vi.plot(f_spec / 1e3, np.imag(X_raw), alpha=0.7, label="Before FIR")
    ax_vi.plot(f_spec / 1e3, np.imag(X_filt), alpha=0.7, label="After FIR")
    ax_vi.set_xlabel("Frequency (kHz)")
    ax_vi.set_ylabel("Im V")
    ax_vi.grid(True, alpha=0.2)

    # Bottom left: FIR frequency response
    ax_fir = fig.add_subplot(outer[1, 0])
    ax_fir.plot(w_hz, 20.0 * np.log10(np.maximum(np.abs(h), 1e-8)), color="C2")
    ax_fir.axvline(
        cutoff_hz, ls="--", lw=0.8, color="0.4",
        label=f"Cutoff = {cutoff_hz:.0f} Hz",
    )
    ax_fir.set_xlim(0.0, fs_hz / 2.0)
    ax_fir.set_xlabel("Frequency (Hz)")
    ax_fir.set_ylabel("|H(f)| (dB)")
    ax_fir.set_title(
        f"FIR low-pass response ({num_taps} taps, cutoff {cutoff_hz:.0f} Hz)"
    )
    ax_fir.legend(loc="upper right", fontsize=8)
    ax_fir.grid(True, alpha=0.2)

    # Bottom right: convolution theorem consistency
    ax_conv = fig.add_subplot(outer[1, 1])
    ax_conv.plot(conv_direct, label="Time-domain convolution")
    ax_conv.plot(
        conv_freq, linestyle="--",
        label=r"IFFT$\{$FFT$(x) \cdot$ FFT$(h)\}$",
    )
    ax_conv.set_xlabel("Sample index")
    ax_conv.set_ylabel("Amplitude")
    ax_conv.set_title(
        f"Convolution theorem check"
    )
    ax_conv.legend(loc="upper right", fontsize=8)
    ax_conv.grid(True, alpha=0.2)

    return fig, {
        "time": ax_time,
        "power": ax_pow,
        "voltage_real": ax_vr,
        "voltage_imag": ax_vi,
        "fir": ax_fir,
        "convolution": ax_conv,
    }


__all__ = [
    "plot_theory_beats",
    "plot_theory_fourier",
    "plot_theory_aliasing",
    "plot_theory_mixer",
]
