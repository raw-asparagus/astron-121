"""E2 bandpass figures for §4.2 (cell 32)."""

from __future__ import annotations

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


_MODE_COLORS = {
    "default@1.0MHz": "C0",
    "default@1.6MHz": "C1",
    "default@2.4MHz": "C2",
    "default@3.2MHz": "C3",
}


def plot_e2_bandpass_default_and_aliased(
    curve_table: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (10, 4.5),
) -> tuple[Figure, dict[str, Axes]]:
    """Render only the default E2 bandpass curves (single panel)."""

    modes = sorted(curve_table["mode"].unique())

    fig, ax_default = plt.subplots(1, 1, figsize=figsize, dpi=300)

    for mode_label in modes:
        sub = curve_table[curve_table["mode"] == mode_label].sort_values("frequency_hz")
        freq_hz = sub["frequency_hz"].to_numpy(dtype=float)
        gain_db = sub["gain_db"].to_numpy(dtype=float)
        fs_nominal = sub["sample_rate_hz_nominal"].iloc[0]
        nyq = fs_nominal / 2.0
        short_label = f"$f_s$ = {fs_nominal/1e6:.1f} MHz"
        c = _MODE_COLORS.get(mode_label, "gray")

        ax_default.plot(
            freq_hz / 1e6, gain_db, color=c, label=short_label, alpha=0.85,
        )
        ax_default.axvline(nyq / 1e6, color=c, ls="--", alpha=0.3, lw=0.8)

    ax_default.set_xlabel("Frequency (MHz)")
    ax_default.set_ylabel("Gain (dB)")
    ax_default.set_title("Default Bandpass Response")
    ax_default.legend(fontsize=8)
    ax_default.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig, {"default": ax_default}


__all__ = ["plot_e2_bandpass_default_and_aliased"]
