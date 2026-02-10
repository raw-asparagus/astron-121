"""E1 alias-mapping figures for §5.1 (cell 26)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_e1_alias_map_combined(
    alias_table: pd.DataFrame,
    e1_t3_physical: pd.DataFrame,
    fs_values: Sequence[float],
    *,
    figsize: tuple[float, float] = (10, 12),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 26: 4x1 stacked alias-map panels, predicted (o) + physical (x)."""

    fig, axes = plt.subplots(
        len(fs_values), 1, figsize=figsize, sharex=True, dpi=300,
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]

    axes_dict: dict[str, Axes] = {}
    for ax, fs in zip(axes, fs_values):
        nyq = fs / 2.0

        # Simulation predicted
        sim_mask = alias_table["sample_rate_hz"] == fs
        sim_sub = alias_table.loc[sim_mask].sort_values("f_true_hz")
        sim_f = sim_sub["f_true_hz"].to_numpy(dtype=float)
        sim_pred = np.abs(sim_sub["predicted_alias_hz"].to_numpy(dtype=float))

        # Physical data — aggregate median per unique f_true
        phys_mask = np.isclose(
            e1_t3_physical["sample_rate_hz"].to_numpy(dtype=float), fs, rtol=1e-3,
        )
        phys_sub = e1_t3_physical.loc[phys_mask].copy()
        phys_sub["measured_alias_hz"] = np.abs(
            phys_sub["measured_alias_hz"].to_numpy(dtype=float)
        )
        phys_agg = (
            phys_sub.groupby("f_true_hz", sort=True)["measured_alias_hz"]
            .median()
            .reset_index()
        )
        phys_f = phys_agg["f_true_hz"].to_numpy(dtype=float)
        phys_meas = phys_agg["measured_alias_hz"].to_numpy(dtype=float)

        ax.axvline(nyq / 1e6, color="0.8", lw=0.8, ls="--", label="Nyquist frequency")

        ax.scatter(
            sim_f / 1e6, sim_pred / 1e6, s=6, alpha=0.7,
            marker="o", color="C0", label="Predicted",
        )
        ax.scatter(
            phys_f / 1e6, phys_meas / 1e6, s=30, alpha=0.8,
            marker="x", color="C1", linewidths=1.5, label="Physical measured",
        )
        ax.set_ylabel("Alias freq (MHz)")
        ax.set_title(
            f"$f_s$ = {fs/1e6:.1f} MHz  (Nyquist = {nyq/1e6:.2f} MHz)",
            fontsize=10,
        )
        ax.set_ylim(0.0, nyq / 1e6)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)
        axes_dict[f"fs_{fs/1e6:.1f}MHz"] = ax

    axes[-1].set_xlabel("True Frequency (MHz)")
    fig.tight_layout()
    return fig, axes_dict


__all__ = ["plot_e1_alias_map_combined"]
