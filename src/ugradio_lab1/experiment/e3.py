"""E3 voltage/power spectrum figures for §5.3 (cells 33, 36)."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_e3_stacked_spectrum_sim(
    freq_hz: np.ndarray,
    spec_v: np.ndarray,
    power_v2: np.ndarray,
    *,
    title: str = "Simulation: Two-tone",
    figsize: tuple[float, float] = (10, 7),
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 33: power row + stacked Re/Im subplots (theory-style)."""

    freq_hz = np.asarray(freq_hz, dtype=float)
    spec_v = np.asarray(spec_v, dtype=np.complex128)
    power_v2 = np.asarray(power_v2, dtype=float)
    if not (freq_hz.size == spec_v.size == power_v2.size):
        raise ValueError("freq_hz, spec_v, and power_v2 must have the same length.")

    fig = plt.figure(figsize=figsize, dpi=300)
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.15)
    ax_pow = fig.add_subplot(outer[0, 0])
    gs_complex = outer[1, 0].subgridspec(2, 1, hspace=0.0)
    ax_re = fig.add_subplot(gs_complex[0, 0], sharex=ax_pow)
    ax_im = fig.add_subplot(gs_complex[1, 0], sharex=ax_pow)

    freq_mhz = freq_hz / 1e6
    power_db = 10.0 * np.log10(np.maximum(power_v2, 1e-30))
    ax_pow.plot(freq_mhz, power_db, color="C0", lw=0.8)
    ax_pow.set_ylabel("Power (dB re 1 V$^2$)")
    ax_pow.set_title(title, fontsize=10)
    ax_pow.axvline(0.0, color="0.65", lw=0.8, ls="--")
    ax_pow.grid(True, alpha=0.2)
    ax_pow.tick_params(labelbottom=False)

    v_re = spec_v.real
    v_im = spec_v.imag
    vmin = float(min(np.min(v_re), np.min(v_im)))
    vmax = float(max(np.max(v_re), np.max(v_im)))
    vpad = 0.05 * (vmax - vmin) if vmax != vmin else 0.01

    ax_re.plot(freq_mhz, v_re, color="C1", lw=0.8)
    ax_re.set_title("Voltage spectrum", fontsize=9)
    ax_re.set_ylabel("Re[V] (V)")
    ax_re.set_ylim(vmin - vpad, vmax + vpad)
    ax_re.axvline(0.0, color="0.65", lw=0.8, ls="--")
    ax_re.grid(True, alpha=0.2)
    ax_re.tick_params(labelbottom=False)

    ax_im.plot(freq_mhz, v_im, color="C2", lw=0.8)
    ax_im.set_xlabel("Frequency (MHz)")
    ax_im.set_ylabel("Im[V] (V)")
    ax_im.set_ylim(vmin - vpad, vmax + vpad)
    ax_im.axvline(0.0, color="0.65", lw=0.8, ls="--")
    ax_im.grid(True, alpha=0.2)

    fig.subplots_adjust(top=0.92)
    return fig, {
        "power": ax_pow,
        "real": ax_re,
        "imag": ax_im,
        # Compatibility aliases
        "complex": ax_re,
        "complex_real": ax_re,
        "complex_imag": ax_im,
    }


def _parse_tones_hz(tones: object) -> list[float]:
    """Parse tone metadata stored as JSON or Python literal string."""
    if isinstance(tones, (list, tuple, np.ndarray)):
        return [float(t) for t in tones]

    tones_str = str(tones)
    parsed: object
    try:
        parsed = json.loads(tones_str)
    except Exception:
        try:
            parsed = ast.literal_eval(tones_str)
        except Exception:
            return []

    if isinstance(parsed, (list, tuple, np.ndarray)):
        out: list[float] = []
        for item in parsed:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    return []


def _format_power_dbm(value: object) -> str:
    try:
        value_float = float(value)
        if np.isfinite(value_float):
            return f"{value_float:.1f} dBm"
    except Exception:
        pass
    return "power=? dBm"


def _run_title(run_catalog: pd.DataFrame, run_id: str) -> str:
    """Build a descriptive title from NPZ metadata."""
    if "run_id" not in run_catalog.columns:
        return str(run_id)

    match = run_catalog[run_catalog["run_id"].astype(str) == str(run_id)]
    if match.empty:
        return str(run_id)

    row = match.iloc[0]
    mode = str(row.get("mode", "unknown"))
    tones = row.get("tones_hz_json", [])
    tone_list = _parse_tones_hz(tones)
    if len(tone_list) > 0:
        tone_str = ", ".join(f"{tone / 1e3:.1f} kHz" for tone in tone_list)
    else:
        tone_str = "tones=unknown"

    if "source_power_dbm_mean" in row and pd.notna(row.get("source_power_dbm_mean")):
        power_str = _format_power_dbm(row.get("source_power_dbm_mean"))
    else:
        power_str = _format_power_dbm(row.get("sg1_power_dbm"))

    return f"{mode}: {tone_str}\n{power_str}"


def _ordered_run_ids(
    spectrum_profile: pd.DataFrame,
    run_catalog: pd.DataFrame,
    run_ids: Sequence[str] | None = None,
) -> list[str]:
    """Prefer run_catalog ordering, then append any unmatched run IDs."""
    available_ids = spectrum_profile["run_id"].astype(str).dropna().unique().tolist()
    available_set = set(available_ids)

    if run_ids is not None:
        ordered = [str(run_id) for run_id in run_ids if str(run_id) in available_set]
        return ordered

    ordered: list[str] = []
    if "run_id" in run_catalog.columns:
        catalog_order = run_catalog["run_id"].astype(str).dropna().tolist()
        ordered = [run_id for run_id in catalog_order if run_id in available_set]

    extras = sorted(run_id for run_id in available_ids if run_id not in set(ordered))
    return ordered + extras


def plot_e3_spectra_grid_physical(
    spectrum_profile: pd.DataFrame,
    run_catalog: pd.DataFrame,
    *,
    run_ids: Sequence[str] | None = None,
    figsize_per_col: float = 5.0,
    figsize_height: float = 4.0,
    suptitle: str | None = "E3 Physical Spectra (block-averaged)",
) -> tuple[Figure, dict[str, Axes]]:
    """Cell 36: power row + stacked Re/Im subplots per run (theory-style)."""

    required_cols = {
        "run_id",
        "frequency_hz",
        "voltage_real_v",
        "voltage_imag_v",
        "power_v2",
    }
    missing_cols = sorted(required_cols - set(spectrum_profile.columns))
    if missing_cols:
        raise ValueError(
            "spectrum_profile is missing required columns: "
            + ", ".join(missing_cols)
        )

    run_id_list = _ordered_run_ids(spectrum_profile, run_catalog, run_ids=run_ids)
    n_runs = len(run_id_list)
    if n_runs == 0:
        raise ValueError("No runs available in spectrum_profile for E3 plotting.")

    fig = plt.figure(figsize=(figsize_per_col * n_runs, figsize_height), dpi=300)
    outer = fig.add_gridspec(
        2, n_runs, height_ratios=[1, 1], hspace=0.18, wspace=0.22,
    )

    # Keep consistent voltage scales across all runs and between Re/Im panels.
    global_vmin = float(
        min(
            spectrum_profile["voltage_real_v"].to_numpy(dtype=float).min(),
            spectrum_profile["voltage_imag_v"].to_numpy(dtype=float).min(),
        )
    )
    global_vmax = float(
        max(
            spectrum_profile["voltage_real_v"].to_numpy(dtype=float).max(),
            spectrum_profile["voltage_imag_v"].to_numpy(dtype=float).max(),
        )
    )
    global_vpad = 0.05 * (global_vmax - global_vmin) if global_vmax != global_vmin else 0.01

    axes_dict: dict[str, Axes] = {}
    for col, run_id in enumerate(run_id_list):
        run_data = spectrum_profile[
            spectrum_profile["run_id"].astype(str) == run_id
        ].copy()
        run_data = run_data.sort_values("frequency_hz").reset_index(drop=True)

        freq_mhz = run_data["frequency_hz"].to_numpy(dtype=float) / 1e6
        v_re = run_data["voltage_real_v"].to_numpy(dtype=float)
        v_im = run_data["voltage_imag_v"].to_numpy(dtype=float)
        pwr = run_data["power_v2"].to_numpy(dtype=float)
        pwr_db = 10.0 * np.log10(np.maximum(pwr, 1e-30))

        ax_pow = fig.add_subplot(outer[0, col])
        gs_complex = outer[1, col].subgridspec(2, 1, hspace=0.0)
        ax_re = fig.add_subplot(gs_complex[0, 0], sharex=ax_pow)
        ax_im = fig.add_subplot(gs_complex[1, 0], sharex=ax_pow)

        ax_pow.plot(freq_mhz, pwr_db, color="C0", lw=0.7)
        ax_pow.axvline(0.0, color="0.7", lw=0.7, ls="--")
        ax_pow.set_ylabel("Power (dB re 1 V$^2$)", fontsize=9)
        ax_pow.set_ylim(-50, np.max(pwr_db) * 1.05)
        ax_pow.grid(True, alpha=0.2)
        ax_pow.tick_params(labelbottom=False)

        ax_re.plot(freq_mhz, v_re, color="C1", lw=0.7)
        ax_re.set_title("Voltage spectrum", fontsize=8)
        ax_re.set_ylabel("Re[V] (V)", fontsize=9)
        ax_re.set_ylim(global_vmin - global_vpad, global_vmax + global_vpad)
        ax_re.axvline(0.0, color="0.7", lw=0.7, ls="--")
        ax_re.grid(True, alpha=0.2)
        ax_re.tick_params(labelbottom=False)

        ax_im.plot(freq_mhz, v_im, color="C2", lw=0.7)
        ax_im.set_xlabel("Frequency (MHz)", fontsize=9)
        ax_im.set_ylabel("Im[V] (V)", fontsize=9)
        ax_im.set_ylim(global_vmin - global_vpad, global_vmax + global_vpad)
        ax_im.axvline(0.0, color="0.7", lw=0.7, ls="--")
        ax_im.grid(True, alpha=0.2)

        ax_pow.set_title(_run_title(run_catalog, run_id), fontsize=8)

        # r0/r1/r2 legacy keys for callers and notebooks.
        axes_dict[f"r0_c{col}"] = ax_pow
        axes_dict[f"r1_c{col}"] = ax_re
        axes_dict[f"r2_c{col}"] = ax_im
        axes_dict[f"power_c{col}"] = ax_pow
        axes_dict[f"complex_c{col}"] = ax_re
        axes_dict[f"real_c{col}"] = ax_re
        axes_dict[f"imag_c{col}"] = ax_im

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
        fig.subplots_adjust(top=0.90)
    else:
        fig.subplots_adjust(top=0.95)
    return fig, axes_dict


__all__ = ["plot_e3_stacked_spectrum_sim", "plot_e3_spectra_grid_physical"]
