"""Simulation generators for DSB/SSB mixer behavior and spur analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal

from ugradio_lab1.analysis.experiments import line_spur_catalog
from ugradio_lab1.analysis.spectra import power_spectrum
from ugradio_lab1.utils.validation import as_1d_array

FFTBackend = Literal["numpy", "ugradio"]
Sideband = Literal["upper", "lower"]


@dataclass(frozen=True)
class DSBSpurSimulation:
    """Container for synthetic DSB spur survey outputs."""

    catalog: pd.DataFrame
    frequency_hz: np.ndarray
    power_v2: np.ndarray
    voltage_v: np.ndarray
    observed_line_hz: np.ndarray
    observed_level_db: np.ndarray


def simulate_dsb_output(
    *,
    f_lo_hz: float,
    f_rf_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    lo_amplitude_v: float = 1.0,
    rf_amplitude_v: float = 1.0,
    lo_phase_rad: float = 0.0,
    rf_phase_rad: float = 0.0,
    noise_std_v: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate ideal DSB mixer output from real LO/RF sinusoids."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if lo_amplitude_v < 0.0 or rf_amplitude_v < 0.0:
        raise ValueError("lo_amplitude_v and rf_amplitude_v must be non-negative.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")

    time_s = np.arange(n_samples, dtype=float) / fs
    lo = float(lo_amplitude_v) * np.cos(2.0 * np.pi * float(f_lo_hz) * time_s + float(lo_phase_rad))
    rf = float(rf_amplitude_v) * np.cos(2.0 * np.pi * float(f_rf_hz) * time_s + float(rf_phase_rad))
    output = lo * rf

    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        output = output + prng.normal(scale=float(noise_std_v), size=n_samples)
    return np.asarray(output, dtype=float)


def simulate_nonlinear_mixer_output(
    *,
    f_lo_hz: float,
    f_rf_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    lo_amplitude_v: float = 1.0,
    rf_amplitude_v: float = 1.0,
    linear_gain: float = 1.0,
    second_order_gain: float = 0.04,
    third_order_gain: float = 0.01,
    lo_leakage_gain: float = 0.015,
    rf_leakage_gain: float = 0.01,
    noise_std_v: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Simulate non-ideal mixer output with low-order nonlinear terms."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")

    time_s = np.arange(n_samples, dtype=float) / fs
    lo = float(lo_amplitude_v) * np.cos(2.0 * np.pi * float(f_lo_hz) * time_s)
    rf = float(rf_amplitude_v) * np.cos(2.0 * np.pi * float(f_rf_hz) * time_s)
    mix = lo * rf

    output = (
        float(linear_gain) * mix
        + float(second_order_gain) * mix**2
        + float(third_order_gain) * mix**3
        + float(lo_leakage_gain) * lo
        + float(rf_leakage_gain) * rf
    )
    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        output = output + prng.normal(scale=float(noise_std_v), size=n_samples)
    return np.asarray(output, dtype=float)


def simulate_dsb_spur_survey(
    *,
    f_lo_hz: float,
    f_rf_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    config: str = "dsb_external",
    orders: int = 3,
    tolerance_hz: float = 500.0,
    n_observed_lines: int = 16,
    min_line_spacing_hz: float = 0.0,
    fft_backend: FFTBackend = "numpy",
    nonlinear_kwargs: dict[str, float] | None = None,
    rng: np.random.Generator | int | None = None,
) -> DSBSpurSimulation:
    """Simulate E6 spur survey and return a T7-ready catalog."""

    if n_observed_lines <= 0:
        raise ValueError("n_observed_lines must be positive.")
    if min_line_spacing_hz < 0.0:
        raise ValueError("min_line_spacing_hz must be non-negative.")

    kwargs = dict(nonlinear_kwargs or {})
    voltage_v = simulate_nonlinear_mixer_output(
        f_lo_hz=f_lo_hz,
        f_rf_hz=f_rf_hz,
        sample_rate_hz=sample_rate_hz,
        n_samples=n_samples,
        rng=rng,
        **kwargs,
    )
    frequency_hz, power_v2 = power_spectrum(
        voltage_v,
        sample_rate_hz=sample_rate_hz,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    observed_line_hz, observed_level_db = pick_top_spectral_lines(
        frequency_hz,
        power_v2,
        n_lines=n_observed_lines,
        min_frequency_hz=0.0,
        min_line_spacing_hz=min_line_spacing_hz,
    )
    catalog = line_spur_catalog(
        config=config,
        f_lo_hz=f_lo_hz,
        f_rf_hz=f_rf_hz,
        observed_hz=observed_line_hz,
        observed_level_db=observed_level_db,
        tolerance_hz=tolerance_hz,
        orders=orders,
    )
    return DSBSpurSimulation(
        catalog=catalog,
        frequency_hz=frequency_hz,
        power_v2=power_v2,
        voltage_v=voltage_v,
        observed_line_hz=observed_line_hz,
        observed_level_db=observed_level_db,
    )


def simulate_ssb_iq(
    *,
    delta_f_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    sideband: Sideband,
    amplitude_v: float = 1.0,
    phase_rad: float = 0.0,
    noise_std_v: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ideal I/Q phasor trajectories for upper/lower sidebands."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if amplitude_v < 0.0:
        raise ValueError("amplitude_v must be non-negative.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")
    if sideband not in {"upper", "lower"}:
        raise ValueError("sideband must be either 'upper' or 'lower'.")

    time_s = np.arange(n_samples, dtype=float) / fs
    phase = 2.0 * np.pi * float(delta_f_hz) * time_s + float(phase_rad)
    sign = 1.0 if sideband == "upper" else -1.0
    i_samples = float(amplitude_v) * np.cos(phase)
    q_samples = float(amplitude_v) * sign * np.sin(phase)

    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        i_samples = i_samples + prng.normal(scale=float(noise_std_v), size=n_samples)
        q_samples = q_samples + prng.normal(scale=float(noise_std_v), size=n_samples)
    return i_samples, q_samples


def simulate_reverted_dsb_iq(
    *,
    delta_f_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    amplitude_v: float = 1.0,
    phase_rad: float = 0.0,
    phase_mismatch_rad: float = 0.0,
    noise_std_v: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate reverted-DSB I/Q traces where quadrature discrimination is lost."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if amplitude_v < 0.0:
        raise ValueError("amplitude_v must be non-negative.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")

    time_s = np.arange(n_samples, dtype=float) / fs
    phase = 2.0 * np.pi * float(delta_f_hz) * time_s + float(phase_rad)
    i_samples = float(amplitude_v) * np.cos(phase)
    q_samples = float(amplitude_v) * np.cos(phase + float(phase_mismatch_rad))

    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        i_samples = i_samples + prng.normal(scale=float(noise_std_v), size=n_samples)
        q_samples = q_samples + prng.normal(scale=float(noise_std_v), size=n_samples)
    return i_samples, q_samples


def simulate_r820t_vs_external(
    *,
    f_lo_hz: float,
    f_rf_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    fft_backend: FFTBackend = "numpy",
    external_nonlinear_kwargs: dict[str, float] | None = None,
    internal_noise_std_v: float = 0.002,
    rng: np.random.Generator | int | None = None,
) -> dict[str, np.ndarray]:
    """Simulate F18-style comparison between external and internal mixing paths."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if internal_noise_std_v < 0.0:
        raise ValueError("internal_noise_std_v must be non-negative.")

    external_voltage = simulate_nonlinear_mixer_output(
        f_lo_hz=f_lo_hz,
        f_rf_hz=f_rf_hz,
        sample_rate_hz=fs,
        n_samples=n_samples,
        rng=rng,
        **dict(external_nonlinear_kwargs or {}),
    )
    ideal_internal = simulate_dsb_output(
        f_lo_hz=f_lo_hz,
        f_rf_hz=f_rf_hz,
        sample_rate_hz=fs,
        n_samples=n_samples,
        noise_std_v=internal_noise_std_v,
        rng=rng,
    )

    diff_hz = abs(float(f_rf_hz) - float(f_lo_hz))
    lowpass_hz = min(max(3.0 * diff_hz, fs * 0.03), fs * 0.45)
    sos = signal.butter(4, lowpass_hz / (fs / 2.0), btype="lowpass", output="sos")
    internal_voltage = signal.sosfiltfilt(sos, ideal_internal)

    frequency_hz_ext, power_ext = power_spectrum(
        external_voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    frequency_hz_int, power_int = power_spectrum(
        internal_voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    if not np.allclose(frequency_hz_ext, frequency_hz_int):
        raise RuntimeError("External/internal frequency grids differ unexpectedly.")

    return {
        "frequency_hz": frequency_hz_ext,
        "external_voltage_v": np.asarray(external_voltage),
        "r820t_voltage_v": np.asarray(internal_voltage),
        "external_power_v2": power_ext,
        "r820t_power_v2": power_int,
    }


def pick_top_spectral_lines(
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
    *,
    n_lines: int,
    min_frequency_hz: float = 0.0,
    min_line_spacing_hz: float = 0.0,
    min_prominence_db: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Select strongest spectral lines with optional minimum spacing."""

    frequency = as_1d_array(frequency_hz, "frequency_hz", dtype=float)
    power = as_1d_array(power_v2, "power_v2", dtype=float)
    if frequency.size != power.size:
        raise ValueError("frequency_hz and power_v2 must have the same length.")
    if n_lines <= 0:
        raise ValueError("n_lines must be positive.")
    if min_line_spacing_hz < 0.0:
        raise ValueError("min_line_spacing_hz must be non-negative.")

    mask = frequency >= float(min_frequency_hz)
    if not np.any(mask):
        raise ValueError("No frequencies satisfy min_frequency_hz.")
    filtered_frequency = frequency[mask]
    filtered_power = power[mask]
    power_db = 10.0 * np.log10(np.maximum(filtered_power, np.finfo(float).tiny))

    peak_indices, _ = signal.find_peaks(power_db, prominence=float(min_prominence_db))
    if peak_indices.size == 0:
        peak_indices = np.arange(filtered_frequency.size, dtype=int)

    sorted_indices = peak_indices[np.argsort(power_db[peak_indices])[::-1]]
    selected: list[int] = []
    for candidate in sorted_indices:
        if len(selected) >= n_lines:
            break
        if min_line_spacing_hz > 0.0:
            if any(
                abs(filtered_frequency[candidate] - filtered_frequency[idx]) < min_line_spacing_hz
                for idx in selected
            ):
                continue
        selected.append(int(candidate))

    if not selected:
        raise RuntimeError("No spectral lines selected; check n_lines/min_line_spacing_hz.")

    selected = sorted(selected, key=lambda idx: filtered_frequency[idx])
    observed_frequency_hz = filtered_frequency[selected]
    observed_level_db = power_db[selected]
    return observed_frequency_hz.astype(float), observed_level_db.astype(float)


def _validate_sample_rate(sample_rate_hz: float) -> float:
    rate = float(sample_rate_hz)
    if rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    return rate


def _resolve_rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


__all__ = [
    "DSBSpurSimulation",
    "pick_top_spectral_lines",
    "simulate_dsb_output",
    "simulate_dsb_spur_survey",
    "simulate_nonlinear_mixer_output",
    "simulate_r820t_vs_external",
    "simulate_reverted_dsb_iq",
    "simulate_ssb_iq",
]
