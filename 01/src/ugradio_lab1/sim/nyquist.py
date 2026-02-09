"""Simulation generators for Nyquist, spectra, leakage, and resolution studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal

from ugradio_lab1.analysis.bandpass import bandpass_curve
from ugradio_lab1.analysis.leakage import nyquist_window_extension
from ugradio_lab1.analysis.nyquist import predict_alias_frequency
from ugradio_lab1.analysis.spectra import power_spectrum
from ugradio_lab1.utils.validation import as_1d_array

FFTBackend = Literal["numpy", "ugradio"]


@dataclass(frozen=True)
class AliasSweepSimulation:
    """Container for synthetic E1 alias sweep outputs."""

    table: pd.DataFrame
    voltage_by_run: dict[str, np.ndarray]


@dataclass(frozen=True)
class LeakageSimulation:
    """Container for synthetic leakage comparison outputs."""

    frequency_hz: np.ndarray
    bin_centered_power_v2: np.ndarray
    off_bin_power_v2: np.ndarray
    bin_centered_voltage_v: np.ndarray
    off_bin_voltage_v: np.ndarray
    bin_centered_frequency_hz: float
    off_bin_frequency_hz: float


@dataclass(frozen=True)
class ResolutionSweepSimulation:
    """Container for synthetic two-tone resolution sweep outputs."""

    table: pd.DataFrame
    voltage_by_n_samples: dict[int, np.ndarray]


def sample_times(
    n_samples: int,
    sample_rate_hz: float,
    *,
    start_time_s: float = 0.0,
) -> np.ndarray:
    """Return evenly spaced sample times for a fixed sample rate."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    fs = _validate_sample_rate(sample_rate_hz)
    return float(start_time_s) + np.arange(n_samples, dtype=float) / fs


def generate_tone(
    *,
    frequency_hz: float,
    sample_rate_hz: float,
    n_samples: int,
    amplitude_v: float = 1.0,
    phase_rad: float = 0.0,
    dc_offset_v: float = 0.0,
    noise_std_v: float = 0.0,
    complex_output: bool = True,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate a sampled sinusoid with optional additive Gaussian noise."""

    if amplitude_v < 0.0:
        raise ValueError("amplitude_v must be non-negative.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")

    time_s = sample_times(n_samples, sample_rate_hz)
    phase = 2.0 * np.pi * float(frequency_hz) * time_s + float(phase_rad)
    if complex_output:
        voltage = float(amplitude_v) * np.exp(1j * phase) + float(dc_offset_v)
    else:
        voltage = float(amplitude_v) * np.cos(phase) + float(dc_offset_v)

    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        if complex_output:
            sigma = float(noise_std_v) / np.sqrt(2.0)
            noise = prng.normal(scale=sigma, size=n_samples) + 1j * prng.normal(
                scale=sigma, size=n_samples
            )
        else:
            noise = prng.normal(scale=float(noise_std_v), size=n_samples)
        voltage = voltage + noise

    return np.asarray(voltage)


def generate_multi_tone(
    *,
    frequencies_hz: np.ndarray | Sequence[float],
    sample_rate_hz: float,
    n_samples: int,
    amplitudes_v: np.ndarray | Sequence[float] | None = None,
    phases_rad: np.ndarray | Sequence[float] | None = None,
    dc_offset_v: float = 0.0,
    noise_std_v: float = 0.0,
    complex_output: bool = True,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate a sampled sum of tones with optional additive Gaussian noise."""

    frequencies = as_1d_array(np.asarray(frequencies_hz, dtype=float), "frequencies_hz", dtype=float)
    if amplitudes_v is None:
        amplitudes = np.ones(frequencies.size, dtype=float)
    else:
        amplitudes = as_1d_array(np.asarray(amplitudes_v, dtype=float), "amplitudes_v", dtype=float)
        if amplitudes.size != frequencies.size:
            raise ValueError("amplitudes_v must match frequencies_hz length.")
    if phases_rad is None:
        phases = np.zeros(frequencies.size, dtype=float)
    else:
        phases = as_1d_array(np.asarray(phases_rad, dtype=float), "phases_rad", dtype=float)
        if phases.size != frequencies.size:
            raise ValueError("phases_rad must match frequencies_hz length.")

    if np.any(amplitudes < 0.0):
        raise ValueError("amplitudes_v entries must be non-negative.")
    if noise_std_v < 0.0:
        raise ValueError("noise_std_v must be non-negative.")

    time_s = sample_times(n_samples, sample_rate_hz)
    if complex_output:
        voltage = np.zeros(n_samples, dtype=np.complex128)
        for frequency_hz, amplitude_v, phase_rad in zip(frequencies, amplitudes, phases):
            phase = 2.0 * np.pi * frequency_hz * time_s + phase_rad
            voltage += amplitude_v * np.exp(1j * phase)
        voltage = voltage + float(dc_offset_v)
    else:
        voltage = np.zeros(n_samples, dtype=float)
        for frequency_hz, amplitude_v, phase_rad in zip(frequencies, amplitudes, phases):
            phase = 2.0 * np.pi * frequency_hz * time_s + phase_rad
            voltage += amplitude_v * np.cos(phase)
        voltage = voltage + float(dc_offset_v)

    if noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        if complex_output:
            sigma = float(noise_std_v) / np.sqrt(2.0)
            noise = prng.normal(scale=sigma, size=n_samples) + 1j * prng.normal(
                scale=sigma, size=n_samples
            )
        else:
            noise = prng.normal(scale=float(noise_std_v), size=n_samples)
        voltage = voltage + noise

    return np.asarray(voltage)


def estimate_peak_frequency(
    voltage_v: np.ndarray,
    *,
    sample_rate_hz: float,
    fft_backend: FFTBackend = "numpy",
    positive_only: bool = False,
) -> float:
    """Estimate dominant tone frequency by locating the max-power FFT bin."""

    frequency_hz, power_v2 = power_spectrum(
        voltage_v,
        sample_rate_hz=sample_rate_hz,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    if positive_only:
        mask = frequency_hz >= 0.0
        if not np.any(mask):
            raise ValueError("positive_only requested but no non-negative frequencies are available.")
        frequency_hz = frequency_hz[mask]
        power_v2 = power_v2[mask]
    peak_index = int(np.argmax(power_v2))
    return float(frequency_hz[peak_index])


def simulate_alias_sweep(
    f_true_hz: np.ndarray | Sequence[float],
    sample_rate_hz: np.ndarray | Sequence[float] | float,
    *,
    n_samples: int,
    amplitude_v: float = 1.0,
    noise_std_v: float = 0.0,
    complex_output: bool = True,
    fft_backend: FFTBackend = "numpy",
    run_id_prefix: str = "sim_e1_",
    rng: np.random.Generator | int | None = None,
) -> AliasSweepSimulation:
    """Simulate E1 alias-map captures and recover measured aliases."""

    f_true = as_1d_array(np.asarray(f_true_hz, dtype=float), "f_true_hz", dtype=float)
    fs_array = np.asarray(sample_rate_hz, dtype=float)
    if fs_array.ndim == 0:
        sample_rates = np.full(f_true.size, float(fs_array), dtype=float)
    else:
        sample_rates = as_1d_array(fs_array, "sample_rate_hz", dtype=float)
        if sample_rates.size != f_true.size:
            raise ValueError("sample_rate_hz must be scalar or match f_true_hz length.")
    if np.any(sample_rates <= 0.0):
        raise ValueError("sample_rate_hz must be positive.")

    predicted_alias = predict_alias_frequency(f_true, sample_rates)

    rows: list[dict[str, float | int | str | bool]] = []
    voltage_by_run: dict[str, np.ndarray] = {}
    for idx, (f_input, fs, predicted) in enumerate(zip(f_true, sample_rates, predicted_alias)):
        run_id = f"{run_id_prefix}{idx:03d}"
        voltage = generate_tone(
            frequency_hz=float(f_input),
            sample_rate_hz=float(fs),
            n_samples=n_samples,
            amplitude_v=amplitude_v,
            noise_std_v=noise_std_v,
            complex_output=complex_output,
            rng=rng,
        )
        measured = estimate_peak_frequency(
            voltage,
            sample_rate_hz=float(fs),
            fft_backend=fft_backend,
            positive_only=not complex_output,
        )
        rows.append(
            {
                "run_id": run_id,
                "f_true_hz": float(f_input),
                "sample_rate_hz": float(fs),
                "predicted_alias_hz": float(predicted),
                "measured_alias_hz": measured,
                "residual_hz": measured - float(predicted),
                "n_samples": int(n_samples),
                "amplitude_v": float(amplitude_v),
                "noise_std_v": float(noise_std_v),
                "complex_output": bool(complex_output),
            }
        )
        voltage_by_run[run_id] = voltage

    table = pd.DataFrame(rows)
    return AliasSweepSimulation(table=table, voltage_by_run=voltage_by_run)


def ideal_bandpass_gain(
    frequency_hz: np.ndarray | Sequence[float],
    *,
    center_hz: float = 0.0,
    passband_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Return a smooth low-pass style gain profile (linear units)."""

    frequency = as_1d_array(np.asarray(frequency_hz, dtype=float), "frequency_hz", dtype=float)
    if passband_hz <= 0.0:
        raise ValueError("passband_hz must be positive.")
    if order < 1:
        raise ValueError("order must be >= 1.")

    half_band = float(passband_hz) / 2.0
    offset = np.abs(frequency - float(center_hz))
    scaled = np.maximum(offset / half_band, 0.0)
    gain = 1.0 / np.sqrt(1.0 + scaled ** (2 * int(order)))
    return gain.astype(float)


def simulate_bandpass_sweep(
    frequency_hz: np.ndarray | Sequence[float],
    *,
    sample_rate_hz: float,
    n_samples: int,
    source_amplitude_v: float = 1.0,
    center_hz: float = 0.0,
    passband_hz: float | None = None,
    order: int = 4,
    measurement_noise_std_v: float = 0.0,
    mode: str = "default",
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Simulate an E2-style bandpass sweep and return a gain curve table."""

    frequencies = as_1d_array(np.asarray(frequency_hz, dtype=float), "frequency_hz", dtype=float)
    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if source_amplitude_v <= 0.0:
        raise ValueError("source_amplitude_v must be positive.")
    if measurement_noise_std_v < 0.0:
        raise ValueError("measurement_noise_std_v must be non-negative.")

    effective_passband = 0.6 * fs if passband_hz is None else float(passband_hz)
    model_gain = ideal_bandpass_gain(
        frequencies,
        center_hz=center_hz,
        passband_hz=effective_passband,
        order=order,
    )
    amplitudes = float(source_amplitude_v) * model_gain

    if measurement_noise_std_v > 0.0:
        prng = _resolve_rng(rng)
        amplitudes = amplitudes + prng.normal(scale=float(measurement_noise_std_v), size=amplitudes.size)
    amplitudes = np.maximum(np.abs(amplitudes), np.finfo(float).tiny)

    curve = bandpass_curve(
        frequencies,
        amplitudes,
        reference_amplitude_v=float(source_amplitude_v),
        mode=mode,
    )
    curve["sample_rate_hz"] = fs
    curve["n_samples"] = int(n_samples)
    curve["model_gain_linear"] = model_gain
    curve["model_gain_db"] = 20.0 * np.log10(np.maximum(model_gain, np.finfo(float).tiny))
    curve["mode_center_hz"] = float(center_hz)
    curve["mode_passband_hz"] = float(effective_passband)
    return curve


def simulate_leakage_experiment(
    *,
    sample_rate_hz: float,
    n_samples: int,
    bin_index: int,
    bin_offset: float = 0.35,
    amplitude_v: float = 1.0,
    noise_std_v: float = 0.0,
    fft_backend: FFTBackend = "numpy",
) -> LeakageSimulation:
    """Simulate bin-centered and off-bin tones for leakage analysis (E4/F7)."""

    fs = _validate_sample_rate(sample_rate_hz)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if bin_index < 0 or bin_index >= n_samples // 2:
        raise ValueError("bin_index must satisfy 0 <= bin_index < n_samples/2.")

    delta_f = fs / float(n_samples)
    bin_centered_frequency_hz = float(bin_index) * delta_f
    off_bin_frequency_hz = (float(bin_index) + float(bin_offset)) * delta_f

    centered_voltage = generate_tone(
        frequency_hz=bin_centered_frequency_hz,
        sample_rate_hz=fs,
        n_samples=n_samples,
        amplitude_v=amplitude_v,
        noise_std_v=noise_std_v,
        complex_output=True,
    )
    off_bin_voltage = generate_tone(
        frequency_hz=off_bin_frequency_hz,
        sample_rate_hz=fs,
        n_samples=n_samples,
        amplitude_v=amplitude_v,
        noise_std_v=noise_std_v,
        complex_output=True,
    )

    frequency_hz, centered_power = power_spectrum(
        centered_voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    _, off_bin_power = power_spectrum(
        off_bin_voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )

    return LeakageSimulation(
        frequency_hz=frequency_hz,
        bin_centered_power_v2=centered_power,
        off_bin_power_v2=off_bin_power,
        bin_centered_voltage_v=centered_voltage,
        off_bin_voltage_v=off_bin_voltage,
        bin_centered_frequency_hz=bin_centered_frequency_hz,
        off_bin_frequency_hz=off_bin_frequency_hz,
    )


def simulate_resolution_sweep(
    *,
    sample_rate_hz: float,
    n_samples_values: np.ndarray | Sequence[int],
    delta_f_hz_candidates: np.ndarray | Sequence[float],
    tone_center_hz: float | None = None,
    tone_amplitude_v: float = 1.0,
    noise_std_v: float = 0.0,
    min_peak_prominence_db: float = 6.0,
    min_valley_depth_db: float = 1.0,
    fft_backend: FFTBackend = "numpy",
) -> ResolutionSweepSimulation:
    """Simulate E4 two-tone resolution runs over sample length."""

    fs = _validate_sample_rate(sample_rate_hz)
    n_values = as_1d_array(np.asarray(n_samples_values, dtype=int), "n_samples_values", dtype=int)
    if np.any(n_values <= 0):
        raise ValueError("n_samples_values entries must be positive.")
    delta_candidates = as_1d_array(
        np.asarray(delta_f_hz_candidates, dtype=float),
        "delta_f_hz_candidates",
        dtype=float,
    )
    if np.any(delta_candidates <= 0.0):
        raise ValueError("delta_f_hz_candidates entries must be positive.")

    sorted_n = np.sort(np.unique(n_values))
    sorted_delta = np.sort(np.unique(delta_candidates))
    center_hz = fs / 8.0 if tone_center_hz is None else float(tone_center_hz)

    rows: list[dict[str, float | int]] = []
    voltage_by_n_samples: dict[int, np.ndarray] = {}
    for n_samples in sorted_n:
        min_resolvable = np.nan
        representative_voltage: np.ndarray | None = None
        delta_f_bin_hz = fs / float(n_samples)

        for delta_f_hz in sorted_delta:
            frequencies = np.array([center_hz - delta_f_hz / 2.0, center_hz + delta_f_hz / 2.0], dtype=float)
            voltage = generate_multi_tone(
                frequencies_hz=frequencies,
                sample_rate_hz=fs,
                n_samples=int(n_samples),
                amplitudes_v=np.array([tone_amplitude_v, tone_amplitude_v], dtype=float),
                complex_output=True,
                noise_std_v=noise_std_v,
            )
            frequency_hz, power_v2 = power_spectrum(
                voltage,
                sample_rate_hz=fs,
                scaling="power",
                center=True,
                fft_backend=fft_backend,
            )
            representative_voltage = voltage
            if _two_tone_resolvable(
                frequency_hz,
                power_v2,
                frequencies[0],
                frequencies[1],
                min_peak_prominence_db=min_peak_prominence_db,
                min_valley_depth_db=min_valley_depth_db,
            ):
                min_resolvable = float(delta_f_hz)
                break

        rows.append(
            {
                "n_samples": int(n_samples),
                "sample_rate_hz": fs,
                "delta_f_bin_hz": delta_f_bin_hz,
                "min_resolvable_delta_f_hz": min_resolvable,
                "resolution_ratio": min_resolvable / delta_f_bin_hz if np.isfinite(min_resolvable) else np.nan,
                "tone_center_hz": center_hz,
                "n_delta_trials": int(sorted_delta.size),
            }
        )
        if representative_voltage is not None:
            voltage_by_n_samples[int(n_samples)] = representative_voltage

    table = pd.DataFrame(rows).sort_values("n_samples", kind="stable").reset_index(drop=True)
    return ResolutionSweepSimulation(table=table, voltage_by_n_samples=voltage_by_n_samples)


def simulate_multi_window_spectrum(
    voltage_v: np.ndarray,
    *,
    sample_rate_hz: float,
    window_indices: Sequence[int] = (-1, 0, 1),
    fft_backend: FFTBackend = "numpy",
) -> pd.DataFrame:
    """Generate an extended Nyquist-window spectrum table (E4/F9 helper)."""

    fs = _validate_sample_rate(sample_rate_hz)
    frequency_hz, power_v2 = power_spectrum(
        voltage_v,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
        fft_backend=fft_backend,
    )
    return nyquist_window_extension(
        frequency_hz,
        power_v2,
        sample_rate_hz=fs,
        window_indices=window_indices,
    )


def _two_tone_resolvable(
    frequency_hz: np.ndarray,
    power_v2: np.ndarray,
    tone_a_hz: float,
    tone_b_hz: float,
    *,
    min_peak_prominence_db: float,
    min_valley_depth_db: float,
) -> bool:
    floor = np.finfo(float).tiny
    power_db = 10.0 * np.log10(np.maximum(power_v2, floor))
    peak_indices, _ = signal.find_peaks(power_db, prominence=min_peak_prominence_db)
    if peak_indices.size < 2:
        return False

    candidate_frequency = frequency_hz[peak_indices]
    idx_a = int(np.argmin(np.abs(candidate_frequency - tone_a_hz)))
    idx_b = int(np.argmin(np.abs(candidate_frequency - tone_b_hz)))
    peak_a = int(peak_indices[idx_a])
    peak_b = int(peak_indices[idx_b])
    if peak_a == peak_b:
        return False

    frequency_tolerance_hz = max(abs(tone_b_hz - tone_a_hz) * 0.35, abs(frequency_hz[1] - frequency_hz[0]) * 1.5)
    if abs(frequency_hz[peak_a] - tone_a_hz) > frequency_tolerance_hz:
        return False
    if abs(frequency_hz[peak_b] - tone_b_hz) > frequency_tolerance_hz:
        return False

    left, right = sorted((peak_a, peak_b))
    if right - left <= 1:
        return True
    valley_db = float(np.min(power_db[left + 1 : right]))
    min_peak_db = float(min(power_db[peak_a], power_db[peak_b]))
    return (min_peak_db - valley_db) >= float(min_valley_depth_db)


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
    "AliasSweepSimulation",
    "LeakageSimulation",
    "ResolutionSweepSimulation",
    "estimate_peak_frequency",
    "generate_multi_tone",
    "generate_tone",
    "ideal_bandpass_gain",
    "sample_times",
    "simulate_alias_sweep",
    "simulate_bandpass_sweep",
    "simulate_leakage_experiment",
    "simulate_multi_window_spectrum",
    "simulate_resolution_sweep",
]
