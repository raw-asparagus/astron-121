"""Simulation generators for noise, ACF, and radiometer-equation studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import signal

from ugradio_lab1.analysis.spectra import autocorrelation, autocorrelation_from_power_spectrum, power_spectrum

FFTBackend = Literal["numpy", "ugradio"]


@dataclass(frozen=True)
class RadiometerSimulation:
    """Container for synthetic radiometer-scaling outputs."""

    table: pd.DataFrame
    block_powers_v2: np.ndarray
    blocks_v: np.ndarray


@dataclass(frozen=True)
class ACFConsistencySimulation:
    """Container for synthetic ACF/spectrum consistency outputs."""

    voltage_v: np.ndarray
    frequency_hz: np.ndarray
    power_v2: np.ndarray
    lag_direct_s: np.ndarray
    autocorrelation_direct: np.ndarray
    lag_from_power_s: np.ndarray
    autocorrelation_from_power: np.ndarray


def generate_gaussian_noise(
    *,
    n_samples: int,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    complex_output: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate Gaussian noise samples (real or complex)."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if std_v < 0.0:
        raise ValueError("std_v must be non-negative.")

    prng = _resolve_rng(rng)
    if complex_output:
        sigma = float(std_v) / np.sqrt(2.0)
        noise = prng.normal(loc=float(mean_v), scale=sigma, size=n_samples) + 1j * prng.normal(
            loc=0.0, scale=sigma, size=n_samples
        )
    else:
        noise = prng.normal(loc=float(mean_v), scale=float(std_v), size=n_samples)
    return np.asarray(noise)


def generate_gaussian_noise_blocks(
    *,
    n_blocks: int,
    n_samples: int,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    complex_output: bool = False,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate 2D Gaussian-noise blocks with shape ``(n_blocks, n_samples)``."""

    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    prng = _resolve_rng(rng)
    if complex_output:
        sigma = float(std_v) / np.sqrt(2.0)
        blocks = prng.normal(loc=float(mean_v), scale=sigma, size=(n_blocks, n_samples)) + 1j * prng.normal(
            loc=0.0, scale=sigma, size=(n_blocks, n_samples)
        )
    else:
        blocks = prng.normal(loc=float(mean_v), scale=float(std_v), size=(n_blocks, n_samples))
    return np.asarray(blocks)


def bandlimit_noise(
    voltage_v: np.ndarray,
    *,
    sample_rate_hz: float,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    order: int = 4,
) -> np.ndarray:
    """Band-limit a noise trace using Butterworth filtering."""

    samples = np.asarray(voltage_v)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("voltage_v must be a non-empty 1D array.")
    fs = _validate_sample_rate(sample_rate_hz)
    if order < 1:
        raise ValueError("order must be >= 1.")

    if low_cut_hz is None and high_cut_hz is None:
        return samples.copy()

    nyquist = fs / 2.0
    if low_cut_hz is not None and low_cut_hz <= 0.0:
        raise ValueError("low_cut_hz must be positive when provided.")
    if high_cut_hz is not None and high_cut_hz <= 0.0:
        raise ValueError("high_cut_hz must be positive when provided.")
    if high_cut_hz is not None and high_cut_hz >= nyquist:
        raise ValueError("high_cut_hz must be < sample_rate_hz/2.")
    if low_cut_hz is not None and high_cut_hz is not None and low_cut_hz >= high_cut_hz:
        raise ValueError("low_cut_hz must be < high_cut_hz.")

    if low_cut_hz is None:
        wn = float(high_cut_hz) / nyquist
        btype = "lowpass"
    elif high_cut_hz is None:
        wn = float(low_cut_hz) / nyquist
        btype = "highpass"
    else:
        wn = [float(low_cut_hz) / nyquist, float(high_cut_hz) / nyquist]
        btype = "bandpass"

    sos = signal.butter(order, wn, btype=btype, output="sos")
    if np.iscomplexobj(samples):
        real = signal.sosfiltfilt(sos, np.real(samples))
        imag = signal.sosfiltfilt(sos, np.imag(samples))
        return real + 1j * imag
    return signal.sosfiltfilt(sos, samples)


def simulate_noise_capture(
    *,
    sample_rate_hz: float,
    n_samples: int,
    n_blocks: int = 1,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    complex_output: bool = False,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    filter_order: int = 4,
    squeeze: bool = True,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Generate synthetic noise capture blocks, optionally band-limited."""

    _validate_sample_rate(sample_rate_hz)
    blocks = generate_gaussian_noise_blocks(
        n_blocks=n_blocks,
        n_samples=n_samples,
        mean_v=mean_v,
        std_v=std_v,
        complex_output=complex_output,
        rng=rng,
    )
    if low_cut_hz is not None or high_cut_hz is not None:
        filtered_rows = [
            bandlimit_noise(
                row,
                sample_rate_hz=sample_rate_hz,
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                order=filter_order,
            )
            for row in blocks
        ]
        blocks = np.vstack([np.asarray(row)[None, :] for row in filtered_rows])

    if squeeze and n_blocks == 1:
        return np.asarray(blocks[0])
    return np.asarray(blocks)


def simulate_radiometer_experiment(
    *,
    sample_rate_hz: float,
    block_size: int,
    n_avg_values: np.ndarray | Sequence[int],
    n_realizations: int = 128,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    complex_output: bool = False,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    filter_order: int = 4,
    rng: np.random.Generator | int | None = None,
) -> RadiometerSimulation:
    """Simulate block averaging and sigma scaling for radiometer analysis."""

    _validate_sample_rate(sample_rate_hz)
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if n_realizations < 2:
        raise ValueError("n_realizations must be >= 2.")

    n_avg = np.asarray(n_avg_values, dtype=int)
    if n_avg.ndim != 1 or n_avg.size == 0:
        raise ValueError("n_avg_values must be a non-empty 1D sequence.")
    if np.any(n_avg <= 0):
        raise ValueError("n_avg_values entries must be positive integers.")

    sorted_n_avg = np.sort(np.unique(n_avg))
    max_n_avg = int(sorted_n_avg[-1])
    total_blocks = int(n_realizations * max_n_avg)

    blocks = simulate_noise_capture(
        sample_rate_hz=sample_rate_hz,
        n_samples=block_size,
        n_blocks=total_blocks,
        mean_v=mean_v,
        std_v=std_v,
        complex_output=complex_output,
        low_cut_hz=low_cut_hz,
        high_cut_hz=high_cut_hz,
        filter_order=filter_order,
        squeeze=False,
        rng=rng,
    )
    block_powers_v2 = np.mean(np.abs(blocks) ** 2, axis=1)

    sigma_reference: float | None = None
    rows: list[dict[str, float | int]] = []
    for n_average in sorted_n_avg:
        usable = int(n_realizations * n_average)
        grouped = block_powers_v2[:usable].reshape(n_realizations, n_average)
        averaged_power = np.mean(grouped, axis=1)
        sigma_power = float(np.std(averaged_power, ddof=1))
        if sigma_reference is None:
            sigma_reference = sigma_power
        expected_sigma_power = sigma_reference / np.sqrt(float(n_average))
        rows.append(
            {
                "n_avg": int(n_average),
                "sigma_power": sigma_power,
                "expected_sigma_power": expected_sigma_power,
                "sample_rate_hz": float(sample_rate_hz),
                "block_size": int(block_size),
                "n_realizations": int(n_realizations),
            }
        )

    table = pd.DataFrame(rows).sort_values("n_avg", kind="stable").reset_index(drop=True)
    return RadiometerSimulation(table=table, block_powers_v2=block_powers_v2, blocks_v=blocks)


def simulate_acf_consistency(
    *,
    sample_rate_hz: float,
    n_samples: int,
    mean_v: float = 0.0,
    std_v: float = 1.0,
    complex_output: bool = False,
    low_cut_hz: float | None = None,
    high_cut_hz: float | None = None,
    filter_order: int = 4,
    fft_backend: FFTBackend = "numpy",
    rng: np.random.Generator | int | None = None,
) -> ACFConsistencySimulation:
    """Simulate E5 ACF/power-spectrum consistency data products."""

    fs = _validate_sample_rate(sample_rate_hz)
    voltage = generate_gaussian_noise(
        n_samples=n_samples,
        mean_v=mean_v,
        std_v=std_v,
        complex_output=complex_output,
        rng=rng,
    )
    if low_cut_hz is not None or high_cut_hz is not None:
        voltage = bandlimit_noise(
            voltage,
            sample_rate_hz=fs,
            low_cut_hz=low_cut_hz,
            high_cut_hz=high_cut_hz,
            order=filter_order,
        )

    frequency_hz, power_v2 = power_spectrum(
        voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=False,
        fft_backend=fft_backend,
    )
    lag_direct_s, autocorrelation_direct = autocorrelation(
        voltage,
        sample_rate_hz=fs,
    )
    lag_from_power_s, autocorrelation_from_power = autocorrelation_from_power_spectrum(
        power_v2,
        sample_rate_hz=fs,
        centered=False,
        normalize="biased",
        fft_backend=fft_backend,
    )

    return ACFConsistencySimulation(
        voltage_v=voltage,
        frequency_hz=frequency_hz,
        power_v2=power_v2,
        lag_direct_s=lag_direct_s,
        autocorrelation_direct=autocorrelation_direct,
        lag_from_power_s=lag_from_power_s,
        autocorrelation_from_power=autocorrelation_from_power,
    )


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
    "ACFConsistencySimulation",
    "RadiometerSimulation",
    "bandlimit_noise",
    "generate_gaussian_noise",
    "generate_gaussian_noise_blocks",
    "simulate_acf_consistency",
    "simulate_noise_capture",
    "simulate_radiometer_experiment",
]
