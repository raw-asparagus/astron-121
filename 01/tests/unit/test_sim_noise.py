"""Unit tests for sim.noise."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.noise import radiometer_fit
from ugradio_lab1.sim.noise import (
    bandlimit_noise,
    generate_gaussian_noise,
    generate_gaussian_noise_blocks,
    simulate_acf_consistency,
    simulate_radiometer_experiment,
)


def test_generate_gaussian_noise_blocks_shape() -> None:
    blocks = generate_gaussian_noise_blocks(
        n_blocks=8,
        n_samples=256,
        std_v=2.0,
        rng=42,
    )
    assert blocks.shape == (8, 256)


def test_bandlimit_noise_reduces_high_frequency_power() -> None:
    fs = 1_000_000.0
    voltage = generate_gaussian_noise(n_samples=8192, std_v=1.0, rng=3)

    filtered = bandlimit_noise(
        voltage,
        sample_rate_hz=fs,
        high_cut_hz=120_000.0,
    )
    spectrum = np.fft.fftshift(np.fft.fft(filtered))
    frequency_hz = np.fft.fftshift(np.fft.fftfreq(filtered.size, d=1.0 / fs))
    power_v2 = np.abs(spectrum) ** 2

    low_band = np.mean(power_v2[np.abs(frequency_hz) <= 80_000.0])
    high_band = np.mean(power_v2[np.abs(frequency_hz) >= 250_000.0])
    assert low_band > high_band


def test_simulate_radiometer_experiment_slope_near_minus_half() -> None:
    simulation = simulate_radiometer_experiment(
        sample_rate_hz=1_000_000.0,
        block_size=512,
        n_avg_values=np.array([1, 2, 4, 8, 16]),
        n_realizations=256,
        std_v=1.0,
        rng=0,
    )
    fit = radiometer_fit(
        simulation.table["n_avg"].to_numpy(dtype=float),
        simulation.table["sigma_power"].to_numpy(dtype=float),
    )
    assert np.isclose(fit["slope"], -0.5, atol=0.15)


def test_simulate_acf_consistency_zero_lag_positive() -> None:
    simulation = simulate_acf_consistency(
        sample_rate_hz=1_000_000.0,
        n_samples=2048,
        std_v=1.0,
        rng=1,
    )

    zero_idx_direct = int(np.where(np.isclose(simulation.lag_direct_s, 0.0))[0][0])
    zero_idx_power = int(np.where(np.isclose(simulation.lag_from_power_s, 0.0))[0][0])
    assert simulation.autocorrelation_direct[zero_idx_direct] > 0.0
    assert simulation.autocorrelation_from_power[zero_idx_power] > 0.0
