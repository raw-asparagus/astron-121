"""Unit tests for analysis.spectra."""

from __future__ import annotations

import numpy as np
import pytest

from ugradio_lab1.analysis.spectra import (
    average_power_spectrum,
    autocorrelation,
    autocorrelation_from_power_spectrum,
    power_spectrum,
    voltage_spectrum,
)


def test_voltage_spectrum_amplitude_scaling_recovers_complex_tone() -> None:
    fs = 1_000_000.0
    n = 2048
    k = 123
    amplitude = 2.3
    frequency = k * fs / n
    time = np.arange(n) / fs
    samples = amplitude * np.exp(1j * 2.0 * np.pi * frequency * time)

    frequency_hz, spectrum = voltage_spectrum(
        samples,
        sample_rate_hz=fs,
        scaling="amplitude",
        center=False,
    )
    peak_index = int(np.argmax(np.abs(spectrum)))

    assert peak_index == k
    assert np.isclose(frequency_hz[peak_index], frequency)
    assert np.isclose(spectrum[peak_index], amplitude + 0j, rtol=1e-3, atol=1e-3)


def test_power_spectrum_sum_matches_mean_square() -> None:
    rng = np.random.default_rng(seed=42)
    samples = rng.normal(size=4096)

    _, power = power_spectrum(samples, sample_rate_hz=3_200_000.0, scaling="power", center=False)
    assert np.isclose(np.sum(power), np.mean(samples**2), rtol=2e-3, atol=1e-8)


def test_autocorrelation_zero_lag_matches_signal_power() -> None:
    rng = np.random.default_rng(seed=123)
    samples = rng.normal(size=2048)

    lag_s, acf = autocorrelation(
        samples,
        sample_rate_hz=2_000_000.0,
    )
    zero_index = int(np.where(np.isclose(lag_s, 0.0))[0][0])

    assert np.isclose(acf[zero_index], np.mean(samples**2), rtol=2e-3, atol=1e-8)


def test_autocorrelation_from_power_spectrum_zero_lag_matches_signal_power() -> None:
    rng = np.random.default_rng(seed=7)
    samples = rng.normal(size=1024)

    _, power = power_spectrum(samples, sample_rate_hz=1_000_000.0, scaling="power", center=False)
    lag_s, acf = autocorrelation_from_power_spectrum(
        power,
        sample_rate_hz=1_000_000.0,
        centered=False,
        normalize="biased",
    )
    zero_index = int(np.where(np.isclose(lag_s, 0.0))[0][0])
    assert np.isclose(acf[zero_index], np.mean(samples**2), rtol=2e-3, atol=1e-8)


def test_average_power_spectrum_block_count_and_shape() -> None:
    rng = np.random.default_rng(seed=9)
    blocks = rng.normal(size=(8, 512))

    result = average_power_spectrum(
        blocks,
        sample_rate_hz=1_000_000.0,
        window="hann",
    )

    assert result.num_blocks == 8
    assert result.frequency_hz.shape == (512,)
    assert result.mean.shape == (512,)
    assert result.std.shape == (512,)


def test_average_power_spectrum_requires_2d_blocks() -> None:
    rng = np.random.default_rng(seed=10)
    one_dimensional_samples = rng.normal(size=512)

    with np.testing.assert_raises(ValueError):
        average_power_spectrum(one_dimensional_samples, sample_rate_hz=1_000_000.0)


def test_invalid_fft_backend_raises_value_error() -> None:
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="fft_backend must be 'numpy' or 'ugradio'"):
        voltage_spectrum(
            samples,
            sample_rate_hz=1_000_000.0,
            fft_backend="invalid",  # type: ignore[arg-type]
        )


def test_numpy_and_ugradio_fft_backends_match_for_voltage_spectrum() -> None:
    pytest.importorskip("ugradio.dft")
    rng = np.random.default_rng(seed=11)
    samples = rng.normal(size=256) + 1j * rng.normal(size=256)

    freq_numpy, spec_numpy = voltage_spectrum(
        samples,
        sample_rate_hz=1_000_000.0,
        center=False,
        fft_backend="numpy",
    )
    freq_ugradio, spec_ugradio = voltage_spectrum(
        samples,
        sample_rate_hz=1_000_000.0,
        center=False,
        fft_backend="ugradio",
    )

    assert np.allclose(freq_numpy, freq_ugradio)
    assert np.allclose(spec_numpy, spec_ugradio)


def test_numpy_and_ugradio_fft_backends_match_for_autocorrelation_from_power() -> None:
    pytest.importorskip("ugradio.dft")
    rng = np.random.default_rng(seed=12)
    samples = rng.normal(size=256)

    _, power = power_spectrum(samples, sample_rate_hz=1_000_000.0, center=True)
    lag_numpy, acf_numpy = autocorrelation_from_power_spectrum(
        power,
        sample_rate_hz=1_000_000.0,
        centered=True,
        normalize="biased",
        fft_backend="numpy",
    )
    lag_ugradio, acf_ugradio = autocorrelation_from_power_spectrum(
        power,
        sample_rate_hz=1_000_000.0,
        centered=True,
        normalize="biased",
        fft_backend="ugradio",
    )

    assert np.allclose(lag_numpy, lag_ugradio)
    assert np.allclose(acf_numpy, acf_ugradio)
