"""Unit tests for sim.nyquist."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.leakage import leakage_metric
from ugradio_lab1.sim.nyquist import (
    estimate_peak_frequency,
    generate_tone,
    simulate_alias_sweep,
    simulate_bandpass_sweep,
    simulate_leakage_experiment,
    simulate_multi_window_spectrum,
    simulate_resolution_sweep,
)


def test_generate_tone_and_peak_estimation() -> None:
    fs = 1_000_000.0
    n = 4096
    k = 321
    f_tone = k * fs / n

    voltage = generate_tone(
        frequency_hz=f_tone,
        sample_rate_hz=fs,
        n_samples=n,
        amplitude_v=2.0,
        complex_output=True,
    )
    peak_hz = estimate_peak_frequency(voltage, sample_rate_hz=fs)
    assert np.isclose(peak_hz, f_tone)


def test_simulate_alias_sweep_residual_small_for_bin_centered_tones() -> None:
    fs = 1_000_000.0
    n = 4096
    f_true = np.array([300, 2500], dtype=float) * fs / n

    simulation = simulate_alias_sweep(
        f_true,
        fs,
        n_samples=n,
        amplitude_v=1.0,
        noise_std_v=0.0,
        complex_output=True,
    )

    residual = simulation.table["residual_hz"].to_numpy(dtype=float)
    assert residual.shape == (2,)
    assert np.all(np.abs(residual) <= (fs / n) * 0.51)


def test_simulate_bandpass_sweep_contains_expected_columns() -> None:
    frequency = np.linspace(-300_000.0, 300_000.0, 61)
    table = simulate_bandpass_sweep(
        frequency,
        sample_rate_hz=1_000_000.0,
        n_samples=4096,
        source_amplitude_v=1.0,
        center_hz=0.0,
        passband_hz=180_000.0,
        order=4,
    )

    assert {"gain_db", "mode", "model_gain_linear", "model_gain_db"}.issubset(table.columns)
    peak_idx = int(np.argmax(table["gain_linear"].to_numpy()))
    assert abs(float(table["frequency_hz"].iloc[peak_idx])) < 10_000.0


def test_simulate_leakage_experiment_has_higher_off_bin_leakage() -> None:
    simulation = simulate_leakage_experiment(
        sample_rate_hz=1_000_000.0,
        n_samples=4096,
        bin_index=250,
        bin_offset=0.33,
    )

    centered = leakage_metric(
        simulation.frequency_hz,
        simulation.bin_centered_power_v2,
        tone_frequency_hz=simulation.bin_centered_frequency_hz,
        main_lobe_half_width_bins=1,
    )
    off_bin = leakage_metric(
        simulation.frequency_hz,
        simulation.off_bin_power_v2,
        tone_frequency_hz=simulation.off_bin_frequency_hz,
        main_lobe_half_width_bins=1,
    )
    assert off_bin["leakage_fraction"] > centered["leakage_fraction"]


def test_simulate_resolution_sweep_tracks_delta_f_bin() -> None:
    simulation = simulate_resolution_sweep(
        sample_rate_hz=1_000_000.0,
        n_samples_values=np.array([512, 1024, 2048]),
        delta_f_hz_candidates=np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0]),
        tone_center_hz=120_000.0,
        min_peak_prominence_db=3.0,
        min_valley_depth_db=0.25,
    )
    table = simulation.table

    assert np.allclose(table["delta_f_bin_hz"].to_numpy(), 1_000_000.0 / table["n_samples"].to_numpy())
    min_resolvable = table["min_resolvable_delta_f_hz"].to_numpy(dtype=float)
    finite = min_resolvable[np.isfinite(min_resolvable)]
    assert finite.size >= 2
    assert finite[-1] <= finite[0]


def test_simulate_multi_window_spectrum_outputs_three_windows() -> None:
    voltage = generate_tone(
        frequency_hz=80_000.0,
        sample_rate_hz=1_000_000.0,
        n_samples=2048,
        complex_output=True,
    )
    table = simulate_multi_window_spectrum(
        voltage,
        sample_rate_hz=1_000_000.0,
        window_indices=(-1, 0, 1),
    )

    assert set(table["window_index"].unique()) == {-1, 0, 1}
