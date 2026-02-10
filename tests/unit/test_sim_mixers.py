"""Unit tests for sim.mixers."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.spectra import power_spectrum
from ugradio_lab1.sim.mixers import (
    simulate_dsb_output,
    simulate_dsb_spur_survey,
    simulate_r820t_vs_external,
    simulate_ssb_iq,
)


def test_simulate_dsb_output_contains_sum_and_difference_lines() -> None:
    fs = 1_000_000.0
    voltage = simulate_dsb_output(
        f_lo_hz=100_000.0,
        f_rf_hz=130_000.0,
        sample_rate_hz=fs,
        n_samples=8192,
        noise_std_v=0.0,
    )
    frequency_hz, power_v2 = power_spectrum(
        voltage,
        sample_rate_hz=fs,
        scaling="power",
        center=True,
    )

    def _power_at(target_hz: float) -> float:
        return float(power_v2[int(np.argmin(np.abs(frequency_hz - target_hz)))])

    floor = float(np.median(power_v2))
    assert _power_at(30_000.0) > 10.0 * floor
    assert _power_at(230_000.0) > 10.0 * floor


def test_simulate_ssb_iq_upper_lower_sign_relation() -> None:
    i_upper, q_upper = simulate_ssb_iq(
        delta_f_hz=20_000.0,
        sample_rate_hz=1_000_000.0,
        n_samples=1024,
        sideband="upper",
        noise_std_v=0.0,
    )
    i_lower, q_lower = simulate_ssb_iq(
        delta_f_hz=20_000.0,
        sample_rate_hz=1_000_000.0,
        n_samples=1024,
        sideband="lower",
        noise_std_v=0.0,
    )
    assert np.allclose(i_upper, i_lower)
    assert np.allclose(q_upper, -q_lower)


def test_simulate_dsb_spur_survey_returns_catalog() -> None:
    simulation = simulate_dsb_spur_survey(
        f_lo_hz=100_000.0,
        f_rf_hz=110_000.0,
        sample_rate_hz=1_000_000.0,
        n_samples=8192,
        tolerance_hz=350.0,
        n_observed_lines=12,
    )
    assert simulation.catalog.shape[0] > 0
    assert {"expected_line_hz", "observed_line_hz", "interpretation"}.issubset(simulation.catalog.columns)
    assert np.any(simulation.catalog["interpretation"] == "expected_dsb_product")


def test_simulate_r820t_vs_external_shapes_match() -> None:
    result = simulate_r820t_vs_external(
        f_lo_hz=100_000.0,
        f_rf_hz=117_500.0,
        sample_rate_hz=1_000_000.0,
        n_samples=4096,
    )
    assert result["frequency_hz"].shape == result["external_power_v2"].shape
    assert result["frequency_hz"].shape == result["r820t_power_v2"].shape
