"""Unit tests for fixed-tier control.acquisition helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ugradio_lab1.control.acquisition import (
    E1AcquisitionConfig,
    E2AcquisitionConfig,
    e1_frequency_grid_hz,
    e1_power_tiers_dbm,
    e2_frequency_grid_hz,
)


def test_e1_frequency_grid_omits_zero_hz_by_default() -> None:
    grid = e1_frequency_grid_hz(1.0e6, n_points=24)
    assert grid.shape == (23,)
    assert grid[0] > 0.0
    assert grid[-1] == pytest.approx(2.0e6)  # 4 * f_Nyquist = 2 * fs


def test_e1_frequency_grid_can_include_zero_hz_when_requested() -> None:
    grid = e1_frequency_grid_hz(1.0e6, n_points=24, include_zero_hz=True)
    assert grid.shape == (24,)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(2.0e6)


def test_e1_power_tiers_match_protocol_defaults() -> None:
    tiers = e1_power_tiers_dbm(E1AcquisitionConfig())
    assert tiers["default"] == (-10.0, 0.0, 10.0)
    assert tiers["alias_hack"] == (-50.0, -40.0, -30.0)


def test_power_tiers_are_configurable() -> None:
    config = E1AcquisitionConfig(
        power_tiers_default_dbm=(-12.0, -2.0, 8.0),
        power_tiers_alias_dbm=(-55.0, -45.0, -35.0),
    )
    tiers = e1_power_tiers_dbm(config)
    assert tiers["default"] == (-12.0, -2.0, 8.0)
    assert tiers["alias_hack"] == (-55.0, -45.0, -35.0)


def test_e2_frequency_grid_is_logspace_with_expected_endpoints() -> None:
    grid = e2_frequency_grid_hz(1.0e6, n_points=50, min_frequency_hz=10_000.0, max_nyquist_multiple=4.0)
    assert grid.shape == (50,)
    assert grid[0] == pytest.approx(10_000.0)
    assert grid[-1] == pytest.approx(2.0e6)  # 4 * f_Nyquist = 2 * fs
    assert np.all(np.diff(grid) > 0.0)
    log_steps = np.diff(np.log10(grid))
    assert np.allclose(log_steps, log_steps[0])


def test_e2_default_contract_matches_requested_setup() -> None:
    config = E2AcquisitionConfig()
    assert config.sample_rates_hz == (1.0e6, 1.6e6, 2.4e6, 3.2e6)
    assert config.n_frequency_points == 50
    assert config.sdr_direct is True
    assert config.fir_coeffs is None
