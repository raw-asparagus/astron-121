"""Unit tests for fixed-tier control.acquisition helpers."""

from __future__ import annotations

import pytest

from ugradio_lab1.control.acquisition import (
    E1AcquisitionConfig,
    e1_frequency_grid_hz,
    e1_power_tiers_dbm,
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
