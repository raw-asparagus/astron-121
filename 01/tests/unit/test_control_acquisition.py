"""Unit tests for control.acquisition target-selection logic."""

from __future__ import annotations

import numpy as np
import pytest

import ugradio_lab1.control.acquisition as acquisition
from ugradio_lab1.control.sdr import ADCSummary, SDRCaptureResult


class _DummySigGen:
    def set_ampl_dbm(self, power_dbm: float) -> None:
        del power_dbm
        return None


def _measurement(
    *,
    rms: float,
    clipped: bool = False,
    guard: bool = True,
    power_dbm: float = -30.0,
) -> acquisition.CaptureMeasurement:
    blocks = np.zeros((10, 8), dtype=np.int8)
    summary = ADCSummary(
        mean_block_rms=float(rms),
        adc_max=127 if clipped else 64,
        adc_min=-128 if clipped else -64,
        is_clipped=bool(clipped),
        passes_guard=bool(guard),
    )
    capture = SDRCaptureResult(
        blocks=blocks,
        summary=summary,
        requested_sample_rate_hz=1.0e6,
        actual_sample_rate_hz=1.0e6,
    )
    return acquisition.CaptureMeasurement(
        requested_power_dbm=float(power_dbm),
        measured_power_dbm=float(power_dbm),
        capture=capture,
        guard_attempts=1,
        rejected_attempts=tuple(),
    )


def test_e1_frequency_grid_is_linear_and_inclusive() -> None:
    grid = acquisition.e1_frequency_grid_hz(1.0e6, n_points=24)
    assert grid.shape == (24,)
    assert grid[0] == pytest.approx(0.0)
    assert grid[-1] == pytest.approx(2.0e6)  # 4 * f_Nyquist = 2 * fs


def test_select_target_returns_unachievable_on_baseline_clip() -> None:
    baseline = _measurement(rms=80.0, clipped=True, power_dbm=-30.0)
    target, status, message = acquisition._select_target_measurement(
        baseline=baseline,
        config=acquisition.E1AcquisitionConfig(),
        controller=_DummySigGen(),
        sdr_factory=None,
        sample_rate_hz=1.0e6,
        signal_frequency_hz=100e3,
        measured_frequency_hz=100e3,
        fir_mode="default",
        fir_coeffs=None,
    )
    assert target is baseline
    assert status == "unachievable"
    assert "clipping" in message


def test_select_target_reuses_baseline_when_already_in_band() -> None:
    baseline = _measurement(rms=65.0, clipped=False, power_dbm=-30.0)
    target, status, message = acquisition._select_target_measurement(
        baseline=baseline,
        config=acquisition.E1AcquisitionConfig(),
        controller=_DummySigGen(),
        sdr_factory=None,
        sample_rate_hz=1.0e6,
        signal_frequency_hz=100e3,
        measured_frequency_hz=100e3,
        fir_mode="default",
        fir_coeffs=None,
    )
    assert target is baseline
    assert status == "ok_target_from_baseline"
    assert "baseline_within_target_band" in message


def test_select_target_returns_closest_when_max_power_still_too_weak(monkeypatch) -> None:
    baseline = _measurement(rms=20.0, clipped=False, power_dbm=-30.0)
    upper = _measurement(rms=40.0, clipped=False, power_dbm=10.0)

    def _fake_capture_measurement(**kwargs):
        power = float(kwargs["requested_power_dbm"])
        if abs(power - 10.0) < 1e-6:
            return upper
        raise AssertionError(f"Unexpected requested_power_dbm={power!r}")

    monkeypatch.setattr(acquisition, "_capture_measurement", _fake_capture_measurement)
    target, status, message = acquisition._select_target_measurement(
        baseline=baseline,
        config=acquisition.E1AcquisitionConfig(),
        controller=_DummySigGen(),
        sdr_factory=None,
        sample_rate_hz=1.0e6,
        signal_frequency_hz=100e3,
        measured_frequency_hz=100e3,
        fir_mode="default",
        fir_coeffs=None,
    )
    assert target is upper
    assert status == "closest_only"
    assert "max_power" in message


def test_select_target_finds_midpoint_by_bisection(monkeypatch) -> None:
    baseline = _measurement(rms=20.0, clipped=False, power_dbm=-30.0)
    upper = _measurement(rms=90.0, clipped=False, power_dbm=10.0)
    mid = _measurement(rms=66.0, clipped=False, power_dbm=-10.0)

    def _fake_capture_measurement(**kwargs):
        power = float(kwargs["requested_power_dbm"])
        if abs(power - 10.0) < 1e-6:
            return upper
        if abs(power + 10.0) < 1e-6:
            return mid
        raise AssertionError(f"Unexpected requested_power_dbm={power!r}")

    monkeypatch.setattr(acquisition, "_capture_measurement", _fake_capture_measurement)
    target, status, message = acquisition._select_target_measurement(
        baseline=baseline,
        config=acquisition.E1AcquisitionConfig(),
        controller=_DummySigGen(),
        sdr_factory=None,
        sample_rate_hz=1.0e6,
        signal_frequency_hz=100e3,
        measured_frequency_hz=100e3,
        fir_mode="default",
        fir_coeffs=None,
    )
    assert target is mid
    assert status == "ok_target"
    assert "bisection" in message
