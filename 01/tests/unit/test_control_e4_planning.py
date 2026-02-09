"""Unit tests for E4 frequency-planning helpers."""

from __future__ import annotations

import pytest

from ugradio_lab1.control.e4_planning import leakage_tone_from_center, resolution_tones_from_center


def test_leakage_tone_from_center_uses_requested_fixed_bin() -> None:
    tone_hz, k_eff, bin_center_hz = leakage_tone_from_center(
        sample_rate_hz=1_000_000.0,
        n_samples=2048,
        center_frequency_hz=125_000.0,
        bin_offset=0.35,
        bin_index=128,
    )
    delta = 1_000_000.0 / 2048.0
    assert k_eff == 128
    assert bin_center_hz == pytest.approx(128.0 * delta)
    assert tone_hz == pytest.approx((128.0 + 0.35) * delta)


def test_leakage_tone_from_center_auto_bin_nearest_to_center() -> None:
    tone_hz, k_eff, _ = leakage_tone_from_center(
        sample_rate_hz=1_000_000.0,
        n_samples=2000,
        center_frequency_hz=100_000.0,
        bin_offset=0.0,
        bin_index=None,
    )
    expected_k = int(round(100_000.0 / (1_000_000.0 / 2000.0)))
    assert k_eff == expected_k
    assert tone_hz == pytest.approx(expected_k * (1_000_000.0 / 2000.0))


def test_leakage_tone_from_center_rejects_bins_that_conflict_with_offset() -> None:
    with pytest.raises(ValueError, match="bin_index"):
        leakage_tone_from_center(
            sample_rate_hz=1_000_000.0,
            n_samples=1024,
            center_frequency_hz=490_000.0,
            bin_offset=1.0,
            bin_index=511,
        )


def test_resolution_tones_from_center_are_symmetric() -> None:
    f1_hz, f2_hz = resolution_tones_from_center(
        center_frequency_hz=125_000.0,
        delta_f_hz=2_000.0,
    )
    assert f1_hz == pytest.approx(124_000.0)
    assert f2_hz == pytest.approx(126_000.0)


def test_resolution_tones_from_center_requires_positive_lower_tone() -> None:
    with pytest.raises(ValueError, match="too low"):
        resolution_tones_from_center(
            center_frequency_hz=500.0,
            delta_f_hz=2_000.0,
        )
