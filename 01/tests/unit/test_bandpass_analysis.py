"""Unit tests for analysis.bandpass."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.bandpass import (
    bandpass_curve,
    bandpass_summary_metrics,
    bandpass_summary_table,
)


def test_bandpass_curve_gain_reference_default() -> None:
    freq = np.array([1.0, 2.0, 3.0])
    amp = np.array([2.0, 1.0, 0.5])

    curve = bandpass_curve(freq, amp)

    assert np.isclose(curve["gain_linear"].max(), 1.0)
    assert np.isclose(curve["gain_db"].iloc[0], 0.0)


def test_bandpass_summary_metrics_basic_fields() -> None:
    freq = np.linspace(100.0, 500.0, 9)
    gain = np.array([-20.0, -8.0, -3.0, -1.0, 0.0, -1.0, -3.0, -8.0, -20.0])

    metrics = bandpass_summary_metrics(freq, gain)

    assert "passband_width_hz" in metrics
    assert "passband_ripple_db" in metrics
    assert metrics["peak_gain_db"] == 0.0
    assert metrics["passband_width_hz"] > 0.0


def test_bandpass_summary_table_has_t4_columns() -> None:
    freq = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    summary = bandpass_summary_table(
        freq,
        {
            "default": np.array([-6.0, -1.0, 0.0, -1.5, -7.0]),
            "fir": np.array([-8.0, -2.0, 0.0, -2.5, -9.0]),
        },
    )

    assert {
        "mode",
        "passband_estimate_hz",
        "rolloff_metric_db_per_hz",
        "ripple_db",
        "fit_residuals_db",
    }.issubset(summary.columns)
    assert summary.shape[0] == 2
