"""Unit tests for analysis.leakage."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.leakage import (
    leakage_metric,
    leakage_resolution_table,
    nyquist_window_extension,
)


def test_leakage_metric_computes_fraction() -> None:
    frequency = np.linspace(-4.0, 3.0, 8)
    power = np.array([0.01, 0.01, 0.02, 10.0, 0.02, 0.01, 0.01, 0.01])

    metrics = leakage_metric(
        frequency,
        power,
        tone_frequency_hz=frequency[3],
        main_lobe_half_width_bins=0,
    )

    assert metrics["main_bin_index"] == 3.0
    assert 0.0 < metrics["leakage_fraction"] < 0.01


def test_leakage_resolution_table_t5_columns() -> None:
    records = [
        {
            "run_id": "r1",
            "n_samples": 256,
            "sample_rate_hz": 1024.0,
            "leakage_metric": 0.1,
            "measured_delta_f_hz": 5.0,
        },
        {
            "run_id": "r2",
            "n_samples": 512,
            "sample_rate_hz": 1024.0,
            "leakage_metric": 0.05,
            "measured_delta_f_hz": 2.5,
        },
    ]

    table = leakage_resolution_table(records)

    assert list(table.columns) == [
        "run_id",
        "n_samples",
        "delta_f_bin_hz",
        "leakage_metric",
        "min_resolvable_delta_f_hz",
    ]
    assert np.allclose(table["delta_f_bin_hz"].to_numpy(), np.array([4.0, 2.0]))


def test_nyquist_window_extension_replicates_windows() -> None:
    frequency = np.array([-1.0, 0.0, 1.0])
    power = np.array([1.0, 2.0, 1.0])

    extended = nyquist_window_extension(
        frequency,
        power,
        sample_rate_hz=4.0,
        window_indices=(-1, 0, 1),
    )

    assert set(extended["window_index"].unique()) == {-1, 0, 1}
    assert extended.shape[0] == 9
