"""Unit tests for analysis.resolution."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.resolution import resolution_vs_n


def test_resolution_vs_n_computes_delta_f_bin() -> None:
    records = [
        {"run_id": "r1", "n_samples": 256, "sample_rate_hz": 1024.0, "measured_delta_f_hz": 5.0},
        {"run_id": "r2", "n_samples": 512, "sample_rate_hz": 1024.0, "measured_delta_f_hz": 3.0},
    ]

    summary = resolution_vs_n(records)

    assert np.allclose(summary["delta_f_bin_hz"].to_numpy(), np.array([4.0, 2.0]))
    assert np.allclose(summary["resolution_ratio"].to_numpy(), np.array([1.25, 1.5]))
