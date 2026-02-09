"""Unit tests for analysis.noise."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.noise import radiometer_fit, radiometer_summary_table


def test_radiometer_fit_recovers_expected_slope() -> None:
    n_avg = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    sigma = 2.0 / np.sqrt(n_avg)

    fit = radiometer_fit(n_avg, sigma)

    assert np.isclose(fit["slope"], -0.5, atol=1e-6)
    assert np.isclose(fit["intercept"], np.log10(2.0), atol=1e-6)
    assert fit["r_squared"] > 0.9999


def test_radiometer_summary_table_t6_columns() -> None:
    n_avg = np.array([1, 2, 4, 8, 16], dtype=float)
    sigma = 1.0 / np.sqrt(n_avg)
    table = radiometer_summary_table(n_avg, sigma, block_size=1024)

    assert list(table.columns) == [
        "block_size",
        "n_avg",
        "sigma_power",
        "fitted_slope",
        "expected_slope",
        "chi2_dof",
    ]
    assert table.shape[0] == n_avg.size
