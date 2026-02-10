"""Unit tests for analysis.noise."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.experiments import radiometer_fit


def test_radiometer_fit_recovers_expected_slope() -> None:
    n_avg = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    sigma = 2.0 / np.sqrt(n_avg)

    fit = radiometer_fit(n_avg, sigma)

    assert np.isclose(fit["slope"], -0.5, atol=1e-6)
    assert np.isclose(fit["intercept"], np.log10(2.0), atol=1e-6)
    assert fit["r_squared"] > 0.9999
