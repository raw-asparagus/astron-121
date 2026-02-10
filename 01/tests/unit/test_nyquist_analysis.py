"""Unit tests for analysis.nyquist."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.experiments import predict_alias_frequency


def test_predict_alias_frequency_principal_zone() -> None:
    f_true = np.array([1.2e6, 2.6e6, -0.2e6])
    fs = 1.0e6
    alias = predict_alias_frequency(f_true, fs)

    assert np.all(alias >= -fs / 2.0 - 1e-9)
    assert np.all(alias < fs / 2.0 + 1e-9)
