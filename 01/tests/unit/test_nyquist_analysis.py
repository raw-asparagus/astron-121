"""Unit tests for analysis.nyquist."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.nyquist import alias_residual_table, predict_alias_frequency


def test_predict_alias_frequency_principal_zone() -> None:
    f_true = np.array([1.2e6, 2.6e6, -0.2e6])
    fs = 1.0e6
    alias = predict_alias_frequency(f_true, fs)

    assert np.all(alias >= -fs / 2.0 - 1e-9)
    assert np.all(alias < fs / 2.0 + 1e-9)


def test_alias_residual_table_columns_and_residual() -> None:
    f_true = np.array([1.2e6, 2.6e6])
    fs = np.array([1.0e6, 1.0e6])
    predicted = predict_alias_frequency(f_true, fs)
    measured = predicted + np.array([100.0, -200.0])

    table = alias_residual_table(
        f_true,
        fs,
        measured,
        predicted_alias_hz=predicted,
        run_id=["r1", "r2"],
        uncertainty_hz=np.array([50.0, 50.0]),
    )

    assert {"run_id", "predicted_alias_hz", "measured_alias_hz", "residual_hz"}.issubset(table.columns)
    assert np.allclose(table["residual_hz"].to_numpy(), np.array([100.0, -200.0]))
