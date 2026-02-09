"""Unit tests for analysis.mixers."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.mixers import (
    expected_dsb_lines,
    line_spur_catalog,
    match_observed_lines,
)


def test_expected_dsb_lines_contains_primary_products() -> None:
    lines = expected_dsb_lines(100.0, 110.0, orders=2)
    expected = lines["expected_hz"].to_numpy()

    assert np.any(np.isclose(expected, 10.0))
    assert np.any(np.isclose(expected, 210.0))


def test_match_observed_lines_with_tolerance() -> None:
    expected = np.array([10.0, 20.0, 30.0])
    observed = np.array([10.2, 29.7, 100.0])

    matched = match_observed_lines(expected, observed, tolerance_hz=0.5)

    expected_rows = matched[matched["family"] == "expected"]
    assert expected_rows["matched"].sum() == 2


def test_line_spur_catalog_has_t7_columns() -> None:
    observed_hz = np.array([10.0, 20.1, 210.0, 300.0])
    observed_level_db = np.array([-10.0, -12.0, -8.0, -30.0])
    catalog = line_spur_catalog(
        config="dsb",
        f_lo_hz=100.0,
        f_rf_hz=110.0,
        observed_hz=observed_hz,
        observed_level_db=observed_level_db,
        tolerance_hz=0.2,
        orders=2,
    )

    assert {"config", "f_lo_hz", "f_rf_hz", "expected_line_hz", "observed_line_hz", "level_db"}.issubset(
        catalog.columns
    )
    assert np.any(catalog["interpretation"] == "expected_dsb_product")
