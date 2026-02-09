"""Unit tests for dataio.catalog."""

from __future__ import annotations

import pandas as pd

from ugradio_lab1.dataio.catalog import (
    append_manifest_rows,
    build_goal_coverage_table,
    build_manifest,
    filter_manifest,
    read_manifest_csv,
)


def _t2_rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "run_002",
            "experiment": "E1",
            "sample_rate_hz": 1_000_000.0,
            "center_frequency_hz": 0.0,
            "tones_hz": "1.2e6",
            "source_power_dbm": -30.0,
            "mixer_config": "none",
            "cable_config": "direct",
            "n_samples": 2048,
        },
        {
            "run_id": "run_001",
            "experiment": "E1",
            "sample_rate_hz": 1_000_000.0,
            "center_frequency_hz": 0.0,
            "tones_hz": "0.8e6",
            "source_power_dbm": -30.0,
            "mixer_config": "none",
            "cable_config": "direct",
            "n_samples": 2048,
        },
    ]


def test_build_manifest_sorts_t2_by_run_id() -> None:
    manifest = build_manifest(_t2_rows(), table_id="T2")
    assert list(manifest["run_id"]) == ["run_001", "run_002"]


def test_append_manifest_rows_and_read_csv(tmp_path) -> None:
    path = tmp_path / "manifest.csv"
    combined = append_manifest_rows(path, _t2_rows(), table_id="T2")
    reloaded = read_manifest_csv(path, table_id="T2")

    assert isinstance(combined, pd.DataFrame)
    assert reloaded.shape[0] == 2


def test_filter_manifest_membership() -> None:
    manifest = build_manifest(_t2_rows(), table_id="T2")
    filtered = filter_manifest(manifest, run_id={"run_001"})
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]["run_id"] == "run_001"


def test_build_goal_coverage_table_t8_shape() -> None:
    table = build_goal_coverage_table(
        ["G1", "G2"],
        evidence_items={"G1": ["F2", "T3"], "G2": "F13"},
        status={"G1": "pass", "G2": "partial"},
    )
    assert list(table.columns) == ["goal_id", "evidence_items", "status", "limitations"]
    assert table.shape[0] == 2
