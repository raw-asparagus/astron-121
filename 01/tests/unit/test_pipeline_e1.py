"""Unit tests for the E1 raw-data pipeline."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from ugradio_lab1.dataio.io_npz import save_npz_dataset
from ugradio_lab1.pipeline.e1 import (
    build_e1_qc_catalog,
    build_e1_run_catalog,
    build_e1_t2_table,
    build_e1_t3_table,
)


def _tone_blocks(
    *,
    frequency_hz: float,
    sample_rate_hz: float,
    nsamples: int = 2048,
    nblocks: int = 5,
    amplitude: float = 40.0,
) -> np.ndarray:
    sample_index = np.arange(nsamples, dtype=float)
    block = amplitude * np.cos(2.0 * np.pi * frequency_hz * sample_index / sample_rate_hz)
    block = np.round(block).astype(np.int8)
    return np.tile(block[np.newaxis, :], (nblocks, 1))


def _write_e1_npz(
    path: Path,
    *,
    run_id: str,
    frequency_hz: float,
    sample_rate_hz: float,
    fir_mode: str,
    power_tier_dbm: float,
    status: str,
    passes_guard: bool,
) -> None:
    blocks = _tone_blocks(frequency_hz=frequency_hz, sample_rate_hz=sample_rate_hz)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": "E1",
        "timestamp_utc": "2026-02-09T00:00:00+00:00",
        "status": status,
        "run_kind": "power_tier_capture",
        "combo": {
            "sample_rate_hz": float(sample_rate_hz),
            "signal_frequency_hz": float(frequency_hz),
            "signal_frequency_measured_hz": float(frequency_hz),
            "fir_mode": fir_mode,
            "fir_coeffs": None if fir_mode == "default" else [0] * 15 + [2047],
            "power_tier_dbm": float(power_tier_dbm),
        },
        "capture_settings": {
            "device_index": 0,
            "direct": True,
            "gain_db": 0.0,
            "nsamples": 2048,
            "nblocks_requested": 6,
            "stale_blocks_dropped": 1,
            "nblocks_saved": 5,
        },
        "signal_generator": {
            "device_path": "/dev/usbtmc0",
            "power_requested_dbm": float(power_tier_dbm),
            "power_measured_dbm": float(power_tier_dbm),
            "settle_time_s": 1.0,
        },
        "sample_rate": {
            "requested_hz": float(sample_rate_hz),
            "actual_hz": float(sample_rate_hz),
            "error_hz": 0.0,
        },
        "adc_summary": {
            "mean_block_rms": float(np.mean(block_rms)),
            "adc_max": int(np.max(blocks)),
            "adc_min": int(np.min(blocks)),
            "is_clipped": bool(np.max(blocks) >= 127 or np.min(blocks) <= -128),
            "passes_guard": bool(passes_guard),
            "guard_attempts": 1,
            "rejected_attempts": [],
        },
    }
    save_npz_dataset(
        path,
        arrays={"adc_counts": blocks, "block_rms": block_rms},
        metadata=metadata,
    )


def _build_tar_source(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "e1"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _write_e1_npz(
        raw_dir / "e1_sr2048_f1300_idx00_fir_default__pp10p0.npz",
        run_id="e1_sr2048_f1300_idx00_fir_default__pp10p0",
        frequency_hz=1300.0,
        sample_rate_hz=2048.0,
        fir_mode="default",
        power_tier_dbm=10.0,
        status="captured",
        passes_guard=True,
    )
    _write_e1_npz(
        raw_dir / "e1_sr2048_f900_idx01_fir_alias_hack__pm50p0.npz",
        run_id="e1_sr2048_f900_idx01_fir_alias_hack__pm50p0",
        frequency_hz=900.0,
        sample_rate_hz=2048.0,
        fir_mode="alias_hack",
        power_tier_dbm=-50.0,
        status="captured_guard_fail",
        passes_guard=False,
    )

    archive_path = tmp_path / "e1.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for npz_path in sorted(raw_dir.glob("*.npz")):
            archive.add(npz_path, arcname=f"e1/{npz_path.name}")
    return archive_path


def test_build_e1_run_catalog_from_tar(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e1_run_catalog(archive_path)

    assert catalog.shape[0] == 2
    assert set(catalog["source_kind"].astype(str)) == {"tar"}
    assert set(catalog["run_kind"].astype(str)) == {"power_tier_capture"}
    assert set(catalog["nblocks_saved"].astype(int)) == {5}


def test_build_e1_qc_and_t2_tables(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e1_run_catalog(archive_path)
    qc = build_e1_qc_catalog(catalog)
    t2 = build_e1_t2_table(qc)

    assert qc.shape[0] == 2
    assert int(qc["qc_analysis_pass"].fillna(False).sum()) == 2
    assert int(qc["qc_recommended_pass"].fillna(False).sum()) == 1
    assert t2.shape[0] == 2
    assert "run_id" in t2.columns
    assert "source_power_dbm" in t2.columns


def test_build_e1_t3_table_estimates_alias(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e1_run_catalog(archive_path)
    qc = build_e1_qc_catalog(catalog)
    t3 = build_e1_t3_table(qc)

    row = t3.loc[t3["run_id"] == "e1_sr2048_f1300_idx00_fir_default__pp10p0"].iloc[0]
    assert np.isclose(float(row["predicted_alias_hz"]), -748.0, atol=0.5)
    assert np.isclose(float(row["measured_alias_hz"]), -748.0, atol=1.0)
