"""Unit tests for the E2 raw-data pipeline."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from ugradio_lab1.dataio.io_npz import save_npz_dataset
from ugradio_lab1.pipeline.e2 import (
    build_e2_bandpass_curve_table,
    build_e2_qc_catalog,
    build_e2_run_catalog,
    build_e2_t2_table,
    build_e2_t4_table,
    write_e2_bandpass_figure,
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
    block = np.clip(np.round(block), -128, 127).astype(np.int8)
    return np.tile(block[np.newaxis, :], (nblocks, 1))


def _write_e2_npz(
    path: Path,
    *,
    run_id: str,
    frequency_hz: float,
    sample_rate_hz: float,
    power_dbm: float,
    status: str,
    passes_guard: bool,
    clipped: bool,
    amplitude: float,
) -> None:
    blocks = _tone_blocks(
        frequency_hz=frequency_hz,
        sample_rate_hz=sample_rate_hz,
        amplitude=amplitude,
    )
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": "E2",
        "timestamp_utc": "2026-02-09T00:00:00+00:00",
        "status": status,
        "run_kind": "bandpass_sweep_capture",
        "combo": {
            "sample_rate_hz": float(sample_rate_hz),
            "signal_frequency_hz": float(frequency_hz),
            "signal_frequency_measured_hz": float(frequency_hz),
            "fir_mode": "default",
            "fir_coeffs": None,
            "power_tier_dbm": float(power_dbm),
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
            "power_requested_dbm": float(power_dbm),
            "power_measured_dbm": float(power_dbm),
            "settle_time_s": 1.0,
        },
        "sample_rate": {
            "requested_hz": float(sample_rate_hz),
            "actual_hz": float(sample_rate_hz + 0.025),
            "error_hz": 0.025,
        },
        "adc_summary": {
            "mean_block_rms": float(np.mean(block_rms)),
            "adc_max": int(np.max(blocks)),
            "adc_min": int(np.min(blocks)),
            "is_clipped": bool(clipped),
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
    raw_dir = tmp_path / "e2"
    raw_dir.mkdir(parents=True, exist_ok=True)

    sample_rates = [1.0e6, 1.6e6]
    frequencies = [1.0e4, 2.0e4, 4.0e4, 8.0e4]

    for sample_rate_hz in sample_rates:
        for idx, frequency_hz in enumerate(frequencies):
            amplitude = 60.0 * (1.0 - 0.45 * (idx / (len(frequencies) - 1)))
            run_id = f"e2_sr{int(sample_rate_hz)}_f{int(frequency_hz)}_idx{idx:02d}_fir_default__pm30p0"
            _write_e2_npz(
                raw_dir / f"{run_id}.npz",
                run_id=run_id,
                frequency_hz=frequency_hz,
                sample_rate_hz=sample_rate_hz,
                power_dbm=-30.0,
                status="captured",
                passes_guard=True,
                clipped=False,
                amplitude=amplitude,
            )

    # Duplicate combo with an alternate power tier; this one is clipped and should not be selected.
    _write_e2_npz(
        raw_dir / "e2_sr1000000_f10000_idx00_fir_default__pm10p0.npz",
        run_id="e2_sr1000000_f10000_idx00_fir_default__pm10p0",
        frequency_hz=1.0e4,
        sample_rate_hz=1.0e6,
        power_dbm=-10.0,
        status="captured_guard_fail",
        passes_guard=False,
        clipped=True,
        amplitude=125.0,
    )

    archive_path = tmp_path / "e2.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for npz_path in sorted(raw_dir.glob("*.npz")):
            archive.add(npz_path, arcname=f"e2/{npz_path.name}")
    return archive_path


def test_build_e2_run_catalog_from_tar(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e2_run_catalog(archive_path)

    assert catalog.shape[0] == 9
    assert set(catalog["source_kind"].astype(str)) == {"tar"}
    assert set(catalog["run_kind"].astype(str)) == {"bandpass_sweep_capture"}
    assert set(catalog["nblocks_saved"].astype(int)) == {5}


def test_build_e2_qc_and_t2_tables(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e2_run_catalog(archive_path)
    qc = build_e2_qc_catalog(catalog)
    t2 = build_e2_t2_table(qc)

    assert qc.shape[0] == 9
    assert int(qc["qc_analysis_pass"].fillna(False).sum()) == 9
    assert int(qc["qc_recommended_pass"].fillna(False).sum()) == 8
    assert t2.shape[0] == 9
    assert "run_id" in t2.columns
    assert "source_power_dbm" in t2.columns


def test_build_e2_curve_t4_and_figure(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e2_run_catalog(archive_path)
    qc = build_e2_qc_catalog(catalog)

    curve_table = build_e2_bandpass_curve_table(qc, preferred_power_dbm=-10.0)
    # Duplicate combo should pick the non-clipped -30 dBm run.
    duplicate_row = curve_table.loc[
        np.isclose(curve_table["sample_rate_hz_nominal"], 1.0e6)
        & np.isclose(curve_table["frequency_hz"], 1.0e4)
    ].iloc[0]
    assert np.isclose(float(duplicate_row["power_tier_dbm"]), -30.0)

    assert curve_table["curve_label"].nunique() == 2
    assert curve_table.shape[0] == 8

    t4 = build_e2_t4_table(curve_table)
    assert t4.shape[0] == 2
    assert "mode" in t4.columns
    assert "passband_estimate_hz" in t4.columns

    figure_path = tmp_path / "F4_physical_test.png"
    saved = write_e2_bandpass_figure(curve_table, figure_path)
    assert saved == figure_path
    assert figure_path.exists()
