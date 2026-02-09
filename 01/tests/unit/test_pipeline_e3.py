"""Unit tests for the E3 raw-data pipeline."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from ugradio_lab1.dataio.io_npz import save_npz_dataset
from ugradio_lab1.pipeline.e3 import (
    build_e3_spectrum_profiles,
    build_e3_qc_catalog,
    build_e3_run_catalog,
    build_e3_spectrum_profile,
    build_e3_t2_table,
    write_e3_voltage_components_figure,
    write_e3_voltage_power_figure,
)


def _tone_blocks(
    *,
    frequencies_hz: tuple[float, ...],
    sample_rate_hz: float,
    nsamples: int = 1024,
    nblocks: int = 5,
    amplitudes: tuple[float, ...] | None = None,
) -> np.ndarray:
    if amplitudes is None:
        amplitudes = tuple(40.0 for _ in frequencies_hz)
    sample_index = np.arange(nsamples, dtype=float)
    block = np.zeros(nsamples, dtype=float)
    for frequency_hz, amplitude in zip(frequencies_hz, amplitudes, strict=True):
        block += amplitude * np.cos(2.0 * np.pi * frequency_hz * sample_index / sample_rate_hz)
    block = np.clip(np.round(block), -128, 127).astype(np.int8)
    return np.tile(block[np.newaxis, :], (nblocks, 1))


def _write_e3_npz(
    path: Path,
    *,
    run_id: str,
    mode: str,
    sample_rate_hz: float,
    frequencies_hz: tuple[float, ...],
    powers_dbm: tuple[float, ...],
    status: str,
    passes_guard: bool,
    clipped: bool,
    timestamp_utc: str,
) -> None:
    blocks = _tone_blocks(
        frequencies_hz=frequencies_hz,
        sample_rate_hz=sample_rate_hz,
    )
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": "E3",
        "timestamp_utc": timestamp_utc,
        "status": status,
        "run_kind": "physical_voltage_power",
        "notes": "unit-test fixture",
        "target_vrms_v": 0.20,
        "signal_generators": [
            {
                "label": f"signal_generator_{idx + 1}",
                "frequency_hz": float(frequency_hz),
                "power_dbm": float(power_dbm),
            }
            for idx, (frequency_hz, power_dbm) in enumerate(zip(frequencies_hz, powers_dbm, strict=True))
        ],
        "signal_setup": "manual_analog",
        "capture_settings": {
            "device_index": 0,
            "direct": True,
            "gain_db": 0.0,
            "nsamples": 1024,
            "nblocks_requested": 6,
            "stale_blocks_dropped": 1,
            "nblocks_saved": 5,
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
            "is_clipped": bool(clipped),
            "passes_guard": bool(passes_guard),
            "guard_attempts": 1,
            "rejected_attempts": [],
        },
        "extra": {"mode": mode},
    }
    save_npz_dataset(
        path,
        arrays={"adc_counts": blocks, "block_rms": block_rms},
        metadata=metadata,
    )


def _build_tar_source(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "e3"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _write_e3_npz(
        raw_dir / "e3_single_good.npz",
        run_id="e3_single_good",
        mode="single_tone",
        sample_rate_hz=1.0e6,
        frequencies_hz=(120_000.0,),
        powers_dbm=(-20.0,),
        status="captured",
        passes_guard=True,
        clipped=False,
        timestamp_utc="2026-02-09T10:00:00+00:00",
    )
    _write_e3_npz(
        raw_dir / "e3_two_tone_guard_fail.npz",
        run_id="e3_two_tone_guard_fail",
        mode="two_tone",
        sample_rate_hz=1.0e6,
        frequencies_hz=(120_000.0, 220_000.0),
        powers_dbm=(-20.0, -18.0),
        status="captured_guard_fail",
        passes_guard=False,
        clipped=True,
        timestamp_utc="2026-02-09T11:00:00+00:00",
    )

    archive_path = tmp_path / "e3.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for npz_path in sorted(raw_dir.glob("*.npz")):
            archive.add(npz_path, arcname=f"e3/{npz_path.name}")
    return archive_path


def test_build_e3_run_catalog_from_tar(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e3_run_catalog(archive_path)

    assert catalog.shape[0] == 2
    assert set(catalog["source_kind"].astype(str)) == {"tar"}
    assert set(catalog["run_kind"].astype(str)) == {"physical_voltage_power"}
    assert set(catalog["nblocks_saved"].astype(int)) == {5}


def test_build_e3_qc_and_t2_tables(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e3_run_catalog(archive_path)
    qc = build_e3_qc_catalog(catalog)
    t2 = build_e3_t2_table(qc)

    assert qc.shape[0] == 2
    assert int(qc["qc_analysis_pass"].fillna(False).sum()) == 2
    assert int(qc["qc_recommended_pass"].fillna(False).sum()) == 1
    assert t2.shape[0] == 2
    assert "run_id" in t2.columns
    assert "source_power_dbm" in t2.columns


def test_build_e3_spectrum_profile_and_figures(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e3_run_catalog(archive_path)
    qc = build_e3_qc_catalog(catalog)

    spectrum = build_e3_spectrum_profile(qc, preferred_mode="single_tone")
    assert spectrum["run_id"].nunique() == 1
    assert str(spectrum["run_id"].iloc[0]) == "e3_single_good"
    assert spectrum.shape[0] > 100

    consistency = np.abs(spectrum["power_consistency_delta_v2"].to_numpy(dtype=float))
    assert np.nanmedian(consistency) < 1e-8

    f5_path = tmp_path / "F5_physical_test.png"
    f6_path = tmp_path / "F6_physical_test.png"
    saved_f5 = write_e3_voltage_components_figure(spectrum, f5_path)
    saved_f6 = write_e3_voltage_power_figure(spectrum, f6_path)

    assert saved_f5 == f5_path
    assert saved_f6 == f6_path
    assert f5_path.exists()
    assert f6_path.exists()


def test_build_e3_spectrum_profiles_all_runs(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e3_run_catalog(archive_path)
    qc = build_e3_qc_catalog(catalog)

    profiles = build_e3_spectrum_profiles(qc)
    run_ids = sorted(profiles["run_id"].astype(str).unique().tolist())

    assert run_ids == ["e3_single_good", "e3_two_tone_guard_fail"]
    assert int(profiles.shape[0]) == 2 * 1024
