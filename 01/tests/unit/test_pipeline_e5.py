"""Unit tests for the E5 raw-data pipeline."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np

from ugradio_lab1.dataio.io_npz import save_npz_dataset
from ugradio_lab1.pipeline.e5 import (
    build_e5_noise_stats_table,
    build_e5_qc_catalog,
    build_e5_radiometer_curve_table,
    build_e5_run_catalog,
    build_e5_t2_table,
    build_e5_t6_table,
    fit_e5_radiometer,
    write_e5_acf_consistency_figure,
    write_e5_noise_histogram_figure,
    write_e5_radiometer_figure,
)


def _noise_blocks(
    *,
    std_counts: float,
    nsamples: int = 2048,
    nblocks: int = 5,
    seed: int = 0,
) -> np.ndarray:
    prng = np.random.default_rng(seed)
    blocks = prng.normal(loc=0.0, scale=float(std_counts), size=(nblocks, nsamples))
    return np.clip(np.round(blocks), -128, 127).astype(np.int8)


def _write_e5_npz(
    path: Path,
    *,
    run_id: str,
    sample_rate_hz: float,
    status: str,
    passes_guard: bool,
    clipped: bool,
    noise_source: str,
    target_vrms_v: float,
    std_counts: float,
    seed: int,
) -> None:
    blocks = _noise_blocks(std_counts=std_counts, seed=seed)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))

    metadata = {
        "run_id": run_id,
        "experiment": "E5",
        "timestamp_utc": "2026-02-09T12:00:00+00:00",
        "status": status,
        "run_kind": "physical_noise_acf",
        "notes": "unit-test fixture",
        "target_vrms_v": float(target_vrms_v),
        "signal_generators": [],
        "capture_settings": {
            "device_index": 0,
            "direct": True,
            "gain_db": 0.0,
            "nsamples": 2048,
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
        "extra": {"noise_source": noise_source},
    }

    save_npz_dataset(
        path,
        arrays={"adc_counts": blocks, "block_rms": block_rms},
        metadata=metadata,
    )


def _build_tar_source(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "e5"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _write_e5_npz(
        raw_dir / "e5_lab_00.npz",
        run_id="e5_lab_00",
        sample_rate_hz=3.2e6,
        status="captured_guard_fail",
        passes_guard=False,
        clipped=False,
        noise_source="lab_noise_generator",
        target_vrms_v=0.014,
        std_counts=7.0,
        seed=11,
    )
    _write_e5_npz(
        raw_dir / "e5_lab_01.npz",
        run_id="e5_lab_01",
        sample_rate_hz=3.2e6,
        status="captured_guard_fail",
        passes_guard=False,
        clipped=False,
        noise_source="lab_noise_generator",
        target_vrms_v=0.014,
        std_counts=7.4,
        seed=17,
    )
    _write_e5_npz(
        raw_dir / "e5_terminated_00.npz",
        run_id="e5_terminated_00",
        sample_rate_hz=3.2e6,
        status="captured_guard_fail",
        passes_guard=False,
        clipped=False,
        noise_source="terminated_input",
        target_vrms_v=0.0,
        std_counts=0.6,
        seed=23,
    )

    archive_path = tmp_path / "e5.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for npz_path in sorted(raw_dir.glob("*.npz")):
            archive.add(npz_path, arcname=f"e5/{npz_path.name}")
    return archive_path


def test_build_e5_run_catalog_from_tar(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e5_run_catalog(archive_path)

    assert catalog.shape[0] == 3
    assert set(catalog["source_kind"].astype(str)) == {"tar"}
    assert set(catalog["run_kind"].astype(str)) == {"physical_noise_acf"}
    assert set(catalog["nblocks_saved"].astype(int)) == {5}


def test_build_e5_qc_and_t2_tables(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e5_run_catalog(archive_path)
    qc = build_e5_qc_catalog(catalog)
    t2 = build_e5_t2_table(qc)

    assert qc.shape[0] == 3
    assert int(qc["qc_analysis_pass"].fillna(False).sum()) == 3
    assert int(qc["qc_recommended_pass"].fillna(False).sum()) == 0
    assert t2.shape[0] == 3
    assert "run_id" in t2.columns
    assert "noise_source" in t2.columns


def test_build_e5_products_and_figures(tmp_path) -> None:
    archive_path = _build_tar_source(tmp_path)
    catalog = build_e5_run_catalog(archive_path)
    qc = build_e5_qc_catalog(catalog)

    stats = build_e5_noise_stats_table(qc)
    assert stats.shape[0] == 2
    assert set(stats["noise_source"].astype(str)) == {"lab_noise_generator"}

    curve = build_e5_radiometer_curve_table(
        qc,
        block_size=256,
        n_avg_values=(1, 2, 4, 8),
    )
    assert curve.shape[0] >= 3

    fit = fit_e5_radiometer(curve)
    assert np.isfinite(float(fit["slope"]))
    assert float(fit["slope"]) < 0.0

    t6 = build_e5_t6_table(curve, fit_result=fit)
    assert t6.shape[0] == curve.shape[0]
    assert "chi2_dof" in t6.columns
    assert "expected_sigma_power" in t6.columns

    f10_path = tmp_path / "F10_physical_test.png"
    f11_path = tmp_path / "F11_physical_test.png"
    f12_path = tmp_path / "F12_physical_test.png"

    saved_f10 = write_e5_noise_histogram_figure(qc, f10_path)
    saved_f11 = write_e5_radiometer_figure(curve, f11_path, fit_result=fit)
    saved_f12 = write_e5_acf_consistency_figure(qc, f12_path)

    assert saved_f10 == f10_path
    assert saved_f11 == f11_path
    assert saved_f12 == f12_path
    assert f10_path.exists()
    assert f11_path.exists()
    assert f12_path.exists()
