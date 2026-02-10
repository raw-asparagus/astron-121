"""Unit tests for the unified catalog pipeline."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ugradio_lab1.dataio.io_npz import save_npz_dataset
from ugradio_lab1.pipeline.catalog import (
    UNIFIED_CATALOG_COLUMNS,
    build_unified_catalog,
    build_unified_qc_catalog,
    record_from_npz_payload,
    write_catalog_parquet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _write_combo_npz(
    path: Path,
    *,
    run_id: str,
    frequency_hz: float,
    sample_rate_hz: float,
    fir_mode: str = "default",
    power_tier_dbm: float = 10.0,
    experiment: str = "E1",
    status: str = "captured",
    passes_guard: bool = True,
) -> None:
    blocks = _tone_blocks(frequency_hz=frequency_hz, sample_rate_hz=sample_rate_hz)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp_utc": "2026-02-09T00:00:00+00:00",
        "status": status,
        "run_kind": "power_tier_capture",
        "combo": {
            "sample_rate_hz": float(sample_rate_hz),
            "signal_frequency_hz": float(frequency_hz),
            "signal_frequency_measured_hz": float(frequency_hz),
            "fir_mode": fir_mode,
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
        },
    }
    save_npz_dataset(path, arrays={"adc_counts": blocks}, metadata=metadata)


def _write_siggen_npz(
    path: Path,
    *,
    run_id: str,
    sample_rate_hz: float = 2048000.0,
    sg1_freq: float = 100000.0,
    sg1_power: float = -10.0,
    mode: str = "single_tone",
) -> None:
    blocks = _tone_blocks(frequency_hz=100.0, sample_rate_hz=2048.0)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": "E3",
        "timestamp_utc": "2026-02-09T01:00:00+00:00",
        "status": "captured",
        "run_kind": "spectrum_capture",
        "extra": {"mode": mode},
        "signal_generators": [
            {"frequency_hz": sg1_freq, "power_dbm": sg1_power},
        ],
        "mixer_config": "direct",
        "cable_config": "short",
        "target_vrms_v": 0.05,
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
            "is_clipped": False,
            "passes_guard": True,
            "guard_attempts": 1,
        },
    }
    save_npz_dataset(path, arrays={"adc_counts": blocks}, metadata=metadata)


def _write_noise_npz(
    path: Path,
    *,
    run_id: str,
    sample_rate_hz: float = 2048000.0,
    noise_source: str = "lab_noise_generator",
) -> None:
    rng = np.random.default_rng(42)
    blocks = rng.integers(-50, 50, size=(5, 2048), dtype=np.int8)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": "E5",
        "timestamp_utc": "2026-02-09T02:00:00+00:00",
        "status": "captured",
        "run_kind": "noise_capture",
        "extra": {"noise_source": noise_source},
        "mixer_config": "direct",
        "cable_config": "short",
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
            "is_clipped": False,
            "passes_guard": True,
            "guard_attempts": 1,
        },
    }
    save_npz_dataset(path, arrays={"adc_counts": blocks}, metadata=metadata)


def _build_tar(tmp_path: Path, subdir: str, npz_files: list[Path]) -> Path:
    archive_path = tmp_path / f"{subdir}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for npz_path in sorted(npz_files):
            archive.add(npz_path, arcname=f"{subdir}/{npz_path.name}")
    return archive_path


# ---------------------------------------------------------------------------
# record_from_npz_payload tests
# ---------------------------------------------------------------------------

class TestRecordFromNpzPayload:
    def test_combo_metadata_populates_signal_fields(self, tmp_path):
        npz_path = tmp_path / "combo.npz"
        _write_combo_npz(npz_path, run_id="r1", frequency_hz=1300.0, sample_rate_hz=2048.0)
        from ugradio_lab1.pipeline._common import load_npz_from_path

        arrays, metadata = load_npz_from_path(npz_path)
        rec = record_from_npz_payload(
            npz_name="combo.npz",
            source_kind="directory",
            source_path=str(tmp_path),
            source_member="combo.npz",
            arrays=arrays,
            metadata=metadata,
            default_experiment="E1",
        )
        assert rec["signal_frequency_hz"] == 1300.0
        assert rec["fir_mode"] == "default"
        assert rec["power_tier_dbm"] == 10.0
        # E3/E5 fields should be NA or nan
        assert rec["noise_source"] is pd.NA
        assert rec["tones_hz_json"] is not pd.NA  # combo fallback populates tones

    def test_siggen_metadata_populates_e3_fields(self, tmp_path):
        npz_path = tmp_path / "siggen.npz"
        _write_siggen_npz(npz_path, run_id="r2", sg1_freq=100000.0, sg1_power=-10.0)
        from ugradio_lab1.pipeline._common import load_npz_from_path

        arrays, metadata = load_npz_from_path(npz_path)
        rec = record_from_npz_payload(
            npz_name="siggen.npz",
            source_kind="directory",
            source_path=str(tmp_path),
            source_member="siggen.npz",
            arrays=arrays,
            metadata=metadata,
            default_experiment="E3",
        )
        assert rec["sg1_frequency_hz"] == 100000.0
        assert rec["sg1_power_dbm"] == -10.0
        assert rec["mode"] == "single_tone"
        assert rec["mixer_config"] == "direct"
        # E1/E2 combo fields should be nan
        assert np.isnan(rec["signal_frequency_hz"])

    def test_noise_metadata_populates_e5_fields(self, tmp_path):
        npz_path = tmp_path / "noise.npz"
        _write_noise_npz(npz_path, run_id="r3", noise_source="lab_noise_generator")
        from ugradio_lab1.pipeline._common import load_npz_from_path

        arrays, metadata = load_npz_from_path(npz_path)
        rec = record_from_npz_payload(
            npz_name="noise.npz",
            source_kind="directory",
            source_path=str(tmp_path),
            source_member="noise.npz",
            arrays=arrays,
            metadata=metadata,
            default_experiment="E5",
        )
        assert rec["noise_source"] == "lab_noise_generator"
        # combo fields should be nan
        assert np.isnan(rec["signal_frequency_hz"])


# ---------------------------------------------------------------------------
# build_unified_catalog tests
# ---------------------------------------------------------------------------

class TestBuildUnifiedCatalog:
    def _make_sources(self, tmp_path):
        e1_dir = tmp_path / "e1_npz"
        e1_dir.mkdir()
        _write_combo_npz(e1_dir / "run_a.npz", run_id="run_a", frequency_hz=500.0,
                         sample_rate_hz=2048.0, experiment="E1")
        e1_tar = _build_tar(tmp_path, "e1", list(e1_dir.glob("*.npz")))

        e3_dir = tmp_path / "e3_npz"
        e3_dir.mkdir()
        _write_siggen_npz(e3_dir / "run_b.npz", run_id="run_b")
        e3_tar = _build_tar(tmp_path, "e3", list(e3_dir.glob("*.npz")))

        e5_dir = tmp_path / "e5_npz"
        e5_dir.mkdir()
        _write_noise_npz(e5_dir / "run_c.npz", run_id="run_c")
        e5_tar = _build_tar(tmp_path, "e5", list(e5_dir.glob("*.npz")))

        return {"E1": e1_tar, "E3": e3_tar, "E5": e5_tar}

    def test_concatenates_multiple_sources(self, tmp_path):
        sources = self._make_sources(tmp_path)
        catalog = build_unified_catalog(sources)
        assert catalog.shape[0] == 3
        assert set(catalog["run_id"]) == {"run_a", "run_b", "run_c"}

    def test_assigns_experiment_tags(self, tmp_path):
        sources = self._make_sources(tmp_path)
        catalog = build_unified_catalog(sources)
        e3_row = catalog.loc[catalog["experiment"] == "E3"].iloc[0]
        assert e3_row["experiment_tags"] == "E3,E4"

    def test_all_columns_present(self, tmp_path):
        sources = self._make_sources(tmp_path)
        catalog = build_unified_catalog(sources)
        for col in UNIFIED_CATALOG_COLUMNS:
            assert col in catalog.columns, f"Missing column: {col}"

    def test_deduplicates_on_run_id(self, tmp_path):
        e1_dir = tmp_path / "e1_npz"
        e1_dir.mkdir()
        _write_combo_npz(e1_dir / "dup.npz", run_id="dup", frequency_hz=500.0,
                         sample_rate_hz=2048.0, experiment="E1")
        e1_tar = _build_tar(tmp_path, "e1", list(e1_dir.glob("*.npz")))

        e2_dir = tmp_path / "e2_npz"
        e2_dir.mkdir()
        _write_combo_npz(e2_dir / "dup.npz", run_id="dup", frequency_hz=600.0,
                         sample_rate_hz=2048.0, experiment="E2")
        e2_tar = _build_tar(tmp_path, "e2", list(e2_dir.glob("*.npz")))

        catalog = build_unified_catalog({"E1": e1_tar, "E2": e2_tar})
        assert catalog.shape[0] == 1


# ---------------------------------------------------------------------------
# build_unified_qc_catalog tests
# ---------------------------------------------------------------------------

class TestBuildUnifiedQcCatalog:
    def _single_source(self, tmp_path, *, passes_guard=True, status="captured"):
        d = tmp_path / "npz"
        d.mkdir()
        _write_combo_npz(d / "r.npz", run_id="r", frequency_hz=500.0,
                         sample_rate_hz=2048.0, passes_guard=passes_guard, status=status)
        tar = _build_tar(tmp_path, "e1", list(d.glob("*.npz")))
        return build_unified_catalog({"E1": tar})

    def test_passing_row(self, tmp_path):
        catalog = self._single_source(tmp_path)
        qc = build_unified_qc_catalog(catalog)
        assert qc.shape[0] == 1
        assert bool(qc["qc_analysis_pass"].iloc[0]) is True
        assert bool(qc["qc_recommended_pass"].iloc[0]) is True
        assert qc["qc_reason"].iloc[0] == "ok"

    def test_clipped_row_passes_analysis_fails_recommended(self, tmp_path):
        d = tmp_path / "npz"
        d.mkdir()
        # Use amplitude that clips
        blocks = np.full((5, 2048), 127, dtype=np.int8)
        block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
        metadata = {
            "run_id": "clip",
            "experiment": "E1",
            "status": "captured",
            "run_kind": "test",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "combo": {"signal_frequency_hz": 500.0, "fir_mode": "default", "power_tier_dbm": 10.0},
            "capture_settings": {
                "device_index": 0, "direct": True, "gain_db": 0.0,
                "nsamples": 2048, "nblocks_requested": 6,
                "stale_blocks_dropped": 1, "nblocks_saved": 5,
            },
            "sample_rate": {"requested_hz": 2048.0, "actual_hz": 2048.0, "error_hz": 0.0},
            "adc_summary": {
                "mean_block_rms": float(np.mean(block_rms)),
                "adc_max": 127, "adc_min": 127,
                "is_clipped": True, "passes_guard": True, "guard_attempts": 1,
            },
        }
        save_npz_dataset(d / "clip.npz", arrays={"adc_counts": blocks}, metadata=metadata)
        tar = _build_tar(tmp_path, "e1", list(d.glob("*.npz")))
        catalog = build_unified_catalog({"E1": tar})
        qc = build_unified_qc_catalog(catalog)
        assert bool(qc["qc_analysis_pass"].iloc[0]) is True
        assert bool(qc["qc_recommended_pass"].iloc[0]) is False

    def test_missing_metadata_fails_analysis(self, tmp_path):
        d = tmp_path / "npz"
        d.mkdir()
        blocks = _tone_blocks(frequency_hz=500.0, sample_rate_hz=2048.0)
        metadata = {
            "run_id": "bad",
            "experiment": "E1",
            "status": "captured",
            "capture_settings": {
                "nblocks_requested": 6, "stale_blocks_dropped": 1, "nblocks_saved": 5,
                "nsamples": 2048, "device_index": 0, "direct": True, "gain_db": 0.0,
            },
            "sample_rate": {},  # missing actual and requested
            "adc_summary": {
                "mean_block_rms": 20.0, "adc_max": 40, "adc_min": -40,
                "is_clipped": False, "passes_guard": True, "guard_attempts": 1,
            },
        }
        save_npz_dataset(d / "bad.npz", arrays={"adc_counts": blocks}, metadata=metadata)
        tar = _build_tar(tmp_path, "e1", list(d.glob("*.npz")))
        catalog = build_unified_catalog({"E1": tar})
        qc = build_unified_qc_catalog(catalog)
        assert bool(qc["qc_analysis_pass"].iloc[0]) is False
        assert "missing_metadata" in qc["qc_reason"].iloc[0]


# ---------------------------------------------------------------------------
# Parquet roundtrip test
# ---------------------------------------------------------------------------

class TestParquetRoundtrip:
    def test_roundtrip_preserves_columns(self, tmp_path):
        d = tmp_path / "npz"
        d.mkdir()
        _write_combo_npz(d / "r.npz", run_id="r", frequency_hz=500.0, sample_rate_hz=2048.0)
        tar = _build_tar(tmp_path, "e1", list(d.glob("*.npz")))
        catalog = build_unified_catalog({"E1": tar})
        qc = build_unified_qc_catalog(catalog)

        pq_path = tmp_path / "out.parquet"
        write_catalog_parquet(qc, pq_path)

        loaded = pd.read_parquet(pq_path)
        for col in UNIFIED_CATALOG_COLUMNS:
            assert col in loaded.columns, f"Missing column after roundtrip: {col}"
        assert loaded.shape[0] == qc.shape[0]
