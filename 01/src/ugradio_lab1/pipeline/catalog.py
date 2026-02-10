"""Unified catalog builder for all Lab 1 experiments."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ugradio_lab1.pipeline._common import (
    coerce_bool,
    coerce_float,
    coerce_int,
    collect_records_from_directory,
    collect_records_from_tar,
    is_tar_archive,
    series_bool,
    series_numeric,
)

UNIFIED_CATALOG_COLUMNS = (
    # Core identity
    "run_id",
    "npz_name",
    "source_kind",
    "source_path",
    "source_member",
    "experiment",
    "experiment_tags",
    "status",
    "run_kind",
    "timestamp_utc",
    # SDR capture settings
    "sample_rate_hz_requested",
    "sample_rate_hz_actual",
    "sample_rate_error_hz",
    "nblocks_requested",
    "stale_blocks_dropped",
    "nblocks_saved",
    "nsamples",
    "device_index",
    "direct",
    "gain_db",
    # ADC summary
    "adc_mean_block_rms",
    "adc_max",
    "adc_min",
    "adc_is_clipped",
    "adc_passes_guard",
    "guard_attempts",
    "array_nblocks",
    "array_nsamples",
    # Signal fields (NA when inapplicable)
    "signal_frequency_hz",
    "signal_frequency_measured_hz",
    "fir_mode",
    "power_tier_dbm",
    # Capture context (NA when inapplicable)
    "mode",
    "target_vrms_v",
    "mixer_config",
    "cable_config",
    "notes",
    # Signal generators (NA for E1/E2/E5)
    "tones_hz_json",
    "n_signal_generators",
    "sg1_frequency_hz",
    "sg1_power_dbm",
    "sg2_frequency_hz",
    "sg2_power_dbm",
    "source_power_dbm_mean",
    # Noise (NA for E1/E2/E3)
    "noise_source",
)

DEFAULT_SOURCES: dict[str, Path] = {
    "E1": Path("data/raw/e1.tar.gz"),
    "E2": Path("data/raw/e2.tar.gz"),
    "E3": Path("data/raw/e3.tar.gz"),
    "E5": Path("data/raw/e5.tar.gz"),
}

_EXPERIMENT_TAG_MAP: dict[str, str] = {
    "E1": "E1",
    "E2": "E2",
    "E3": "E3,E4",
    "E5": "E5",
}


# ---------------------------------------------------------------------------
# Tone/power extraction (ported from e3.py)
# ---------------------------------------------------------------------------

def _extract_tones_and_powers(
    metadata: dict[str, Any],
) -> tuple[list[float], list[float]]:
    tones_hz: list[float] = []
    powers_dbm: list[float] = []

    signal_generators = metadata.get("signal_generators", [])
    if isinstance(signal_generators, list):
        for entry in signal_generators:
            if not isinstance(entry, dict):
                continue
            frequency = coerce_float(entry.get("frequency_hz"))
            power = coerce_float(entry.get("power_dbm"))
            if np.isfinite(frequency):
                tones_hz.append(float(frequency))
            if np.isfinite(power):
                powers_dbm.append(float(power))

    combo = metadata.get("combo", {})
    if isinstance(combo, dict):
        frequency = coerce_float(combo.get("signal_frequency_hz"))
        power = coerce_float(combo.get("power_tier_dbm"))
        if np.isfinite(frequency) and len(tones_hz) == 0:
            tones_hz.append(float(frequency))
        if np.isfinite(power) and len(powers_dbm) == 0:
            powers_dbm.append(float(power))

    return tones_hz, powers_dbm


# ---------------------------------------------------------------------------
# Unified record extractor
# ---------------------------------------------------------------------------

def record_from_npz_payload(
    *,
    npz_name: str,
    source_kind: str,
    source_path: str,
    source_member: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    default_experiment: str,
) -> dict[str, Any]:
    capture_settings = metadata.get("capture_settings", {})
    sample_rate = metadata.get("sample_rate", {})
    adc_summary = metadata.get("adc_summary", {})
    adc_counts = np.asarray(arrays.get("adc_counts", np.empty((0, 0))), dtype=np.int8)

    # --- Combo path (E1/E2) ---
    combo = metadata.get("combo", {})
    if not isinstance(combo, dict):
        combo = {}

    # --- Signal generators path (E3) ---
    tones_hz, power_dbm_values = _extract_tones_and_powers(metadata)
    signal_generators = metadata.get("signal_generators", [])
    sg1 = (
        signal_generators[0]
        if isinstance(signal_generators, list)
        and len(signal_generators) >= 1
        and isinstance(signal_generators[0], dict)
        else {}
    )
    sg2 = (
        signal_generators[1]
        if isinstance(signal_generators, list)
        and len(signal_generators) >= 2
        and isinstance(signal_generators[1], dict)
        else {}
    )

    extra = metadata.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    mode_value = extra.get("mode", metadata.get("mode", pd.NA))

    # --- Noise path (E5) ---
    noise_source = extra.get("noise_source", metadata.get("noise_source", pd.NA))

    # --- Shared context (E3/E5) ---
    mixer_config_value = metadata.get("mixer_config", pd.NA)
    cable_config_value = metadata.get("cable_config", pd.NA)
    notes_value = metadata.get("notes", pd.NA)

    return {
        # Core identity
        "run_id": str(metadata.get("run_id", Path(npz_name).stem)),
        "npz_name": npz_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_member": source_member,
        "experiment": metadata.get("experiment", default_experiment),
        "status": metadata.get("status", pd.NA),
        "run_kind": metadata.get("run_kind", pd.NA),
        "timestamp_utc": metadata.get("timestamp_utc", pd.NA),
        # SDR capture settings
        "sample_rate_hz_requested": coerce_float(sample_rate.get("requested_hz")),
        "sample_rate_hz_actual": coerce_float(sample_rate.get("actual_hz")),
        "sample_rate_error_hz": coerce_float(sample_rate.get("error_hz")),
        "nblocks_requested": coerce_int(capture_settings.get("nblocks_requested")),
        "stale_blocks_dropped": coerce_int(capture_settings.get("stale_blocks_dropped")),
        "nblocks_saved": coerce_int(capture_settings.get("nblocks_saved")),
        "nsamples": coerce_int(capture_settings.get("nsamples")),
        "device_index": coerce_int(capture_settings.get("device_index")),
        "direct": coerce_bool(capture_settings.get("direct")),
        "gain_db": coerce_float(capture_settings.get("gain_db")),
        # ADC summary
        "adc_mean_block_rms": coerce_float(adc_summary.get("mean_block_rms")),
        "adc_max": coerce_int(adc_summary.get("adc_max")),
        "adc_min": coerce_int(adc_summary.get("adc_min")),
        "adc_is_clipped": coerce_bool(adc_summary.get("is_clipped")),
        "adc_passes_guard": coerce_bool(adc_summary.get("passes_guard")),
        "guard_attempts": coerce_int(adc_summary.get("guard_attempts")),
        "array_nblocks": int(adc_counts.shape[0]) if adc_counts.ndim == 2 else 0,
        "array_nsamples": int(adc_counts.shape[1]) if adc_counts.ndim == 2 else 0,
        # Signal fields (combo path)
        "signal_frequency_hz": coerce_float(combo.get("signal_frequency_hz")),
        "signal_frequency_measured_hz": coerce_float(combo.get("signal_frequency_measured_hz")),
        "fir_mode": combo.get("fir_mode", pd.NA),
        "power_tier_dbm": coerce_float(combo.get("power_tier_dbm")),
        # Capture context
        "mode": mode_value,
        "target_vrms_v": coerce_float(metadata.get("target_vrms_v")),
        "mixer_config": mixer_config_value,
        "cable_config": cable_config_value,
        "notes": notes_value,
        # Signal generators
        "tones_hz_json": json.dumps([float(v) for v in tones_hz]) if tones_hz else pd.NA,
        "n_signal_generators": len(tones_hz) if tones_hz else pd.NA,
        "sg1_frequency_hz": coerce_float(sg1.get("frequency_hz")),
        "sg1_power_dbm": coerce_float(sg1.get("power_dbm")),
        "sg2_frequency_hz": coerce_float(sg2.get("frequency_hz")),
        "sg2_power_dbm": coerce_float(sg2.get("power_dbm")),
        "source_power_dbm_mean": (
            float(np.mean(power_dbm_values)) if power_dbm_values else float("nan")
        ),
        # Noise
        "noise_source": noise_source,
    }


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------

def build_unified_catalog(
    sources: dict[str, str | Path] | None = None,
) -> pd.DataFrame:
    if sources is None:
        sources = DEFAULT_SOURCES

    all_records: list[dict[str, Any]] = []
    for tag, raw_path in sources.items():
        source = Path(raw_path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found for {tag}: {source}")

        record_fn = partial(record_from_npz_payload, default_experiment=tag)

        if is_tar_archive(source):
            records = collect_records_from_tar(source, record_fn)
        elif source.is_dir():
            records = collect_records_from_directory(source, record_fn)
        else:
            raise ValueError(f"Unsupported source for {tag}: {source}")

        experiment_tags = _EXPERIMENT_TAG_MAP.get(tag, tag)
        for rec in records:
            rec["experiment_tags"] = experiment_tags

        all_records.extend(records)

    if not all_records:
        raise ValueError("No NPZ runs found in any source.")

    frame = pd.DataFrame(all_records)
    for col in UNIFIED_CATALOG_COLUMNS:
        if col not in frame.columns:
            frame[col] = pd.NA
    frame = frame.loc[:, list(UNIFIED_CATALOG_COLUMNS)].copy()
    frame = frame.drop_duplicates(subset=["run_id"], keep="last")
    frame = frame.sort_values(["run_id", "source_member"], kind="stable").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# QC catalog
# ---------------------------------------------------------------------------

def _build_qc_reason(
    *,
    has_required_metadata: bool,
    has_adc_shape: bool,
    nblocks_policy_ok: bool,
    status_ok: bool,
    adc_passes_guard: bool,
    adc_is_clipped: bool,
) -> str:
    failures: list[str] = []
    if not has_required_metadata:
        failures.append("missing_metadata")
    if not has_adc_shape:
        failures.append("missing_adc_blocks")
    if not nblocks_policy_ok:
        failures.append("nblocks_policy_mismatch")
    if not status_ok:
        failures.append("status_not_captured")
    if not adc_passes_guard:
        failures.append("guard_fail")
    if adc_is_clipped:
        failures.append("clipped")
    return "ok" if not failures else ";".join(failures)


def build_unified_qc_catalog(run_catalog: pd.DataFrame) -> pd.DataFrame:
    frame = run_catalog.copy()

    has_metadata = (
        frame["run_id"].notna()
        & frame["sample_rate_hz_actual"].notna()
        & frame["sample_rate_hz_requested"].notna()
    )
    has_adc_shape = (
        (series_numeric(frame, "array_nblocks") > 0)
        & (series_numeric(frame, "array_nsamples") > 0)
    )
    nblocks_policy_ok = (
        (series_numeric(frame, "nblocks_requested") == 6)
        & (series_numeric(frame, "stale_blocks_dropped") == 1)
        & (series_numeric(frame, "nblocks_saved") == 5)
    )
    status_ok = frame["status"].astype(str).isin({"captured", "captured_guard_fail"})
    adc_passes_guard = series_bool(frame, "adc_passes_guard")
    adc_is_clipped = series_bool(frame, "adc_is_clipped")

    qc_analysis_pass = has_metadata & has_adc_shape & nblocks_policy_ok & status_ok
    qc_recommended_pass = qc_analysis_pass & adc_passes_guard & (~adc_is_clipped)

    qc = frame.copy()
    qc["qc_has_required_metadata"] = has_metadata
    qc["qc_has_adc_shape"] = has_adc_shape
    qc["qc_nblocks_policy_ok"] = nblocks_policy_ok
    qc["qc_status_ok"] = status_ok
    qc["qc_analysis_pass"] = qc_analysis_pass
    qc["qc_recommended_pass"] = qc_recommended_pass
    qc["qc_reason"] = [
        _build_qc_reason(
            has_required_metadata=bool(has_metadata.iloc[idx]),
            has_adc_shape=bool(has_adc_shape.iloc[idx]),
            nblocks_policy_ok=bool(nblocks_policy_ok.iloc[idx]),
            status_ok=bool(status_ok.iloc[idx]),
            adc_passes_guard=bool(adc_passes_guard.iloc[idx]),
            adc_is_clipped=bool(adc_is_clipped.iloc[idx]),
        )
        for idx in range(len(qc))
    ]
    return qc.sort_values("run_id", kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def write_catalog_parquet(
    catalog: pd.DataFrame,
    path: str | Path,
    *,
    compression: str = "snappy",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(destination, engine="pyarrow", compression=compression, index=False)
    return destination
