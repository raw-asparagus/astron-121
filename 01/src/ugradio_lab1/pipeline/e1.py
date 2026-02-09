"""Experiment 1 raw-to-notebook pipeline helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ugradio_lab1.analysis.nyquist import alias_residual_table, predict_alias_frequency
from ugradio_lab1.analysis.spectra import power_spectrum
from ugradio_lab1.dataio.catalog import build_manifest, write_manifest_csv
from ugradio_lab1.dataio.schema import empty_table
from ugradio_lab1.plotting.figure_builders import AliasMapFigureBuilder

_NPZ_METADATA_KEY = "__metadata_json__"

DEFAULT_E1_RAW_SOURCE = Path("data/raw/e1.tar.gz")
DEFAULT_E1_RUN_CATALOG_PATH = Path("data/interim/e1/run_catalog.csv")
DEFAULT_E1_QC_CATALOG_PATH = Path("data/interim/e1/qc_catalog.csv")
DEFAULT_E1_T2_TABLE_PATH = Path("data/processed/e1/tables/T2_e1_runs.csv")
DEFAULT_E1_T3_TABLE_PATH = Path("data/processed/e1/tables/T3_e1_alias_residuals.csv")
DEFAULT_E1_F2_FIGURE_PATH = Path("report/figures/F2_alias_map_physical.png")

_RUN_CATALOG_COLUMNS = (
    "run_id",
    "npz_name",
    "source_kind",
    "source_path",
    "source_member",
    "experiment",
    "status",
    "run_kind",
    "timestamp_utc",
    "sample_rate_hz_requested",
    "sample_rate_hz_actual",
    "sample_rate_error_hz",
    "signal_frequency_hz",
    "signal_frequency_measured_hz",
    "fir_mode",
    "power_tier_dbm",
    "nblocks_requested",
    "stale_blocks_dropped",
    "nblocks_saved",
    "nsamples",
    "device_index",
    "direct",
    "gain_db",
    "adc_mean_block_rms",
    "adc_max",
    "adc_min",
    "adc_is_clipped",
    "adc_passes_guard",
    "guard_attempts",
    "array_nblocks",
    "array_nsamples",
)


def build_e1_run_catalog(raw_source: str | Path = DEFAULT_E1_RAW_SOURCE) -> pd.DataFrame:
    """Build a normalized E1 run catalog from either ``.tar.gz`` or a directory."""

    source = Path(raw_source)
    if not source.exists():
        raise FileNotFoundError(source)

    if _is_tar_archive(source):
        records = _collect_records_from_tar(source)
    elif source.is_dir():
        records = _collect_records_from_directory(source)
    else:
        raise ValueError(f"Unsupported E1 raw source: {source}")

    if len(records) == 0:
        raise ValueError(f"No NPZ runs found in source: {source}")

    frame = pd.DataFrame(records)
    for column in _RUN_CATALOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, list(_RUN_CATALOG_COLUMNS)].copy()
    frame = frame.drop_duplicates(subset=["run_id"], keep="last")
    frame = frame.sort_values(["run_id", "source_member"], kind="stable").reset_index(drop=True)
    return frame


def build_e1_qc_catalog(run_catalog: pd.DataFrame) -> pd.DataFrame:
    """Compute E1 QC labels from a normalized run catalog."""

    frame = run_catalog.copy()
    required = {
        "run_id",
        "status",
        "sample_rate_hz_actual",
        "sample_rate_hz_requested",
        "signal_frequency_hz",
        "fir_mode",
        "power_tier_dbm",
        "nblocks_requested",
        "stale_blocks_dropped",
        "nblocks_saved",
        "array_nblocks",
        "array_nsamples",
        "adc_is_clipped",
        "adc_passes_guard",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run_catalog is missing required columns: {missing}")

    has_metadata = (
        frame["run_id"].notna()
        & frame["sample_rate_hz_actual"].notna()
        & frame["sample_rate_hz_requested"].notna()
        & frame["signal_frequency_hz"].notna()
        & frame["fir_mode"].notna()
        & frame["power_tier_dbm"].notna()
    )
    has_adc_shape = (
        _series_numeric(frame, "array_nblocks") > 0
    ) & (_series_numeric(frame, "array_nsamples") > 0)
    nblocks_policy_ok = (
        (_series_numeric(frame, "nblocks_requested") == 6)
        & (_series_numeric(frame, "stale_blocks_dropped") == 1)
        & (_series_numeric(frame, "nblocks_saved") == 5)
    )
    status_ok = frame["status"].astype(str).isin({"captured", "captured_guard_fail"})
    adc_passes_guard = _series_bool(frame, "adc_passes_guard")
    adc_is_clipped = _series_bool(frame, "adc_is_clipped")

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


def build_e1_t2_table(run_catalog_or_qc: pd.DataFrame) -> pd.DataFrame:
    """Build T2-style E1 manifest rows from the normalized run catalog."""

    frame = run_catalog_or_qc.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))
        rows.append(
            {
                "run_id": str(row.get("run_id")),
                "experiment": row.get("experiment", "E1"),
                "sample_rate_hz": sample_rate_hz,
                "center_frequency_hz": 0.0,
                "tones_hz": json.dumps([_coerce_float(row.get("signal_frequency_hz"))]),
                "source_power_dbm": _coerce_float(row.get("power_tier_dbm")),
                "mixer_config": f"direct_sdr:{row.get('fir_mode', 'unknown')}",
                "cable_config": "siggen_to_sdr_direct",
                "n_samples": _coerce_int(row.get("nsamples")),
                "run_kind": row.get("run_kind", pd.NA),
                "status": row.get("status", pd.NA),
                "n_blocks_saved": _coerce_int(row.get("nblocks_saved")),
                "sample_rate_hz_requested": _coerce_float(row.get("sample_rate_hz_requested")),
                "sample_rate_hz_actual": _coerce_float(row.get("sample_rate_hz_actual")),
                "sample_rate_error_hz": _coerce_float(row.get("sample_rate_error_hz")),
                "qc_analysis_pass": row.get("qc_analysis_pass", pd.NA),
                "qc_recommended_pass": row.get("qc_recommended_pass", pd.NA),
            }
        )

    if len(rows) == 0:
        return empty_table("T2")
    return build_manifest(rows, table_id="T2", allow_extra=True)


def build_e1_t3_table(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
) -> pd.DataFrame:
    """Build T3-style alias residual rows from E1 raw NPZ captures."""

    frame = run_catalog_or_qc.copy()
    required = {"run_id", "source_kind", "source_path", "source_member"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run_catalog_or_qc is missing required columns: {missing}")

    if use_qc_analysis_pass and "qc_analysis_pass" in frame.columns:
        frame = frame.loc[_series_bool(frame, "qc_analysis_pass")].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "f_true_hz",
                "sample_rate_hz",
                "predicted_alias_hz",
                "measured_alias_hz",
                "residual_hz",
                "abs_residual_hz",
                "uncertainty_hz",
            ]
        )

    grouped = frame.groupby(["source_kind", "source_path"], sort=False)
    measurements: list[dict[str, Any]] = []
    for (source_kind, source_path), group in grouped:
        source_kind_value = str(source_kind)
        source_path_value = Path(str(source_path))
        if source_kind_value == "tar":
            with tarfile.open(source_path_value, "r:gz") as archive:
                members = {member.name: member for member in archive.getmembers() if member.isfile()}
                for _, row in group.iterrows():
                    arrays, metadata = _load_npz_from_tar_member(
                        archive,
                        members=members,
                        member_name=str(row["source_member"]),
                    )
                    measurements.append(_measure_alias_row(row, arrays, metadata))
        elif source_kind_value == "directory":
            for _, row in group.iterrows():
                arrays, metadata = _load_npz_from_directory(
                    source_path_value,
                    member_name=str(row["source_member"]),
                )
                measurements.append(_measure_alias_row(row, arrays, metadata))
        else:
            raise ValueError(f"Unsupported source_kind: {source_kind_value!r}")

    if len(measurements) == 0:
        raise ValueError("No analyzable E1 runs were found for T3 table construction.")

    measured = pd.DataFrame(measurements).sort_values("run_id", kind="stable").reset_index(drop=True)
    base = alias_residual_table(
        measured["f_true_hz"].to_numpy(dtype=float),
        measured["sample_rate_hz"].to_numpy(dtype=float),
        measured["measured_alias_hz"].to_numpy(dtype=float),
        predicted_alias_hz=measured["predicted_alias_hz"].to_numpy(dtype=float),
        run_id=measured["run_id"].tolist(),
        uncertainty_hz=measured["uncertainty_hz"].to_numpy(dtype=float),
    )
    merged = base.merge(
        measured.drop(columns=["f_true_hz", "sample_rate_hz", "predicted_alias_hz", "measured_alias_hz", "uncertainty_hz"]),
        on="run_id",
        how="left",
    )
    return merged.sort_values("run_id", kind="stable").reset_index(drop=True)


def write_e1_alias_figure(
    t3_table: pd.DataFrame,
    path: str | Path = DEFAULT_E1_F2_FIGURE_PATH,
    *,
    use_recommended_only: bool = False,
) -> Path:
    """Render and save F2-style alias map from processed T3 rows."""

    table = t3_table.copy()
    if use_recommended_only and "qc_recommended_pass" in table.columns:
        table = table.loc[_series_bool(table, "qc_recommended_pass")].copy()
    if table.empty:
        raise ValueError("t3_table has no rows to plot.")

    required = {"f_true_hz", "measured_alias_hz", "predicted_alias_hz", "residual_hz"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"t3_table is missing required plotting columns: {missing}")

    builder = AliasMapFigureBuilder()
    figure, _ = builder.build(
        table["f_true_hz"].to_numpy(dtype=float),
        table["measured_alias_hz"].to_numpy(dtype=float),
        predicted_alias_hz=table["predicted_alias_hz"].to_numpy(dtype=float),
        residual_hz=table["residual_hz"].to_numpy(dtype=float),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def _is_tar_archive(path: Path) -> bool:
    suffixes = "".join(path.suffixes).lower()
    return suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar")


def _collect_records_from_tar(source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with tarfile.open(source, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".npz"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            arrays, metadata = _load_npz_from_bytes(extracted.read())
            records.append(
                _record_from_npz_payload(
                    npz_name=Path(member.name).name,
                    source_kind="tar",
                    source_path=str(source),
                    source_member=member.name,
                    arrays=arrays,
                    metadata=metadata,
                )
            )
    return records


def _collect_records_from_directory(source: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for npz_path in sorted(source.rglob("*.npz")):
        arrays, metadata = _load_npz_from_path(npz_path)
        records.append(
            _record_from_npz_payload(
                npz_name=npz_path.name,
                source_kind="directory",
                source_path=str(source),
                source_member=str(npz_path.relative_to(source)),
                arrays=arrays,
                metadata=metadata,
            )
        )
    return records


def _load_npz_from_path(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        return _unpack_npz_payload(payload)


def _load_npz_from_bytes(raw: bytes) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        return _unpack_npz_payload(payload)


def _load_npz_from_tar_member(
    archive: tarfile.TarFile,
    *,
    members: dict[str, tarfile.TarInfo],
    member_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if member_name not in members:
        raise FileNotFoundError(f"Tar member not found: {member_name}")
    extracted = archive.extractfile(members[member_name])
    if extracted is None:
        raise FileNotFoundError(f"Unable to extract tar member: {member_name}")
    return _load_npz_from_bytes(extracted.read())


def _load_npz_from_directory(source_dir: Path, *, member_name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = source_dir / member_name
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_npz_from_path(path)


def _unpack_npz_payload(payload: np.lib.npyio.NpzFile) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for key in payload.files:
        if key == _NPZ_METADATA_KEY:
            metadata = json.loads(str(payload[key].item()))
        else:
            arrays[key] = np.asarray(payload[key])
    return arrays, metadata


def _record_from_npz_payload(
    *,
    npz_name: str,
    source_kind: str,
    source_path: str,
    source_member: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    combo = metadata.get("combo", {})
    capture_settings = metadata.get("capture_settings", {})
    sample_rate = metadata.get("sample_rate", {})
    adc_summary = metadata.get("adc_summary", {})
    adc_counts = np.asarray(arrays.get("adc_counts", np.empty((0, 0))), dtype=np.int8)

    return {
        "run_id": str(metadata.get("run_id", Path(npz_name).stem)),
        "npz_name": npz_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_member": source_member,
        "experiment": metadata.get("experiment", "E1"),
        "status": metadata.get("status", pd.NA),
        "run_kind": metadata.get("run_kind", pd.NA),
        "timestamp_utc": metadata.get("timestamp_utc", pd.NA),
        "sample_rate_hz_requested": _coerce_float(sample_rate.get("requested_hz")),
        "sample_rate_hz_actual": _coerce_float(sample_rate.get("actual_hz")),
        "sample_rate_error_hz": _coerce_float(sample_rate.get("error_hz")),
        "signal_frequency_hz": _coerce_float(combo.get("signal_frequency_hz")),
        "signal_frequency_measured_hz": _coerce_float(combo.get("signal_frequency_measured_hz")),
        "fir_mode": combo.get("fir_mode", pd.NA),
        "power_tier_dbm": _coerce_float(combo.get("power_tier_dbm")),
        "nblocks_requested": _coerce_int(capture_settings.get("nblocks_requested")),
        "stale_blocks_dropped": _coerce_int(capture_settings.get("stale_blocks_dropped")),
        "nblocks_saved": _coerce_int(capture_settings.get("nblocks_saved")),
        "nsamples": _coerce_int(capture_settings.get("nsamples")),
        "device_index": _coerce_int(capture_settings.get("device_index")),
        "direct": _coerce_bool(capture_settings.get("direct")),
        "gain_db": _coerce_float(capture_settings.get("gain_db")),
        "adc_mean_block_rms": _coerce_float(adc_summary.get("mean_block_rms")),
        "adc_max": _coerce_int(adc_summary.get("adc_max")),
        "adc_min": _coerce_int(adc_summary.get("adc_min")),
        "adc_is_clipped": _coerce_bool(adc_summary.get("is_clipped")),
        "adc_passes_guard": _coerce_bool(adc_summary.get("passes_guard")),
        "guard_attempts": _coerce_int(adc_summary.get("guard_attempts")),
        "array_nblocks": int(adc_counts.shape[0]) if adc_counts.ndim == 2 else 0,
        "array_nsamples": int(adc_counts.shape[1]) if adc_counts.ndim == 2 else 0,
    }


def _measure_alias_row(
    row: pd.Series,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    blocks = np.asarray(arrays.get("adc_counts"), dtype=np.int8)
    if blocks.ndim != 2:
        raise ValueError(f"run_id={row.get('run_id')} has invalid adc_counts shape: {blocks.shape!r}")
    if blocks.shape[0] < 1:
        raise ValueError(f"run_id={row.get('run_id')} has no saved blocks.")

    sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
    if not np.isfinite(sample_rate_hz):
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(f"run_id={row.get('run_id')} has invalid sample rate: {sample_rate_hz}")

    f_true_hz = _coerce_float(row.get("signal_frequency_measured_hz"))
    if not np.isfinite(f_true_hz):
        f_true_hz = _coerce_float(row.get("signal_frequency_hz"))
    if not np.isfinite(f_true_hz):
        raise ValueError(f"run_id={row.get('run_id')} has invalid signal frequency.")

    predicted_alias_hz = float(predict_alias_frequency(np.array([f_true_hz]), np.array([sample_rate_hz]))[0])
    measured_alias_hz, uncertainty_hz = _estimate_alias_from_blocks(
        blocks,
        sample_rate_hz=sample_rate_hz,
        predicted_alias_hz=predicted_alias_hz,
    )

    return {
        "run_id": str(row.get("run_id")),
        "f_true_hz": float(f_true_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "predicted_alias_hz": float(predicted_alias_hz),
        "measured_alias_hz": float(measured_alias_hz),
        "uncertainty_hz": float(uncertainty_hz),
        "fir_mode": row.get("fir_mode", pd.NA),
        "power_tier_dbm": _coerce_float(row.get("power_tier_dbm")),
        "status": metadata.get("status", row.get("status", pd.NA)),
        "adc_passes_guard": metadata.get("adc_summary", {}).get("passes_guard", row.get("adc_passes_guard", pd.NA)),
        "adc_is_clipped": metadata.get("adc_summary", {}).get("is_clipped", row.get("adc_is_clipped", pd.NA)),
        "qc_analysis_pass": row.get("qc_analysis_pass", pd.NA),
        "qc_recommended_pass": row.get("qc_recommended_pass", pd.NA),
    }


def _estimate_alias_from_blocks(
    blocks: np.ndarray,
    *,
    sample_rate_hz: float,
    predicted_alias_hz: float,
    search_half_width_bins: int = 8,
) -> tuple[float, float]:
    peaks: list[float] = []
    for block in blocks:
        frequency_hz, power_v2 = power_spectrum(
            block,
            sample_rate_hz=float(sample_rate_hz),
            window=None,
            detrend=True,
            scaling="power",
            center=True,
            fft_backend="numpy",
        )
        center_idx = int(np.argmin(np.abs(frequency_hz - float(predicted_alias_hz))))
        lo = max(0, center_idx - int(search_half_width_bins))
        hi = min(power_v2.size, center_idx + int(search_half_width_bins) + 1)
        if hi <= lo:
            peak_idx = int(np.argmax(power_v2))
        else:
            local_idx = int(np.argmax(power_v2[lo:hi]))
            peak_idx = int(lo + local_idx)
        measured_peak_hz = float(frequency_hz[peak_idx])
        aligned_peak_hz = _align_peak_sign(measured_peak_hz, predicted_alias_hz)
        peaks.append(float(aligned_peak_hz))

    if len(peaks) == 0:
        raise ValueError("No blocks available for alias estimation.")

    peaks_array = np.asarray(peaks, dtype=float)
    mean_peak_hz = float(np.mean(peaks_array))
    if peaks_array.size > 1:
        uncertainty_hz = float(np.std(peaks_array, ddof=1))
    else:
        uncertainty_hz = float("nan")
    return mean_peak_hz, uncertainty_hz


def _align_peak_sign(measured_peak_hz: float, predicted_alias_hz: float) -> float:
    if not np.isfinite(measured_peak_hz):
        return measured_peak_hz
    if not np.isfinite(predicted_alias_hz):
        return measured_peak_hz
    direct_distance = abs(measured_peak_hz - predicted_alias_hz)
    flipped_distance = abs((-measured_peak_hz) - predicted_alias_hz)
    return -measured_peak_hz if flipped_distance < direct_distance else measured_peak_hz


def _series_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(False, index=frame.index)
    return values.fillna(False).astype(bool)


def _series_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


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
    if len(failures) == 0:
        return "ok"
    return ";".join(failures)


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return int(-1)


def _coerce_bool(value: Any) -> bool | Any:
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y"}:
            return True
        if normalized in {"false", "f", "0", "no", "n"}:
            return False
    return pd.NA


def write_dataframe_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV, creating parent directories."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def write_table_manifest_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a table/manifest DataFrame with standard CSV behavior."""

    return write_manifest_csv(frame, path)


__all__ = [
    "DEFAULT_E1_F2_FIGURE_PATH",
    "DEFAULT_E1_QC_CATALOG_PATH",
    "DEFAULT_E1_RAW_SOURCE",
    "DEFAULT_E1_RUN_CATALOG_PATH",
    "DEFAULT_E1_T2_TABLE_PATH",
    "DEFAULT_E1_T3_TABLE_PATH",
    "build_e1_qc_catalog",
    "build_e1_run_catalog",
    "build_e1_t2_table",
    "build_e1_t3_table",
    "write_dataframe_csv",
    "write_e1_alias_figure",
    "write_table_manifest_csv",
]
