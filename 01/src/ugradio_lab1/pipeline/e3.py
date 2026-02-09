"""Experiment 3 raw-to-notebook pipeline helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ugradio_lab1.analysis.spectra import average_power_spectrum, voltage_spectrum
from ugradio_lab1.dataio.catalog import build_manifest, write_manifest_csv
from ugradio_lab1.dataio.schema import empty_table
from ugradio_lab1.plotting.figure_builders import (
    ComplexVoltageComponentsFigureBuilder,
    VoltagePowerComparisonFigureBuilder,
)

_NPZ_METADATA_KEY = "__metadata_json__"

DEFAULT_E3_RAW_SOURCE = Path("data/raw/e3.tar.gz")
DEFAULT_E3_RUN_CATALOG_PATH = Path("data/interim/e3/run_catalog.csv")
DEFAULT_E3_QC_CATALOG_PATH = Path("data/interim/e3/qc_catalog.csv")
DEFAULT_E3_SPECTRUM_TABLE_PATH = Path("data/interim/e3/spectrum_profile.csv")
DEFAULT_E3_T2_TABLE_PATH = Path("data/processed/e3/tables/T2_e3_runs.csv")
DEFAULT_E3_F5_FIGURE_PATH = Path("report/figures/F5_complex_voltage_components_physical.png")
DEFAULT_E3_F6_FIGURE_PATH = Path("report/figures/F6_voltage_vs_power_physical.png")

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
    "mode",
    "target_vrms_v",
    "mixer_config",
    "cable_config",
    "notes",
    "tones_hz_json",
    "n_signal_generators",
    "sg1_frequency_hz",
    "sg1_power_dbm",
    "sg2_frequency_hz",
    "sg2_power_dbm",
    "source_power_dbm_mean",
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


def build_e3_run_catalog(raw_source: str | Path = DEFAULT_E3_RAW_SOURCE) -> pd.DataFrame:
    """Build a normalized E3 run catalog from either ``.tar.gz`` or a directory."""

    source = Path(raw_source)
    if not source.exists():
        raise FileNotFoundError(source)

    if _is_tar_archive(source):
        records = _collect_records_from_tar(source)
    elif source.is_dir():
        records = _collect_records_from_directory(source)
    else:
        raise ValueError(f"Unsupported E3 raw source: {source}")

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


def build_e3_qc_catalog(run_catalog: pd.DataFrame) -> pd.DataFrame:
    """Compute E3 QC labels from a normalized run catalog."""

    frame = run_catalog.copy()
    required = {
        "run_id",
        "status",
        "sample_rate_hz_actual",
        "sample_rate_hz_requested",
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


def build_e3_t2_table(run_catalog_or_qc: pd.DataFrame) -> pd.DataFrame:
    """Build T2-style E3 manifest rows from the normalized run catalog."""

    frame = run_catalog_or_qc.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))

        tones_json = row.get("tones_hz_json", "[]")
        try:
            parsed_tones = json.loads(str(tones_json))
            if not isinstance(parsed_tones, list):
                parsed_tones = []
        except Exception:
            parsed_tones = []

        rows.append(
            {
                "run_id": str(row.get("run_id")),
                "experiment": row.get("experiment", "E3"),
                "sample_rate_hz": sample_rate_hz,
                "center_frequency_hz": 0.0,
                "tones_hz": json.dumps([float(value) for value in parsed_tones]),
                "source_power_dbm": _coerce_float(row.get("source_power_dbm_mean")),
                "mixer_config": row.get("mixer_config", "manual_capture"),
                "cable_config": row.get("cable_config", "manual_capture"),
                "n_samples": _coerce_int(row.get("nsamples")),
                "run_kind": row.get("run_kind", pd.NA),
                "status": row.get("status", pd.NA),
                "n_blocks_saved": _coerce_int(row.get("nblocks_saved")),
                "sample_rate_hz_requested": _coerce_float(row.get("sample_rate_hz_requested")),
                "sample_rate_hz_actual": _coerce_float(row.get("sample_rate_hz_actual")),
                "sample_rate_error_hz": _coerce_float(row.get("sample_rate_error_hz")),
                "target_vrms_v": _coerce_float(row.get("target_vrms_v")),
                "mode": row.get("mode", pd.NA),
                "notes": row.get("notes", pd.NA),
                "qc_analysis_pass": row.get("qc_analysis_pass", pd.NA),
                "qc_recommended_pass": row.get("qc_recommended_pass", pd.NA),
            }
        )

    if len(rows) == 0:
        return empty_table("T2")
    return build_manifest(rows, table_id="T2", allow_extra=True)


def select_e3_analysis_runs(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    mode_filter: str | None = None,
    run_ids: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Select E3 run rows for spectrum-profile construction."""

    frame = run_catalog_or_qc.copy()
    if frame.empty:
        raise ValueError("run_catalog_or_qc has no rows.")

    required = {
        "run_id",
        "source_kind",
        "source_path",
        "source_member",
        "sample_rate_hz_requested",
        "sample_rate_hz_actual",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run_catalog_or_qc is missing required columns: {missing}")

    if use_qc_analysis_pass and "qc_analysis_pass" in frame.columns:
        frame = frame.loc[_series_bool(frame, "qc_analysis_pass")].copy()

    if mode_filter is not None and "mode" in frame.columns:
        mode_values = frame["mode"].astype(str).str.strip().str.lower()
        frame = frame.loc[mode_values == str(mode_filter).strip().lower()].copy()

    if run_ids is not None:
        requested = {str(value).strip() for value in run_ids if str(value).strip()}
        frame = frame.loc[frame["run_id"].astype(str).isin(requested)].copy()

    if frame.empty:
        raise ValueError("No E3 runs remain after requested filtering.")

    frame["qc_recommended_pass_bool"] = _series_bool(frame, "qc_recommended_pass")
    frame["adc_is_clipped_bool"] = _series_bool(frame, "adc_is_clipped")
    frame["status_rank"] = frame["status"].astype(str).map(
        {
            "captured": 0,
            "captured_guard_fail": 1,
        }
    ).fillna(2).astype(int)
    frame["timestamp_sort"] = pd.to_datetime(
        frame.get("timestamp_utc", pd.Series(pd.NA, index=frame.index)),
        errors="coerce",
        utc=True,
    )

    ordered = frame.sort_values(
        [
            "run_id",
            "qc_recommended_pass_bool",
            "status_rank",
            "adc_is_clipped_bool",
            "timestamp_sort",
        ],
        ascending=[True, False, True, True, False],
        kind="stable",
    )

    selected = ordered.drop_duplicates(subset=["run_id"], keep="first").copy()
    selected = selected.sort_values(
        ["timestamp_sort", "run_id"],
        ascending=[True, True],
        kind="stable",
    )
    return selected.reset_index(drop=True)


def select_e3_reference_run(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    preferred_mode: str | None = "single_tone",
) -> pd.Series:
    """Select one best E3 run row for physical F5/F6 figure generation."""

    selected = select_e3_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
    )
    frame = selected.copy()

    if preferred_mode is not None and "mode" in frame.columns:
        mode_values = frame["mode"].astype(str).str.strip().str.lower()
        frame["preferred_mode_match"] = mode_values == str(preferred_mode).strip().lower()
    else:
        frame["preferred_mode_match"] = True

    ordered = frame.sort_values(
        [
            "preferred_mode_match",
            "qc_recommended_pass_bool",
            "status_rank",
            "adc_is_clipped_bool",
            "timestamp_sort",
            "run_id",
        ],
        ascending=[False, False, True, True, False, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_e3_spectrum_profiles(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    mode_filter: str | None = None,
    run_ids: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build per-bin E3 spectrum profiles for all selected physical runs."""

    selected_rows = select_e3_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        mode_filter=mode_filter,
        run_ids=run_ids,
    )

    profile_rows: list[pd.DataFrame] = []
    for _, selected in selected_rows.iterrows():
        arrays = _load_arrays_for_catalog_row(selected)
        profile_rows.append(_build_profile_for_selected_run(selected, arrays))

    if len(profile_rows) == 0:
        raise ValueError("No E3 runs were selected for spectrum-profile construction.")

    table = pd.concat(profile_rows, ignore_index=True)
    return table.sort_values(["run_id", "frequency_hz"], kind="stable").reset_index(drop=True)


def build_e3_spectrum_profile(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    preferred_mode: str | None = "single_tone",
) -> pd.DataFrame:
    """Build a per-bin E3 spectrum profile from one selected physical run."""

    selected = select_e3_reference_run(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        preferred_mode=preferred_mode,
    )
    return build_e3_spectrum_profiles(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        run_ids=[str(selected.get("run_id"))],
    )


def write_e3_voltage_components_figure(
    spectrum_profile: pd.DataFrame,
    path: str | Path = DEFAULT_E3_F5_FIGURE_PATH,
    *,
    run_id: str | None = None,
) -> Path:
    """Render and save F5-style complex voltage components from E3 rows."""

    table = _select_single_run_profile(spectrum_profile, run_id=run_id)
    frequency_hz = table["frequency_hz"].to_numpy(dtype=float)
    spectrum_v = table["voltage_real_v"].to_numpy(dtype=float) + 1j * table["voltage_imag_v"].to_numpy(dtype=float)

    builder = ComplexVoltageComponentsFigureBuilder()
    figure, _ = builder.build(frequency_hz, spectrum_v)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_e3_voltage_power_figure(
    spectrum_profile: pd.DataFrame,
    path: str | Path = DEFAULT_E3_F6_FIGURE_PATH,
    *,
    power_db: bool = False,
    run_id: str | None = None,
) -> Path:
    """Render and save F6-style voltage-vs-power figure from E3 rows."""

    table = _select_single_run_profile(spectrum_profile, run_id=run_id)
    frequency_hz = table["frequency_hz"].to_numpy(dtype=float)
    spectrum_v = table["voltage_real_v"].to_numpy(dtype=float) + 1j * table["voltage_imag_v"].to_numpy(dtype=float)
    power_v2 = table["power_v2"].to_numpy(dtype=float)

    builder = VoltagePowerComparisonFigureBuilder()
    figure, _ = builder.build(
        frequency_hz,
        spectrum_v,
        power_v2,
        voltage_component="magnitude",
        power_db=bool(power_db),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def _select_single_run_profile(
    spectrum_profile: pd.DataFrame,
    *,
    run_id: str | None = None,
) -> pd.DataFrame:
    if spectrum_profile.empty:
        raise ValueError("spectrum_profile has no rows.")

    required = {"run_id", "frequency_hz", "voltage_real_v", "voltage_imag_v", "power_v2"}
    missing = sorted(required - set(spectrum_profile.columns))
    if missing:
        raise ValueError(f"spectrum_profile is missing required plotting columns: {missing}")

    table = spectrum_profile.copy()
    run_ids = table["run_id"].astype(str).dropna().unique().tolist()
    if len(run_ids) == 0:
        raise ValueError("spectrum_profile has no valid run_id rows.")

    if run_id is not None:
        target_run_id = str(run_id)
        table = table.loc[table["run_id"].astype(str) == target_run_id].copy()
        if table.empty:
            raise ValueError(f"run_id={target_run_id!r} is not present in spectrum_profile.")
    elif len(run_ids) > 1:
        first_run_id = sorted(run_ids)[0]
        table = table.loc[table["run_id"].astype(str) == first_run_id].copy()

    return table.sort_values("frequency_hz", kind="stable").reset_index(drop=True)


def _build_profile_for_selected_run(
    selected: pd.Series,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    blocks = np.asarray(arrays.get("adc_counts"), dtype=np.int8)
    if blocks.ndim != 2:
        raise ValueError(f"run_id={selected.get('run_id')} has invalid adc_counts shape: {blocks.shape!r}")
    if blocks.shape[0] < 1:
        raise ValueError(f"run_id={selected.get('run_id')} has no saved blocks.")

    sample_rate_hz = _coerce_float(selected.get("sample_rate_hz_actual"))
    if not np.isfinite(sample_rate_hz):
        sample_rate_hz = _coerce_float(selected.get("sample_rate_hz_requested"))
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError(f"run_id={selected.get('run_id')} has invalid sample rate: {sample_rate_hz}")

    spectrum_blocks: list[np.ndarray] = []
    frequency_hz: np.ndarray | None = None
    for block in blocks:
        current_frequency_hz, current_spectrum = voltage_spectrum(
            block.astype(float),
            sample_rate_hz=float(sample_rate_hz),
            window=None,
            detrend=True,
            scaling="amplitude",
            center=True,
            fft_backend="numpy",
        )
        if frequency_hz is None:
            frequency_hz = current_frequency_hz
        spectrum_blocks.append(np.asarray(current_spectrum, dtype=np.complex128))

    if frequency_hz is None:
        raise ValueError("Unable to construct E3 voltage spectrum frequency axis.")

    stacked_spectrum = np.vstack(spectrum_blocks)
    ddof = 1 if stacked_spectrum.shape[0] > 1 else 0
    spectrum_mean = np.mean(stacked_spectrum, axis=0)
    spectrum_mag_std = np.std(np.abs(stacked_spectrum), axis=0, ddof=ddof)

    averaged_power = average_power_spectrum(
        blocks.astype(float),
        sample_rate_hz=float(sample_rate_hz),
        window=None,
        detrend=True,
        scaling="power",
        center=True,
        fft_backend="numpy",
    )
    power_mean = np.asarray(averaged_power.mean, dtype=float)
    power_std = np.asarray(averaged_power.std, dtype=float)

    if power_mean.shape != spectrum_mean.shape:
        raise ValueError("Voltage and power spectra have inconsistent bin counts.")

    power_from_voltage = np.abs(spectrum_mean) ** 2
    return pd.DataFrame(
        {
            "run_id": str(selected.get("run_id")),
            "mode": selected.get("mode", pd.NA),
            "sample_rate_hz": float(sample_rate_hz),
            "frequency_hz": np.asarray(frequency_hz, dtype=float),
            "voltage_real_v": np.real(spectrum_mean),
            "voltage_imag_v": np.imag(spectrum_mean),
            "voltage_mag_v": np.abs(spectrum_mean),
            "voltage_phase_rad": np.angle(spectrum_mean),
            "voltage_mag_block_std_v": spectrum_mag_std,
            "power_v2": power_mean,
            "power_block_std_v2": power_std,
            "power_from_voltage_v2": power_from_voltage,
            "power_consistency_delta_v2": power_mean - power_from_voltage,
            "n_blocks_used": int(stacked_spectrum.shape[0]),
            "qc_analysis_pass": selected.get("qc_analysis_pass", pd.NA),
            "qc_recommended_pass": selected.get("qc_recommended_pass", pd.NA),
            "adc_passes_guard": selected.get("adc_passes_guard", pd.NA),
            "adc_is_clipped": selected.get("adc_is_clipped", pd.NA),
            "source_power_dbm": _coerce_float(selected.get("source_power_dbm_mean")),
            "target_vrms_v": _coerce_float(selected.get("target_vrms_v")),
            "tones_hz_json": selected.get("tones_hz_json", "[]"),
        }
    )


def _load_arrays_for_catalog_row(row: pd.Series) -> dict[str, np.ndarray]:
    source_kind = str(row.get("source_kind"))
    source_path = Path(str(row.get("source_path")))
    source_member = str(row.get("source_member"))

    if source_kind == "tar":
        with tarfile.open(source_path, "r:gz") as archive:
            members = {member.name: member for member in archive.getmembers() if member.isfile()}
            arrays, _ = _load_npz_from_tar_member(
                archive,
                members=members,
                member_name=source_member,
            )
    elif source_kind == "directory":
        arrays, _ = _load_npz_from_directory(source_path, member_name=source_member)
    else:
        raise ValueError(f"Unsupported source_kind: {source_kind!r}")
    return arrays


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
    capture_settings = metadata.get("capture_settings", {})
    sample_rate = metadata.get("sample_rate", {})
    adc_summary = metadata.get("adc_summary", {})

    tones_hz, power_dbm_values = _extract_tones_and_powers(metadata)
    signal_generators = metadata.get("signal_generators", [])
    sg1 = signal_generators[0] if len(signal_generators) >= 1 and isinstance(signal_generators[0], dict) else {}
    sg2 = signal_generators[1] if len(signal_generators) >= 2 and isinstance(signal_generators[1], dict) else {}

    extra = metadata.get("extra", {})
    mode_value = extra.get("mode", metadata.get("mode", pd.NA)) if isinstance(extra, dict) else metadata.get("mode", pd.NA)
    mixer_config_value = metadata.get("mixer_config", pd.NA)
    cable_config_value = metadata.get("cable_config", pd.NA)
    notes_value = metadata.get("notes", pd.NA)

    adc_counts = np.asarray(arrays.get("adc_counts", np.empty((0, 0))), dtype=np.int8)

    return {
        "run_id": str(metadata.get("run_id", Path(npz_name).stem)),
        "npz_name": npz_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_member": source_member,
        "experiment": metadata.get("experiment", "E3"),
        "status": metadata.get("status", pd.NA),
        "run_kind": metadata.get("run_kind", pd.NA),
        "timestamp_utc": metadata.get("timestamp_utc", pd.NA),
        "sample_rate_hz_requested": _coerce_float(sample_rate.get("requested_hz")),
        "sample_rate_hz_actual": _coerce_float(sample_rate.get("actual_hz")),
        "sample_rate_error_hz": _coerce_float(sample_rate.get("error_hz")),
        "mode": mode_value,
        "target_vrms_v": _coerce_float(metadata.get("target_vrms_v")),
        "mixer_config": mixer_config_value,
        "cable_config": cable_config_value,
        "notes": notes_value,
        "tones_hz_json": json.dumps([float(value) for value in tones_hz]),
        "n_signal_generators": len(tones_hz),
        "sg1_frequency_hz": _coerce_float(sg1.get("frequency_hz")),
        "sg1_power_dbm": _coerce_float(sg1.get("power_dbm")),
        "sg2_frequency_hz": _coerce_float(sg2.get("frequency_hz")),
        "sg2_power_dbm": _coerce_float(sg2.get("power_dbm")),
        "source_power_dbm_mean": float(np.mean(power_dbm_values)) if len(power_dbm_values) > 0 else float("nan"),
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


def _extract_tones_and_powers(metadata: dict[str, Any]) -> tuple[list[float], list[float]]:
    tones_hz: list[float] = []
    powers_dbm: list[float] = []

    signal_generators = metadata.get("signal_generators", [])
    if isinstance(signal_generators, list):
        for entry in signal_generators:
            if not isinstance(entry, dict):
                continue
            frequency = _coerce_float(entry.get("frequency_hz"))
            power = _coerce_float(entry.get("power_dbm"))
            if np.isfinite(frequency):
                tones_hz.append(float(frequency))
            if np.isfinite(power):
                powers_dbm.append(float(power))

    combo = metadata.get("combo", {})
    if isinstance(combo, dict):
        frequency = _coerce_float(combo.get("signal_frequency_hz"))
        power = _coerce_float(combo.get("power_tier_dbm"))
        if np.isfinite(frequency) and len(tones_hz) == 0:
            tones_hz.append(float(frequency))
        if np.isfinite(power) and len(powers_dbm) == 0:
            powers_dbm.append(float(power))

    return tones_hz, powers_dbm


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
