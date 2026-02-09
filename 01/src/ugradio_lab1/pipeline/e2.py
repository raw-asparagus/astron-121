"""Experiment 2 raw-to-notebook pipeline helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ugradio_lab1.analysis.bandpass import bandpass_curve, bandpass_summary_metrics
from ugradio_lab1.dataio.catalog import build_manifest, write_manifest_csv
from ugradio_lab1.dataio.schema import empty_table, validate_table
from ugradio_lab1.plotting.figure_builders import BandpassFigureBuilder

_NPZ_METADATA_KEY = "__metadata_json__"

DEFAULT_E2_RAW_SOURCE = Path("data/raw/e2.tar.gz")
DEFAULT_E2_RUN_CATALOG_PATH = Path("data/interim/e2/run_catalog.csv")
DEFAULT_E2_QC_CATALOG_PATH = Path("data/interim/e2/qc_catalog.csv")
DEFAULT_E2_CURVE_TABLE_PATH = Path("data/interim/e2/bandpass_curves.csv")
DEFAULT_E2_T2_TABLE_PATH = Path("data/processed/e2/tables/T2_e2_runs.csv")
DEFAULT_E2_T4_TABLE_PATH = Path("data/processed/e2/tables/T4_e2_bandpass_summary.csv")
DEFAULT_E2_F4_FIGURE_PATH = Path("report/figures/F4_bandpass_curves_physical.png")

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


def build_e2_run_catalog(raw_source: str | Path = DEFAULT_E2_RAW_SOURCE) -> pd.DataFrame:
    """Build a normalized E2 run catalog from either ``.tar.gz`` or a directory."""

    source = Path(raw_source)
    if not source.exists():
        raise FileNotFoundError(source)

    if _is_tar_archive(source):
        records = _collect_records_from_tar(source)
    elif source.is_dir():
        records = _collect_records_from_directory(source)
    else:
        raise ValueError(f"Unsupported E2 raw source: {source}")

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


def build_e2_qc_catalog(run_catalog: pd.DataFrame) -> pd.DataFrame:
    """Compute E2 QC labels from a normalized run catalog."""

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


def build_e2_t2_table(run_catalog_or_qc: pd.DataFrame) -> pd.DataFrame:
    """Build T2-style E2 manifest rows from the normalized run catalog."""

    frame = run_catalog_or_qc.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))
        rows.append(
            {
                "run_id": str(row.get("run_id")),
                "experiment": row.get("experiment", "E2"),
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


def select_e2_analysis_runs(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    preferred_power_dbm: float = -10.0,
) -> pd.DataFrame:
    """Select one analysis run per ``(sample_rate, frequency, fir_mode)`` combo."""

    frame = run_catalog_or_qc.copy()
    if frame.empty:
        return frame

    required = {
        "run_id",
        "sample_rate_hz_requested",
        "sample_rate_hz_actual",
        "signal_frequency_hz",
        "signal_frequency_measured_hz",
        "fir_mode",
        "power_tier_dbm",
        "timestamp_utc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"run_catalog_or_qc is missing required columns: {missing}")

    frame["sample_rate_hz_nominal"] = _effective_sample_rate(frame)
    frame["signal_frequency_hz_effective"] = _effective_signal_frequency(frame)
    frame["power_tier_dbm_num"] = _series_numeric(frame, "power_tier_dbm")

    finite_mask = (
        np.isfinite(frame["sample_rate_hz_nominal"].to_numpy(dtype=float))
        & np.isfinite(frame["signal_frequency_hz_effective"].to_numpy(dtype=float))
    )
    frame = frame.loc[finite_mask].copy()
    if frame.empty:
        return frame

    frame["qc_analysis_pass_bool"] = (
        _series_bool(frame, "qc_analysis_pass") if "qc_analysis_pass" in frame.columns else True
    )
    if use_qc_analysis_pass:
        frame = frame.loc[frame["qc_analysis_pass_bool"]].copy()
    if frame.empty:
        return frame

    frame["qc_recommended_pass_bool"] = _series_bool(frame, "qc_recommended_pass")
    frame["adc_is_clipped_bool"] = _series_bool(frame, "adc_is_clipped")
    frame["adc_not_clipped_bool"] = ~frame["adc_is_clipped_bool"]

    frame["sample_rate_key_hz"] = np.round(frame["sample_rate_hz_nominal"].to_numpy(dtype=float), 3)
    frame["frequency_key_hz"] = np.round(frame["signal_frequency_hz_effective"].to_numpy(dtype=float), 6)
    frame["power_distance_to_preferred"] = np.abs(
        frame["power_tier_dbm_num"].to_numpy(dtype=float) - float(preferred_power_dbm)
    )

    score = (
        frame.groupby(["sample_rate_key_hz", "fir_mode", "power_tier_dbm_num"], dropna=False)
        .agg(
            combo_count=("frequency_key_hz", "nunique"),
            recommended_count=("qc_recommended_pass_bool", "sum"),
            analysis_count=("qc_analysis_pass_bool", "sum"),
            not_clipped_count=("adc_not_clipped_bool", "sum"),
        )
        .reset_index()
    )
    score["power_distance_to_preferred"] = np.abs(
        score["power_tier_dbm_num"].to_numpy(dtype=float) - float(preferred_power_dbm)
    )
    score = score.sort_values(
        [
            "sample_rate_key_hz",
            "fir_mode",
            "combo_count",
            "recommended_count",
            "analysis_count",
            "not_clipped_count",
            "power_distance_to_preferred",
        ],
        ascending=[True, True, False, False, False, False, True],
        kind="stable",
    )
    preferred_by_mode = (
        score.drop_duplicates(subset=["sample_rate_key_hz", "fir_mode"], keep="first")
        .loc[:, ["sample_rate_key_hz", "fir_mode", "power_tier_dbm_num"]]
        .rename(columns={"power_tier_dbm_num": "preferred_mode_power_dbm"})
    )

    frame = frame.merge(preferred_by_mode, on=["sample_rate_key_hz", "fir_mode"], how="left")
    frame["is_preferred_mode_power"] = np.isclose(
        frame["power_tier_dbm_num"].to_numpy(dtype=float),
        frame["preferred_mode_power_dbm"].to_numpy(dtype=float),
        atol=1e-9,
        rtol=0.0,
    )
    frame["power_distance_to_mode_preference"] = np.abs(
        frame["power_tier_dbm_num"].to_numpy(dtype=float)
        - frame["preferred_mode_power_dbm"].to_numpy(dtype=float)
    )

    timestamp = frame["timestamp_utc"] if "timestamp_utc" in frame.columns else pd.Series(pd.NA, index=frame.index)
    frame["timestamp_sort"] = pd.to_datetime(timestamp, errors="coerce", utc=True)

    ordered = frame.sort_values(
        [
            "sample_rate_key_hz",
            "frequency_key_hz",
            "fir_mode",
            "is_preferred_mode_power",
            "qc_recommended_pass_bool",
            "qc_analysis_pass_bool",
            "adc_is_clipped_bool",
            "power_distance_to_mode_preference",
            "power_distance_to_preferred",
            "timestamp_sort",
            "run_id",
        ],
        ascending=[True, True, True, False, False, False, True, True, True, False, True],
        kind="stable",
    )

    selected = ordered.drop_duplicates(
        subset=["sample_rate_key_hz", "frequency_key_hz", "fir_mode"], keep="first"
    ).copy()
    selected["selection_preferred_power_dbm"] = selected["preferred_mode_power_dbm"]
    selected["selection_used_preferred_mode_power"] = selected["is_preferred_mode_power"]

    return selected.sort_values(
        ["sample_rate_key_hz", "fir_mode", "frequency_key_hz"],
        kind="stable",
    ).reset_index(drop=True)


def build_e2_bandpass_curve_table(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    preferred_power_dbm: float = -10.0,
) -> pd.DataFrame:
    """Build per-point E2 bandpass curve rows from selected analysis runs."""

    selected = select_e2_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        preferred_power_dbm=preferred_power_dbm,
    )
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "curve_label",
                "mode",
                "fir_mode",
                "sample_rate_hz_nominal",
                "frequency_hz",
                "frequency_normalized_to_nyquist",
                "frequency_normalized_to_fs",
                "amplitude_v",
                "gain_linear",
                "gain_db",
                "reference_amplitude_v",
                "power_tier_dbm",
                "status",
                "qc_analysis_pass",
                "qc_recommended_pass",
            ]
        )

    curve_rows: list[pd.DataFrame] = []
    grouped = selected.groupby(["sample_rate_key_hz", "fir_mode"], sort=False)
    for (sample_rate_key_hz, fir_mode), group in grouped:
        ordered = group.sort_values("frequency_key_hz", kind="stable").reset_index(drop=True)

        frequency_hz = ordered["signal_frequency_hz_effective"].to_numpy(dtype=float)
        amplitude = _series_numeric(ordered, "adc_mean_block_rms").to_numpy(dtype=float)
        valid = np.isfinite(frequency_hz) & np.isfinite(amplitude) & (amplitude > 0.0)
        if int(np.sum(valid)) < 2:
            continue

        filtered = ordered.loc[valid].reset_index(drop=True)
        frequency_valid = frequency_hz[valid]
        amplitude_valid = amplitude[valid]

        curve_label = _curve_label(str(fir_mode), float(sample_rate_key_hz))
        curve = bandpass_curve(frequency_valid, amplitude_valid, mode=curve_label)

        merged = curve.copy()
        merged["run_id"] = filtered["run_id"].to_numpy(dtype=object)
        merged["curve_label"] = curve_label
        merged["mode"] = curve_label
        merged["fir_mode"] = str(fir_mode)
        merged["sample_rate_hz_nominal"] = float(sample_rate_key_hz)
        merged["power_tier_dbm"] = _series_numeric(filtered, "power_tier_dbm").to_numpy(dtype=float)
        merged["status"] = filtered.get("status", pd.Series(pd.NA, index=filtered.index)).to_numpy(dtype=object)
        merged["qc_analysis_pass"] = (
            filtered.get("qc_analysis_pass", pd.Series(pd.NA, index=filtered.index)).to_numpy(dtype=object)
        )
        merged["qc_recommended_pass"] = (
            filtered.get("qc_recommended_pass", pd.Series(pd.NA, index=filtered.index)).to_numpy(dtype=object)
        )
        merged["selection_preferred_power_dbm"] = _series_numeric(
            filtered, "selection_preferred_power_dbm"
        ).to_numpy(dtype=float)
        merged["selection_used_preferred_mode_power"] = filtered.get(
            "selection_used_preferred_mode_power", pd.Series(pd.NA, index=filtered.index)
        ).to_numpy(dtype=object)
        merged["source_kind"] = filtered.get("source_kind", pd.Series(pd.NA, index=filtered.index)).to_numpy(
            dtype=object
        )
        merged["source_path"] = filtered.get("source_path", pd.Series(pd.NA, index=filtered.index)).to_numpy(
            dtype=object
        )
        merged["source_member"] = filtered.get(
            "source_member", pd.Series(pd.NA, index=filtered.index)
        ).to_numpy(dtype=object)

        sample_rate = float(sample_rate_key_hz)
        merged["frequency_normalized_to_nyquist"] = merged["frequency_hz"].to_numpy(dtype=float) / (
            sample_rate / 2.0
        )
        merged["frequency_normalized_to_fs"] = merged["frequency_hz"].to_numpy(dtype=float) / sample_rate

        curve_rows.append(merged)

    if len(curve_rows) == 0:
        raise ValueError("No analyzable E2 runs were found for bandpass curve construction.")

    table = pd.concat(curve_rows, ignore_index=True)
    return table.sort_values(["curve_label", "frequency_hz"], kind="stable").reset_index(drop=True)


def build_e2_t4_table(curve_table: pd.DataFrame) -> pd.DataFrame:
    """Build a T4-style E2 bandpass summary table from per-point curve rows."""

    if curve_table.empty:
        return empty_table("T4")

    required = {"curve_label", "frequency_hz", "gain_db", "sample_rate_hz_nominal", "fir_mode", "power_tier_dbm"}
    missing = sorted(required - set(curve_table.columns))
    if missing:
        raise ValueError(f"curve_table is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for curve_label, group in curve_table.groupby("curve_label", sort=False):
        ordered = group.sort_values("frequency_hz", kind="stable").reset_index(drop=True)
        frequency = ordered["frequency_hz"].to_numpy(dtype=float)
        gain_db = ordered["gain_db"].to_numpy(dtype=float)
        metrics = bandpass_summary_metrics(frequency, gain_db)
        rolloff_values = np.array(
            [metrics.get("left_rolloff_db_per_hz", np.nan), metrics.get("right_rolloff_db_per_hz", np.nan)],
            dtype=float,
        )
        rows.append(
            {
                "mode": str(curve_label),
                "passband_estimate_hz": float(metrics.get("passband_width_hz", np.nan)),
                "rolloff_metric_db_per_hz": float(np.nanmean(np.abs(rolloff_values))),
                "ripple_db": float(metrics.get("passband_ripple_db", np.nan)),
                "fit_residuals_db": float(metrics.get("fit_residual_rms_db", np.nan)),
                "sample_rate_hz": float(_coerce_float(ordered["sample_rate_hz_nominal"].iloc[0])),
                "fir_mode": str(ordered["fir_mode"].iloc[0]),
                "selected_power_dbm": float(np.nanmedian(_series_numeric(ordered, "power_tier_dbm"))),
                "selected_point_count": int(len(ordered)),
                "recommended_fraction": float(_series_bool(ordered, "qc_recommended_pass").mean()),
                "peak_frequency_hz": float(metrics.get("peak_frequency_hz", np.nan)),
                "peak_gain_db": float(metrics.get("peak_gain_db", np.nan)),
            }
        )

    if len(rows) == 0:
        return empty_table("T4")

    table = validate_table("T4", rows, allow_extra=True, fill_missing=False)
    return table.sort_values(["sample_rate_hz", "mode"], kind="stable").reset_index(drop=True)


def write_e2_bandpass_figure(
    curve_table: pd.DataFrame,
    path: str | Path = DEFAULT_E2_F4_FIGURE_PATH,
    *,
    reference_sample_rate_hz: float = 1.0e6,
) -> Path:
    """Render and save F4-style bandpass curves from processed E2 rows."""

    frequency_hz, gain_db_by_mode = _build_plot_ready_gain_map(
        curve_table,
        reference_sample_rate_hz=reference_sample_rate_hz,
    )
    builder = BandpassFigureBuilder()
    figure, axes = builder.build(frequency_hz, gain_db_by_mode)
    axes["main"].set_title("F4 Physical: SDR Bandpass Curves (E2)")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def _build_plot_ready_gain_map(
    curve_table: pd.DataFrame,
    *,
    reference_sample_rate_hz: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    required = {"curve_label", "frequency_hz", "gain_db", "sample_rate_hz_nominal"}
    missing = sorted(required - set(curve_table.columns))
    if missing:
        raise ValueError(f"curve_table is missing required plotting columns: {missing}")
    if curve_table.empty:
        raise ValueError("curve_table has no rows to plot.")

    grouped: dict[str, pd.DataFrame] = {}
    for curve_label, group in curve_table.groupby("curve_label", sort=False):
        ordered = group.sort_values("frequency_hz", kind="stable").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        grouped[str(curve_label)] = ordered

    if len(grouped) == 0:
        raise ValueError("curve_table has no curves with at least two points.")

    overlap_min = max(float(group["frequency_hz"].min()) for group in grouped.values())
    overlap_max = min(float(group["frequency_hz"].max()) for group in grouped.values())
    if overlap_max <= overlap_min:
        overlap_min = min(float(group["frequency_hz"].min()) for group in grouped.values())
        overlap_max = max(float(group["frequency_hz"].max()) for group in grouped.values())

    reference_label = _choose_reference_curve(
        grouped,
        reference_sample_rate_hz=float(reference_sample_rate_hz),
    )
    reference_frequency = grouped[reference_label]["frequency_hz"].to_numpy(dtype=float)
    in_overlap = (reference_frequency >= overlap_min) & (reference_frequency <= overlap_max)
    reference_frequency = reference_frequency[in_overlap]
    if reference_frequency.size < 2:
        union_frequency = np.unique(
            np.concatenate(
                [
                    group["frequency_hz"].to_numpy(dtype=float)
                    for group in grouped.values()
                ]
            )
        )
        reference_frequency = union_frequency[(union_frequency >= overlap_min) & (union_frequency <= overlap_max)]
    if reference_frequency.size < 2:
        raise ValueError("Unable to build a common frequency axis for E2 bandpass plotting.")

    gain_db_by_mode: dict[str, np.ndarray] = {}
    for curve_label, group in grouped.items():
        frequency = group["frequency_hz"].to_numpy(dtype=float)
        gain_db = group["gain_db"].to_numpy(dtype=float)

        unique_frequency, unique_index = np.unique(frequency, return_index=True)
        gain_unique = gain_db[unique_index]

        if unique_frequency.size == reference_frequency.size and np.allclose(unique_frequency, reference_frequency):
            interpolated = gain_unique
        else:
            interpolated = np.interp(reference_frequency, unique_frequency, gain_unique, left=np.nan, right=np.nan)

        if np.all(np.isnan(interpolated)):
            continue
        gain_db_by_mode[curve_label] = interpolated

    if len(gain_db_by_mode) == 0:
        raise ValueError("No plottable E2 gain curves remain after interpolation.")

    return reference_frequency, gain_db_by_mode


def _choose_reference_curve(
    grouped: dict[str, pd.DataFrame],
    *,
    reference_sample_rate_hz: float,
) -> str:
    choices: list[tuple[float, int, str]] = []
    for curve_label, group in grouped.items():
        sample_rate = float(_coerce_float(group["sample_rate_hz_nominal"].iloc[0]))
        points = int(len(group))
        choices.append((abs(sample_rate - reference_sample_rate_hz), -points, str(curve_label)))
    choices.sort(key=lambda item: (item[0], item[1], item[2]))
    return choices[0][2]


def _curve_label(fir_mode: str, sample_rate_hz: float) -> str:
    sample_rate_mhz = float(sample_rate_hz) / 1.0e6
    return f"{fir_mode}@{sample_rate_mhz:.1f}MHz"


def _effective_sample_rate(frame: pd.DataFrame) -> pd.Series:
    actual = _series_numeric(frame, "sample_rate_hz_actual")
    requested = _series_numeric(frame, "sample_rate_hz_requested")
    result = actual.copy()
    fallback = ~np.isfinite(result.to_numpy(dtype=float))
    result.loc[fallback] = requested.loc[fallback]
    return result


def _effective_signal_frequency(frame: pd.DataFrame) -> pd.Series:
    measured = _series_numeric(frame, "signal_frequency_measured_hz")
    requested = _series_numeric(frame, "signal_frequency_hz")
    result = measured.copy()
    fallback = ~np.isfinite(result.to_numpy(dtype=float))
    result.loc[fallback] = requested.loc[fallback]
    return result


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
        "experiment": metadata.get("experiment", "E2"),
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


def _series_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column] if column in frame.columns else pd.Series(False, index=frame.index)
    return values.fillna(False).astype(bool)


def _series_numeric(frame: pd.DataFrame | pd.Series, column: str | None = None) -> pd.Series:
    if isinstance(frame, pd.Series):
        values = frame
    else:
        if column is None:
            raise ValueError("column is required when frame is a DataFrame.")
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
