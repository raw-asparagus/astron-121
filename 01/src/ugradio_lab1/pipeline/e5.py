"""Experiment 5 raw-to-notebook pipeline helpers."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from ugradio_lab1.analysis.noise import radiometer_fit, radiometer_summary_table
from ugradio_lab1.analysis.spectra import autocorrelation, power_spectrum
from ugradio_lab1.dataio.catalog import build_manifest, write_manifest_csv
from ugradio_lab1.dataio.schema import empty_table, validate_table
from ugradio_lab1.plotting.figure_builders import (
    ACFSpectrumConsistencyFigureBuilder,
    NoiseHistogramFigureBuilder,
    RadiometerFigureBuilder,
)

_NPZ_METADATA_KEY = "__metadata_json__"

DEFAULT_E5_RAW_SOURCE = Path("data/raw/e5.tar.gz")
DEFAULT_E5_RUN_CATALOG_PATH = Path("data/interim/e5/run_catalog.csv")
DEFAULT_E5_QC_CATALOG_PATH = Path("data/interim/e5/qc_catalog.csv")
DEFAULT_E5_STATS_TABLE_PATH = Path("data/interim/e5/noise_stats.csv")
DEFAULT_E5_CURVE_TABLE_PATH = Path("data/interim/e5/radiometer_curve.csv")
DEFAULT_E5_T2_TABLE_PATH = Path("data/processed/e5/tables/T2_e5_runs.csv")
DEFAULT_E5_T6_TABLE_PATH = Path("data/processed/e5/tables/T6_e5_radiometer_summary.csv")
DEFAULT_E5_F10_FIGURE_PATH = Path("report/figures/F10_noise_histogram_physical.png")
DEFAULT_E5_F11_FIGURE_PATH = Path("report/figures/F11_radiometer_scaling_physical.png")
DEFAULT_E5_F12_FIGURE_PATH = Path("report/figures/F12_acf_spectrum_consistency_physical.png")

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
    "noise_source",
    "target_vrms_v",
    "mixer_config",
    "cable_config",
    "notes",
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


def build_e5_run_catalog(raw_source: str | Path = DEFAULT_E5_RAW_SOURCE) -> pd.DataFrame:
    """Build a normalized E5 run catalog from either ``.tar.gz`` or a directory."""

    source = Path(raw_source)
    if not source.exists():
        raise FileNotFoundError(source)

    if _is_tar_archive(source):
        records = _collect_records_from_tar(source)
    elif source.is_dir():
        records = _collect_records_from_directory(source)
    else:
        raise ValueError(f"Unsupported E5 raw source: {source}")

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


def build_e5_qc_catalog(run_catalog: pd.DataFrame) -> pd.DataFrame:
    """Compute E5 QC labels from a normalized run catalog."""

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


def build_e5_t2_table(run_catalog_or_qc: pd.DataFrame) -> pd.DataFrame:
    """Build T2-style E5 manifest rows from the normalized run catalog."""

    frame = run_catalog_or_qc.copy()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))

        rows.append(
            {
                "run_id": str(row.get("run_id")),
                "experiment": row.get("experiment", "E5"),
                "sample_rate_hz": sample_rate_hz,
                "center_frequency_hz": 0.0,
                "tones_hz": json.dumps([]),
                "source_power_dbm": float("nan"),
                "mixer_config": row.get("mixer_config", "manual_noise_source"),
                "cable_config": row.get("cable_config", "manual_noise_source"),
                "n_samples": _coerce_int(row.get("nsamples")),
                "run_kind": row.get("run_kind", pd.NA),
                "status": row.get("status", pd.NA),
                "n_blocks_saved": _coerce_int(row.get("nblocks_saved")),
                "sample_rate_hz_requested": _coerce_float(row.get("sample_rate_hz_requested")),
                "sample_rate_hz_actual": _coerce_float(row.get("sample_rate_hz_actual")),
                "sample_rate_error_hz": _coerce_float(row.get("sample_rate_error_hz")),
                "target_vrms_v": _coerce_float(row.get("target_vrms_v")),
                "noise_source": row.get("noise_source", pd.NA),
                "notes": row.get("notes", pd.NA),
                "qc_analysis_pass": row.get("qc_analysis_pass", pd.NA),
                "qc_recommended_pass": row.get("qc_recommended_pass", pd.NA),
            }
        )

    if len(rows) == 0:
        return empty_table("T2")
    return build_manifest(rows, table_id="T2", allow_extra=True)


def select_e5_analysis_runs(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Select E5 rows for physical noise/radiometer analysis."""

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

    frame["sample_rate_hz_nominal"] = _effective_sample_rate(frame)
    finite = np.isfinite(frame["sample_rate_hz_nominal"].to_numpy(dtype=float))
    frame = frame.loc[finite].copy()
    if frame.empty:
        raise ValueError("No finite E5 sample_rate_hz values were found.")

    if use_qc_analysis_pass and "qc_analysis_pass" in frame.columns:
        frame = frame.loc[_series_bool(frame, "qc_analysis_pass")].copy()
    if frame.empty:
        raise ValueError("No E5 runs remain after qc_analysis_pass filtering.")

    normalized_noise_filter = _normalize_noise_source_filter(noise_source_filter)
    if normalized_noise_filter is not None and "noise_source" in frame.columns:
        noise_values = frame["noise_source"].astype(str).str.strip().str.lower()
        frame = frame.loc[noise_values == normalized_noise_filter].copy()
    if frame.empty:
        raise ValueError("No E5 runs remain after noise_source filtering.")

    if sample_rate_hz is None:
        rounded = np.round(frame["sample_rate_hz_nominal"].to_numpy(dtype=float), 3)
        counts = pd.Series(rounded).value_counts(sort=True)
        chosen_rate = float(counts.index[0])
    else:
        chosen_rate = float(sample_rate_hz)

    frame = frame.loc[
        np.isclose(
            frame["sample_rate_hz_nominal"].to_numpy(dtype=float),
            chosen_rate,
            atol=max(1e-6, abs(chosen_rate) * 1e-9),
            rtol=0.0,
        )
    ].copy()
    if frame.empty:
        raise ValueError("No E5 runs remain after sample-rate filtering.")

    frame["qc_recommended_pass_bool"] = _series_bool(frame, "qc_recommended_pass")
    frame["adc_is_clipped_bool"] = _series_bool(frame, "adc_is_clipped")
    frame["adc_mean_block_rms_num"] = _series_numeric(frame, "adc_mean_block_rms")
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
            "qc_recommended_pass_bool",
            "status_rank",
            "adc_is_clipped_bool",
            "adc_mean_block_rms_num",
            "timestamp_sort",
            "run_id",
        ],
        ascending=[False, True, True, False, False, True],
        kind="stable",
    )
    selected = ordered.drop_duplicates(subset=["run_id"], keep="first").copy()
    selected["selection_noise_source_filter"] = normalized_noise_filter or "all"
    selected["selection_sample_rate_hz"] = chosen_rate
    return selected.sort_values(["timestamp_sort", "run_id"], kind="stable").reset_index(drop=True)


def select_e5_reference_run(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
) -> pd.Series:
    """Select one best E5 run row for physical F12 figure generation."""

    selected = select_e5_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        noise_source_filter=noise_source_filter,
        sample_rate_hz=sample_rate_hz,
    )
    ordered = selected.sort_values(
        [
            "qc_recommended_pass_bool",
            "status_rank",
            "adc_is_clipped_bool",
            "adc_mean_block_rms_num",
            "timestamp_sort",
            "run_id",
        ],
        ascending=[False, True, True, False, False, True],
        kind="stable",
    )
    return ordered.iloc[0]


def build_e5_noise_stats_table(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
) -> pd.DataFrame:
    """Build per-run E5 noise-statistic rows from selected physical captures."""

    selected = select_e5_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        noise_source_filter=noise_source_filter,
        sample_rate_hz=sample_rate_hz,
    )

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        arrays = _load_arrays_for_catalog_row(row)
        blocks = np.asarray(arrays.get("adc_counts"), dtype=np.int8)
        if blocks.ndim != 2 or blocks.shape[0] < 1 or blocks.shape[1] < 1:
            raise ValueError(f"run_id={row.get('run_id')} has invalid adc_counts shape: {blocks.shape!r}")

        flat = blocks.astype(float).reshape(-1)
        block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
        rows.append(
            {
                "run_id": str(row.get("run_id")),
                "noise_source": row.get("noise_source", pd.NA),
                "sample_rate_hz": _coerce_float(row.get("sample_rate_hz_nominal")),
                "n_blocks": int(blocks.shape[0]),
                "n_samples_per_block": int(blocks.shape[1]),
                "n_total_samples": int(flat.size),
                "mean_counts": float(np.mean(flat)),
                "std_counts": float(np.std(flat, ddof=1 if flat.size > 1 else 0)),
                "rms_counts": float(np.sqrt(np.mean(np.square(flat)))),
                "median_counts": float(np.median(flat)),
                "p05_counts": float(np.quantile(flat, 0.05)),
                "p95_counts": float(np.quantile(flat, 0.95)),
                "mean_block_rms_counts": float(np.mean(block_rms)),
                "qc_analysis_pass": row.get("qc_analysis_pass", pd.NA),
                "qc_recommended_pass": row.get("qc_recommended_pass", pd.NA),
            }
        )

    if len(rows) == 0:
        raise ValueError("No E5 runs were selected for noise-stat table construction.")

    return pd.DataFrame(rows).sort_values("run_id", kind="stable").reset_index(drop=True)


def build_e5_radiometer_curve_table(
    run_catalog_or_qc: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
    block_size: int = 256,
    n_avg_values: tuple[int, ...] = (1, 2, 4, 8, 16),
    min_groups: int = 2,
) -> pd.DataFrame:
    """Build E5 radiometer-curve rows from selected runs."""

    selected = select_e5_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        noise_source_filter=noise_source_filter,
        sample_rate_hz=sample_rate_hz,
    )
    blocks, effective_sample_rate_hz = _stack_selected_blocks(selected)

    segments = _segment_blocks(blocks, block_size=block_size)
    block_powers_v2 = np.mean(np.square(segments.astype(float)), axis=1)

    candidates = sorted({int(value) for value in n_avg_values if int(value) > 0})
    if len(candidates) == 0:
        raise ValueError("n_avg_values must include at least one positive integer.")
    if min_groups < 2:
        raise ValueError("min_groups must be >= 2.")

    rows: list[dict[str, Any]] = []
    sigma_reference: float | None = None
    for n_avg in candidates:
        n_groups = int(block_powers_v2.size // n_avg)
        if n_groups < int(min_groups):
            continue

        grouped = block_powers_v2[: n_groups * n_avg].reshape(n_groups, n_avg)
        averaged_power = np.mean(grouped, axis=1)
        sigma_power = float(np.std(averaged_power, ddof=1))
        if not np.isfinite(sigma_power) or sigma_power <= 0.0:
            continue

        if sigma_reference is None:
            sigma_reference = sigma_power * np.sqrt(float(n_avg))
        expected_sigma_power = float(sigma_reference / np.sqrt(float(n_avg)))

        rows.append(
            {
                "n_avg": int(n_avg),
                "sigma_power": float(sigma_power),
                "expected_sigma_power": expected_sigma_power,
                "n_groups": int(n_groups),
                "total_segments": int(block_powers_v2.size),
                "block_size": int(block_size),
                "sample_rate_hz": float(effective_sample_rate_hz),
                "num_runs_used": int(selected.shape[0]),
                "noise_source_filter": _normalize_noise_source_filter(noise_source_filter) or "all",
                "run_ids_json": json.dumps(selected["run_id"].astype(str).tolist()),
            }
        )

    if len(rows) < 2:
        raise ValueError("Need at least two E5 radiometer points after grouping filters.")

    return pd.DataFrame(rows).sort_values("n_avg", kind="stable").reset_index(drop=True)


def fit_e5_radiometer(radiometer_curve: pd.DataFrame) -> dict[str, float]:
    """Fit log-log radiometer slope from a curve table."""

    table = radiometer_curve.copy()
    required = {"n_avg", "sigma_power"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"radiometer_curve is missing required columns: {missing}")
    return radiometer_fit(
        table["n_avg"].to_numpy(dtype=float),
        table["sigma_power"].to_numpy(dtype=float),
    )


def build_e5_t6_table(
    radiometer_curve: pd.DataFrame,
    *,
    fit_result: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build processed T6 summary rows from an E5 radiometer curve table."""

    curve = radiometer_curve.copy()
    required = {"n_avg", "sigma_power", "block_size"}
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"radiometer_curve is missing required columns: {missing}")

    fit = fit_result or fit_e5_radiometer(curve)
    block_size = int(curve["block_size"].iloc[0])

    t6 = radiometer_summary_table(
        curve["n_avg"].to_numpy(dtype=float),
        curve["sigma_power"].to_numpy(dtype=float),
        block_size=block_size,
        fit_result=fit,
    )

    extras = curve.loc[
        :, [
            "n_avg",
            "expected_sigma_power",
            "n_groups",
            "total_segments",
            "sample_rate_hz",
            "num_runs_used",
            "noise_source_filter",
            "run_ids_json",
        ]
    ].copy()
    merged = t6.merge(extras, on="n_avg", how="left")
    merged["fitted_intercept"] = float(fit["intercept"])
    merged["r_squared"] = float(fit["r_squared"])
    merged["slope_ci_low"] = float(fit.get("slope_ci_low", np.nan))
    merged["slope_ci_high"] = float(fit.get("slope_ci_high", np.nan))
    return validate_table("T6", merged, allow_extra=True)


def write_e5_noise_histogram_figure(
    run_catalog_or_qc: pd.DataFrame,
    path: str | Path = DEFAULT_E5_F10_FIGURE_PATH,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
    bins: int = 70,
) -> Path:
    """Render and save F10-style physical noise histogram."""

    selected = select_e5_analysis_runs(
        run_catalog_or_qc,
        use_qc_analysis_pass=use_qc_analysis_pass,
        noise_source_filter=noise_source_filter,
        sample_rate_hz=sample_rate_hz,
    )
    blocks, _ = _stack_selected_blocks(selected)
    samples = blocks.reshape(-1)

    builder = NoiseHistogramFigureBuilder()
    figure, _ = builder.build(samples, bins=int(bins))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_e5_radiometer_figure(
    radiometer_curve: pd.DataFrame,
    path: str | Path = DEFAULT_E5_F11_FIGURE_PATH,
    *,
    fit_result: dict[str, float] | None = None,
) -> Path:
    """Render and save F11-style physical radiometer scaling figure."""

    curve = radiometer_curve.copy()
    required = {"n_avg", "sigma_power"}
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"radiometer_curve is missing required columns: {missing}")

    fit = fit_result or fit_e5_radiometer(curve)

    builder = RadiometerFigureBuilder()
    figure, _ = builder.build(
        curve["n_avg"].to_numpy(dtype=float),
        curve["sigma_power"].to_numpy(dtype=float),
        fit_result=fit,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def write_e5_acf_consistency_figure(
    run_catalog_or_qc: pd.DataFrame,
    path: str | Path = DEFAULT_E5_F12_FIGURE_PATH,
    *,
    use_qc_analysis_pass: bool = True,
    noise_source_filter: str | None = "lab_noise_generator",
    sample_rate_hz: float | None = None,
    run_id: str | None = None,
) -> Path:
    """Render and save F12-style physical ACF/spectrum consistency figure."""

    frame = run_catalog_or_qc.copy()
    if run_id is not None:
        frame = frame.loc[frame["run_id"].astype(str) == str(run_id)].copy()
        if frame.empty:
            raise ValueError(f"run_id={run_id!r} is not present in run_catalog_or_qc.")
        reference = select_e5_reference_run(
            frame,
            use_qc_analysis_pass=False,
            noise_source_filter=None,
            sample_rate_hz=sample_rate_hz,
        )
    else:
        reference = select_e5_reference_run(
            frame,
            use_qc_analysis_pass=use_qc_analysis_pass,
            noise_source_filter=noise_source_filter,
            sample_rate_hz=sample_rate_hz,
        )

    arrays = _load_arrays_for_catalog_row(reference)
    blocks = np.asarray(arrays.get("adc_counts"), dtype=np.int8)
    if blocks.ndim != 2 or blocks.shape[0] < 1 or blocks.shape[1] < 1:
        raise ValueError(f"run_id={reference.get('run_id')} has invalid adc_counts shape: {blocks.shape!r}")

    sample_rate_hz_value = _coerce_float(reference.get("sample_rate_hz_actual"))
    if not np.isfinite(sample_rate_hz_value):
        sample_rate_hz_value = _coerce_float(reference.get("sample_rate_hz_requested"))
    if not np.isfinite(sample_rate_hz_value) or sample_rate_hz_value <= 0.0:
        raise ValueError(f"run_id={reference.get('run_id')} has invalid sample rate: {sample_rate_hz_value}")

    voltage = blocks.astype(float).reshape(-1)
    frequency_hz, power_v2 = power_spectrum(
        voltage,
        sample_rate_hz=float(sample_rate_hz_value),
        window=None,
        detrend=True,
        scaling="power",
        center=False,
        fft_backend="numpy",
    )
    lag_s, autocorrelation_values = autocorrelation(
        voltage,
        sample_rate_hz=float(sample_rate_hz_value),
        detrend=True,
    )

    builder = ACFSpectrumConsistencyFigureBuilder()
    figure, _ = builder.build(
        lag_s,
        np.real(np.asarray(autocorrelation_values)),
        frequency_hz,
        power_v2,
    )
    figure.suptitle(f"F12 Physical E5 | run_id={reference.get('run_id')}", y=1.02)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return destination


def _stack_selected_blocks(selected_rows: pd.DataFrame) -> tuple[np.ndarray, float]:
    if selected_rows.empty:
        raise ValueError("selected_rows has no rows.")

    blocks_list: list[np.ndarray] = []
    sample_rates: list[float] = []
    for _, row in selected_rows.iterrows():
        arrays = _load_arrays_for_catalog_row(row)
        blocks = np.asarray(arrays.get("adc_counts"), dtype=np.int8)
        if blocks.ndim != 2 or blocks.shape[0] < 1 or blocks.shape[1] < 1:
            raise ValueError(f"run_id={row.get('run_id')} has invalid adc_counts shape: {blocks.shape!r}")

        blocks_list.append(blocks.astype(float))
        sample_rate_hz = _coerce_float(row.get("sample_rate_hz_nominal"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_actual"))
        if not np.isfinite(sample_rate_hz):
            sample_rate_hz = _coerce_float(row.get("sample_rate_hz_requested"))
        sample_rates.append(sample_rate_hz)

    if len(blocks_list) == 0:
        raise ValueError("No E5 blocks could be loaded from selected_rows.")

    valid_rates = [value for value in sample_rates if np.isfinite(value) and value > 0.0]
    if len(valid_rates) == 0:
        raise ValueError("No valid sample_rate_hz values found for selected E5 runs.")
    reference_rate = float(valid_rates[0])
    for value in valid_rates[1:]:
        if not np.isclose(value, reference_rate, atol=max(1e-6, abs(reference_rate) * 1e-9), rtol=0.0):
            raise ValueError("Selected E5 runs have mismatched sample rates.")

    return np.vstack(blocks_list), reference_rate


def _segment_blocks(blocks: np.ndarray, *, block_size: int) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if blocks.ndim != 2 or blocks.shape[0] < 1 or blocks.shape[1] < 1:
        raise ValueError("blocks must be a non-empty 2D array.")

    n_segments_per_block = int(blocks.shape[1] // block_size)
    if n_segments_per_block < 1:
        raise ValueError(
            f"block_size={block_size} is too large for nsamples={blocks.shape[1]} in selected E5 runs."
        )

    usable_samples = n_segments_per_block * int(block_size)
    trimmed = blocks[:, :usable_samples]
    return trimmed.reshape(trimmed.shape[0] * n_segments_per_block, int(block_size))


def _normalize_noise_source_filter(noise_source_filter: str | None) -> str | None:
    if noise_source_filter is None:
        return None
    normalized = str(noise_source_filter).strip().lower()
    if normalized in {"", "all", "*", "any"}:
        return None
    return normalized


def _effective_sample_rate(frame: pd.DataFrame) -> pd.Series:
    actual = _series_numeric(frame, "sample_rate_hz_actual")
    requested = _series_numeric(frame, "sample_rate_hz_requested")
    return actual.where(np.isfinite(actual), requested)


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
    extra = metadata.get("extra", {}) if isinstance(metadata.get("extra", {}), dict) else {}

    adc_counts = np.asarray(arrays.get("adc_counts", np.empty((0, 0))), dtype=np.int8)

    return {
        "run_id": str(metadata.get("run_id", Path(npz_name).stem)),
        "npz_name": npz_name,
        "source_kind": source_kind,
        "source_path": source_path,
        "source_member": source_member,
        "experiment": metadata.get("experiment", "E5"),
        "status": metadata.get("status", pd.NA),
        "run_kind": metadata.get("run_kind", pd.NA),
        "timestamp_utc": metadata.get("timestamp_utc", pd.NA),
        "sample_rate_hz_requested": _coerce_float(sample_rate.get("requested_hz")),
        "sample_rate_hz_actual": _coerce_float(sample_rate.get("actual_hz")),
        "sample_rate_error_hz": _coerce_float(sample_rate.get("error_hz")),
        "noise_source": extra.get("noise_source", metadata.get("noise_source", pd.NA)),
        "target_vrms_v": _coerce_float(metadata.get("target_vrms_v")),
        "mixer_config": metadata.get("mixer_config", "manual_noise_source"),
        "cable_config": metadata.get("cable_config", "manual_noise_source"),
        "notes": metadata.get("notes", pd.NA),
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
