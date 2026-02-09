"""Acquisition orchestration for synchronized instrument runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ugradio_lab1.control.sdr import (
    ADCSummary,
    SDRCaptureConfig,
    SDRCaptureResult,
    acquire_sdr_capture,
    alias_hack_fir_coeffs,
)
from ugradio_lab1.control.siggen import N9310AUSBTMC, SigGenRetryPolicy
from ugradio_lab1.dataio.catalog import append_manifest_rows, read_manifest_csv
from ugradio_lab1.dataio.io_npz import load_npz_dataset, save_npz_dataset


DEFAULT_E1_RAW_DIR = Path("data/raw/e1")
DEFAULT_E1_T2_MANIFEST_PATH = Path("data/manifests/t2_e1_runs.csv")
DEFAULT_E1_PROGRESS_PATH = Path("data/manifests/e1_progress.csv")

_COMPLETED_STATUSES = {"captured", "captured_guard_fail", "skip_existing_npz"}
_PROGRESS_COLUMNS = (
    "timestamp_utc",
    "run_id",
    "combo_key",
    "experiment",
    "sample_rate_hz",
    "frequency_hz",
    "fir_mode",
    "power_dbm",
    "final_status",
    "message",
    "npz_path",
    "adc_rms",
    "adc_max",
    "adc_min",
    "is_clipped",
    "guard_passed",
    "requested_sample_rate_hz",
    "actual_sample_rate_hz",
)


@dataclass(frozen=True)
class E1AcquisitionConfig:
    """Config for Experiment 1 physical data acquisition."""

    sample_rates_hz: tuple[float, ...] = (1.0e6, 1.6e6, 2.4e6, 3.2e6)
    n_frequency_points: int = 24
    nsamples: int = 2048
    nblocks: int = 6
    stale_blocks: int = 1

    power_tiers_default_dbm: tuple[float, ...] = (-10.0, 0.0, 10.0)
    power_tiers_alias_dbm: tuple[float, ...] = (-50.0, -40.0, -30.0)

    experiment_id: str = "E1"
    raw_dir: Path = DEFAULT_E1_RAW_DIR
    t2_manifest_path: Path = DEFAULT_E1_T2_MANIFEST_PATH
    progress_path: Path = DEFAULT_E1_PROGRESS_PATH

    siggen_device_path: Path = Path("/dev/usbtmc0")
    siggen_retry: SigGenRetryPolicy = field(default_factory=SigGenRetryPolicy)
    siggen_settle_s: float = 1.0

    sdr_device_index: int = 0
    sdr_direct: bool = True
    sdr_gain: float = 0.0
    sdr_timeout_s: float = 10.0
    sdr_max_retries: int = 3
    sdr_retry_sleep_s: float = 0.25
    guard_max_attempts: int = 3

    center_frequency_hz: float = 0.0
    cable_config: str = "siggen_to_sdr_direct"
    mixer_config_prefix: str = "direct_sdr"


@dataclass(frozen=True)
class CaptureMeasurement:
    """One accepted capture measurement at one generator power."""

    requested_power_dbm: float
    measured_power_dbm: float
    capture: SDRCaptureResult
    guard_attempts: int
    rejected_attempts: tuple[ADCSummary, ...]


def e1_frequency_grid_hz(
    sample_rate_hz: float,
    *,
    n_points: int = 24,
    include_zero_hz: bool = False,
) -> np.ndarray:
    """Return linear points over ``[0, 4 f_Nyquist]``.

    By default, ``0 Hz`` is omitted because the connected signal generator
    cannot set an exact zero-frequency CW output.
    """

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    nyquist_hz = sample_rate_hz / 2.0
    grid = np.linspace(0.0, 4.0 * nyquist_hz, n_points, endpoint=True, dtype=float)
    if include_zero_hz:
        return grid
    return grid[grid > 0.0]


def e1_fir_modes() -> dict[str, np.ndarray | None]:
    """FIR mode mapping for E1: default + manual alias-hack coefficients."""

    return {
        "default": None,
        "alias_hack": alias_hack_fir_coeffs(),
    }


def e1_power_tiers_dbm(config: E1AcquisitionConfig) -> dict[str, tuple[float, ...]]:
    """Power tiers per FIR mode for fixed-tier E1 acquisition."""

    return {
        "default": tuple(float(value) for value in config.power_tiers_default_dbm),
        "alias_hack": tuple(float(value) for value in config.power_tiers_alias_dbm),
    }


def run_e1_acquisition(
    config: E1AcquisitionConfig = E1AcquisitionConfig(),
    *,
    siggen: N9310AUSBTMC | None = None,
    sdr_factory: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Run E1 fixed-tier physical acquisition sweep.

    Protocol:
    - For each (sample rate, frequency, FIR mode), capture all fixed power tiers.
    - No bisection/target search is performed.
    - ``f=0 Hz`` is skipped by construction.
    """

    _validate_config(config)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.t2_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.progress_path.parent.mkdir(parents=True, exist_ok=True)

    progress = _read_progress(config.progress_path)
    controller = siggen or N9310AUSBTMC(device_path=config.siggen_device_path, retry=config.siggen_retry)
    fir_modes = e1_fir_modes()
    power_tiers = e1_power_tiers_dbm(config)

    controller.rf_on()
    try:
        for sample_rate_hz in config.sample_rates_hz:
            frequency_grid = e1_frequency_grid_hz(
                sample_rate_hz,
                n_points=config.n_frequency_points,
                include_zero_hz=False,
            )
            for fir_mode, fir_coeffs in fir_modes.items():
                tiers = power_tiers[fir_mode]
                for frequency_index, signal_frequency_hz in enumerate(frequency_grid):
                    combo_key = _combo_key(
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        fir_mode=fir_mode,
                        frequency_index=frequency_index,
                    )

                    timestamp_utc = _utc_now()
                    try:
                        controller.set_freq_mhz(signal_frequency_hz / 1e6)
                        measured_frequency_hz = float(signal_frequency_hz)
                    except Exception as error:
                        for requested_power_dbm in tiers:
                            run_id = f"{combo_key}__p{_power_tag_dbm(requested_power_dbm)}"
                            if _is_run_completed(progress, run_id):
                                continue
                            progress = _upsert_progress_row(
                                progress,
                                {
                                    "timestamp_utc": timestamp_utc,
                                    "run_id": run_id,
                                    "combo_key": combo_key,
                                    "experiment": config.experiment_id,
                                    "sample_rate_hz": float(sample_rate_hz),
                                    "frequency_hz": float(signal_frequency_hz),
                                    "fir_mode": fir_mode,
                                    "power_dbm": float(requested_power_dbm),
                                    "final_status": "error_io",
                                    "message": f"siggen_frequency_error:{error}",
                                },
                            )
                        _write_progress(config.progress_path, progress)
                        continue

                    for requested_power_dbm in tiers:
                        run_id = f"{combo_key}__p{_power_tag_dbm(requested_power_dbm)}"
                        npz_path = config.raw_dir / f"{run_id}.npz"
                        if _is_run_completed(progress, run_id):
                            continue

                        loaded = _load_measurement_from_npz(npz_path) if npz_path.exists() else None
                        if loaded is not None:
                            final_status = _measurement_status(loaded)
                            message = "loaded_existing_npz"
                            measurement = loaded
                        else:
                            measurement = _capture_measurement(
                                config=config,
                                controller=controller,
                                sdr_factory=sdr_factory,
                                sample_rate_hz=sample_rate_hz,
                                fir_coeffs=fir_coeffs,
                                requested_power_dbm=requested_power_dbm,
                            )
                            if measurement is None:
                                progress = _upsert_progress_row(
                                    progress,
                                    {
                                        "timestamp_utc": _utc_now(),
                                        "run_id": run_id,
                                        "combo_key": combo_key,
                                        "experiment": config.experiment_id,
                                        "sample_rate_hz": float(sample_rate_hz),
                                        "frequency_hz": float(signal_frequency_hz),
                                        "fir_mode": fir_mode,
                                        "power_dbm": float(requested_power_dbm),
                                        "final_status": "error_io",
                                        "message": "capture_failed",
                                        "npz_path": str(npz_path),
                                    },
                                )
                                _write_progress(config.progress_path, progress)
                                continue
                            final_status = _measurement_status(measurement)
                            message = (
                                "capture_complete"
                                if final_status == "captured"
                                else "capture_complete_guard_fail"
                            )
                            _save_measurement_npz(
                                run_id=run_id,
                                npz_path=npz_path,
                                run_kind="power_tier_capture",
                                config=config,
                                sample_rate_hz=sample_rate_hz,
                                signal_frequency_hz=signal_frequency_hz,
                                measured_frequency_hz=measured_frequency_hz,
                                fir_mode=fir_mode,
                                fir_coeffs=fir_coeffs,
                                measurement=measurement,
                                status=final_status,
                            )

                        _append_t2_manifest_row(
                            config=config,
                            run_id=run_id,
                            sample_rate_hz=sample_rate_hz,
                            signal_frequency_hz=signal_frequency_hz,
                            requested_power_dbm=requested_power_dbm,
                            fir_mode=fir_mode,
                        )
                        progress = _upsert_progress_row(
                            progress,
                            {
                                "timestamp_utc": _utc_now(),
                                "run_id": run_id,
                                "combo_key": combo_key,
                                "experiment": config.experiment_id,
                                "sample_rate_hz": float(sample_rate_hz),
                                "frequency_hz": float(signal_frequency_hz),
                                "fir_mode": fir_mode,
                                "power_dbm": float(requested_power_dbm),
                                "final_status": final_status,
                                "message": message,
                                "npz_path": str(npz_path),
                                "adc_rms": measurement.capture.summary.mean_block_rms,
                                "adc_max": measurement.capture.summary.adc_max,
                                "adc_min": measurement.capture.summary.adc_min,
                                "is_clipped": measurement.capture.summary.is_clipped,
                                "guard_passed": measurement.capture.summary.passes_guard,
                                "requested_sample_rate_hz": measurement.capture.requested_sample_rate_hz,
                                "actual_sample_rate_hz": measurement.capture.actual_sample_rate_hz,
                            },
                        )
                        _write_progress(config.progress_path, progress)
    finally:
        try:
            controller.rf_off()
        except Exception:
            pass
    return progress.copy()


def _measurement_status(measurement: CaptureMeasurement) -> str:
    return "captured" if measurement.capture.summary.passes_guard else "captured_guard_fail"


def _capture_measurement(
    *,
    config: E1AcquisitionConfig,
    controller: N9310AUSBTMC,
    sdr_factory: Callable[..., Any] | None,
    sample_rate_hz: float,
    fir_coeffs: np.ndarray | None,
    requested_power_dbm: float,
) -> CaptureMeasurement | None:
    try:
        controller.set_ampl_dbm(requested_power_dbm)
        measured_power_dbm = float(requested_power_dbm)
    except Exception:
        return None

    rejected: list[ADCSummary] = []
    last_capture: SDRCaptureResult | None = None
    for guard_attempt in range(1, config.guard_max_attempts + 1):
        capture_config = SDRCaptureConfig(
            sample_rate_hz=float(sample_rate_hz),
            device_index=config.sdr_device_index,
            direct=config.sdr_direct,
            gain=float(config.sdr_gain),
            fir_coeffs=fir_coeffs,
            nsamples=int(config.nsamples),
            nblocks=int(config.nblocks),
            stale_blocks=int(config.stale_blocks),
            timeout_s=float(config.sdr_timeout_s),
            max_retries=int(config.sdr_max_retries),
            retry_sleep_s=float(config.sdr_retry_sleep_s),
        )
        try:
            capture = acquire_sdr_capture(capture_config, sdr_factory=sdr_factory)
        except Exception:
            return None

        last_capture = capture
        if capture.summary.passes_guard:
            return CaptureMeasurement(
                requested_power_dbm=float(requested_power_dbm),
                measured_power_dbm=measured_power_dbm,
                capture=capture,
                guard_attempts=guard_attempt,
                rejected_attempts=tuple(rejected),
            )
        rejected.append(capture.summary)

    if last_capture is None:
        return None
    return CaptureMeasurement(
        requested_power_dbm=float(requested_power_dbm),
        measured_power_dbm=measured_power_dbm,
        capture=last_capture,
        guard_attempts=config.guard_max_attempts,
        rejected_attempts=tuple(rejected),
    )


def _save_measurement_npz(
    *,
    run_id: str,
    npz_path: Path,
    run_kind: str,
    config: E1AcquisitionConfig,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    measured_frequency_hz: float,
    fir_mode: str,
    fir_coeffs: np.ndarray | None,
    measurement: CaptureMeasurement,
    status: str,
) -> None:
    if npz_path.exists():
        return

    blocks = np.asarray(measurement.capture.blocks, dtype=np.int8)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    metadata = {
        "run_id": run_id,
        "experiment": config.experiment_id,
        "timestamp_utc": _utc_now(),
        "status": status,
        "run_kind": run_kind,
        "combo": {
            "sample_rate_hz": float(sample_rate_hz),
            "signal_frequency_hz": float(signal_frequency_hz),
            "signal_frequency_measured_hz": float(measured_frequency_hz),
            "fir_mode": fir_mode,
            "fir_coeffs": None if fir_coeffs is None else np.asarray(fir_coeffs).tolist(),
            "power_tier_dbm": float(measurement.requested_power_dbm),
        },
        "capture_settings": {
            "device_index": int(config.sdr_device_index),
            "direct": bool(config.sdr_direct),
            "gain_db": float(config.sdr_gain),
            "nsamples": int(config.nsamples),
            "nblocks_requested": int(config.nblocks),
            "stale_blocks_dropped": int(config.stale_blocks),
            "nblocks_saved": int(config.nblocks - config.stale_blocks),
        },
        "signal_generator": {
            "device_path": str(config.siggen_device_path),
            "power_requested_dbm": float(measurement.requested_power_dbm),
            "power_measured_dbm": float(measurement.measured_power_dbm),
            "settle_time_s": float(config.siggen_settle_s),
        },
        "sample_rate": {
            "requested_hz": float(measurement.capture.requested_sample_rate_hz),
            "actual_hz": float(measurement.capture.actual_sample_rate_hz),
            "error_hz": float(
                measurement.capture.actual_sample_rate_hz - measurement.capture.requested_sample_rate_hz
            ),
        },
        "adc_summary": {
            "mean_block_rms": float(measurement.capture.summary.mean_block_rms),
            "adc_max": int(measurement.capture.summary.adc_max),
            "adc_min": int(measurement.capture.summary.adc_min),
            "is_clipped": bool(measurement.capture.summary.is_clipped),
            "passes_guard": bool(measurement.capture.summary.passes_guard),
            "guard_attempts": int(measurement.guard_attempts),
            "rejected_attempts": [
                {
                    "mean_block_rms": float(summary.mean_block_rms),
                    "adc_max": int(summary.adc_max),
                    "adc_min": int(summary.adc_min),
                    "is_clipped": bool(summary.is_clipped),
                    "passes_guard": bool(summary.passes_guard),
                }
                for summary in measurement.rejected_attempts
            ],
        },
    }
    save_npz_dataset(
        npz_path,
        arrays={
            "adc_counts": blocks,
            "block_rms": np.asarray(block_rms, dtype=float),
        },
        metadata=metadata,
        overwrite=False,
    )


def _load_measurement_from_npz(npz_path: Path) -> CaptureMeasurement | None:
    try:
        arrays, metadata = load_npz_dataset(npz_path)
    except Exception:
        return None
    if "adc_counts" not in arrays:
        return None
    blocks = np.asarray(arrays["adc_counts"], dtype=np.int8)
    summary = ADCSummary(
        mean_block_rms=float(metadata.get("adc_summary", {}).get("mean_block_rms", np.nan)),
        adc_max=int(metadata.get("adc_summary", {}).get("adc_max", int(np.max(blocks)))),
        adc_min=int(metadata.get("adc_summary", {}).get("adc_min", int(np.min(blocks)))),
        is_clipped=bool(metadata.get("adc_summary", {}).get("is_clipped", False)),
        passes_guard=bool(metadata.get("adc_summary", {}).get("passes_guard", False)),
    )
    capture = SDRCaptureResult(
        blocks=blocks,
        summary=summary,
        requested_sample_rate_hz=float(metadata.get("sample_rate", {}).get("requested_hz", np.nan)),
        actual_sample_rate_hz=float(metadata.get("sample_rate", {}).get("actual_hz", np.nan)),
    )
    return CaptureMeasurement(
        requested_power_dbm=float(metadata.get("signal_generator", {}).get("power_requested_dbm", np.nan)),
        measured_power_dbm=float(metadata.get("signal_generator", {}).get("power_measured_dbm", np.nan)),
        capture=capture,
        guard_attempts=int(metadata.get("adc_summary", {}).get("guard_attempts", 1)),
        rejected_attempts=tuple(),
    )


def _append_t2_manifest_row(
    *,
    config: E1AcquisitionConfig,
    run_id: str,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    requested_power_dbm: float,
    fir_mode: str,
) -> None:
    if config.t2_manifest_path.exists():
        existing = read_manifest_csv(config.t2_manifest_path, table_id="T2")
        if not existing.empty and run_id in set(existing["run_id"].astype(str)):
            return

    row = {
        "run_id": run_id,
        "experiment": config.experiment_id,
        "sample_rate_hz": float(sample_rate_hz),
        "center_frequency_hz": float(config.center_frequency_hz),
        "tones_hz": json.dumps([float(signal_frequency_hz)]),
        "source_power_dbm": float(requested_power_dbm),
        "mixer_config": f"{config.mixer_config_prefix}:{fir_mode}",
        "cable_config": config.cable_config,
        "n_samples": int(config.nsamples),
        "run_kind": "physical_power_tier",
        "n_blocks_saved": int(config.nblocks - config.stale_blocks),
    }
    append_manifest_rows(config.t2_manifest_path, [row], table_id="T2", allow_extra=True)


def _combo_key(
    *,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    fir_mode: str,
    frequency_index: int,
) -> str:
    sr_tag = int(round(sample_rate_hz))
    freq_tag = int(round(signal_frequency_hz))
    return f"e1_sr{sr_tag}_f{freq_tag}_idx{frequency_index:02d}_fir_{fir_mode}"


def _power_tag_dbm(power_dbm: float) -> str:
    sign = "p" if power_dbm >= 0.0 else "m"
    magnitude = str(f"{abs(float(power_dbm)):.1f}").replace(".", "p")
    return f"{sign}{magnitude}"


def _is_run_completed(progress: pd.DataFrame, run_id: str) -> bool:
    if progress.empty:
        return False
    matched = progress.loc[progress["run_id"].astype(str) == run_id]
    if matched.empty:
        return False
    latest = matched.iloc[-1]
    return str(latest.get("final_status", "")) in _COMPLETED_STATUSES


def _read_progress(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(_PROGRESS_COLUMNS))
    frame = pd.read_csv(path)
    for column in _PROGRESS_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame.loc[:, list(_PROGRESS_COLUMNS)].copy()


def _write_progress(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


def _upsert_progress_row(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    normalized = {column: row.get(column, pd.NA) for column in _PROGRESS_COLUMNS}
    run_id = str(normalized["run_id"])
    if frame.empty:
        return pd.DataFrame([normalized], columns=list(_PROGRESS_COLUMNS))

    updated = frame.copy()
    mask = updated["run_id"].astype(str) == run_id
    if mask.any():
        updated.loc[mask, :] = pd.DataFrame([normalized], columns=list(_PROGRESS_COLUMNS)).iloc[0].to_numpy()
        return updated.reset_index(drop=True)
    updated = pd.concat([updated, pd.DataFrame([normalized])], ignore_index=True)
    return updated.loc[:, list(_PROGRESS_COLUMNS)].reset_index(drop=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_config(config: E1AcquisitionConfig) -> None:
    if len(config.sample_rates_hz) == 0:
        raise ValueError("sample_rates_hz cannot be empty.")
    if any(rate <= 0.0 for rate in config.sample_rates_hz):
        raise ValueError("sample_rates_hz must all be positive.")
    if config.n_frequency_points < 2:
        raise ValueError("n_frequency_points must be >= 2.")
    if config.nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if config.nblocks <= 0:
        raise ValueError("nblocks must be positive.")
    if config.nblocks <= config.stale_blocks:
        raise ValueError("nblocks must be greater than stale_blocks.")
    if config.guard_max_attempts < 1:
        raise ValueError("guard_max_attempts must be >= 1.")
    if len(config.power_tiers_default_dbm) == 0:
        raise ValueError("power_tiers_default_dbm cannot be empty.")
    if len(config.power_tiers_alias_dbm) == 0:
        raise ValueError("power_tiers_alias_dbm cannot be empty.")


__all__ = [
    "DEFAULT_E1_PROGRESS_PATH",
    "DEFAULT_E1_RAW_DIR",
    "DEFAULT_E1_T2_MANIFEST_PATH",
    "E1AcquisitionConfig",
    "e1_fir_modes",
    "e1_frequency_grid_hz",
    "e1_power_tiers_dbm",
    "run_e1_acquisition",
]
