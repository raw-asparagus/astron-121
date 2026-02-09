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

_COMPLETED_STATUSES = {"ok_target", "ok_target_from_baseline", "closest_only", "unachievable"}
_PROGRESS_COLUMNS = (
    "timestamp_utc",
    "combo_key",
    "experiment",
    "sample_rate_hz",
    "frequency_hz",
    "fir_mode",
    "final_status",
    "message",
    "baseline_run_id",
    "baseline_npz_path",
    "baseline_power_dbm",
    "baseline_adc_rms",
    "baseline_adc_max",
    "baseline_adc_min",
    "baseline_is_clipped",
    "baseline_guard_passed",
    "baseline_requested_sample_rate_hz",
    "baseline_actual_sample_rate_hz",
    "target_run_id",
    "target_npz_path",
    "target_power_dbm",
    "target_adc_rms",
    "target_adc_max",
    "target_adc_min",
    "target_is_clipped",
    "target_guard_passed",
    "target_requested_sample_rate_hz",
    "target_actual_sample_rate_hz",
)


@dataclass(frozen=True)
class E1AcquisitionConfig:
    """Config for Experiment 1 physical data acquisition."""

    sample_rates_hz: tuple[float, ...] = (1.0e6, 1.6e6, 2.4e6, 3.2e6)
    n_frequency_points: int = 24
    nsamples: int = 2048
    nblocks: int = 11
    stale_blocks: int = 1

    baseline_power_dbm: float = -30.0
    max_power_dbm: float = 10.0
    bisection_precision_dbm: float = 0.1
    target_adc_rms_center: float = 65.0
    target_adc_rms_tolerance: float = 5.0

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
    expected_idn_substring: str = "N9310A"


@dataclass(frozen=True)
class CaptureMeasurement:
    """One accepted capture measurement at one generator power."""

    requested_power_dbm: float
    measured_power_dbm: float
    capture: SDRCaptureResult
    guard_attempts: int
    rejected_attempts: tuple[ADCSummary, ...]

    @property
    def adc_rms(self) -> float:
        return self.capture.summary.mean_block_rms


def e1_frequency_grid_hz(sample_rate_hz: float, *, n_points: int = 24) -> np.ndarray:
    """Return 24 linear points over ``[0, 4 f_Nyquist]`` (inclusive)."""

    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if n_points < 2:
        raise ValueError("n_points must be >= 2.")
    nyquist_hz = sample_rate_hz / 2.0
    return np.linspace(0.0, 4.0 * nyquist_hz, n_points, endpoint=True, dtype=float)


def e1_fir_modes() -> dict[str, np.ndarray | None]:
    """FIR mode mapping for E1: default + manual alias-hack coefficients."""

    return {
        "default": None,
        "alias_hack": alias_hack_fir_coeffs(),
    }


def run_e1_acquisition(
    config: E1AcquisitionConfig = E1AcquisitionConfig(),
    *,
    siggen: N9310AUSBTMC | None = None,
    sdr_factory: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Run E1 acquisition sweep with baseline + target captures per combination.

    Resume policy:
    - Each combination has deterministic IDs/paths.
    - Completed combinations are skipped.
    - Existing per-run NPZ files are reused and never overwritten.
    """

    _validate_config(config)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.t2_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.progress_path.parent.mkdir(parents=True, exist_ok=True)

    progress = _read_progress(config.progress_path)
    controller = siggen or N9310AUSBTMC(device_path=config.siggen_device_path, retry=config.siggen_retry)
    controller.validate_identity(expected_substring=config.expected_idn_substring)
    controller.set_rf_output_verified(True)

    try:
        for sample_rate_hz in config.sample_rates_hz:
            frequency_grid = e1_frequency_grid_hz(sample_rate_hz, n_points=config.n_frequency_points)
            for fir_mode, fir_coeffs in e1_fir_modes().items():
                for frequency_index, signal_frequency_hz in enumerate(frequency_grid):
                    combo_key = _combo_key(
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        fir_mode=fir_mode,
                        frequency_index=frequency_index,
                    )
                    if _is_combo_completed(progress, combo_key):
                        continue

                    timestamp_utc = _utc_now()
                    try:
                        measured_frequency_hz = controller.set_frequency_hz_verified(
                            signal_frequency_hz,
                            tolerance_hz=1.0,
                        )
                    except Exception as error:
                        progress = _upsert_progress_row(
                            progress,
                            {
                                "timestamp_utc": timestamp_utc,
                                "combo_key": combo_key,
                                "experiment": config.experiment_id,
                                "sample_rate_hz": float(sample_rate_hz),
                                "frequency_hz": float(signal_frequency_hz),
                                "fir_mode": fir_mode,
                                "final_status": "error_io",
                                "message": f"siggen_frequency_error:{error}",
                            },
                        )
                        _write_progress(config.progress_path, progress)
                        continue

                    baseline_run_id = f"{combo_key}__baseline"
                    target_run_id = f"{combo_key}__target"
                    baseline_path = config.raw_dir / f"{baseline_run_id}.npz"
                    target_path = config.raw_dir / f"{target_run_id}.npz"

                    baseline_measurement = _load_or_capture_measurement(
                        run_id=baseline_run_id,
                        npz_path=baseline_path,
                        run_kind="baseline_-30dBm",
                        config=config,
                        controller=controller,
                        sdr_factory=sdr_factory,
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        measured_frequency_hz=measured_frequency_hz,
                        fir_mode=fir_mode,
                        fir_coeffs=fir_coeffs,
                        requested_power_dbm=config.baseline_power_dbm,
                    )

                    if baseline_measurement is None:
                        progress = _upsert_progress_row(
                            progress,
                            {
                                "timestamp_utc": _utc_now(),
                                "combo_key": combo_key,
                                "experiment": config.experiment_id,
                                "sample_rate_hz": float(sample_rate_hz),
                                "frequency_hz": float(signal_frequency_hz),
                                "fir_mode": fir_mode,
                                "final_status": "error_io",
                                "message": "baseline_capture_failed",
                            },
                        )
                        _write_progress(config.progress_path, progress)
                        continue

                    _append_t2_manifest_row(
                        config=config,
                        run_id=baseline_run_id,
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        requested_power_dbm=baseline_measurement.requested_power_dbm,
                        fir_mode=fir_mode,
                    )

                    target_measurement, final_status, message = _select_target_measurement(
                        baseline=baseline_measurement,
                        config=config,
                        controller=controller,
                        sdr_factory=sdr_factory,
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        measured_frequency_hz=measured_frequency_hz,
                        fir_mode=fir_mode,
                        fir_coeffs=fir_coeffs,
                    )

                    if target_measurement is None:
                        progress = _upsert_progress_row(
                            progress,
                            {
                                "timestamp_utc": _utc_now(),
                                "combo_key": combo_key,
                                "experiment": config.experiment_id,
                                "sample_rate_hz": float(sample_rate_hz),
                                "frequency_hz": float(signal_frequency_hz),
                                "fir_mode": fir_mode,
                                "final_status": "error_io",
                                "message": message,
                                "baseline_run_id": baseline_run_id,
                                "baseline_npz_path": str(baseline_path),
                            },
                        )
                        _write_progress(config.progress_path, progress)
                        continue

                    if final_status == "ok_target_from_baseline" or final_status == "unachievable":
                        _save_measurement_npz(
                            run_id=target_run_id,
                            npz_path=target_path,
                            run_kind="target_65rms",
                            config=config,
                            sample_rate_hz=sample_rate_hz,
                            signal_frequency_hz=signal_frequency_hz,
                            measured_frequency_hz=measured_frequency_hz,
                            fir_mode=fir_mode,
                            fir_coeffs=fir_coeffs,
                            measurement=target_measurement,
                            status=final_status,
                            duplicate_of_run_id=baseline_run_id,
                        )
                    else:
                        _save_measurement_npz(
                            run_id=target_run_id,
                            npz_path=target_path,
                            run_kind="target_65rms",
                            config=config,
                            sample_rate_hz=sample_rate_hz,
                            signal_frequency_hz=signal_frequency_hz,
                            measured_frequency_hz=measured_frequency_hz,
                            fir_mode=fir_mode,
                            fir_coeffs=fir_coeffs,
                            measurement=target_measurement,
                            status=final_status,
                            duplicate_of_run_id=None,
                        )

                    _append_t2_manifest_row(
                        config=config,
                        run_id=target_run_id,
                        sample_rate_hz=sample_rate_hz,
                        signal_frequency_hz=signal_frequency_hz,
                        requested_power_dbm=target_measurement.requested_power_dbm,
                        fir_mode=fir_mode,
                    )

                    progress = _upsert_progress_row(
                        progress,
                        {
                            "timestamp_utc": _utc_now(),
                            "combo_key": combo_key,
                            "experiment": config.experiment_id,
                            "sample_rate_hz": float(sample_rate_hz),
                            "frequency_hz": float(signal_frequency_hz),
                            "fir_mode": fir_mode,
                            "final_status": final_status,
                            "message": message,
                            "baseline_run_id": baseline_run_id,
                            "baseline_npz_path": str(baseline_path),
                            "baseline_power_dbm": baseline_measurement.requested_power_dbm,
                            "baseline_adc_rms": baseline_measurement.capture.summary.mean_block_rms,
                            "baseline_adc_max": baseline_measurement.capture.summary.adc_max,
                            "baseline_adc_min": baseline_measurement.capture.summary.adc_min,
                            "baseline_is_clipped": baseline_measurement.capture.summary.is_clipped,
                            "baseline_guard_passed": baseline_measurement.capture.summary.passes_guard,
                            "baseline_requested_sample_rate_hz": baseline_measurement.capture.requested_sample_rate_hz,
                            "baseline_actual_sample_rate_hz": baseline_measurement.capture.actual_sample_rate_hz,
                            "target_run_id": target_run_id,
                            "target_npz_path": str(target_path),
                            "target_power_dbm": target_measurement.requested_power_dbm,
                            "target_adc_rms": target_measurement.capture.summary.mean_block_rms,
                            "target_adc_max": target_measurement.capture.summary.adc_max,
                            "target_adc_min": target_measurement.capture.summary.adc_min,
                            "target_is_clipped": target_measurement.capture.summary.is_clipped,
                            "target_guard_passed": target_measurement.capture.summary.passes_guard,
                            "target_requested_sample_rate_hz": target_measurement.capture.requested_sample_rate_hz,
                            "target_actual_sample_rate_hz": target_measurement.capture.actual_sample_rate_hz,
                        },
                    )
                    _write_progress(config.progress_path, progress)
    finally:
        try:
            controller.set_rf_output(False)
        except Exception:
            pass
    return progress.copy()


def _select_target_measurement(
    *,
    baseline: CaptureMeasurement,
    config: E1AcquisitionConfig,
    controller: N9310AUSBTMC,
    sdr_factory: Callable[..., Any] | None,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    measured_frequency_hz: float,
    fir_mode: str,
    fir_coeffs: np.ndarray | None,
) -> tuple[CaptureMeasurement | None, str, str]:
    lower_bound = config.target_adc_rms_center - config.target_adc_rms_tolerance
    upper_bound = config.target_adc_rms_center + config.target_adc_rms_tolerance

    if baseline.capture.summary.is_clipped:
        return baseline, "unachievable", "clipping_at_baseline_-30dBm"

    if _is_within_target_band(baseline.capture.summary.mean_block_rms, config=config):
        return baseline, "ok_target_from_baseline", "baseline_within_target_band"

    attempts: list[CaptureMeasurement] = [baseline]
    if baseline.capture.summary.mean_block_rms > upper_bound:
        return baseline, "closest_only", "baseline_above_target_at_min_power"

    upper = _capture_measurement(
        config=config,
        controller=controller,
        sdr_factory=sdr_factory,
        sample_rate_hz=sample_rate_hz,
        signal_frequency_hz=signal_frequency_hz,
        measured_frequency_hz=measured_frequency_hz,
        fir_mode=fir_mode,
        fir_coeffs=fir_coeffs,
        requested_power_dbm=config.max_power_dbm,
    )
    if upper is None:
        return None, "error_io", "target_search_failed_at_max_power"
    attempts.append(upper)

    if _is_within_target_band(upper.capture.summary.mean_block_rms, config=config):
        return upper, "ok_target", "target_found_at_max_power"

    if upper.capture.summary.mean_block_rms < lower_bound and not upper.capture.summary.is_clipped:
        closest = _closest_measurement(attempts, target_rms=config.target_adc_rms_center)
        return closest, "closest_only", "target_not_reached_by_max_power"

    low_power = float(config.baseline_power_dbm)
    high_power = float(config.max_power_dbm)
    low_rms = baseline.capture.summary.mean_block_rms
    high_rms = upper.capture.summary.mean_block_rms

    while (high_power - low_power) > config.bisection_precision_dbm:
        mid_power = round((low_power + high_power) / 2.0, 3)
        mid = _capture_measurement(
            config=config,
            controller=controller,
            sdr_factory=sdr_factory,
            sample_rate_hz=sample_rate_hz,
            signal_frequency_hz=signal_frequency_hz,
            measured_frequency_hz=measured_frequency_hz,
            fir_mode=fir_mode,
            fir_coeffs=fir_coeffs,
            requested_power_dbm=mid_power,
        )
        if mid is None:
            break
        attempts.append(mid)

        mid_rms = mid.capture.summary.mean_block_rms
        if _is_within_target_band(mid_rms, config=config):
            return mid, "ok_target", "target_found_by_bisection"

        if mid.capture.summary.is_clipped or mid_rms > high_rms:
            high_power = mid_power
            high_rms = mid_rms
            continue

        if mid_rms < low_rms:
            low_power = mid_power
            low_rms = mid_rms
            continue

        if mid_rms < lower_bound:
            low_power = mid_power
            low_rms = mid_rms
        else:
            high_power = mid_power
            high_rms = mid_rms

    closest = _closest_measurement(attempts, target_rms=config.target_adc_rms_center)
    return closest, "closest_only", "target_not_reached_within_precision"


def _closest_measurement(
    attempts: list[CaptureMeasurement],
    *,
    target_rms: float,
) -> CaptureMeasurement:
    guard_valid = [item for item in attempts if item.capture.summary.passes_guard]
    candidates = guard_valid if guard_valid else attempts
    return min(candidates, key=lambda item: abs(item.capture.summary.mean_block_rms - target_rms))


def _is_within_target_band(adc_rms: float, *, config: E1AcquisitionConfig) -> bool:
    lower = config.target_adc_rms_center - config.target_adc_rms_tolerance
    upper = config.target_adc_rms_center + config.target_adc_rms_tolerance
    return bool(lower <= adc_rms <= upper)


def _load_or_capture_measurement(
    *,
    run_id: str,
    npz_path: Path,
    run_kind: str,
    config: E1AcquisitionConfig,
    controller: N9310AUSBTMC,
    sdr_factory: Callable[..., Any] | None,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    measured_frequency_hz: float,
    fir_mode: str,
    fir_coeffs: np.ndarray | None,
    requested_power_dbm: float,
) -> CaptureMeasurement | None:
    if npz_path.exists():
        loaded = _load_measurement_from_npz(npz_path)
        if loaded is not None:
            return loaded

    measurement = _capture_measurement(
        config=config,
        controller=controller,
        sdr_factory=sdr_factory,
        sample_rate_hz=sample_rate_hz,
        signal_frequency_hz=signal_frequency_hz,
        measured_frequency_hz=measured_frequency_hz,
        fir_mode=fir_mode,
        fir_coeffs=fir_coeffs,
        requested_power_dbm=requested_power_dbm,
    )
    if measurement is None:
        return None

    _save_measurement_npz(
        run_id=run_id,
        npz_path=npz_path,
        run_kind=run_kind,
        config=config,
        sample_rate_hz=sample_rate_hz,
        signal_frequency_hz=signal_frequency_hz,
        measured_frequency_hz=measured_frequency_hz,
        fir_mode=fir_mode,
        fir_coeffs=fir_coeffs,
        measurement=measurement,
        status="captured",
        duplicate_of_run_id=None,
    )
    return measurement


def _capture_measurement(
    *,
    config: E1AcquisitionConfig,
    controller: N9310AUSBTMC,
    sdr_factory: Callable[..., Any] | None,
    sample_rate_hz: float,
    signal_frequency_hz: float,
    measured_frequency_hz: float,
    fir_mode: str,
    fir_coeffs: np.ndarray | None,
    requested_power_dbm: float,
) -> CaptureMeasurement | None:
    del signal_frequency_hz, measured_frequency_hz, fir_mode  # carried in caller metadata
    try:
        measured_power_dbm = controller.set_power_dbm_verified(requested_power_dbm, tolerance_dbm=0.05)
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
                measured_power_dbm=float(measured_power_dbm),
                capture=capture,
                guard_attempts=guard_attempt,
                rejected_attempts=tuple(rejected),
            )
        rejected.append(capture.summary)

    if last_capture is None:
        return None
    return CaptureMeasurement(
        requested_power_dbm=float(requested_power_dbm),
        measured_power_dbm=float(measured_power_dbm),
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
    duplicate_of_run_id: str | None,
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
        "duplicate_of_run_id": duplicate_of_run_id,
        "combo": {
            "sample_rate_hz": float(sample_rate_hz),
            "signal_frequency_hz": float(signal_frequency_hz),
            "signal_frequency_measured_hz": float(measured_frequency_hz),
            "fir_mode": fir_mode,
            "fir_coeffs": None if fir_coeffs is None else np.asarray(fir_coeffs).tolist(),
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
        # extras retained by allow_extra=True for downstream filtering
        "run_kind": "physical_acquire",
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


def _is_combo_completed(progress: pd.DataFrame, combo_key: str) -> bool:
    if progress.empty:
        return False
    matched = progress.loc[progress["combo_key"].astype(str) == combo_key]
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
    output = frame.copy()
    output.to_csv(path, index=False)


def _upsert_progress_row(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    normalized = {column: row.get(column, pd.NA) for column in _PROGRESS_COLUMNS}
    combo_key = str(normalized["combo_key"])
    if frame.empty:
        return pd.DataFrame([normalized], columns=list(_PROGRESS_COLUMNS))

    updated = frame.copy()
    mask = updated["combo_key"].astype(str) == combo_key
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
    if config.nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if config.nblocks <= 0:
        raise ValueError("nblocks must be positive.")
    if config.nblocks <= config.stale_blocks:
        raise ValueError("nblocks must be greater than stale_blocks.")
    if config.guard_max_attempts < 1:
        raise ValueError("guard_max_attempts must be >= 1.")
    if config.bisection_precision_dbm <= 0.0:
        raise ValueError("bisection_precision_dbm must be positive.")
    if config.max_power_dbm < config.baseline_power_dbm:
        raise ValueError("max_power_dbm must be >= baseline_power_dbm.")


__all__ = [
    "DEFAULT_E1_PROGRESS_PATH",
    "DEFAULT_E1_RAW_DIR",
    "DEFAULT_E1_T2_MANIFEST_PATH",
    "E1AcquisitionConfig",
    "e1_fir_modes",
    "e1_frequency_grid_hz",
    "run_e1_acquisition",
]
