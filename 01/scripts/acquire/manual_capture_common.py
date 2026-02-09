#!/usr/bin/env python3
"""Shared helpers for one-shot physical acquisition scripts (E3-E7)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

# Allow running scripts directly from the repo without installing the package.
_LAB_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _LAB_ROOT / "src"
if _SRC_ROOT.exists() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from ugradio_lab1.control.sdr import ADCSummary, SDRCaptureConfig, SDRCaptureResult, acquire_sdr_capture
from ugradio_lab1.dataio.catalog import append_manifest_rows
from ugradio_lab1.dataio.io_npz import save_npz_dataset


@dataclass(frozen=True)
class ToneParams:
    """One manually configured tone description."""

    label: str
    frequency_hz: float
    power_dbm: float


@dataclass(frozen=True)
class OneShotCaptureParams:
    """One physical run configuration for E3-E7 style captures."""

    experiment_id: str
    run_kind: str
    run_id: str
    raw_dir: Path
    t2_manifest_path: Path
    sample_rate_hz: float
    nsamples: int
    nblocks: int
    stale_blocks: int
    guard_max_attempts: int
    sdr_device_index: int
    sdr_direct: bool
    sdr_gain_db: float
    sdr_timeout_s: float
    sdr_max_retries: int
    sdr_retry_sleep_s: float
    vrms_target_v: float | None
    signal_generators: tuple[ToneParams, ...]
    center_frequency_hz: float = 0.0
    mixer_config: str = "direct_sdr"
    cable_config: str = "siggen_to_sdr_direct"
    notes: str = ""
    extra_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class OneShotCaptureResult:
    """Result bundle from a one-shot capture."""

    npz_path: Path
    manifest_path: Path
    capture: SDRCaptureResult
    guard_attempts: int
    rejected_attempts: tuple[ADCSummary, ...]


def default_run_id(experiment_id: str) -> str:
    """Build a deterministic timestamp run id prefix."""

    tag = str(experiment_id).strip().lower()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{tag}_{stamp}"


def resolve_required_choice(
    value: str | None,
    *,
    name: str,
    choices: Sequence[str],
    prompt: str,
) -> str:
    normalized_choices = tuple(str(choice) for choice in choices)
    if value is None:
        value = _prompt_if_interactive(prompt)
    normalized = str(value).strip()
    if normalized not in normalized_choices:
        allowed = ", ".join(normalized_choices)
        raise ValueError(f"{name} must be one of: {allowed}.")
    return normalized


def resolve_required_float(
    value: float | None,
    *,
    name: str,
    prompt: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if value is None:
        value = _prompt_float(prompt)
    parsed = float(value)
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return parsed


def resolve_required_int(
    value: int | None,
    *,
    name: str,
    prompt: str,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    if value is None:
        value = _prompt_int(prompt)
    parsed = int(value)
    if min_value is not None and parsed < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and parsed > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return parsed


def run_one_shot_capture(params: OneShotCaptureParams) -> OneShotCaptureResult:
    """Capture SDR blocks and persist NPZ + T2 row."""

    _validate_capture_params(params)
    params.raw_dir.mkdir(parents=True, exist_ok=True)
    params.t2_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    npz_path = params.raw_dir / f"{params.run_id}.npz"
    if npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run file: {npz_path}")

    capture, guard_attempts, rejected_attempts = _capture_with_guard(params)

    blocks = np.asarray(capture.blocks, dtype=np.int8)
    block_rms = np.sqrt(np.mean(np.square(blocks.astype(float)), axis=1))
    status = "captured" if capture.summary.passes_guard else "captured_guard_fail"

    metadata = {
        "run_id": params.run_id,
        "experiment": params.experiment_id,
        "timestamp_utc": _utc_now(),
        "status": status,
        "run_kind": params.run_kind,
        "notes": params.notes,
        "target_vrms_v": None if params.vrms_target_v is None else float(params.vrms_target_v),
        "signal_generators": [
            {
                "label": setting.label,
                "frequency_hz": float(setting.frequency_hz),
                "power_dbm": float(setting.power_dbm),
            }
            for setting in params.signal_generators
        ],
        "signal_setup": "manual_analog",
        "capture_settings": {
            "device_index": int(params.sdr_device_index),
            "direct": bool(params.sdr_direct),
            "gain_db": float(params.sdr_gain_db),
            "nsamples": int(params.nsamples),
            "nblocks_requested": int(params.nblocks),
            "stale_blocks_dropped": int(params.stale_blocks),
            "nblocks_saved": int(params.nblocks - params.stale_blocks),
        },
        "sample_rate": {
            "requested_hz": float(capture.requested_sample_rate_hz),
            "actual_hz": float(capture.actual_sample_rate_hz),
            "error_hz": float(capture.actual_sample_rate_hz - capture.requested_sample_rate_hz),
        },
        "adc_summary": {
            "mean_block_rms": float(capture.summary.mean_block_rms),
            "adc_max": int(capture.summary.adc_max),
            "adc_min": int(capture.summary.adc_min),
            "is_clipped": bool(capture.summary.is_clipped),
            "passes_guard": bool(capture.summary.passes_guard),
            "guard_attempts": int(guard_attempts),
            "rejected_attempts": [
                {
                    "mean_block_rms": float(summary.mean_block_rms),
                    "adc_max": int(summary.adc_max),
                    "adc_min": int(summary.adc_min),
                    "is_clipped": bool(summary.is_clipped),
                    "passes_guard": bool(summary.passes_guard),
                }
                for summary in rejected_attempts
            ],
        },
        "extra": dict(params.extra_metadata or {}),
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

    tones_hz = [float(setting.frequency_hz) for setting in params.signal_generators]
    powers = [float(setting.power_dbm) for setting in params.signal_generators]
    source_power_dbm = float(np.mean(powers)) if powers else np.nan
    manifest_row = {
        "run_id": params.run_id,
        "experiment": params.experiment_id,
        "sample_rate_hz": float(capture.requested_sample_rate_hz),
        "center_frequency_hz": float(params.center_frequency_hz),
        "tones_hz": json.dumps(tones_hz),
        "source_power_dbm": source_power_dbm,
        "mixer_config": params.mixer_config,
        "cable_config": params.cable_config,
        "n_samples": int(params.nsamples),
        "run_kind": params.run_kind,
        "n_blocks_saved": int(params.nblocks - params.stale_blocks),
        "target_vrms_v": None if params.vrms_target_v is None else float(params.vrms_target_v),
        "status": status,
        "notes": params.notes,
    }
    append_manifest_rows(params.t2_manifest_path, [manifest_row], table_id="T2", allow_extra=True)
    return OneShotCaptureResult(
        npz_path=npz_path,
        manifest_path=params.t2_manifest_path,
        capture=capture,
        guard_attempts=guard_attempts,
        rejected_attempts=rejected_attempts,
    )


def _capture_with_guard(
    params: OneShotCaptureParams,
) -> tuple[SDRCaptureResult, int, tuple[ADCSummary, ...]]:
    rejected: list[ADCSummary] = []
    last_capture: SDRCaptureResult | None = None
    for guard_attempt in range(1, params.guard_max_attempts + 1):
        capture = acquire_sdr_capture(
            SDRCaptureConfig(
                sample_rate_hz=float(params.sample_rate_hz),
                device_index=int(params.sdr_device_index),
                direct=bool(params.sdr_direct),
                gain=float(params.sdr_gain_db),
                fir_coeffs=None,
                nsamples=int(params.nsamples),
                nblocks=int(params.nblocks),
                stale_blocks=int(params.stale_blocks),
                timeout_s=float(params.sdr_timeout_s),
                max_retries=int(params.sdr_max_retries),
                retry_sleep_s=float(params.sdr_retry_sleep_s),
            )
        )
        last_capture = capture
        if capture.summary.passes_guard:
            return capture, guard_attempt, tuple(rejected)
        rejected.append(capture.summary)

    if last_capture is None:
        raise RuntimeError("Capture did not produce any data.")
    return last_capture, params.guard_max_attempts, tuple(rejected)


def _validate_capture_params(params: OneShotCaptureParams) -> None:
    if params.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if params.nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if params.nblocks <= 0:
        raise ValueError("nblocks must be positive.")
    if params.nblocks <= params.stale_blocks:
        raise ValueError("nblocks must be greater than stale_blocks.")
    if params.guard_max_attempts < 1:
        raise ValueError("guard_max_attempts must be >= 1.")
    if not str(params.run_id).strip():
        raise ValueError("run_id cannot be empty.")
    for setting in params.signal_generators:
        if setting.frequency_hz <= 0.0:
            raise ValueError(f"{setting.label} frequency_hz must be positive.")
        if setting.power_dbm < -130.0 or setting.power_dbm > 25.0:
            raise ValueError(f"{setting.label} power_dbm must be in [-130, 25].")


def _prompt_if_interactive(prompt: str) -> str:
    if not sys.stdin.isatty():
        raise ValueError(
            f"Missing required parameter ({prompt}). Provide via CLI when stdin is non-interactive."
        )
    return input(f"{prompt}: ")


def _prompt_float(prompt: str) -> float:
    raw = _prompt_if_interactive(prompt).strip()
    try:
        return float(raw)
    except ValueError as error:
        raise ValueError(f"Expected float for {prompt!r}.") from error


def _prompt_int(prompt: str) -> int:
    raw = _prompt_if_interactive(prompt).strip()
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"Expected integer for {prompt!r}.") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_common_capture_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_raw_dir: str,
    default_manifest_path: str,
) -> None:
    """Add common one-shot capture CLI arguments."""

    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamp-based).")
    parser.add_argument("--vrms", type=float, default=None, help="Target Vrms for this run.")
    parser.add_argument("--sample-rate-mhz", type=float, default=None, help="SDR sample rate in MHz.")
    parser.add_argument("--nsamples", type=int, default=2048, help="Samples per block (default: 2048).")
    parser.add_argument(
        "--nblocks",
        type=int,
        default=6,
        help="Requested SDR blocks before stale-drop (default: 6).",
    )
    parser.add_argument(
        "--stale-blocks",
        type=int,
        default=1,
        help="Number of stale leading blocks to drop (default: 1).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(default_raw_dir),
        help=f"Output directory for NPZ capture files (default: {default_raw_dir}).",
    )
    parser.add_argument(
        "--t2-manifest-path",
        type=Path,
        default=Path(default_manifest_path),
        help=f"T2 CSV manifest path (default: {default_manifest_path}).",
    )
    parser.add_argument(
        "--mixer-config",
        type=str,
        default="direct_sdr",
        help="Mixer configuration label stored in manifest metadata.",
    )
    parser.add_argument(
        "--cable-config",
        type=str,
        default="siggen_to_sdr_direct",
        help="Cable configuration label stored in manifest metadata.",
    )
    parser.add_argument("--notes", type=str, default="", help="Free-text run notes.")
    parser.add_argument("--sdr-device-index", type=int, default=0, help="SDR device index (default: 0).")
    parser.add_argument("--gain-db", type=float, default=0.0, help="SDR gain in dB (default: 0.0).")
    parser.add_argument(
        "--direct",
        dest="direct",
        action="store_true",
        default=True,
        help="Use SDR direct-sampling mode (default: true).",
    )
    parser.add_argument(
        "--no-direct",
        dest="direct",
        action="store_false",
        help="Disable SDR direct-sampling mode.",
    )
    parser.add_argument("--timeout-s", type=float, default=10.0, help="I/O timeout seconds (default: 10.0).")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries (default: 3).")
    parser.add_argument(
        "--retry-sleep-s",
        type=float,
        default=0.25,
        help="Retry sleep seconds (default: 0.25).",
    )
    parser.add_argument(
        "--guard-max-attempts",
        type=int,
        default=3,
        help="Guard-based recapture attempts (default: 3).",
    )


def add_signal_generator_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_sg2: bool = True,
) -> None:
    """Add SG1/SG2 CLI arguments (optional, validated later)."""

    parser.add_argument("--sg1-frequency-hz", type=float, default=None, help="Signal generator 1 frequency in Hz.")
    parser.add_argument("--sg1-power-dbm", type=float, default=None, help="Signal generator 1 power in dBm.")
    if include_sg2:
        parser.add_argument(
            "--sg2-frequency-hz",
            type=float,
            default=None,
            help="Signal generator 2 frequency in Hz.",
        )
        parser.add_argument("--sg2-power-dbm", type=float, default=None, help="Signal generator 2 power in dBm.")


def resolve_manual_tone(
    *,
    label: str,
    frequency_hz: float | None,
    power_dbm: float | None,
) -> ToneParams:
    """Resolve one manual tone config from CLI value or interactive prompt."""

    frequency = resolve_required_float(
        frequency_hz,
        name=f"{label} frequency_hz",
        prompt=f"{label} frequency (Hz)",
        min_value=1.0,
    )
    power = resolve_required_float(
        power_dbm,
        name=f"{label} power_dbm",
        prompt=f"{label} power (dBm)",
        min_value=-130.0,
        max_value=25.0,
    )
    return ToneParams(
        label=label,
        frequency_hz=frequency,
        power_dbm=power,
    )
