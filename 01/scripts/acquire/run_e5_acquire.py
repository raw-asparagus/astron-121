#!/usr/bin/env python3
"""Run Experiment 5 one-shot physical acquisition (noise/ACF/radiometer)."""

from __future__ import annotations

import argparse

from manual_capture_common import (
    OneShotCaptureParams,
    add_common_capture_arguments,
    default_run_id,
    resolve_required_choice,
    resolve_required_float,
    run_one_shot_capture,
)
from ugradio_lab1.control.siggen import SigGenRetryPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-source",
        type=str,
        default=None,
        help="Noise source label: lab_noise_generator or terminated_input (prompted if omitted).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e5",
        default_manifest_path="data/manifests/t2_e5_runs.csv",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        noise_source = resolve_required_choice(
            args.noise_source,
            name="noise_source",
            choices=("lab_noise_generator", "terminated_input"),
            prompt="Noise source [lab_noise_generator|terminated_input]",
        )
        sample_rate_mhz = resolve_required_float(
            args.sample_rate_mhz,
            name="sample_rate_mhz",
            prompt="Sample rate (MHz)",
            min_value=0.001,
        )
        vrms = resolve_required_float(
            args.vrms,
            name="vrms",
            prompt="Target Vrms (V)",
            min_value=0.0,
        )
    except ValueError as error:
        parser.error(str(error))
        return

    retry = SigGenRetryPolicy(
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
        retry_sleep_s=float(args.retry_sleep_s),
        settle_time_s=1.0,
    )
    run_id = args.run_id or default_run_id("E5")
    params = OneShotCaptureParams(
        experiment_id="E5",
        run_kind="physical_noise_acf",
        run_id=run_id,
        raw_dir=args.raw_dir,
        t2_manifest_path=args.t2_manifest_path,
        sample_rate_hz=float(sample_rate_mhz) * 1e6,
        nsamples=int(args.nsamples),
        nblocks=int(args.nblocks),
        stale_blocks=int(args.stale_blocks),
        guard_max_attempts=int(args.guard_max_attempts),
        sdr_device_index=int(args.sdr_device_index),
        sdr_direct=bool(args.direct),
        sdr_gain_db=float(args.gain_db),
        sdr_timeout_s=float(args.timeout_s),
        sdr_max_retries=int(args.max_retries),
        sdr_retry_sleep_s=float(args.retry_sleep_s),
        vrms_target_v=float(vrms),
        siggen_retry=retry,
        signal_generators=tuple(),
        mixer_config=str(args.mixer_config),
        cable_config=str(args.cable_config),
        notes=str(args.notes),
        extra_metadata={"noise_source": noise_source},
    )
    result = run_one_shot_capture(params)
    print(f"Run ID: {run_id}")
    print(f"Status: {'captured' if result.capture.summary.passes_guard else 'captured_guard_fail'}")
    print(f"Guard attempts: {result.guard_attempts}")
    print(f"ADC RMS: {result.capture.summary.mean_block_rms:.3f}")
    print(f"ADC max/min: {result.capture.summary.adc_max} / {result.capture.summary.adc_min}")
    print(f"NPZ path: {result.npz_path}")
    print(f"T2 manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
