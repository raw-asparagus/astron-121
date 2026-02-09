#!/usr/bin/env python3
"""Run Experiment 6 one-shot physical acquisition (DSB/intermod)."""

from __future__ import annotations

import argparse

from manual_capture_common import (
    OneShotCaptureParams,
    add_common_capture_arguments,
    add_signal_generator_arguments,
    default_run_id,
    resolve_required_float,
    resolve_siggen,
    run_one_shot_capture,
)
from ugradio_lab1.control.siggen import SigGenRetryPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help="Sideband offset delta-f in Hz (optional; computed from SG frequencies if omitted).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e6",
        default_manifest_path="data/manifests/t2_e6_runs.csv",
    )
    add_signal_generator_arguments(parser, include_sg2=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
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
        lo = resolve_siggen(
            label="signal_generator_1",
            device_path=args.sg1_device_path,
            frequency_hz=args.sg1_frequency_hz,
            power_dbm=args.sg1_power_dbm,
            default_device_path="/dev/usbtmc0",
        )
        rf = resolve_siggen(
            label="signal_generator_2",
            device_path=args.sg2_device_path,
            frequency_hz=args.sg2_frequency_hz,
            power_dbm=args.sg2_power_dbm,
            default_device_path="/dev/usbtmc1",
        )
        delta_f_hz = (
            float(args.delta_f_hz)
            if args.delta_f_hz is not None
            else abs(float(rf.frequency_hz) - float(lo.frequency_hz))
        )
        if delta_f_hz <= 0.0:
            raise ValueError("delta_f_hz must be positive.")
    except ValueError as error:
        parser.error(str(error))
        return

    retry = SigGenRetryPolicy(
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
        retry_sleep_s=float(args.retry_sleep_s),
        settle_time_s=1.0,
    )
    run_id = args.run_id or default_run_id("E6")
    params = OneShotCaptureParams(
        experiment_id="E6",
        run_kind="physical_dsb_intermod",
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
        signal_generators=(lo, rf),
        mixer_config=str(args.mixer_config),
        cable_config=str(args.cable_config),
        notes=str(args.notes),
        extra_metadata={
            "signal_generator_1_role": "LO",
            "signal_generator_2_role": "RF",
            "delta_f_hz": float(delta_f_hz),
        },
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
