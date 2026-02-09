#!/usr/bin/env python3
"""Run Experiment 4 one-shot physical acquisition (leakage/resolution)."""

from __future__ import annotations

import argparse

from manual_capture_common import (
    OneShotCaptureParams,
    add_common_capture_arguments,
    add_signal_generator_arguments,
    default_run_id,
    resolve_required_choice,
    resolve_required_float,
    resolve_required_int,
    resolve_siggen,
    run_one_shot_capture,
)
from ugradio_lab1.control.siggen import SigGenRetryPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="E4 mode: leakage or resolution (prompted if omitted).",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="N bins/samples for this run (<16384). Prompted if omitted.",
    )
    parser.add_argument(
        "--bin-index",
        type=int,
        default=None,
        help="Primary FFT bin index for leakage-style runs.",
    )
    parser.add_argument(
        "--bin-offset",
        type=float,
        default=None,
        help="Off-bin fractional offset epsilon for leakage-style runs (e.g. 0.35).",
    )
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help="Tone spacing in Hz for resolution-style runs.",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e4",
        default_manifest_path="data/manifests/t2_e4_runs.csv",
    )
    add_signal_generator_arguments(parser, include_sg2=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        mode = resolve_required_choice(
            args.mode,
            name="mode",
            choices=("leakage", "resolution"),
            prompt="E4 mode [leakage|resolution]",
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
        n_bins = resolve_required_int(
            args.n_bins,
            name="n_bins",
            prompt="Number of bins/samples (N, must be <16384)",
            min_value=2,
            max_value=16383,
        )
        bin_index = args.bin_index
        bin_offset = args.bin_offset
        delta_f_hz = args.delta_f_hz
        if mode == "leakage":
            bin_index = resolve_required_int(
                bin_index,
                name="bin_index",
                prompt="Leakage bin index (k)",
                min_value=1,
                max_value=(n_bins // 2) - 1,
            )
            bin_offset = resolve_required_float(
                bin_offset,
                name="bin_offset",
                prompt="Leakage bin offset epsilon",
                min_value=0.0,
            )
        else:
            delta_f_hz = resolve_required_float(
                delta_f_hz,
                name="delta_f_hz",
                prompt="Resolution delta-f (Hz)",
                min_value=1.0,
            )

        sg1 = resolve_siggen(
            label="signal_generator_1",
            device_path=args.sg1_device_path,
            frequency_hz=args.sg1_frequency_hz,
            power_dbm=args.sg1_power_dbm,
            default_device_path="/dev/usbtmc0",
        )
        signal_generators = [sg1]
        if mode == "resolution":
            sg2 = resolve_siggen(
                label="signal_generator_2",
                device_path=args.sg2_device_path,
                frequency_hz=args.sg2_frequency_hz,
                power_dbm=args.sg2_power_dbm,
                default_device_path="/dev/usbtmc1",
            )
            signal_generators.append(sg2)
    except ValueError as error:
        parser.error(str(error))
        return

    retry = SigGenRetryPolicy(
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
        retry_sleep_s=float(args.retry_sleep_s),
        settle_time_s=1.0,
    )
    run_id = args.run_id or default_run_id("E4")
    params = OneShotCaptureParams(
        experiment_id="E4",
        run_kind="physical_leakage" if mode == "leakage" else "physical_resolution",
        run_id=run_id,
        raw_dir=args.raw_dir,
        t2_manifest_path=args.t2_manifest_path,
        sample_rate_hz=float(sample_rate_mhz) * 1e6,
        nsamples=int(n_bins),
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
        signal_generators=tuple(signal_generators),
        mixer_config=str(args.mixer_config),
        cable_config=str(args.cable_config),
        notes=str(args.notes),
        extra_metadata={
            "mode": mode,
            "n_bins": int(n_bins),
            "bin_index": None if bin_index is None else int(bin_index),
            "bin_offset": None if bin_offset is None else float(bin_offset),
            "delta_f_hz": None if delta_f_hz is None else float(delta_f_hz),
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
