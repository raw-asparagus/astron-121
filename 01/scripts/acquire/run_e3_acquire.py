#!/usr/bin/env python3
"""Run Experiment 3 one-shot physical acquisition (voltage/power spectra)."""

from __future__ import annotations

import argparse

from manual_capture_common import (
    OneShotCaptureParams,
    add_common_capture_arguments,
    add_signal_generator_arguments,
    default_run_id,
    resolve_manual_tone,
    resolve_required_choice,
    resolve_required_float,
    run_one_shot_capture,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="E3 mode: single_tone or two_tone (prompted if omitted).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e3",
        default_manifest_path="data/manifests/t2_e3_runs.csv",
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
            choices=("single_tone", "two_tone"),
            prompt="E3 mode [single_tone|two_tone]",
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
        sg1 = resolve_manual_tone(
            label="signal_generator_1",
            frequency_hz=args.sg1_frequency_hz,
            power_dbm=args.sg1_power_dbm,
        )
        signal_generators = [sg1]
        if mode == "two_tone":
            sg2 = resolve_manual_tone(
                label="signal_generator_2",
                frequency_hz=args.sg2_frequency_hz,
                power_dbm=args.sg2_power_dbm,
            )
            signal_generators.append(sg2)
    except ValueError as error:
        parser.error(str(error))
        return

    run_id = args.run_id or default_run_id("E3")
    params = OneShotCaptureParams(
        experiment_id="E3",
        run_kind="physical_voltage_power",
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
        signal_generators=tuple(signal_generators),
        mixer_config=str(args.mixer_config),
        cable_config=str(args.cable_config),
        notes=str(args.notes),
        extra_metadata={"mode": mode},
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
