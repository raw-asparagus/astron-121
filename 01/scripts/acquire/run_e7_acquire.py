#!/usr/bin/env python3
"""Run Experiment 7 one-shot physical acquisition (SSB/reverted DSB/R820T)."""

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
        help="E7 mode: ssb_external, reverted_dsb, or r820t_internal (prompted if omitted).",
    )
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help="Sideband offset delta-f in Hz (optional metadata; computed from SG frequencies if possible).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e7",
        default_manifest_path="data/manifests/t2_e7_runs.csv",
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
            choices=("ssb_external", "reverted_dsb", "r820t_internal"),
            prompt="E7 mode [ssb_external|reverted_dsb|r820t_internal]",
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
        role_1 = "LO" if mode != "r820t_internal" else "RF"
        role_2 = None
        if mode != "r820t_internal":
            sg2 = resolve_manual_tone(
                label="signal_generator_2",
                frequency_hz=args.sg2_frequency_hz,
                power_dbm=args.sg2_power_dbm,
            )
            signal_generators.append(sg2)
            role_2 = "RF"

        if args.delta_f_hz is not None:
            delta_f_hz = float(args.delta_f_hz)
        elif len(signal_generators) >= 2:
            delta_f_hz = abs(
                float(signal_generators[1].frequency_hz) - float(signal_generators[0].frequency_hz)
            )
        else:
            delta_f_hz = None
        if delta_f_hz is not None and delta_f_hz <= 0.0:
            raise ValueError("delta_f_hz must be positive when provided.")
    except ValueError as error:
        parser.error(str(error))
        return

    run_id = args.run_id or default_run_id("E7")
    params = OneShotCaptureParams(
        experiment_id="E7",
        run_kind=f"physical_{mode}",
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
        extra_metadata={
            "mode": mode,
            "signal_generator_1_role": role_1,
            "signal_generator_2_role": role_2,
            "delta_f_hz": delta_f_hz,
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
