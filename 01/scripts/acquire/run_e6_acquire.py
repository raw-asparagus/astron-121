#!/usr/bin/env python3
"""Run Experiment 6 physical acquisition sweep (+/- delta-nu, DSB/intermod)."""

from __future__ import annotations

import argparse

from manual_capture_common import (
    OneShotCaptureParams,
    ToneParams,
    add_common_capture_arguments,
    add_signal_generator_arguments,
    build_signed_delta_sweep,
    default_run_id,
    match_tone_to_reference,
    resolve_manual_tone,
    resolve_required_float,
    resolve_required_int,
    run_one_shot_capture,
)


def _delta_tag_hz(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    abs_value = abs(float(value))
    if abs_value.is_integer():
        magnitude = str(int(abs_value))
    else:
        magnitude = f"{abs_value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{sign}{magnitude}hz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help="Maximum absolute sideband offset |delta-nu| in Hz for sweep endpoints.",
    )
    parser.add_argument(
        "--delta-points",
        type=int,
        default=2,
        help="Number of signed delta points between -|delta-nu| and +|delta-nu| (default: 2).",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        default=False,
        help="Include delta=0 in the sweep when delta-points > 2.",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e6",
        default_manifest_path="data/manifests/t2_e6_runs.csv",
    )
    add_signal_generator_arguments(parser, include_sg1=False, include_sg2=True)
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
        rf = resolve_manual_tone(
            label="signal_generator_2",
            frequency_hz=args.sg2_frequency_hz,
            power_dbm=args.sg2_power_dbm,
        )
        lo_reference = match_tone_to_reference(rf, label="signal_generator_1")
        delta_max_hz = resolve_required_float(
            args.delta_f_hz,
            name="delta_f_hz",
            prompt="Sweep endpoint |delta-nu| (Hz)",
            min_value=1.0,
        )
        delta_points = resolve_required_int(
            args.delta_points,
            name="delta_points",
            prompt="Delta sweep points",
            min_value=2,
        )
        signed_delta_values = build_signed_delta_sweep(
            float(delta_max_hz),
            n_points=int(delta_points),
            include_zero=bool(args.include_zero),
        )
        for signed_delta in signed_delta_values:
            lo_frequency_hz = float(rf.frequency_hz) - float(signed_delta)
            if lo_frequency_hz <= 0.0:
                raise ValueError(
                    f"SG1 frequency became non-positive for signed delta {signed_delta} Hz."
                )
    except ValueError as error:
        parser.error(str(error))
        return

    run_id_base = args.run_id or default_run_id("E6")
    total_runs = len(signed_delta_values)
    print(f"E6 sweep runs={total_runs} sample_rate_mhz={sample_rate_mhz:.6g}")
    print(f"SG2 reference: f={rf.frequency_hz:.6g} Hz, p={rf.power_dbm:.6g} dBm")
    print("SG1 power is matched to SG2; SG1 frequency is swept via signed delta-nu.")

    captured = 0
    captured_guard_fail = 0
    skipped_existing = 0
    errors = 0
    for sweep_index, signed_delta_hz in enumerate(signed_delta_values, start=1):
        lo_frequency_hz = float(rf.frequency_hz) - float(signed_delta_hz)
        lo = ToneParams(
            label=str(lo_reference.label),
            frequency_hz=float(lo_frequency_hz),
            power_dbm=float(lo_reference.power_dbm),
        )
        delta_tag = _delta_tag_hz(float(signed_delta_hz))
        run_id = (
            run_id_base
            if total_runs == 1
            else f"{run_id_base}__d{delta_tag}__r{sweep_index:03d}"
        )

        params = OneShotCaptureParams(
            experiment_id="E6",
            run_kind="physical_dsb_intermod_sweep",
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
            signal_generators=(lo, rf),
            mixer_config=str(args.mixer_config),
            cable_config=str(args.cable_config),
            notes=str(args.notes),
            extra_metadata={
                "signal_generator_1_role": "LO",
                "signal_generator_2_role": "RF",
                "signed_delta_f_hz": float(signed_delta_hz),
                "delta_f_hz": float(abs(signed_delta_hz)),
                "sweep_index": int(sweep_index),
                "sweep_total_runs": int(total_runs),
                "signal_generator_1_matched_to_signal_generator_2_power": True,
                "signal_generator_1_frequency_rule": "f1 = f2 - signed_delta_f_hz",
            },
        )
        try:
            result = run_one_shot_capture(params)
        except FileExistsError:
            skipped_existing += 1
            print(f"[{sweep_index:02d}/{total_runs:02d}] skip existing run: {run_id}")
            continue
        except Exception as error:
            errors += 1
            print(f"[{sweep_index:02d}/{total_runs:02d}] error run={run_id}: {error}")
            continue

        status = "captured" if result.capture.summary.passes_guard else "captured_guard_fail"
        if result.capture.summary.passes_guard:
            captured += 1
        else:
            captured_guard_fail += 1
        print(
            f"[{sweep_index:02d}/{total_runs:02d}] run={run_id} "
            f"status={status} signed_delta_hz={signed_delta_hz:.6g} "
            f"f_lo_hz={lo.frequency_hz:.6g} f_rf_hz={rf.frequency_hz:.6g}"
        )

    print(f"Sweep complete: {total_runs} planned runs.")
    print(f"Captured: {captured}")
    print(f"Captured (guard-fail): {captured_guard_fail}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Errors: {errors}")
    print(f"T2 manifest: {args.t2_manifest_path}")
    print(f"Raw output dir: {args.raw_dir}")


if __name__ == "__main__":
    main()
