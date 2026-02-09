#!/usr/bin/env python3
"""Run Experiment 7 physical acquisition sweep (+/- delta-nu, SSB/DSB/R820T)."""

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
    resolve_required_choice,
    resolve_required_float,
    resolve_required_int,
    run_one_shot_capture,
)

_R820T_NU_HZ = 1_420_405_751.768  # 1420.405751768 MHz (1.420405751768 GHz)
_R820T_DELTA_FRACTION = 0.05


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
        "--mode",
        type=str,
        default=None,
        help="E7 mode: ssb_external, reverted_dsb, or r820t_internal (prompted if omitted).",
    )
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help=(
            "Maximum absolute sideband offset |delta-nu| in Hz for sweep endpoints. "
            "If omitted in r820t_internal mode, defaults to 0.05 * r820t_nu_hz."
        ),
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
    parser.add_argument(
        "--iq-component",
        type=str,
        default=None,
        help="Captured component with one SDR: I (real) or Q (imaginary). Prompted if omitted.",
    )
    parser.add_argument(
        "--r820t-nu-hz",
        type=float,
        default=_R820T_NU_HZ,
        help=(
            "R820T LO center frequency nu in Hz for r820t_internal mode "
            "(default: 1420.405751768 MHz)."
        ),
    )
    parser.add_argument(
        "--r820t-delta-fraction",
        type=float,
        default=_R820T_DELTA_FRACTION,
        help="Default |delta-nu| fraction of nu for r820t_internal mode (default: 0.05).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e7",
        default_manifest_path="data/manifests/t2_e7_runs.csv",
    )
    add_signal_generator_arguments(parser, include_sg1=False, include_sg2=True)
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
        iq_choice = resolve_required_choice(
            args.iq_component,
            name="iq_component",
            choices=("i", "q", "real", "imaginary"),
            prompt="E7 captured component [I|Q|real|imaginary]",
            case_sensitive=False,
        )
        iq_key = str(iq_choice).strip().lower()
        iq_component = "I" if iq_key in ("i", "real") else "Q"
        voltage_component = "real" if iq_component == "I" else "imaginary"
        delta_points = resolve_required_int(
            args.delta_points,
            name="delta_points",
            prompt="Delta sweep points",
            min_value=2,
        )

        if mode == "r820t_internal":
            r820t_nu_hz = resolve_required_float(
                args.r820t_nu_hz,
                name="r820t_nu_hz",
                prompt="R820T LO nu (Hz)",
                min_value=1.0,
            )
            delta_fraction = resolve_required_float(
                args.r820t_delta_fraction,
                name="r820t_delta_fraction",
                prompt="R820T delta fraction",
                min_value=1e-9,
            )
            default_delta_hz = float(delta_fraction) * float(r820t_nu_hz)
            if default_delta_hz <= 0.0:
                raise ValueError("Computed default delta_f_hz must be positive.")
            delta_max_hz = (
                float(args.delta_f_hz)
                if args.delta_f_hz is not None
                else float(default_delta_hz)
            )
            if delta_max_hz <= 0.0:
                raise ValueError("delta_f_hz must be positive.")
            sg2_power_dbm = resolve_required_float(
                args.sg2_power_dbm,
                name="signal_generator_2 power_dbm",
                prompt="signal_generator_2 power (dBm)",
                min_value=-130.0,
                max_value=25.0,
            )
            sg2_reference = ToneParams(
                label="signal_generator_2",
                frequency_hz=float(r820t_nu_hz),
                power_dbm=float(sg2_power_dbm),
            )
            signed_delta_values = build_signed_delta_sweep(
                float(delta_max_hz),
                n_points=int(delta_points),
                include_zero=bool(args.include_zero),
            )
            for signed_delta_hz in signed_delta_values:
                sg2_frequency_hz = float(r820t_nu_hz) + float(signed_delta_hz)
                if sg2_frequency_hz <= 0.0:
                    raise ValueError(
                        f"SG2 frequency became non-positive for signed delta {signed_delta_hz} Hz."
                    )
        else:
            sg2_reference = resolve_manual_tone(
                label="signal_generator_2",
                frequency_hz=args.sg2_frequency_hz,
                power_dbm=args.sg2_power_dbm,
            )
            sg1_reference = match_tone_to_reference(sg2_reference, label="signal_generator_1")
            delta_max_hz = resolve_required_float(
                args.delta_f_hz,
                name="delta_f_hz",
                prompt="Sweep endpoint |delta-nu| (Hz)",
                min_value=1.0,
            )
            signed_delta_values = build_signed_delta_sweep(
                float(delta_max_hz),
                n_points=int(delta_points),
                include_zero=bool(args.include_zero),
            )
            for signed_delta_hz in signed_delta_values:
                sg1_frequency_hz = float(sg2_reference.frequency_hz) - float(signed_delta_hz)
                if sg1_frequency_hz <= 0.0:
                    raise ValueError(
                        f"SG1 frequency became non-positive for signed delta {signed_delta_hz} Hz."
                    )
    except ValueError as error:
        parser.error(str(error))
        return

    run_id_base = args.run_id or default_run_id("E7")
    total_runs = len(signed_delta_values)
    print(f"E7 sweep mode={mode} runs={total_runs} sample_rate_mhz={sample_rate_mhz:.6g}")
    print(f"Captured component: {iq_component} ({voltage_component} voltage)")
    print(
        f"SG2 reference: f={sg2_reference.frequency_hz:.6g} Hz, "
        f"p={sg2_reference.power_dbm:.6g} dBm"
    )
    if mode == "r820t_internal":
        print(
            "R820T profile: direct=False, center_frequency_hz=nu, "
            "SG2 frequency swept as nu + signed_delta."
        )
    else:
        print("SG1 power is matched to SG2; SG1 frequency is swept via signed delta-nu.")

    captured = 0
    captured_guard_fail = 0
    skipped_existing = 0
    errors = 0
    for sweep_index, signed_delta_hz in enumerate(signed_delta_values, start=1):
        if mode == "r820t_internal":
            sg2_frequency_hz = float(sg2_reference.frequency_hz) + float(signed_delta_hz)
            signal_generators = (
                ToneParams(
                    label=str(sg2_reference.label),
                    frequency_hz=float(sg2_frequency_hz),
                    power_dbm=float(sg2_reference.power_dbm),
                ),
            )
            role_1 = None
            role_2 = "RF"
            center_frequency_hz = float(sg2_reference.frequency_hz)
            sdr_direct = False
        else:
            sg1_frequency_hz = float(sg2_reference.frequency_hz) - float(signed_delta_hz)
            sg1 = ToneParams(
                label=str(sg1_reference.label),
                frequency_hz=float(sg1_frequency_hz),
                power_dbm=float(sg1_reference.power_dbm),
            )
            signal_generators = (sg1, sg2_reference)
            role_1 = "LO"
            role_2 = "RF"
            center_frequency_hz = 0.0
            sdr_direct = bool(args.direct)

        delta_tag = _delta_tag_hz(float(signed_delta_hz))
        run_id = (
            run_id_base
            if total_runs == 1
            else f"{run_id_base}__d{delta_tag}__r{sweep_index:03d}"
        )
        params = OneShotCaptureParams(
            experiment_id="E7",
            run_kind=f"physical_{mode}_sweep",
            run_id=run_id,
            raw_dir=args.raw_dir,
            t2_manifest_path=args.t2_manifest_path,
            sample_rate_hz=float(sample_rate_mhz) * 1e6,
            nsamples=int(args.nsamples),
            nblocks=int(args.nblocks),
            stale_blocks=int(args.stale_blocks),
            guard_max_attempts=int(args.guard_max_attempts),
            sdr_device_index=int(args.sdr_device_index),
            sdr_direct=bool(sdr_direct),
            sdr_gain_db=float(args.gain_db),
            sdr_timeout_s=float(args.timeout_s),
            sdr_max_retries=int(args.max_retries),
            sdr_retry_sleep_s=float(args.retry_sleep_s),
            vrms_target_v=float(vrms),
            signal_generators=tuple(signal_generators),
            center_frequency_hz=float(center_frequency_hz),
            mixer_config=str(args.mixer_config),
            cable_config=str(args.cable_config),
            notes=str(args.notes),
            extra_metadata={
                "mode": mode,
                "iq_component": iq_component,
                "voltage_component": voltage_component,
                "signal_generator_1_role": role_1,
                "signal_generator_2_role": role_2,
                "signed_delta_f_hz": float(signed_delta_hz),
                "delta_f_hz": float(abs(signed_delta_hz)),
                "sweep_index": int(sweep_index),
                "sweep_total_runs": int(total_runs),
                "signal_generator_2_reference_only": bool(mode == "r820t_internal"),
                "sdr_direct_forced_false": bool(mode == "r820t_internal"),
                "sdr_lo_center_frequency_hz": float(center_frequency_hz),
                "r820t_delta_fraction": (
                    None if mode != "r820t_internal" else float(args.r820t_delta_fraction)
                ),
                "signal_generator_1_matched_to_signal_generator_2_power": bool(
                    mode != "r820t_internal"
                ),
                "signal_generator_1_frequency_rule": (
                    "f1 = f2 - signed_delta_f_hz" if mode != "r820t_internal" else None
                ),
                "signal_generator_2_frequency_rule": (
                    "f2 = nu + signed_delta_f_hz" if mode == "r820t_internal" else None
                ),
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
            f"center_hz={center_frequency_hz:.6g} component={iq_component}"
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
