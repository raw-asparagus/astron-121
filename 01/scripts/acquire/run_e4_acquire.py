#!/usr/bin/env python3
"""Run Experiment 4 physical acquisition sweep (auto leakage/resolution planning)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from manual_capture_common import (
    OneShotCaptureParams,
    ToneParams,
    add_common_capture_arguments,
    default_run_id,
    resolve_required_choice,
    resolve_required_float,
    resolve_required_int,
    run_one_shot_capture,
)

from ugradio_lab1.control.e4_planning import leakage_tone_from_center, resolution_tones_from_center
from ugradio_lab1.control.siggen import N9310AUSBTMC, SigGenRetryPolicy


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _parse_n_bins_list(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("n_bins_list must contain at least one integer.")
    return tuple(values)


def _pick_evenly_spaced(values: Sequence[int], count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return []
    if len(values) < count:
        raise ValueError("Not enough candidate values to sample.")
    if count == 1:
        return [int(values[len(values) // 2])]

    chosen: list[int] = []
    used_indices: set[int] = set()
    last = len(values) - 1
    for i in range(count):
        target = int(round((i * last) / (count - 1)))
        index = target
        while index in used_indices and index < last:
            index += 1
        if index in used_indices:
            index = target
            while index in used_indices and index > 0:
                index -= 1
        if index in used_indices:
            raise ValueError("Failed to choose unique evenly spaced values.")
        used_indices.add(index)
        chosen.append(int(values[index]))
    return chosen


def build_n_bins_sweep(
    *,
    total_runs: int,
    min_bins: int,
    max_bins: int,
    required_min_bins: int,
) -> tuple[int, ...]:
    if total_runs < 1:
        raise ValueError("total_runs must be >= 1.")
    if max_bins > 16383:
        raise ValueError("max_bins must be <= 16383.")
    if min_bins < 2:
        raise ValueError("min_bins must be >= 2.")
    low = max(int(min_bins), int(required_min_bins))
    high = int(max_bins)
    if low > high:
        raise ValueError("min_bins/required_min_bins exceed max_bins.")

    pow2_candidates = [value for value in range(low, high + 1) if _is_power_of_two(value)]
    nonpow_candidates = [value for value in range(low, high + 1) if not _is_power_of_two(value)]

    if not pow2_candidates and not nonpow_candidates:
        raise ValueError("No valid bin counts available for the requested bounds.")
    if total_runs > (len(pow2_candidates) + len(nonpow_candidates)):
        raise ValueError("Requested total_runs exceeds available unique bin counts.")

    if len(pow2_candidates) >= total_runs:
        return tuple(sorted(_pick_evenly_spaced(pow2_candidates, total_runs)))

    pow2_selected = list(pow2_candidates)
    nonpow_needed = total_runs - len(pow2_selected)
    nonpow_selected = _pick_evenly_spaced(nonpow_candidates, nonpow_needed)
    values = sorted(pow2_selected + nonpow_selected)
    return tuple(int(value) for value in values)


def _resolve_n_bins_values(
    args: argparse.Namespace,
    *,
    required_min_bins: int,
) -> tuple[int, ...]:
    if int(args.max_bins) > 16383:
        raise ValueError("max_bins must be <= 16383.")
    if args.n_bins is not None:
        n_value = resolve_required_int(
            int(args.n_bins),
            name="n_bins",
            prompt="Number of bins/samples (N)",
            min_value=required_min_bins,
            max_value=int(args.max_bins),
        )
        return (int(n_value),)
    if args.n_bins_list is not None:
        values = _parse_n_bins_list(args.n_bins_list)
        checked: list[int] = []
        for value in values:
            checked.append(
                resolve_required_int(
                    int(value),
                    name="n_bins",
                    prompt="Number of bins/samples (N)",
                    min_value=required_min_bins,
                    max_value=int(args.max_bins),
                )
            )
        return tuple(checked)
    return build_n_bins_sweep(
        total_runs=int(args.sweep_total_runs),
        min_bins=int(args.min_bins),
        max_bins=int(args.max_bins),
        required_min_bins=required_min_bins,
    )


def _build_siggen_controller(
    *,
    device_path: Path,
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
    settle_time_s: float,
) -> N9310AUSBTMC:
    retry = SigGenRetryPolicy(
        timeout_s=float(timeout_s),
        max_retries=int(max_retries),
        retry_sleep_s=float(retry_sleep_s),
        settle_time_s=float(settle_time_s),
    )
    return N9310AUSBTMC(device_path=Path(device_path), retry=retry)


def _program_signal_generator(controller: N9310AUSBTMC, tone: ToneParams) -> None:
    controller.set_freq_mhz(float(tone.frequency_hz) / 1e6)
    controller.set_ampl_dbm(float(tone.power_dbm))
    controller.rf_on()


def _safe_rf_off(controller: N9310AUSBTMC | None, *, label: str) -> None:
    if controller is None:
        return
    try:
        controller.rf_off()
    except Exception as error:
        print(f"[warn] failed to disable RF for {label}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="E4 mode: leakage or resolution (prompted if omitted).",
    )
    parser.add_argument(
        "--center-frequency-hz",
        type=float,
        default=None,
        help="Center/reference frequency in Hz used to auto-derive SG frequencies.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Run one capture at this single N (<16384). If omitted, a sweep is generated.",
    )
    parser.add_argument(
        "--n-bins-list",
        type=str,
        default=None,
        help="Explicit comma-separated N list for sweep (overrides auto-generated sweep).",
    )
    parser.add_argument(
        "--sweep-total-runs",
        type=int,
        default=50,
        help="Total runs when --n-bins/--n-bins-list is omitted (default: 50).",
    )
    parser.add_argument(
        "--min-bins",
        type=int,
        default=64,
        help="Minimum N for auto-generated sweep (default: 64).",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=16383,
        help="Maximum N for sweep/validation (must be <=16383; default: 16383).",
    )
    parser.add_argument(
        "--bin-index",
        type=int,
        default=None,
        help="Leakage mode fixed bin index k. If omitted, k is auto-derived from center-frequency per N.",
    )
    parser.add_argument(
        "--bin-offset",
        type=float,
        default=0.35,
        help="Leakage mode fractional offset epsilon (default: 0.35). Use 0.0 for bin-centered.",
    )
    parser.add_argument(
        "--delta-f-hz",
        type=float,
        default=None,
        help="Resolution mode two-tone spacing in Hz (required for mode=resolution).",
    )
    parser.add_argument(
        "--sg2-power-dbm",
        type=float,
        default=None,
        help="SG2 output power in dBm (required).",
    )
    parser.add_argument(
        "--sg1-power-dbm",
        type=float,
        default=None,
        help="SG1 power in dBm for resolution mode (default: same as SG2).",
    )
    parser.add_argument(
        "--auto-program-siggen",
        action="store_true",
        default=False,
        help="Program SG frequencies/powers each run via USBTMC instead of metadata-only manual setup.",
    )
    parser.add_argument(
        "--sg2-device-path",
        type=Path,
        default=Path("/dev/usbtmc0"),
        help="USBTMC device path for SG2 (default: /dev/usbtmc0).",
    )
    parser.add_argument(
        "--sg1-device-path",
        type=Path,
        default=Path("/dev/usbtmc1"),
        help="USBTMC device path for SG1 in resolution mode (default: /dev/usbtmc1).",
    )
    parser.add_argument(
        "--sg-settle-s",
        type=float,
        default=1.0,
        help="Signal-generator settle time after writes in seconds (default: 1.0).",
    )
    add_common_capture_arguments(
        parser,
        default_raw_dir="data/raw/e4",
        default_manifest_path="data/manifests/t2_e4_runs.csv",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sg1_power_dbm: float | None = None
    delta_f_hz: float | None = None
    bin_index: int | None = None
    bin_offset: float | None = None
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
        sample_rate_hz = float(sample_rate_mhz) * 1e6
        vrms = resolve_required_float(
            args.vrms,
            name="vrms",
            prompt="Target Vrms (V)",
            min_value=0.0,
        )
        center_frequency_hz = resolve_required_float(
            args.center_frequency_hz,
            name="center_frequency_hz",
            prompt="Center frequency (Hz)",
            min_value=1.0,
        )
        sg2_power_dbm = resolve_required_float(
            args.sg2_power_dbm,
            name="sg2_power_dbm",
            prompt="SG2 power (dBm)",
            min_value=-130.0,
            max_value=25.0,
        )

        required_min_bins = 2
        if mode == "leakage":
            if args.bin_index is not None:
                bin_index = resolve_required_int(
                    int(args.bin_index),
                    name="bin_index",
                    prompt="Leakage fixed bin index (k)",
                    min_value=1,
                )
                required_min_bins = max(2, (2 * int(bin_index)) + 2)
            bin_offset = resolve_required_float(
                args.bin_offset,
                name="bin_offset",
                prompt="Leakage bin offset epsilon",
                min_value=0.0,
            )
        else:
            delta_f_hz = resolve_required_float(
                args.delta_f_hz,
                name="delta_f_hz",
                prompt="Resolution delta-f (Hz)",
                min_value=1e-9,
            )
            sg1_power_dbm = (
                float(sg2_power_dbm)
                if args.sg1_power_dbm is None
                else resolve_required_float(
                    args.sg1_power_dbm,
                    name="sg1_power_dbm",
                    prompt="SG1 power (dBm)",
                    min_value=-130.0,
                    max_value=25.0,
                )
            )

        n_bins_values = _resolve_n_bins_values(args, required_min_bins=required_min_bins)
        if mode == "leakage" and bin_index is not None:
            for value in n_bins_values:
                max_index = (int(value) // 2) - 1
                if int(bin_index) > max_index:
                    raise ValueError(
                        f"bin_index={int(bin_index)} is too large for N={int(value)} (max {max_index})."
                    )
        if bool(args.auto_program_siggen) and mode == "resolution":
            if Path(args.sg1_device_path) == Path(args.sg2_device_path):
                raise ValueError(
                    "sg1_device_path and sg2_device_path must differ for resolution auto-programming."
                )
    except ValueError as error:
        parser.error(str(error))
        return

    run_id_base = args.run_id or default_run_id("E4")
    total_runs = len(n_bins_values)
    print(
        f"E4 sweep mode={mode} runs={total_runs} sample_rate_mhz={sample_rate_mhz:.6g} "
        f"center_frequency_hz={center_frequency_hz:.6g}"
    )
    print(f"SG2 power={sg2_power_dbm:.6g} dBm")
    if mode == "leakage":
        print(
            f"Leakage config: bin_index={'auto' if bin_index is None else bin_index}, "
            f"bin_offset={float(bin_offset):.6g}"
        )
    else:
        print(f"Resolution config: delta_f_hz={float(delta_f_hz):.6g}, SG1 power={float(sg1_power_dbm):.6g} dBm")
    if bool(args.auto_program_siggen):
        print(
            "Signal-generator auto-programming: enabled "
            f"(SG2={Path(args.sg2_device_path)}, "
            f"{'SG1=' + str(Path(args.sg1_device_path)) if mode == 'resolution' else 'single-SG leakage mode'})"
        )
    else:
        print("Signal-generator auto-programming: disabled (metadata-only manual analog setup).")

    sg1_controller: N9310AUSBTMC | None = None
    sg2_controller: N9310AUSBTMC | None = None
    if bool(args.auto_program_siggen):
        sg2_controller = _build_siggen_controller(
            device_path=Path(args.sg2_device_path),
            timeout_s=float(args.timeout_s),
            max_retries=int(args.max_retries),
            retry_sleep_s=float(args.retry_sleep_s),
            settle_time_s=float(args.sg_settle_s),
        )
        if mode == "resolution":
            sg1_controller = _build_siggen_controller(
                device_path=Path(args.sg1_device_path),
                timeout_s=float(args.timeout_s),
                max_retries=int(args.max_retries),
                retry_sleep_s=float(args.retry_sleep_s),
                settle_time_s=float(args.sg_settle_s),
            )

    captured = 0
    captured_guard_fail = 0
    skipped_existing = 0
    errors = 0
    try:
        for sweep_index, n_bins in enumerate(n_bins_values, start=1):
            run_id = run_id_base if total_runs == 1 else f"{run_id_base}__n{int(n_bins):05d}__r{sweep_index:03d}"
            try:
                if mode == "leakage":
                    tone_frequency_hz, effective_bin_index, bin_center_hz = leakage_tone_from_center(
                        sample_rate_hz=float(sample_rate_hz),
                        n_samples=int(n_bins),
                        center_frequency_hz=float(center_frequency_hz),
                        bin_offset=float(bin_offset),
                        bin_index=None if bin_index is None else int(bin_index),
                    )
                    sg2_tone = ToneParams(
                        label="signal_generator_2",
                        frequency_hz=float(tone_frequency_hz),
                        power_dbm=float(sg2_power_dbm),
                    )
                    sg1_tone: ToneParams | None = None
                    signal_generators = (sg2_tone,)
                    run_metadata = {
                        "mode": mode,
                        "sweep_index": int(sweep_index),
                        "sweep_total_runs": int(total_runs),
                        "n_bins": int(n_bins),
                        "n_bins_power_of_two": bool(_is_power_of_two(int(n_bins))),
                        "center_frequency_hz": float(center_frequency_hz),
                        "bin_index_requested": None if bin_index is None else int(bin_index),
                        "bin_index_effective": int(effective_bin_index),
                        "bin_offset": float(bin_offset),
                        "delta_f_hz": None,
                        "bin_center_frequency_hz": float(bin_center_hz),
                        "tone_frequency_hz": float(tone_frequency_hz),
                        "signal_generator_roles": ["tone"],
                        "siggen_programmed": bool(args.auto_program_siggen),
                    }
                else:
                    tone_low_hz, tone_high_hz = resolution_tones_from_center(
                        center_frequency_hz=float(center_frequency_hz),
                        delta_f_hz=float(delta_f_hz),
                    )
                    sg1_tone = ToneParams(
                        label="signal_generator_1",
                        frequency_hz=float(tone_low_hz),
                        power_dbm=float(sg1_power_dbm),
                    )
                    sg2_tone = ToneParams(
                        label="signal_generator_2",
                        frequency_hz=float(tone_high_hz),
                        power_dbm=float(sg2_power_dbm),
                    )
                    signal_generators = (sg1_tone, sg2_tone)
                    run_metadata = {
                        "mode": mode,
                        "sweep_index": int(sweep_index),
                        "sweep_total_runs": int(total_runs),
                        "n_bins": int(n_bins),
                        "n_bins_power_of_two": bool(_is_power_of_two(int(n_bins))),
                        "center_frequency_hz": float(center_frequency_hz),
                        "bin_index_requested": None,
                        "bin_index_effective": None,
                        "bin_offset": None,
                        "delta_f_hz": float(delta_f_hz),
                        "signal_generator_1_role": "tone_lower",
                        "signal_generator_2_role": "tone_upper",
                        "signal_generator_frequency_rule": "f1 = center - delta/2, f2 = center + delta/2",
                        "siggen_programmed": bool(args.auto_program_siggen),
                    }
            except Exception as error:
                errors += 1
                print(f"[{sweep_index:02d}/{total_runs:02d}] error planning run={run_id}: {error}")
                continue

            if sg2_controller is not None:
                try:
                    if sg1_controller is not None and sg1_tone is not None:
                        _program_signal_generator(sg1_controller, sg1_tone)
                    _program_signal_generator(sg2_controller, sg2_tone)
                except Exception as error:
                    errors += 1
                    print(f"[{sweep_index:02d}/{total_runs:02d}] error programming SG run={run_id}: {error}")
                    continue

            params = OneShotCaptureParams(
                experiment_id="E4",
                run_kind="physical_leakage_sweep" if mode == "leakage" else "physical_resolution_sweep",
                run_id=run_id,
                raw_dir=args.raw_dir,
                t2_manifest_path=args.t2_manifest_path,
                sample_rate_hz=float(sample_rate_hz),
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
                signal_generators=signal_generators,
                center_frequency_hz=float(center_frequency_hz),
                mixer_config=str(args.mixer_config),
                cable_config=str(args.cable_config),
                notes=str(args.notes),
                extra_metadata=run_metadata,
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
            if mode == "leakage":
                print(
                    f"[{sweep_index:02d}/{total_runs:02d}] run={run_id} status={status} "
                    f"n_bins={int(n_bins)} k={int(run_metadata['bin_index_effective'])} "
                    f"f_tone_hz={sg2_tone.frequency_hz:.6g} adc_rms={result.capture.summary.mean_block_rms:.3f}"
                )
            else:
                print(
                    f"[{sweep_index:02d}/{total_runs:02d}] run={run_id} status={status} "
                    f"n_bins={int(n_bins)} f1_hz={sg1_tone.frequency_hz:.6g} "
                    f"f2_hz={sg2_tone.frequency_hz:.6g} adc_rms={result.capture.summary.mean_block_rms:.3f}"
                )
    finally:
        _safe_rf_off(sg1_controller, label="SG1")
        _safe_rf_off(sg2_controller, label="SG2")

    print(f"Sweep complete: {total_runs} planned runs.")
    print(f"Captured: {captured}")
    print(f"Captured (guard-fail): {captured_guard_fail}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Errors: {errors}")
    print(f"T2 manifest: {args.t2_manifest_path}")
    print(f"Raw output dir: {args.raw_dir}")


if __name__ == "__main__":
    main()
