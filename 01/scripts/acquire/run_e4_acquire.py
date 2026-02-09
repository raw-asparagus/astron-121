#!/usr/bin/env python3
"""Run Experiment 4 physical acquisition sweep (leakage/resolution)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from manual_capture_common import (
    OneShotCaptureParams,
    add_common_capture_arguments,
    add_signal_generator_arguments,
    default_run_id,
    match_tone_to_reference,
    resolve_manual_tone,
    resolve_required_choice,
    resolve_required_float,
    resolve_required_int,
    run_one_shot_capture,
)


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
        help="Total number of E4 sweep runs when --n-bins/--n-bins-list is omitted (default: 50).",
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
        help="Maximum N for sweep and validation (must be <=16383; default: 16383).",
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
    add_signal_generator_arguments(parser, include_sg1=False, include_sg2=True)
    return parser


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
        sg2 = resolve_manual_tone(
            label="signal_generator_2",
            frequency_hz=args.sg2_frequency_hz,
            power_dbm=args.sg2_power_dbm,
        )
        sg1 = match_tone_to_reference(sg2, label="signal_generator_1")

        bin_index = args.bin_index
        bin_offset = args.bin_offset
        delta_f_hz = args.delta_f_hz
        required_min_bins = 2
        if mode == "leakage":
            bin_index = resolve_required_int(
                bin_index,
                name="bin_index",
                prompt="Leakage bin index (k)",
                min_value=1,
            )
            bin_offset = resolve_required_float(
                bin_offset,
                name="bin_offset",
                prompt="Leakage bin offset epsilon",
                min_value=0.0,
            )
            required_min_bins = max(2, (2 * int(bin_index)) + 2)
        else:
            delta_f_hz = resolve_required_float(
                delta_f_hz,
                name="delta_f_hz",
                prompt="Resolution delta-f (Hz)",
                min_value=0.0,
            )
        n_bins_values = _resolve_n_bins_values(args, required_min_bins=required_min_bins)
        if mode == "leakage":
            for value in n_bins_values:
                max_index = (int(value) // 2) - 1
                if int(bin_index) > max_index:
                    raise ValueError(
                        f"bin_index={int(bin_index)} is too large for N={int(value)} (max {max_index})."
                    )

    except ValueError as error:
        parser.error(str(error))
        return

    run_id_base = args.run_id or default_run_id("E4")
    total_runs = len(n_bins_values)
    print(f"E4 sweep mode={mode} runs={total_runs} sample_rate_mhz={sample_rate_mhz:.6g}")
    print(f"SG2 reference: f={sg2.frequency_hz:.6g} Hz, p={sg2.power_dbm:.6g} dBm")
    print("SG1 is auto-matched to SG2 for every run.")

    captured = 0
    captured_guard_fail = 0
    skipped_existing = 0
    errors = 0
    for sweep_index, n_bins in enumerate(n_bins_values, start=1):
        run_id = run_id_base if total_runs == 1 else f"{run_id_base}__n{int(n_bins):05d}__r{sweep_index:03d}"
        params = OneShotCaptureParams(
            experiment_id="E4",
            run_kind="physical_leakage_sweep" if mode == "leakage" else "physical_resolution_sweep",
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
            signal_generators=(sg1, sg2),
            mixer_config=str(args.mixer_config),
            cable_config=str(args.cable_config),
            notes=str(args.notes),
            extra_metadata={
                "mode": mode,
                "sweep_index": int(sweep_index),
                "sweep_total_runs": int(total_runs),
                "n_bins": int(n_bins),
                "n_bins_power_of_two": bool(_is_power_of_two(int(n_bins))),
                "bin_index": None if bin_index is None else int(bin_index),
                "bin_offset": None if bin_offset is None else float(bin_offset),
                "delta_f_hz": None if delta_f_hz is None else float(delta_f_hz),
                "signal_generator_1_matched_to_signal_generator_2": True,
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
            f"status={status} n_bins={int(n_bins)} adc_rms={result.capture.summary.mean_block_rms:.3f}"
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
