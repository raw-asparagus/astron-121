#!/usr/bin/env python3
"""Run Experiment 1 physical acquisition (N9310A + SDR direct mode)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.control.acquisition import E1AcquisitionConfig, run_e1_acquisition
from ugradio_lab1.control.siggen import SigGenRetryPolicy


def _parse_sample_rates_mhz(raw: str) -> tuple[float, ...]:
    rates: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        rates.append(float(token) * 1e6)
    if len(rates) == 0:
        raise argparse.ArgumentTypeError("At least one sample rate is required.")
    return tuple(rates)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-rates-mhz",
        type=_parse_sample_rates_mhz,
        default=(1.0e6, 1.6e6, 2.4e6, 3.2e6),
        help="Comma-separated SDR sample rates in MHz (default: 1.0,1.6,2.4,3.2).",
    )
    parser.add_argument(
        "--n-frequency-points",
        type=int,
        default=24,
        help="Linear points over [0, 4 f_Nyquist] per sample rate (default: 24).",
    )
    parser.add_argument("--nsamples", type=int, default=2048, help="Samples per block (default: 2048).")
    parser.add_argument(
        "--nblocks",
        type=int,
        default=11,
        help="Requested SDR blocks before stale-drop (default: 11; first block dropped -> 10 saved).",
    )
    parser.add_argument(
        "--stale-blocks",
        type=int,
        default=1,
        help="Number of stale leading blocks to drop (default: 1).",
    )
    parser.add_argument(
        "--device-path",
        type=Path,
        default=Path("/dev/usbtmc0"),
        help="N9310A USBTMC path (default: /dev/usbtmc0).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/e1"),
        help="Output directory for NPZ capture files.",
    )
    parser.add_argument(
        "--t2-manifest-path",
        type=Path,
        default=Path("data/manifests/t2_e1_runs.csv"),
        help="T2 CSV manifest path.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=Path("data/manifests/e1_progress.csv"),
        help="Progress CSV used for resume/skip logic.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=10.0,
        help="Timeout (seconds) for siggen query and SDR capture call.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for siggen operations and SDR captures.",
    )
    parser.add_argument(
        "--guard-max-attempts",
        type=int,
        default=3,
        help="Guard-based recapture attempts before accepting a weak/clipped block.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retry = SigGenRetryPolicy(
        timeout_s=float(args.timeout_s),
        max_retries=int(args.max_retries),
        settle_time_s=1.0,
    )
    config = E1AcquisitionConfig(
        sample_rates_hz=tuple(float(rate) for rate in args.sample_rates_mhz),
        n_frequency_points=int(args.n_frequency_points),
        nsamples=int(args.nsamples),
        nblocks=int(args.nblocks),
        stale_blocks=int(args.stale_blocks),
        raw_dir=Path(args.raw_dir),
        t2_manifest_path=Path(args.t2_manifest_path),
        progress_path=Path(args.progress_path),
        siggen_device_path=Path(args.device_path),
        siggen_retry=retry,
        sdr_timeout_s=float(args.timeout_s),
        sdr_max_retries=int(args.max_retries),
        guard_max_attempts=int(args.guard_max_attempts),
    )
    progress = run_e1_acquisition(config)
    completed = progress.loc[progress["final_status"].isin(["ok_target", "ok_target_from_baseline"])]
    closest_only = progress.loc[progress["final_status"] == "closest_only"]
    unachievable = progress.loc[progress["final_status"] == "unachievable"]
    error_io = progress.loc[progress["final_status"] == "error_io"]
    print(f"Progress rows: {len(progress)}")
    print(f"Completed target-band combos: {len(completed)}")
    print(f"Closest-only combos: {len(closest_only)}")
    print(f"Unachievable combos: {len(unachievable)}")
    print(f"IO-error combos: {len(error_io)}")
    print(f"Progress CSV: {config.progress_path}")
    print(f"T2 manifest CSV: {config.t2_manifest_path}")
    print(f"Raw NPZ dir: {config.raw_dir}")


if __name__ == "__main__":
    main()
