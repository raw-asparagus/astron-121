#!/usr/bin/env python3
"""Run Experiment 2 physical acquisition (logspace bandpass sweep)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.control.acquisition import E2AcquisitionConfig, run_e2_acquisition
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
        default=50,
        help="Logspace points per sample rate (default: 50).",
    )
    parser.add_argument(
        "--min-frequency-hz",
        type=float,
        default=10_000.0,
        help="Minimum sweep frequency in Hz (default: 10000).",
    )
    parser.add_argument(
        "--max-nyquist-multiple",
        type=float,
        default=4.0,
        help="Sweep up to this multiple of f_Nyquist (default: 4.0 -> 2*fs).",
    )
    parser.add_argument(
        "--source-power-dbm",
        type=float,
        default=-10.0,
        help="Constant signal-generator power in dBm for the sweep (default: -10).",
    )
    parser.add_argument(
        "--fir-mode",
        type=str,
        default="default",
        help="Metadata label for FIR mode (default: default; fir_coeffs is fixed to None).",
    )
    parser.add_argument("--nsamples", type=int, default=2048, help="Samples per block (default: 2048).")
    parser.add_argument(
        "--nblocks",
        type=int,
        default=6,
        help="Requested SDR blocks before stale-drop (default: 6; first block dropped -> 5 saved).",
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
        default=Path("data/raw/e2"),
        help="Output directory for NPZ capture files.",
    )
    parser.add_argument(
        "--t2-manifest-path",
        type=Path,
        default=Path("data/manifests/t2_e2_runs.csv"),
        help="T2 CSV manifest path.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=Path("data/manifests/e2_progress.csv"),
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
    config = E2AcquisitionConfig(
        sample_rates_hz=tuple(float(rate) for rate in args.sample_rates_mhz),
        n_frequency_points=int(args.n_frequency_points),
        log_min_frequency_hz=float(args.min_frequency_hz),
        log_max_nyquist_multiple=float(args.max_nyquist_multiple),
        source_power_dbm=float(args.source_power_dbm),
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
        fir_mode=str(args.fir_mode),
        fir_coeffs=None,
        sdr_direct=True,
    )
    progress = run_e2_acquisition(config)
    captured = progress.loc[progress["final_status"] == "captured"]
    captured_guard_fail = progress.loc[progress["final_status"] == "captured_guard_fail"]
    skipped_existing = progress.loc[progress["final_status"] == "skip_existing_npz"]
    error_io = progress.loc[progress["final_status"] == "error_io"]
    print(f"Progress rows: {len(progress)}")
    print(f"Captured runs: {len(captured)}")
    print(f"Captured runs (guard-fail): {len(captured_guard_fail)}")
    print(f"Skipped existing NPZ runs: {len(skipped_existing)}")
    print(f"IO-error runs: {len(error_io)}")
    print(f"FIR mode: {config.fir_mode}")
    print(f"FIR coefficients: {config.fir_coeffs}")
    print(f"SDR direct mode: {config.sdr_direct}")
    print(f"Progress CSV: {config.progress_path}")
    print(f"T2 manifest CSV: {config.t2_manifest_path}")
    print(f"Raw NPZ dir: {config.raw_dir}")


if __name__ == "__main__":
    main()
