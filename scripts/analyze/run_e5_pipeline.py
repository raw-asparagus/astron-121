#!/usr/bin/env python3
"""Build E5 interim/processed artifacts from raw NPZ input."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.pipeline.e5 import (
    DEFAULT_E5_CURVE_TABLE_PATH,
    DEFAULT_E5_F10_FIGURE_PATH,
    DEFAULT_E5_F11_FIGURE_PATH,
    DEFAULT_E5_F12_FIGURE_PATH,
    DEFAULT_E5_QC_CATALOG_PATH,
    DEFAULT_E5_RAW_SOURCE,
    DEFAULT_E5_RUN_CATALOG_PATH,
    DEFAULT_E5_STATS_TABLE_PATH,
    DEFAULT_E5_T2_TABLE_PATH,
    DEFAULT_E5_T6_TABLE_PATH,
    build_e5_noise_stats_table,
    build_e5_qc_catalog,
    build_e5_radiometer_curve_table,
    build_e5_run_catalog,
    build_e5_t2_table,
    build_e5_t6_table,
    fit_e5_radiometer,
    select_e5_analysis_runs,
    write_dataframe_csv,
    write_e5_acf_consistency_figure,
    write_e5_noise_histogram_figure,
    write_e5_radiometer_figure,
    write_table_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_E5_RAW_SOURCE,
        help="E5 raw input source (.tar.gz or directory of .npz files).",
    )
    parser.add_argument(
        "--run-catalog-out",
        type=Path,
        default=DEFAULT_E5_RUN_CATALOG_PATH,
        help="Output CSV for normalized run catalog.",
    )
    parser.add_argument(
        "--qc-catalog-out",
        type=Path,
        default=DEFAULT_E5_QC_CATALOG_PATH,
        help="Output CSV for QC catalog.",
    )
    parser.add_argument(
        "--noise-stats-out",
        type=Path,
        default=DEFAULT_E5_STATS_TABLE_PATH,
        help="Output CSV for selected per-run physical noise statistics.",
    )
    parser.add_argument(
        "--curve-table-out",
        type=Path,
        default=DEFAULT_E5_CURVE_TABLE_PATH,
        help="Output CSV for selected radiometer-curve rows.",
    )
    parser.add_argument(
        "--t2-out",
        type=Path,
        default=DEFAULT_E5_T2_TABLE_PATH,
        help="Output CSV for processed T2 E5 table.",
    )
    parser.add_argument(
        "--t6-out",
        type=Path,
        default=DEFAULT_E5_T6_TABLE_PATH,
        help="Output CSV for processed T6 E5 table.",
    )
    parser.add_argument(
        "--f10-out",
        type=Path,
        default=DEFAULT_E5_F10_FIGURE_PATH,
        help="Output figure path for physical F10 noise histogram.",
    )
    parser.add_argument(
        "--f11-out",
        type=Path,
        default=DEFAULT_E5_F11_FIGURE_PATH,
        help="Output figure path for physical F11 radiometer scaling.",
    )
    parser.add_argument(
        "--f12-out",
        type=Path,
        default=DEFAULT_E5_F12_FIGURE_PATH,
        help="Output figure path for physical F12 ACF/spectrum consistency.",
    )
    parser.add_argument(
        "--analysis-noise-source",
        type=str,
        default="lab_noise_generator",
        help="Noise source for E5 analysis products (default: lab_noise_generator; use 'all' for no filter).",
    )
    parser.add_argument(
        "--include-qc-fail-in-analysis",
        action="store_true",
        help="If set, include non-analysis-pass runs in E5 analysis selection.",
    )
    parser.add_argument(
        "--radiometer-block-size",
        type=int,
        default=256,
        help="Sub-block size (samples) used for E5 radiometer grouping (default: 256).",
    )
    parser.add_argument(
        "--n-avg-values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Averaging factors used for E5 radiometer curve (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--f10-bins",
        type=int,
        default=70,
        help="Histogram bin count for physical F10 (default: 70).",
    )
    parser.add_argument(
        "--f12-run-id",
        type=str,
        default=None,
        help="Optional run_id override for physical F12 reference run.",
    )
    return parser


def _normalize_noise_source_filter(value: str) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "all", "*", "any"}:
        return None
    return normalized


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    noise_source_filter = _normalize_noise_source_filter(str(args.analysis_noise_source))

    run_catalog = build_e5_run_catalog(args.raw_source)
    write_dataframe_csv(run_catalog, args.run_catalog_out)

    qc_catalog = build_e5_qc_catalog(run_catalog)
    write_dataframe_csv(qc_catalog, args.qc_catalog_out)

    noise_stats = build_e5_noise_stats_table(
        qc_catalog,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_analysis),
        noise_source_filter=noise_source_filter,
    )
    write_dataframe_csv(noise_stats, args.noise_stats_out)

    radiometer_curve = build_e5_radiometer_curve_table(
        qc_catalog,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_analysis),
        noise_source_filter=noise_source_filter,
        block_size=int(args.radiometer_block_size),
        n_avg_values=tuple(int(value) for value in args.n_avg_values),
    )
    write_dataframe_csv(radiometer_curve, args.curve_table_out)

    radiometer_fit_result = fit_e5_radiometer(radiometer_curve)

    t2_table = build_e5_t2_table(qc_catalog)
    write_table_manifest_csv(t2_table, args.t2_out)

    t6_table = build_e5_t6_table(radiometer_curve, fit_result=radiometer_fit_result)
    write_table_manifest_csv(t6_table, args.t6_out)

    f10_path = write_e5_noise_histogram_figure(
        qc_catalog,
        args.f10_out,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_analysis),
        noise_source_filter=noise_source_filter,
        bins=int(args.f10_bins),
    )
    f11_path = write_e5_radiometer_figure(
        radiometer_curve,
        args.f11_out,
        fit_result=radiometer_fit_result,
    )
    f12_path = write_e5_acf_consistency_figure(
        qc_catalog,
        args.f12_out,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_analysis),
        noise_source_filter=noise_source_filter,
        run_id=args.f12_run_id,
    )

    selected = select_e5_analysis_runs(
        qc_catalog,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_analysis),
        noise_source_filter=noise_source_filter,
    )

    print(f"Run catalog rows: {len(run_catalog)}")
    print(f"QC analysis-pass rows: {int(qc_catalog['qc_analysis_pass'].fillna(False).sum())}")
    print(f"QC recommended-pass rows: {int(qc_catalog['qc_recommended_pass'].fillna(False).sum())}")
    print(f"Selected analysis runs: {len(selected)}")
    print(f"Noise stats rows: {len(noise_stats)}")
    print(f"Radiometer curve rows: {len(radiometer_curve)}")
    print(f"Radiometer slope: {float(radiometer_fit_result['slope']):.6f}")
    print(f"Radiometer expected slope: {float(radiometer_fit_result['expected_slope']):.6f}")
    print(f"T2 rows: {len(t2_table)}")
    print(f"T6 rows: {len(t6_table)}")
    if "status" in run_catalog.columns:
        print("Run status counts:")
        counts = run_catalog["status"].value_counts(dropna=False)
        for key, count in counts.items():
            print(f"  {key}: {int(count)}")
    print(f"Run catalog CSV: {args.run_catalog_out}")
    print(f"QC catalog CSV: {args.qc_catalog_out}")
    print(f"Noise stats CSV: {args.noise_stats_out}")
    print(f"Radiometer curve CSV: {args.curve_table_out}")
    print(f"T2 table CSV: {args.t2_out}")
    print(f"T6 table CSV: {args.t6_out}")
    print(f"F10 figure: {f10_path}")
    print(f"F11 figure: {f11_path}")
    print(f"F12 figure: {f12_path}")


if __name__ == "__main__":
    main()
