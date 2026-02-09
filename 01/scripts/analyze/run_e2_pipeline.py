#!/usr/bin/env python3
"""Build E2 interim/processed artifacts from raw NPZ input."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.pipeline.e2 import (
    DEFAULT_E2_CURVE_TABLE_PATH,
    DEFAULT_E2_F4_FIGURE_PATH,
    DEFAULT_E2_QC_CATALOG_PATH,
    DEFAULT_E2_RAW_SOURCE,
    DEFAULT_E2_RUN_CATALOG_PATH,
    DEFAULT_E2_T2_TABLE_PATH,
    DEFAULT_E2_T4_TABLE_PATH,
    build_e2_bandpass_curve_table,
    build_e2_qc_catalog,
    build_e2_run_catalog,
    build_e2_t2_table,
    build_e2_t4_table,
    write_dataframe_csv,
    write_e2_bandpass_figure,
    write_table_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_E2_RAW_SOURCE,
        help="E2 raw input source (.tar.gz or directory of .npz files).",
    )
    parser.add_argument(
        "--run-catalog-out",
        type=Path,
        default=DEFAULT_E2_RUN_CATALOG_PATH,
        help="Output CSV for normalized run catalog.",
    )
    parser.add_argument(
        "--qc-catalog-out",
        type=Path,
        default=DEFAULT_E2_QC_CATALOG_PATH,
        help="Output CSV for QC catalog.",
    )
    parser.add_argument(
        "--curve-table-out",
        type=Path,
        default=DEFAULT_E2_CURVE_TABLE_PATH,
        help="Output CSV for selected per-point E2 bandpass rows.",
    )
    parser.add_argument(
        "--t2-out",
        type=Path,
        default=DEFAULT_E2_T2_TABLE_PATH,
        help="Output CSV for processed T2 E2 table.",
    )
    parser.add_argument(
        "--t4-out",
        type=Path,
        default=DEFAULT_E2_T4_TABLE_PATH,
        help="Output CSV for processed T4 E2 table.",
    )
    parser.add_argument(
        "--f4-out",
        type=Path,
        default=DEFAULT_E2_F4_FIGURE_PATH,
        help="Output figure path for physical F4 bandpass curves.",
    )
    parser.add_argument(
        "--preferred-power-dbm",
        type=float,
        default=-10.0,
        help="Preferred source power for duplicate-run tie-breaks (default: -10).",
    )
    parser.add_argument(
        "--include-qc-fail-in-curves",
        action="store_true",
        help="If set, include non-analysis-pass runs in E2 curve selection.",
    )
    parser.add_argument(
        "--reference-sample-rate-hz",
        type=float,
        default=1.0e6,
        help="Reference sample rate used for common F4 plotting axis (default: 1e6).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_catalog = build_e2_run_catalog(args.raw_source)
    write_dataframe_csv(run_catalog, args.run_catalog_out)

    qc_catalog = build_e2_qc_catalog(run_catalog)
    write_dataframe_csv(qc_catalog, args.qc_catalog_out)

    curve_table = build_e2_bandpass_curve_table(
        qc_catalog,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_curves),
        preferred_power_dbm=float(args.preferred_power_dbm),
    )
    write_dataframe_csv(curve_table, args.curve_table_out)

    t2_table = build_e2_t2_table(qc_catalog)
    write_table_manifest_csv(t2_table, args.t2_out)

    t4_table = build_e2_t4_table(curve_table)
    write_table_manifest_csv(t4_table, args.t4_out)

    figure_path = write_e2_bandpass_figure(
        curve_table,
        args.f4_out,
        reference_sample_rate_hz=float(args.reference_sample_rate_hz),
    )

    print(f"Run catalog rows: {len(run_catalog)}")
    print(f"QC analysis-pass rows: {int(qc_catalog['qc_analysis_pass'].fillna(False).sum())}")
    print(f"QC recommended-pass rows: {int(qc_catalog['qc_recommended_pass'].fillna(False).sum())}")
    print(f"Curve rows: {len(curve_table)}")
    print(f"T2 rows: {len(t2_table)}")
    print(f"T4 rows: {len(t4_table)}")
    if "status" in run_catalog.columns:
        print("Run status counts:")
        counts = run_catalog["status"].value_counts(dropna=False)
        for key, count in counts.items():
            print(f"  {key}: {int(count)}")
    print(f"Run catalog CSV: {args.run_catalog_out}")
    print(f"QC catalog CSV: {args.qc_catalog_out}")
    print(f"Curve table CSV: {args.curve_table_out}")
    print(f"T2 table CSV: {args.t2_out}")
    print(f"T4 table CSV: {args.t4_out}")
    print(f"F4 figure: {figure_path}")


if __name__ == "__main__":
    main()
