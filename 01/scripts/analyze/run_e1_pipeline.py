#!/usr/bin/env python3
"""Build E1 interim/processed artifacts from raw NPZ input."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.pipeline.e1 import (
    DEFAULT_E1_F2_FIGURE_PATH,
    DEFAULT_E1_QC_CATALOG_PATH,
    DEFAULT_E1_RAW_SOURCE,
    DEFAULT_E1_RUN_CATALOG_PATH,
    DEFAULT_E1_T2_TABLE_PATH,
    DEFAULT_E1_T3_TABLE_PATH,
    build_e1_qc_catalog,
    build_e1_run_catalog,
    build_e1_t2_table,
    build_e1_t3_table,
    write_dataframe_csv,
    write_e1_alias_figure,
    write_table_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_E1_RAW_SOURCE,
        help="E1 raw input source (.tar.gz or directory of .npz files).",
    )
    parser.add_argument(
        "--run-catalog-out",
        type=Path,
        default=DEFAULT_E1_RUN_CATALOG_PATH,
        help="Output CSV for normalized run catalog.",
    )
    parser.add_argument(
        "--qc-catalog-out",
        type=Path,
        default=DEFAULT_E1_QC_CATALOG_PATH,
        help="Output CSV for QC catalog.",
    )
    parser.add_argument(
        "--t2-out",
        type=Path,
        default=DEFAULT_E1_T2_TABLE_PATH,
        help="Output CSV for processed T2 E1 table.",
    )
    parser.add_argument(
        "--t3-out",
        type=Path,
        default=DEFAULT_E1_T3_TABLE_PATH,
        help="Output CSV for processed T3 E1 table.",
    )
    parser.add_argument(
        "--f2-out",
        type=Path,
        default=DEFAULT_E1_F2_FIGURE_PATH,
        help="Output figure path for physical F2 alias map.",
    )
    parser.add_argument(
        "--plot-recommended-only",
        action="store_true",
        help="If set, render F2 using only QC-recommended runs.",
    )
    parser.add_argument(
        "--include-qc-fail-in-t3",
        action="store_true",
        help="If set, include non-analysis-pass runs in T3 construction.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_catalog = build_e1_run_catalog(args.raw_source)
    write_dataframe_csv(run_catalog, args.run_catalog_out)

    qc_catalog = build_e1_qc_catalog(run_catalog)
    write_dataframe_csv(qc_catalog, args.qc_catalog_out)

    t2_table = build_e1_t2_table(qc_catalog)
    write_table_manifest_csv(t2_table, args.t2_out)

    t3_table = build_e1_t3_table(
        qc_catalog,
        use_qc_analysis_pass=not bool(args.include_qc_fail_in_t3),
    )
    write_table_manifest_csv(t3_table, args.t3_out)

    figure_path = write_e1_alias_figure(
        t3_table,
        args.f2_out,
        use_recommended_only=bool(args.plot_recommended_only),
    )

    print(f"Run catalog rows: {len(run_catalog)}")
    print(f"QC analysis-pass rows: {int(qc_catalog['qc_analysis_pass'].fillna(False).sum())}")
    print(f"QC recommended-pass rows: {int(qc_catalog['qc_recommended_pass'].fillna(False).sum())}")
    print(f"T2 rows: {len(t2_table)}")
    print(f"T3 rows: {len(t3_table)}")
    if "status" in run_catalog.columns:
        print("Run status counts:")
        counts = run_catalog["status"].value_counts(dropna=False)
        for key, count in counts.items():
            print(f"  {key}: {int(count)}")
    print(f"Run catalog CSV: {args.run_catalog_out}")
    print(f"QC catalog CSV: {args.qc_catalog_out}")
    print(f"T2 table CSV: {args.t2_out}")
    print(f"T3 table CSV: {args.t3_out}")
    print(f"F2 figure: {figure_path}")


if __name__ == "__main__":
    main()
