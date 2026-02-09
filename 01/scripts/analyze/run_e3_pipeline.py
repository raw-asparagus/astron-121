#!/usr/bin/env python3
"""Build E3 interim/processed artifacts from raw NPZ input."""

from __future__ import annotations

import argparse
from pathlib import Path

from ugradio_lab1.pipeline.e3 import (
    DEFAULT_E3_F5_FIGURE_PATH,
    DEFAULT_E3_F6_FIGURE_PATH,
    DEFAULT_E3_QC_CATALOG_PATH,
    DEFAULT_E3_RAW_SOURCE,
    DEFAULT_E3_RUN_CATALOG_PATH,
    DEFAULT_E3_SPECTRUM_TABLE_PATH,
    DEFAULT_E3_T2_TABLE_PATH,
    build_e3_qc_catalog,
    build_e3_run_catalog,
    build_e3_spectrum_profile,
    build_e3_spectrum_profiles,
    build_e3_t2_table,
    write_dataframe_csv,
    write_e3_voltage_components_figure,
    write_e3_voltage_power_figure,
    write_table_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_E3_RAW_SOURCE,
        help="E3 raw input source (.tar.gz or directory of .npz files).",
    )
    parser.add_argument(
        "--run-catalog-out",
        type=Path,
        default=DEFAULT_E3_RUN_CATALOG_PATH,
        help="Output CSV for normalized run catalog.",
    )
    parser.add_argument(
        "--qc-catalog-out",
        type=Path,
        default=DEFAULT_E3_QC_CATALOG_PATH,
        help="Output CSV for QC catalog.",
    )
    parser.add_argument(
        "--spectrum-out",
        type=Path,
        default=DEFAULT_E3_SPECTRUM_TABLE_PATH,
        help="Output CSV for selected E3 per-bin spectrum profile.",
    )
    parser.add_argument(
        "--t2-out",
        type=Path,
        default=DEFAULT_E3_T2_TABLE_PATH,
        help="Output CSV for processed T2 E3 table.",
    )
    parser.add_argument(
        "--f5-out",
        type=Path,
        default=DEFAULT_E3_F5_FIGURE_PATH,
        help="Output figure path for physical F5 complex-voltage components.",
    )
    parser.add_argument(
        "--f6-out",
        type=Path,
        default=DEFAULT_E3_F6_FIGURE_PATH,
        help="Output figure path for physical F6 voltage-vs-power comparison.",
    )
    parser.add_argument(
        "--preferred-mode",
        type=str,
        default="single_tone",
        help="Preferred E3 mode used when selecting one reference run (default: single_tone).",
    )
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="If set, build spectrum profile from only one selected reference run.",
    )
    parser.add_argument(
        "--include-qc-fail-in-spectrum",
        action="store_true",
        help="If set, include non-analysis-pass runs during reference-run selection.",
    )
    parser.add_argument(
        "--f6-power-db",
        action="store_true",
        help="If set, plot F6 power axis in dB instead of linear V^2.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_catalog = build_e3_run_catalog(args.raw_source)
    write_dataframe_csv(run_catalog, args.run_catalog_out)

    qc_catalog = build_e3_qc_catalog(run_catalog)
    write_dataframe_csv(qc_catalog, args.qc_catalog_out)

    if bool(args.reference_only):
        spectrum_profile = build_e3_spectrum_profile(
            qc_catalog,
            use_qc_analysis_pass=not bool(args.include_qc_fail_in_spectrum),
            preferred_mode=str(args.preferred_mode),
        )
    else:
        spectrum_profile = build_e3_spectrum_profiles(
            qc_catalog,
            use_qc_analysis_pass=not bool(args.include_qc_fail_in_spectrum),
        )
    write_dataframe_csv(spectrum_profile, args.spectrum_out)

    t2_table = build_e3_t2_table(qc_catalog)
    write_table_manifest_csv(t2_table, args.t2_out)

    f5_path = write_e3_voltage_components_figure(spectrum_profile, args.f5_out)
    f6_path = write_e3_voltage_power_figure(
        spectrum_profile,
        args.f6_out,
        power_db=bool(args.f6_power_db),
    )

    unique_run_ids = sorted(spectrum_profile["run_id"].astype(str).dropna().unique().tolist())
    run_id = unique_run_ids[0] if len(unique_run_ids) > 0 else ""

    print(f"Run catalog rows: {len(run_catalog)}")
    print(f"QC analysis-pass rows: {int(qc_catalog['qc_analysis_pass'].fillna(False).sum())}")
    print(f"QC recommended-pass rows: {int(qc_catalog['qc_recommended_pass'].fillna(False).sum())}")
    print(f"Spectrum profile rows: {len(spectrum_profile)}")
    print(f"Spectrum profile run IDs: {len(unique_run_ids)}")
    print(f"Reference run ID: {run_id}")
    print(f"T2 rows: {len(t2_table)}")
    if "status" in run_catalog.columns:
        print("Run status counts:")
        counts = run_catalog["status"].value_counts(dropna=False)
        for key, count in counts.items():
            print(f"  {key}: {int(count)}")
    print(f"Run catalog CSV: {args.run_catalog_out}")
    print(f"QC catalog CSV: {args.qc_catalog_out}")
    print(f"Spectrum profile CSV: {args.spectrum_out}")
    print(f"T2 table CSV: {args.t2_out}")
    print(f"F5 figure: {f5_path}")
    print(f"F6 figure: {f6_path}")


if __name__ == "__main__":
    main()
