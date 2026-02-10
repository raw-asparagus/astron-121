#!/usr/bin/env python3
"""Build E4 interim/processed artifacts from raw NPZ input.

Default raw source is intentionally ``data/raw/e3.tar.gz`` so E4 can be
bootstrapped from E3 single-tone/two-tone captures when dedicated E4 captures
are not available yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ugradio_lab1.pipeline.e4 import (
    DEFAULT_E4_F7_FIGURE_PATH,
    DEFAULT_E4_F8_FIGURE_PATH,
    DEFAULT_E4_F9_FIGURE_PATH,
    DEFAULT_E4_LEAKAGE_TABLE_PATH,
    DEFAULT_E4_QC_CATALOG_PATH,
    DEFAULT_E4_RAW_SOURCE,
    DEFAULT_E4_RESOLUTION_TABLE_PATH,
    DEFAULT_E4_RUN_CATALOG_PATH,
    DEFAULT_E4_T2_TABLE_PATH,
    DEFAULT_E4_T5_TABLE_PATH,
    DEFAULT_E4_WINDOW_TABLE_PATH,
    build_e4_leakage_metrics,
    build_e4_qc_catalog,
    build_e4_resolution_curve,
    build_e4_run_catalog,
    build_e4_t2_table,
    build_e4_t5_table,
    build_e4_window_spectrum_table,
    write_dataframe_csv,
    write_e4_leakage_figure,
    write_e4_multi_window_figure,
    write_e4_resolution_figure,
    write_table_manifest_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-source",
        type=Path,
        default=DEFAULT_E4_RAW_SOURCE,
        help="E4 raw input source (.tar.gz or directory of .npz files). Default is E3 raw bootstrap source.",
    )
    parser.add_argument(
        "--run-catalog-out",
        type=Path,
        default=DEFAULT_E4_RUN_CATALOG_PATH,
        help="Output CSV for normalized run catalog.",
    )
    parser.add_argument(
        "--qc-catalog-out",
        type=Path,
        default=DEFAULT_E4_QC_CATALOG_PATH,
        help="Output CSV for QC catalog.",
    )
    parser.add_argument(
        "--leakage-out",
        type=Path,
        default=DEFAULT_E4_LEAKAGE_TABLE_PATH,
        help="Output CSV for leakage metric rows.",
    )
    parser.add_argument(
        "--resolution-out",
        type=Path,
        default=DEFAULT_E4_RESOLUTION_TABLE_PATH,
        help="Output CSV for resolution curve rows.",
    )
    parser.add_argument(
        "--window-out",
        type=Path,
        default=DEFAULT_E4_WINDOW_TABLE_PATH,
        help="Output CSV for multi-window spectrum rows.",
    )
    parser.add_argument(
        "--t2-out",
        type=Path,
        default=DEFAULT_E4_T2_TABLE_PATH,
        help="Output CSV for processed T2 E4 table.",
    )
    parser.add_argument(
        "--t5-out",
        type=Path,
        default=DEFAULT_E4_T5_TABLE_PATH,
        help="Output CSV for processed T5 E4 table.",
    )
    parser.add_argument(
        "--f7-out",
        type=Path,
        default=DEFAULT_E4_F7_FIGURE_PATH,
        help="Output figure path for physical F7 leakage comparison.",
    )
    parser.add_argument(
        "--f8-out",
        type=Path,
        default=DEFAULT_E4_F8_FIGURE_PATH,
        help="Output figure path for physical F8 resolution-vs-N.",
    )
    parser.add_argument(
        "--f9-out",
        type=Path,
        default=DEFAULT_E4_F9_FIGURE_PATH,
        help="Output figure path for physical F9 multi-window spectra.",
    )
    parser.add_argument(
        "--include-qc-fail-in-analysis",
        action="store_true",
        help="If set, include non-analysis-pass runs in E4 derived tables/figures.",
    )
    parser.add_argument(
        "--main-lobe-half-width-bins",
        type=int,
        default=1,
        help="Half-width in bins used for leakage main-lobe integration (default: 1).",
    )
    parser.add_argument(
        "--bin-center-tolerance",
        type=float,
        default=0.02,
        help="Absolute bin-offset threshold used to label a tone as bin-centered (default: 0.02 bins).",
    )
    parser.add_argument(
        "--min-peak-prominence-db",
        type=float,
        default=4.0,
        help="Peak-finding prominence threshold in dB for two-tone resolution extraction (default: 4.0).",
    )
    parser.add_argument(
        "--min-valley-depth-db",
        type=float,
        default=0.4,
        help="Minimum valley depth in dB between two peaks to mark resolvable tones (default: 0.4).",
    )
    parser.add_argument(
        "--f9-run-id",
        type=str,
        default=None,
        help="Optional run_id override for F9 window comparison reference run.",
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        choices=("numpy", "ugradio"),
        default="numpy",
        help="FFT backend for windowed spectrum construction (default: numpy).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    run_catalog = build_e4_run_catalog(
        args.raw_source,
        bin_center_tolerance=float(args.bin_center_tolerance),
    )
    write_dataframe_csv(run_catalog, args.run_catalog_out)

    qc_catalog = build_e4_qc_catalog(run_catalog)
    write_dataframe_csv(qc_catalog, args.qc_catalog_out)

    use_qc_analysis_pass = not bool(args.include_qc_fail_in_analysis)

    leakage_metrics = _build_leakage_table(
        qc_catalog,
        use_qc_analysis_pass=use_qc_analysis_pass,
        main_lobe_half_width_bins=int(args.main_lobe_half_width_bins),
        bin_center_tolerance=float(args.bin_center_tolerance),
    )
    write_dataframe_csv(leakage_metrics, args.leakage_out)

    resolution_curve = _build_resolution_table(
        qc_catalog,
        use_qc_analysis_pass=use_qc_analysis_pass,
        min_peak_prominence_db=float(args.min_peak_prominence_db),
        min_valley_depth_db=float(args.min_valley_depth_db),
    )
    write_dataframe_csv(resolution_curve, args.resolution_out)

    window_table = _build_window_table(
        qc_catalog,
        use_qc_analysis_pass=use_qc_analysis_pass,
        run_id=args.f9_run_id,
        fft_backend=str(args.fft_backend),
    )
    write_dataframe_csv(window_table, args.window_out)

    t2_table = build_e4_t2_table(qc_catalog)
    write_table_manifest_csv(t2_table, args.t2_out)

    t5_table = build_e4_t5_table(leakage_metrics, resolution_curve)
    write_table_manifest_csv(t5_table, args.t5_out)

    f7_path = _write_f7(
        qc_catalog,
        leakage_metrics=leakage_metrics,
        path=args.f7_out,
        use_qc_analysis_pass=use_qc_analysis_pass,
    )
    f8_path = _write_f8(resolution_curve, path=args.f8_out)
    f9_path = _write_f9(window_table, path=args.f9_out, run_id=args.f9_run_id)

    print(f"Run catalog rows: {len(run_catalog)}")
    print(f"QC analysis-pass rows: {int(qc_catalog['qc_analysis_pass'].fillna(False).sum())}")
    print(f"QC recommended-pass rows: {int(qc_catalog['qc_recommended_pass'].fillna(False).sum())}")
    print(f"Leakage rows: {len(leakage_metrics)}")
    print(f"Resolution rows: {len(resolution_curve)}")
    print(f"Window spectrum rows: {len(window_table)}")
    print(f"T2 rows: {len(t2_table)}")
    print(f"T5 rows: {len(t5_table)}")
    if "status" in run_catalog.columns:
        print("Run status counts:")
        counts = run_catalog["status"].value_counts(dropna=False)
        for key, count in counts.items():
            print(f"  {key}: {int(count)}")
    print(f"Run catalog CSV: {args.run_catalog_out}")
    print(f"QC catalog CSV: {args.qc_catalog_out}")
    print(f"Leakage CSV: {args.leakage_out}")
    print(f"Resolution CSV: {args.resolution_out}")
    print(f"Window spectra CSV: {args.window_out}")
    print(f"T2 table CSV: {args.t2_out}")
    print(f"T5 table CSV: {args.t5_out}")
    print(f"F7 figure: {f7_path if f7_path is not None else '<skipped>'}")
    print(f"F8 figure: {f8_path if f8_path is not None else '<skipped>'}")
    print(f"F9 figure: {f9_path if f9_path is not None else '<skipped>'}")


def _build_leakage_table(
    qc_catalog: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool,
    main_lobe_half_width_bins: int,
    bin_center_tolerance: float,
) -> pd.DataFrame:
    try:
        return build_e4_leakage_metrics(
            qc_catalog,
            use_qc_analysis_pass=use_qc_analysis_pass,
            main_lobe_half_width_bins=main_lobe_half_width_bins,
            bin_center_tolerance=bin_center_tolerance,
        )
    except ValueError as error:
        print(f"[warn] leakage table not built: {error}")
        return pd.DataFrame()


def _build_resolution_table(
    qc_catalog: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool,
    min_peak_prominence_db: float,
    min_valley_depth_db: float,
) -> pd.DataFrame:
    try:
        return build_e4_resolution_curve(
            qc_catalog,
            use_qc_analysis_pass=use_qc_analysis_pass,
            min_peak_prominence_db=min_peak_prominence_db,
            min_valley_depth_db=min_valley_depth_db,
        )
    except ValueError as error:
        print(f"[warn] resolution table not built: {error}")
        return pd.DataFrame()


def _build_window_table(
    qc_catalog: pd.DataFrame,
    *,
    use_qc_analysis_pass: bool,
    run_id: str | None,
    fft_backend: str,
) -> pd.DataFrame:
    try:
        return build_e4_window_spectrum_table(
            qc_catalog,
            use_qc_analysis_pass=use_qc_analysis_pass,
            run_id=run_id,
            fft_backend=fft_backend,
        )
    except ValueError as error:
        print(f"[warn] window table not built: {error}")
        return pd.DataFrame()


def _write_f7(
    qc_catalog: pd.DataFrame,
    *,
    leakage_metrics: pd.DataFrame,
    path: Path,
    use_qc_analysis_pass: bool,
) -> Path | None:
    if leakage_metrics.empty:
        return None
    return write_e4_leakage_figure(
        qc_catalog,
        path,
        leakage_metrics=leakage_metrics,
        use_qc_analysis_pass=use_qc_analysis_pass,
        db=True,
    )


def _write_f8(
    resolution_curve: pd.DataFrame,
    *,
    path: Path,
) -> Path | None:
    if resolution_curve.empty:
        return None
    return write_e4_resolution_figure(resolution_curve, path)


def _write_f9(
    window_table: pd.DataFrame,
    *,
    path: Path,
    run_id: str | None,
) -> Path | None:
    if window_table.empty:
        return None
    return write_e4_multi_window_figure(window_table, path, run_id=run_id, db=True)


if __name__ == "__main__":
    main()

