# Analysis Scripts

Batch analysis pipelines that transform raw captures into processed notebook inputs.

## E1 Physical Pipeline

`run_e1_pipeline.py` consumes raw E1 NPZ input (default `data/raw/e1.tar.gz`) and writes:
- `data/interim/e1/run_catalog.csv`
- `data/interim/e1/qc_catalog.csv`
- `data/processed/e1/tables/T2_e1_runs.csv`
- `data/processed/e1/tables/T3_e1_alias_residuals.csv`
- `report/figures/F2_alias_map_physical.png`

Example:

```bash
cd labs/01
/Users/junruiting/GitHub/ugradio/.venv/bin/python3 scripts/analyze/run_e1_pipeline.py
```

## E2 Physical Pipeline

`run_e2_pipeline.py` consumes raw E2 NPZ input (default `data/raw/e2.tar.gz`) and writes:
- `data/interim/e2/run_catalog.csv`
- `data/interim/e2/qc_catalog.csv`
- `data/interim/e2/bandpass_curves.csv`
- `data/processed/e2/tables/T2_e2_runs.csv`
- `data/processed/e2/tables/T4_e2_bandpass_summary.csv`
- `report/figures/F4_bandpass_curves_physical.png`

Example:

```bash
cd labs/01
/Users/junruiting/GitHub/ugradio/.venv/bin/python3 scripts/analyze/run_e2_pipeline.py
```

## E3 Physical Pipeline

`run_e3_pipeline.py` consumes raw E3 NPZ input (default `data/raw/e3.tar.gz`) and writes:
- `data/interim/e3/run_catalog.csv`
- `data/interim/e3/qc_catalog.csv`
- `data/interim/e3/spectrum_profile.csv`
- `data/processed/e3/tables/T2_e3_runs.csv`
- `report/figures/F5_complex_voltage_components_physical.png`
- `report/figures/F6_voltage_vs_power_physical.png`

Example:

```bash
cd labs/01
/Users/junruiting/GitHub/ugradio/.venv/bin/python3 scripts/analyze/run_e3_pipeline.py
```

## E5 Physical Pipeline

`run_e5_pipeline.py` consumes raw E5 NPZ input (default `data/raw/e5.tar.gz`) and writes:
- `data/interim/e5/run_catalog.csv`
- `data/interim/e5/qc_catalog.csv`
- `data/interim/e5/noise_stats.csv`
- `data/interim/e5/radiometer_curve.csv`
- `data/processed/e5/tables/T2_e5_runs.csv`
- `data/processed/e5/tables/T6_e5_radiometer_summary.csv`
- `report/figures/F10_noise_histogram_physical.png`
- `report/figures/F11_radiometer_scaling_physical.png`
- `report/figures/F12_acf_spectrum_consistency_physical.png`

Example:

```bash
cd labs/01
/Users/junruiting/GitHub/ugradio/.venv/bin/python3 scripts/analyze/run_e5_pipeline.py
```
