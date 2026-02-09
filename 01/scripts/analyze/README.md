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
