I wa# Traceability: `lab_mixers/allmixers.tex`

Map lab-manual goals and required tasks to package modules, scripts, tests, and report outputs.

## Goals (`allmixers.tex` \S Goals)

| Goal | Implementation Location | Evidence |
| --- | --- | --- |
| Nyquist sampling and aliasing | `analysis/nyquist.py`, `sim/nyquist.py`, `scripts/sim/` | Figures + quantitative answer in `01.ipynb` |
| SDR bandpass characterization | `analysis/bandpass.py`, `scripts/acquire/` | Bandpass plots and fit summaries |
| Power and voltage spectra | `analysis/spectra.py` | Spectrum comparisons and captions |
| Spectral leakage and frequency resolution | `analysis/leakage.py`, `analysis/resolution.py` | Leakage and resolution plots |
| Noise and radiometer equation | `analysis/noise.py`, `sim/noise.py` | Averaging/SNR scaling results |
| DSB/SSB mixer operation | `analysis/mixers.py`, `sim/mixers.py`, `scripts/acquire/` | Sideband discrimination figures |
| Quantitative diagnostics/tests | `tests/unit/`, `tests/integration/` | Test report + diagnostic plots |
| Installable tested package in VCS | `pyproject.toml`, `src/`, `tests/` | GitHub URL + install command in notebook |

## Software Engineering (`allmixers.tex` \S Software Engineering)

- Acquisition scripts live in `scripts/acquire/`.
- Shared reusable logic lives in `src/ugradio_lab1/`.
- Notebook analysis imports package code from `src/`.
- Tests are separated by scope in `tests/`.

## Report Constraints (`allmixers.tex` \S Writing the Lab Report)

- Main report notebook: `01.ipynb`
- Renderable figure assets: `report/figures/`
- Caption drafts and notes: `report/captions/`
