# Lab 1 Agent Context (`labs/01`)

## Scope
This directory contains:
- The Lab 1 report notebook: `labs/01/01.ipynb`
- The installable package: `labs/01/src/ugradio_lab1`
- Scripts/config/data/tests/docs needed for SDR + mixer simulation and analysis

Primary lab manual reference:
- `lab_mixers/allmixers.tex`

## Current Status
Core spectrum and plotting modules are scaffolded and implemented.
- Analysis module: `labs/01/src/ugradio_lab1/analysis/spectra.py`
- Plotting module: `labs/01/src/ugradio_lab1/plotting/axes_plots.py`
- FFT backend selection is implemented in spectrum routines (`numpy` default, optional `ugradio`).
- P0 analysis modules are implemented for deliverable-oriented metrics:
  - `analysis/nyquist.py` (`predict_alias_frequency`, `alias_residual_table`)
  - `analysis/bandpass.py` (`bandpass_curve`, `bandpass_summary_metrics`)
  - `analysis/resolution.py` (`resolution_vs_n`)
  - `analysis/noise.py` (`radiometer_fit`)
  - `analysis/mixers.py` (`expected_dsb_lines`, `match_observed_lines`)
- P0 plotting functions are implemented in `plotting/axes_plots.py`:
  - `plot_alias_map`, `plot_bandpass_curves`, `plot_resolution_vs_n`,
    `plot_radiometer_fit`, `plot_spur_survey`, `plot_iq_phase_trajectory`
- OOP GridSpec figure builders are implemented in `plotting/figure_builders.py` for:
  - Full figure blueprint coverage F1 through F18

P1 blueprint coverage is now implemented for the remaining figures/tables:
- Analysis additions:
  - `analysis/leakage.py` (`leakage_metric`, `leakage_resolution_table`, `nyquist_window_extension`)
  - `analysis/bandpass.py` (`bandpass_summary_table`)
  - `analysis/noise.py` (`radiometer_summary_table`)
  - `analysis/mixers.py` (`line_spur_catalog`, plus observed-index matching detail)
- Data I/O and schema infrastructure:
  - `dataio/schema.py` defines and validates T1–T8 table schemas
  - `dataio/io_npz.py` implements NPZ + metadata read/write helpers
  - `dataio/catalog.py` implements manifest build/read/write/append/filter and T8 goal-coverage table helpers
- Plotting additions:
  - New Axes-level functions in `plotting/axes_plots.py`:
    - `plot_time_series_comparison`, `plot_windowed_spectra`,
      `plot_waveform_comparison`, `plot_power_spectrum_comparison`
  - Alias-map plotting supports scatter-style predicted overlays; F2 now uses
    scatter for predicted alias points (no connecting line), and the residual
    inset uses scatter markers instead of a connected line.
  - Time-domain figure builders now support sample slicing and default to a
    readable first-samples view (to avoid over-dense traces):
    - `TimeDomainComparisonFigureBuilder`
    - `FilteredWaveformFigureBuilder`
  - `plotting/style.py` now provides reusable Matplotlib style helpers
  - `plotting/figure_builders.py` now includes builders for all F1–F18 figure IDs
    (in addition to existing P0 builders)

Unit test coverage has been expanded with dedicated tests for:
- `analysis/leakage.py`
- `dataio/schema.py`, `dataio/io_npz.py`, `dataio/catalog.py`
- `plotting/style.py`
- New plotting axes functions and figure builders

Simulation generators are now fully implemented in `src/ugradio_lab1/sim`:
- `sim/nyquist.py` (E1-E4 support):
  - sampled time vectors, tone/multi-tone synthesis,
    alias sweep simulation with recovered peaks,
    bandpass sweep simulation,
    leakage experiment simulation,
    resolution sweep simulation,
    multi-window spectrum table generation.
- `sim/noise.py` (E5 support):
  - Gaussian noise/noise-block generators,
    optional band-limiting filter,
    synthetic noise capture blocks,
    radiometer averaging-law simulation,
    ACF/spectrum consistency simulation.
- `sim/mixers.py` (E6-E7 support):
  - ideal DSB output synthesis,
    nonlinear mixer/spur synthesis,
    spectral-line picker,
    DSB spur survey + T7-ready catalog simulation,
    SSB IQ sideband simulation,
    reverted-DSB IQ simulation,
    R820T-vs-external comparison simulation.

Simulation tests were added:
- `tests/unit/test_sim_nyquist.py`
- `tests/unit/test_sim_noise.py`
- `tests/unit/test_sim_mixers.py`

Notebook integration updates:
- `labs/01/01.ipynb` now includes SIM runner code blocks for E1-E7 and
  simulation caption markdown blocks for all generated simulation figures
  (F2 through F18).
- `labs/01/01.ipynb` now includes an E1 physical pipeline integration block
  (`PIPELINE_RUNNER`) that can ingest `data/raw/e1.tar.gz`, rebuild/write
  E1 interim/processed artifacts, and display the physical F2 output inline.
- E1 writeup sections in `labs/01/01.ipynb` are synchronized with the current
  physical acquisition contract:
  - `nblocks=6`, drop first stale block, save/use `5` blocks.
  - fixed FIR-mode power tiers (`default: -10/0/+10 dBm`, `alias_hack: -50/-40/-30 dBm`).
  - `0 Hz` omitted in physical E1 runs; no bisection/target-RMS search.
- F3 was moved from E1 outputs to Calibration/Pre-Checks context in the
  notebook structure and SIM runner flow.
- Section 3 ("Theory and Modeling Framework") was expanded into a full
  derivation-backed framework covering:
  - waves/superposition,
  - sampling/aliasing/Nyquist zones,
  - CTFT/IFT + Parseval and practical voltage/power spectrum usage,
  - DFT/FFT scaling with optional `ugradio.dft` backend comparison,
  - convolution theorem, heterodyne mixing products, and FIR filtering.
  The section now includes a symbol table and executable matplotlib demos.

Shared validation helpers were centralized in:
- `labs/01/src/ugradio_lab1/utils/validation.py`

Unit tests currently pass for `labs/01/tests` using the repo virtualenv.

Physical acquisition infrastructure for E1 is now implemented:
- Signal-generator wrapper:
  - `control/siggen.py` provides N9310A direct USBTMC control on `/dev/usbtmc0`
    with a minimal command set and retry/timeout logic.
  - Public API is intentionally minimal:
    - `set_freq_mhz`, `get_freq`
    - `set_ampl_dbm`, `get_ampl`
    - `rf_on`, `rf_off`, `rf_state`
  - E1 detector runtime now assumes setpoints sent to the generator are the
    effective setpoints (no query-back enforcement for startup/freq/power).
  - Query polling treats USBTMC read timeouts (`Errno 110`) as transient misses
    within the query timeout budget instead of immediate hard-fail.
- SDR wrapper:
  - `control/sdr.py` provides direct-sampling capture with stale-buffer handling
    (`nblocks=6`, drop first, keep 5), ADC guard metrics, and capture retries.
- Acquisition orchestrator:
  - `control/acquisition.py` implements E1 sweep execution with fixed power tiers
    per FIR mode, run-level resume-safe progress logging, NPZ metadata dumps,
    and T2 manifest append/skip behavior.
  - E1 acquisition omits exact `0 Hz` frequency points in physical capture
    because the connected signal generator cannot set `0 Hz`.
  - Default contracts are established in-package:
    - raw output dir: `data/raw/e1`
    - T2 manifest: `data/manifests/t2_e1_runs.csv`
    - progress log: `data/manifests/e1_progress.csv`
- Acquisition script:
  - `scripts/acquire/run_e1_acquire.py` runs physical E1 acquisition from CLI.
- Unit tests added for acquisition/control logic:
  - `tests/unit/test_control_sdr.py`
  - `tests/unit/test_control_acquisition.py`
  - `tests/unit/test_control_siggen.py`
- E1 post-acquisition data pipeline is implemented:
  - Reusable module: `src/ugradio_lab1/pipeline/e1.py`
  - CLI runner: `scripts/analyze/run_e1_pipeline.py`
  - Pipeline can ingest E1 raw data directly from `data/raw/e1.tar.gz` (or a raw NPZ directory).
  - Generated artifacts:
    - `data/interim/e1/run_catalog.csv`
    - `data/interim/e1/qc_catalog.csv`
    - `data/processed/e1/tables/T2_e1_runs.csv`
    - `data/processed/e1/tables/T3_e1_alias_residuals.csv`
    - `report/figures/F2_alias_map_physical.png`
- Pipeline unit tests added:
  - `tests/unit/test_pipeline_e1.py`

Physical acquisition infrastructure for E2 is now implemented:
- Acquisition orchestrator:
  - `control/acquisition.py` now includes `E2AcquisitionConfig`,
    `e2_frequency_grid_hz`, and `run_e2_acquisition`.
  - E2 sweep policy is logspace by default and uses one constant-power
    capture per `(sample_rate, signal_frequency)` combination.
  - Default E2 contracts are established in-package:
    - raw output dir: `data/raw/e2`
    - T2 manifest: `data/manifests/t2_e2_runs.csv`
    - progress log: `data/manifests/e2_progress.csv`
- Acquisition script:
  - `scripts/acquire/run_e2_acquire.py` runs physical E2 acquisition from CLI.
- Control package exports now include E2 acquisition symbols in
  `control/__init__.py`.
- E2 post-acquisition data pipeline is implemented:
  - Reusable module: `src/ugradio_lab1/pipeline/e2.py`
  - CLI runner: `scripts/analyze/run_e2_pipeline.py`
  - Pipeline can ingest E2 raw data directly from `data/raw/e2.tar.gz` (or a raw NPZ directory).
  - Generated artifacts:
    - `data/interim/e2/run_catalog.csv`
    - `data/interim/e2/qc_catalog.csv`
    - `data/interim/e2/bandpass_curves.csv`
    - `data/processed/e2/tables/T2_e2_runs.csv`
    - `data/processed/e2/tables/T4_e2_bandpass_summary.csv`
    - `report/figures/F4_bandpass_curves_physical.png`
- Pipeline unit tests added:
  - `tests/unit/test_pipeline_e2.py`
- Notebook integration updates:
  - `labs/01/01.ipynb` now includes an E2 physical pipeline integration block
    (`PIPELINE_RUNNER`) that ingests `data/raw/e2.tar.gz`, writes E2
    interim/processed artifacts, and renders the physical F4 output inline.
  - Reproducibility appendix now includes the E2 physical pipeline command.

Physical acquisition scripts for E3-E7 are now implemented:
- New one-shot acquisition scripts:
  - `scripts/acquire/run_e3_acquire.py`
  - `scripts/acquire/run_e4_acquire.py`
  - `scripts/acquire/run_e5_acquire.py`
  - `scripts/acquire/run_e6_acquire.py`
  - `scripts/acquire/run_e7_acquire.py`
- Shared helper module:
  - `scripts/acquire/manual_capture_common.py` centralizes:
    - CLI + interactive prompt resolution for required parameters,
    - SG1/SG2 validation,
    - one-shot SDR guarded capture,
    - NPZ metadata persistence,
    - T2 manifest append.
  - E3-E7 contract assumes manual analog SG setup only; SG frequencies/powers
    are captured as metadata and are not programmed from the RPi.

## Locked-In API Decisions (Do Not Revert)
These decisions were requested explicitly by the user:

1. `sample_rate_hz` is required for core spectrum/ACF functions.
- No timestamp-driven sample rate inference.
- `infer_sample_rate` was removed.

2. `average_power_spectrum` uses pre-blocked SDR data.
- Input is a 2D array: `(n_blocks, n_samples)`.
- Removed `block_size` and `overlap_fraction` arguments.

3. `autocorrelation` API is simplified.
- Removed `mode` and `normalize` arguments.
- Behavior is fixed to full-lag, biased normalization.

4. `power_spectrum_from_autocorrelation` was removed.
- Workflow is voltage-data-first.

5. No window should be applied by default.
- Default `window=None` in windowed analysis functions.

6. Array validators are shared utilities.
- Use `as_1d_array` and `as_2d_array` from `ugradio_lab1.utils.validation`.
- Avoid reintroducing local duplicates in modules.

7. FFT backend is selectable where FFT/IFFT is used in spectra analysis.
- Use `fft_backend=\"numpy\"` by default.
- Support `fft_backend=\"ugradio\"` for `ugradio.dft`/`ugradio.dft.idft` parity checks and analysis.

8. Time-domain plots should use slices, not full records, by default.
- In notebook runners, pass an explicit sample slice for time-domain figures.
- In figure builders and standalone time-series helpers, default to a tighter
  readable subset (`~300` samples) unless explicitly overridden.

9. SDR acquisition pre-check guards must run before accepting captures.
- Treat clipping/undriven checks as calibration/preflight gates, not post-hoc diagnostics.
- Reject candidate capture blocks when any condition is true:
  - `adc_rms < 10`
  - `adc_max >= 127`
  - `adc_min <= -128`
- Future acquisition scripts should implement a try/retry loop that logs rejected
  attempts and only returns data after a valid block passes all guards.

10. E1 physical acquisition contract is locked for current runs.
- Signal generator model/control: N9310A via direct USBTMC (`/dev/usbtmc0`).
- Signal-generator API for physical scripts is minimal and fixed:
  - `set_freq_mhz`, `get_freq`, `set_ampl_dbm`, `get_ampl`,
    `rf_on`, `rf_off`, `rf_state`.
- Detector runtime assumes requested signal-generator frequency/power values
  are the capture metadata values (no query-back verification requirement).
- Instrument and SDR call timeout/retry policy: timeout `10s`, max retries `3`.
- Signal-generator settling delay: `1s` after set commands.
- SDR E1 capture settings:
  - `device_index=0`, `direct=True`, `gain=0.0`, `nsamples=2048`
  - request `nblocks=6`, drop first stale block, save/use `5` blocks.
- E1 FIR modes must be captured as two sets:
  - default `fir_coeffs=None`
  - alias-hack `fir_coeffs=[0, ..., 0, 2047]`.
- E1 sweep grid:
  - sample rates: `1.0, 1.6, 2.4, 3.2 MHz`
  - per sample rate, signal frequencies are `24` linear points over
    `[0, 4 f_Nyquist]`, but `0 Hz` is omitted in physical runs.
- Fixed power-tier policy per `(sample_rate, signal_frequency, fir_mode)`:
  - no bisection search and no target-RMS scanning.
  - capture all tiers for each FIR mode:
    - `default` FIR: `-10, 0, +10 dBm`
    - `alias_hack` FIR: `-50, -40, -30 dBm`.
  - each tier run keeps ADC quality labels (`passes_guard`, `is_clipped`) in
    metadata/progress for downstream filtering.
- Persistence/resume policy:
  - deterministic run IDs/paths per combination and power tier;
  - verbose metadata written in every NPZ;
  - completed run IDs are skipped on resume (duplicate-safe behavior).

11. E2 physical acquisition contract is established for current runs.
- Signal generator model/control: N9310A via direct USBTMC (`/dev/usbtmc0`).
- Instrument and SDR call timeout/retry policy: timeout `10s`, max retries `3`.
- Signal-generator settling delay: `1s` after set commands.
- SDR E2 capture settings:
  - `device_index=0`, `direct=True`, `gain=0.0`, `nsamples=2048`
  - request `nblocks=6`, drop first stale block, save/use `5` blocks.
- E2 FIR/default detector configuration:
  - `fir_mode="default"`
  - `fir_coeffs=None`.
- E2 sweep grid policy:
  - sample rates: `1.0, 1.6, 2.4, 3.2 MHz`
  - per sample rate, signal frequencies are `50` logspace points over
    `[10 kHz, 4 f_Nyquist]` (upper bound is `2 * f_s`).
- E2 source power policy:
  - one constant power per sweep point (`-10 dBm` default, configurable).
- Persistence/resume policy:
  - deterministic run IDs/paths per sweep point;
  - verbose metadata written in every NPZ;
  - completed run IDs are skipped on resume (duplicate-safe behavior).

12. E3-E7 physical acquisition scripts are one-shot (non-sweep) CLIs.
- Scripts must validate CLI arguments and prompt for missing required values.
- Required run-level parameters include `Vrms` and SG settings where applicable.
- SG usage by experiment:
  - E3: SG1 required, SG2 optional/required by mode.
  - E4: SG1 required, SG2 required in resolution mode.
  - E5: no SG required.
  - E6: SG1 + SG2 required.
  - E7: SG1 required, SG2 required for external modes.

13. E1 post-processing entrypoint and artifact contract is established.
- Raw starting point defaults to `data/raw/e1.tar.gz` (direct acquisition output tarball).
- The reproducible E1 processing runner is `scripts/analyze/run_e1_pipeline.py`.
- Required pipeline outputs for notebook/report integration:
  - `data/interim/e1/run_catalog.csv`
  - `data/interim/e1/qc_catalog.csv`
  - `data/processed/e1/tables/T2_e1_runs.csv`
  - `data/processed/e1/tables/T3_e1_alias_residuals.csv`
  - `report/figures/F2_alias_map_physical.png`.

14. E2 post-processing entrypoint and artifact contract is established.
- Raw starting point defaults to `data/raw/e2.tar.gz` (direct acquisition output tarball).
- The reproducible E2 processing runner is `scripts/analyze/run_e2_pipeline.py`.
- Required pipeline outputs for notebook/report integration:
  - `data/interim/e2/run_catalog.csv`
  - `data/interim/e2/qc_catalog.csv`
  - `data/interim/e2/bandpass_curves.csv`
  - `data/processed/e2/tables/T2_e2_runs.csv`
  - `data/processed/e2/tables/T4_e2_bandpass_summary.csv`
  - `report/figures/F4_bandpass_curves_physical.png`.

## Plotting Requirements
Plotting functions should stay Axes-first and composable:
- `axes_plots.py` functions take a Matplotlib `Axes` and return artist handles.
- Figure assembly/rendering logic should live separately (GridSpec + OOP) in figure builder modules.

## Testing / Verification
Use the project virtualenv interpreter:
- `/Users/junruiting/GitHub/ugradio/.venv/bin/python3 -m pytest -q labs/01/tests`
- `/Users/junruiting/GitHub/ugradio/.venv/bin/python3 -m compileall labs/01/src labs/01/tests`

## Working Norms for Future Sessions
1. Keep `labs/01/01.ipynb` untouched unless explicitly asked.
2. Prefer reusable package code in `src/ugradio_lab1` over notebook-only logic.
3. Preserve the SDR data assumption: data arrive as blocks, sample rate comes from metadata/config.
4. Keep the API simple and explicit; avoid adding optional complexity unless requested.

## Maintenance Rule (Required)
After any meaningful code or API change in `labs/01`, update this file (`labs/01/AGENTS.md`) in the same session:
1. Update `Current Status` with what changed.
2. Update `Locked-In API Decisions` if any user-level decisions changed.
3. Add/remove verification commands only if the project workflow changed.
