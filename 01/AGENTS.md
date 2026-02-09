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
    with query-back verification, 10s timeout, and retry logic.
- SDR wrapper:
  - `control/sdr.py` provides direct-sampling capture with stale-buffer handling
    (`nblocks=11`, drop first, keep 10), ADC guard metrics, and capture retries.
- Acquisition orchestrator:
  - `control/acquisition.py` implements E1 sweep execution, baseline/target run
    pairing, bisection search, resume-safe progress logging, NPZ metadata dumps,
    and T2 manifest append/skip behavior.
  - Default contracts are established in-package:
    - raw output dir: `data/raw/e1`
    - T2 manifest: `data/manifests/t2_e1_runs.csv`
    - progress log: `data/manifests/e1_progress.csv`
- Acquisition script:
  - `scripts/acquire/run_e1_acquire.py` runs physical E1 acquisition from CLI.
- Unit tests added for acquisition/control logic:
  - `tests/unit/test_control_sdr.py`
  - `tests/unit/test_control_acquisition.py`

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
- Instrument and SDR call timeout/retry policy: timeout `10s`, max retries `3`.
- Signal-generator settling delay: `1s` after set commands.
- SDR E1 capture settings:
  - `device_index=0`, `direct=True`, `gain=0.0`, `nsamples=2048`
  - request `nblocks=11`, drop first stale block, save/use `10` blocks.
- E1 FIR modes must be captured as two sets:
  - default `fir_coeffs=None`
  - alias-hack `fir_coeffs=[0, ..., 0, 2047]`.
- E1 sweep grid:
  - sample rates: `1.0, 1.6, 2.4, 3.2 MHz`
  - per sample rate, signal frequencies are `24` linear points over
    `[0, 4 f_Nyquist]` inclusive.
- ADC power-search policy per `(sample_rate, signal_frequency, fir_mode)`:
  - always capture/save baseline at `-30 dBm`.
  - target setpoint uses bisection in `[-30, +10] dBm`, precision `0.1 dBm`.
  - stop on the first valid capture with `ADCrms in 65 +/- 5`.
  - if baseline is already in target band, duplicate baseline as target run.
  - if clipping occurs at `-30 dBm`, classify as `unachievable` and retain
    baseline + closest available target record.
  - if target band is never reached by `+10 dBm`, classify as `closest_only`.
- Persistence/resume policy:
  - deterministic run IDs/paths per combination and run kind;
  - verbose metadata written in every NPZ;
  - completed combinations are skipped on resume (duplicate-safe behavior).

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
