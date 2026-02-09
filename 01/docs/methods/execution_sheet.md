# Lab 1 One-Day Execution Sheet (Physical + Simulation)

Use this sheet during data-taking to keep physical and simulation tracks synchronized.

## 1. Global Pre-Flight (Must Complete First)

- [ ] Warm-up complete; instrument clocks stable.
- [ ] SDR baseline capture with terminated input recorded.
- [ ] Signal generator frequency check recorded.
- [ ] Signal amplitude safety check on scope recorded.
- [ ] SDR guard thresholds validated in pre-check:
  - reject if `ADC RMS < 10`
  - reject if `ADC max >= 127` or `ADC min <= -128`
- [ ] Retry policy validated: rejected captures trigger re-read until guard conditions pass.
- [ ] Data path/metadata path verified (run manifest writable).
- [ ] First-capture stale-buffer policy applied (discard first capture).

## 2. Experiment Run Order

1. E1 Nyquist/Aliasing
2. E2 Bandpass
3. E3 Voltage/Power Spectra
4. E4 Leakage/Resolution/Nyquist Windows
5. E5 Noise/ACF/Radiometer
6. E6 DSB/Intermod
7. E7 SSB/Reverted DSB/R820T

## 3. Per-Experiment Compact Checklist

### E1 Nyquist/Aliasing

**Physical**
- [ ] Sweep `(f_true, f_s)` grid and capture all runs.
- [ ] Repeat one anchor run at end for drift check.

**Simulation**
- [ ] Compute predicted alias frequencies for full grid.
- [ ] Generate synthetic sampled tones with matched `N` and `f_s`.

**Must-Have Output**
- [ ] F2 alias map (+ residual inset)
- [ ] T3 alias residual table

### E2 SDR Bandpass

**Physical**
- [ ] Constant-amplitude frequency sweep in default mode.
- [ ] Repeat sweep in alias/FIR-modified mode.

**Simulation**
- [ ] Apply empirical/ideal bandpass model to synthetic tones.
- [ ] Compare modeled vs measured rolloff and passband.

**Must-Have Output**
- [ ] F4 bandpass curves
- [ ] T4 bandpass summary metrics

### E3 Voltage vs Power Spectra

**Physical**
- [ ] Single-tone and two-tone captures with logged settings.
- [ ] Compute complex voltage and power spectra.

**Simulation**
- [ ] Real and complex synthetic signals with known frequencies.
- [ ] Confirm symmetry/sign conventions with same pipeline.

**Must-Have Output**
- [ ] F5 complex voltage spectrum panels
- [ ] F6 voltage vs power comparison

### E4 Leakage/Resolution/Nyquist Windows

**Physical**
- [ ] Off-bin and bin-centered tone captures.
- [ ] Two-tone runs over increasing `N`.
- [ ] Multi-window spectral output for selected runs.

**Simulation**
- [ ] Off-bin leakage and `N`-dependent resolution simulations.
- [ ] Compare expected and measured `delta f` separability.

**Must-Have Output**
- [ ] F7 leakage comparison
- [ ] F8 resolution vs `N`
- [ ] F9 multi-window spectra
- [ ] T5 leakage/resolution metrics

### E5 Noise/ACF/Radiometer

**Physical**
- [ ] Long noise captures with anti-alias filtering.
- [ ] Histogram/statistics and block-averaged spectra.
- [ ] ACF via direct and transform routes.

**Simulation**
- [ ] Matched Gaussian-noise datasets.
- [ ] Radiometer scaling fit and ACF/spectrum cross-check.

**Must-Have Output**
- [ ] F10 histogram + Gaussian fit
- [ ] F11 `sigma` scaling log-log fit
- [ ] F12 ACF/spectrum consistency
- [ ] T6 noise/radiometer fit summary

### E6 DSB/Intermod

**Physical**
- [ ] LO/RF runs for `f_RF = f_LO +- delta f`.
- [ ] Wide dynamic-range spectrum for spur survey.

**Simulation**
- [ ] Ideal mixing products and nonlinear spur model.
- [ ] Expected vs observed line matching.

**Must-Have Output**
- [ ] F13 DSB output spectrum
- [ ] F15 spur survey
- [ ] T7 line and spur catalog (DSB rows)

### E7 SSB/Reverted DSB/R820T

**Physical**
- [ ] SSB IQ captures for both sidebands.
- [ ] Revert to DSB and repeat reference capture.
- [ ] R820T internal-mixer comparison capture.

**Simulation**
- [ ] IQ phasor model for sideband sign.
- [ ] DSB reversion and R820T-style comparison curves.

**Must-Have Output**
- [ ] F16 SSB IQ behavior
- [ ] F17 reverted-DSB comparison
- [ ] F18 R820T vs external comparison
- [ ] T7 line and spur catalog (SSB/R820T rows)

## 4. End-of-Day Closure

- [ ] Raw data backed up to second location.
- [ ] Run manifest complete and consistent with figures.
- [ ] Missing/failed runs documented with reason.
- [ ] Notebook placeholders updated with run IDs.
- [ ] Remaining analysis items tagged as `post-lab`.
