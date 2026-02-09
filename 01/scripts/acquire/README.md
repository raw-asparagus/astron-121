# Acquisition Scripts

Place SDR/signal-generator control scripts here for lab data collection.

## E1 Runner

`run_e1_acquire.py` runs Experiment 1 physical data capture with:
- N9310A via direct USBTMC (`/dev/usbtmc0`)
- SDR direct mode (`device_index=0`, `direct=True`, `gain=0`)
- stale-buffer policy (`nblocks=6`, drop first, keep 5)
- fixed power tiers by FIR mode:
  - `default`: `-10, 0, +10 dBm`
  - `alias_hack`: `-50, -40, -30 dBm`
- `0 Hz` frequency point omitted for physical runs
- resume-safe skip logic from progress CSV

## E2 Runner

`run_e2_acquire.py` runs Experiment 2 physical bandpass capture with:
- N9310A via direct USBTMC (`/dev/usbtmc0`)
- SDR direct mode (`device_index=0`, `direct=True`, `gain=0`)
- stale-buffer policy (`nblocks=6`, drop first, keep 5)
- fixed FIR configuration by default:
  - `fir_mode="default"`
  - `fir_coeffs=None`
- logspace frequency scan per sample rate:
  - `50` points
  - lower bound `10 kHz`
  - upper bound `4 f_Nyquist` (i.e., `2 * f_s`)
- default sample rates: `1.0, 1.6, 2.4, 3.2 MHz`
- constant source power (`-10 dBm` default)
- resume-safe skip logic from progress CSV

## E3-E7 Runners

These scripts are for E3-E7 physical captures:
- `run_e3_acquire.py`
- `run_e4_acquire.py`
- `run_e5_acquire.py`
- `run_e6_acquire.py`
- `run_e7_acquire.py`

Shared behavior:
- Validate CLI inputs.
- Prompt interactively for missing required values (e.g., `Vrms`, SG settings).
- Capture SDR data with guard-based recapture attempts.
- Save NPZ data + metadata and append one T2 manifest row.
- Signal setup is manual analog by default.
- `run_e4_acquire.py` additionally supports optional SG USBTMC auto-programming.

Parameter patterns:
- E3: `Vrms`, SG1 required; SG2 required in `two_tone` mode.
- E4: sweep runner by default (`50` runs) with mixed powers-of-two/non-powers-of-two bins (`N < 16384`).
  Tone frequencies are auto-derived from `center_frequency_hz`:
  - leakage mode (single-SG): `f = (k + epsilon) * (f_s / N)`
  - resolution mode (two-SG): `f1 = f_center - delta/2`, `f2 = f_center + delta/2`
  Optional SG programming can be enabled with `--auto-program-siggen`.
- E5: `Vrms` and noise-source mode; no signal generator required.
- E6: signed-`delta_nu` sweep between `- |delta|` and `+ |delta|`; SG2 is manual RF reference and
  SG1 (LO) power matches SG2 while frequency follows `f1 = f2 - signed_delta`.
- E7: signed-`delta_nu` sweep between `- |delta|` and `+ |delta|`; prompts whether the single SDR
  capture is `I`/real or `Q`/imaginary voltage.
  - `r820t_internal` mode defaults to `|delta| = 0.05 * nu` with
    `nu = 1420.405751768 MHz`, forces `direct=False`, and sets SDR LO center to `nu`.
