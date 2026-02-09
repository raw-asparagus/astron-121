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

## E3-E7 One-Shot Runners

These scripts are for non-sweep, one-shot physical captures:
- `run_e3_acquire.py`
- `run_e4_acquire.py`
- `run_e5_acquire.py`
- `run_e6_acquire.py`
- `run_e7_acquire.py`

Shared behavior:
- Validate CLI inputs.
- Prompt interactively for missing required values (e.g., `Vrms`, SG settings).
- Capture one run with SDR guard-based recapture attempts.
- Save NPZ data + metadata and append one T2 manifest row.
- Signal setup is manual analog for E3-E7 (SG values are metadata inputs only).

Parameter patterns:
- E3: `Vrms`, SG1 required; SG2 required in `two_tone` mode.
- E4: sweep runner by default (`50` runs) with mixed powers-of-two/non-powers-of-two bins (`N < 16384`);
  SG2 is the manual reference input and SG1 is auto-matched to SG2.
- E5: `Vrms` and noise-source mode; no signal generator required.
- E6: `Vrms`, SG2 manual reference required; SG1 (LO) auto-matches SG2.
- E7: `Vrms`, SG2 manual reference required; SG1 auto-matches SG2 (SG2 used as RF in external modes).
