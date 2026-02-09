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
