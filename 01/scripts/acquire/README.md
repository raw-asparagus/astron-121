# Acquisition Scripts

Place SDR/signal-generator control scripts here for lab data collection.

## E1 Runner

`run_e1_acquire.py` runs Experiment 1 physical data capture with:
- N9310A via direct USBTMC (`/dev/usbtmc0`)
- SDR direct mode (`device_index=0`, `direct=True`, `gain=0`)
- stale-buffer policy (`nblocks=11`, drop first, keep 10)
- baseline + target run pairing per combination
- resume-safe skip logic from progress CSV
