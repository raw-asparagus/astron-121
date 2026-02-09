"""Control subpackage for Lab 1 physical acquisition workflows."""

from ugradio_lab1.control.acquisition import (
    DEFAULT_E1_PROGRESS_PATH,
    DEFAULT_E1_RAW_DIR,
    DEFAULT_E1_T2_MANIFEST_PATH,
    E1AcquisitionConfig,
    e1_fir_modes,
    e1_frequency_grid_hz,
    e1_power_tiers_dbm,
    run_e1_acquisition,
)
from ugradio_lab1.control.sdr import (
    ADCSummary,
    SDRCaptureConfig,
    SDRCaptureResult,
    acquire_sdr_capture,
    alias_hack_fir_coeffs,
)
from ugradio_lab1.control.siggen import N9310AUSBTMC, SigGenIOError, SigGenRetryPolicy

__all__ = [
    "ADCSummary",
    "DEFAULT_E1_PROGRESS_PATH",
    "DEFAULT_E1_RAW_DIR",
    "DEFAULT_E1_T2_MANIFEST_PATH",
    "E1AcquisitionConfig",
    "N9310AUSBTMC",
    "SDRCaptureConfig",
    "SDRCaptureResult",
    "SigGenIOError",
    "SigGenRetryPolicy",
    "acquire_sdr_capture",
    "alias_hack_fir_coeffs",
    "e1_fir_modes",
    "e1_frequency_grid_hz",
    "e1_power_tiers_dbm",
    "run_e1_acquisition",
]
