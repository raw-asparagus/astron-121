"""Experiment-level plotting functions for Lab 1.

Each submodule corresponds to one experiment section and exposes
thin wrappers that accept pre-computed arrays and return
``(Figure, dict[str, Axes])``.
"""

from ugradio_lab1.experiment.theory import (
    plot_theory_aliasing,
    plot_theory_beats,
    plot_theory_fourier,
    plot_theory_mixer,
)
from ugradio_lab1.experiment.calibration import plot_calibration_precheck
from ugradio_lab1.experiment.e1 import plot_e1_alias_map_combined
from ugradio_lab1.experiment.e2 import plot_e2_bandpass_default_and_aliased
from ugradio_lab1.experiment.e3 import (
    plot_e3_spectra_grid_physical,
    plot_e3_stacked_spectrum_sim,
)
from ugradio_lab1.experiment.e4 import (
    plot_e4_acf_physical,
    plot_e4_acf_sim,
    plot_e4_leakage_and_resolution_physical,
    plot_e4_leakage_and_resolution_sim,
    plot_e4_leakage_physical,
    plot_e4_leakage_sim,
    plot_e4_resolution_physical,
    plot_e4_resolution_sim,
    plot_e4_windows_physical,
    plot_e4_windows_sim,
)
from ugradio_lab1.experiment.e5 import (
    plot_e5_histogram_and_radiometer_physical,
    plot_e5_histogram_and_radiometer_sim,
    plot_e5_histogram_physical,
    plot_e5_histogram_sim,
    plot_e5_radiometer_physical,
    plot_e5_radiometer_sim,
)
from ugradio_lab1.experiment.e6 import (
    plot_e6_dsb_spectrum,
    plot_e6_filtered_waveform,
    plot_e6_spur_survey,
)
from ugradio_lab1.experiment.e7 import (
    plot_e7_r820t_comparison,
    plot_e7_reverted_dsb,
    plot_e7_ssb_iq,
)

__all__ = [
    "plot_theory_beats",
    "plot_theory_fourier",
    "plot_theory_aliasing",
    "plot_theory_mixer",
    "plot_calibration_precheck",
    "plot_e1_alias_map_combined",
    "plot_e2_bandpass_default_and_aliased",
    "plot_e3_stacked_spectrum_sim",
    "plot_e3_spectra_grid_physical",
    "plot_e4_acf_physical",
    "plot_e4_acf_sim",
    "plot_e4_leakage_and_resolution_physical",
    "plot_e4_leakage_and_resolution_sim",
    "plot_e4_leakage_physical",
    "plot_e4_leakage_sim",
    "plot_e4_resolution_physical",
    "plot_e4_resolution_sim",
    "plot_e4_windows_physical",
    "plot_e4_windows_sim",
    "plot_e5_histogram_and_radiometer_physical",
    "plot_e5_histogram_and_radiometer_sim",
    "plot_e5_histogram_physical",
    "plot_e5_histogram_sim",
    "plot_e5_radiometer_physical",
    "plot_e5_radiometer_sim",
    "plot_e6_dsb_spectrum",
    "plot_e6_filtered_waveform",
    "plot_e6_spur_survey",
    "plot_e7_ssb_iq",
    "plot_e7_reverted_dsb",
    "plot_e7_r820t_comparison",
]
