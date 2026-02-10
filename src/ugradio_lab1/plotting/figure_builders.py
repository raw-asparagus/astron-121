"""OOP figure builders that use GridSpec for final render layouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.axes_plots import (
    plot_alias_map,
    plot_autocorrelation,
    plot_bandpass_curves,
    plot_histogram_with_gaussian,
    plot_iq_phase_trajectory,
    plot_power_spectrum,
    plot_power_spectrum_comparison,
    plot_radiometer_fit,
    plot_resolution_vs_n,
    plot_spur_survey,
    plot_time_series_comparison,
    plot_voltage_spectrum,
    plot_waveform_comparison,
    plot_windowed_spectra,
)

SampleSlice = slice | tuple[int, int] | None


@dataclass
class SetupDiagramFigureBuilder:
    """GridSpec builder for F1 setup image + signal-chain notes."""

    figsize: tuple[float, float] = (12.0, 6.0)

    def build(
        self,
        setup_image: np.ndarray,
        *,
        signal_chain: Sequence[str] | None = None,
        annotations: Sequence[str] | None = None,
        title: str = "Setup + Signal Chain",
    ) -> tuple[Figure, dict[str, Axes]]:
        image = np.asarray(setup_image)
        if image.ndim not in {2, 3}:
            raise ValueError("setup_image must be a 2D grayscale or 3D RGB array.")

        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.15)
        ax_image = figure.add_subplot(grid[0, 0])
        ax_notes = figure.add_subplot(grid[0, 1])

        ax_image.imshow(image, aspect="auto")
        ax_image.set_title(title)
        ax_image.axis("off")

        lines: list[str] = []
        if signal_chain:
            lines.append("Signal Chain")
            lines.extend([f"- {item}" for item in signal_chain])
        if annotations:
            if lines:
                lines.append("")
            lines.append("Annotations")
            lines.extend([f"- {item}" for item in annotations])
        if not lines:
            lines = ["No notes supplied."]

        ax_notes.axis("off")
        ax_notes.text(0.0, 1.0, "\n".join(lines), va="top", ha="left")
        return figure, {"image": ax_image, "notes": ax_notes}


@dataclass
class AliasMapFigureBuilder:
    """GridSpec builder for F2 alias map + residual inset."""

    figsize: tuple[float, float] = (10.0, 7.0)

    def build(
        self,
        true_frequency_hz: np.ndarray,
        measured_alias_hz: np.ndarray,
        *,
        predicted_alias_hz: np.ndarray | None = None,
        residual_hz: np.ndarray | None = None,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08)
        ax_main = figure.add_subplot(grid[0, 0])
        ax_residual = figure.add_subplot(grid[1, 0], sharex=ax_main)

        plot_alias_map(
            ax_main,
            true_frequency_hz,
            measured_alias_hz,
            predicted_alias_hz=predicted_alias_hz,
            predicted_as_scatter=True,
            residual_ax=ax_residual,
            residual_hz=residual_hz,
            title="Alias Map",
        )
        ax_main.legend(loc="best")
        return figure, {"main": ax_main, "residual": ax_residual}


@dataclass
class TimeDomainComparisonFigureBuilder:
    """GridSpec builder for F3 good-vs-bad time-domain traces."""

    figsize: tuple[float, float] = (10.0, 4.5)

    def build(
        self,
        time_s: np.ndarray,
        good_voltage_v: np.ndarray,
        bad_voltage_v: np.ndarray,
        *,
        clip_limits_v: tuple[float, float] | None = None,
        sample_slice: SampleSlice = slice(0, 100),
    ) -> tuple[Figure, dict[str, Axes]]:
        """Build figure from a readable time slice (default first 300 samples)."""

        time_values, good_values, bad_values = _apply_sample_slice(
            time_s, good_voltage_v, bad_voltage_v, sample_slice
        )
        figure = plt.figure(figsize=self.figsize, dpi=300)
        ax = figure.add_subplot(1, 1, 1)

        plot_time_series_comparison(
            ax,
            time_values,
            good_values,
            bad_values,
            reference_label="Good run",
            comparison_label="Bad run",
            clip_limits_v=clip_limits_v,
            title="Signal clipping (Time-Domain)",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class BandpassFigureBuilder:
    """GridSpec builder for F4 bandpass curves."""

    figsize: tuple[float, float] = (9.0, 5.5)

    def build(
        self,
        frequency_hz: np.ndarray,
        gain_db_by_mode: dict[str, np.ndarray],
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)

        plot_bandpass_curves(ax, frequency_hz, gain_db_by_mode, title="SDR Bandpass Curves")
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class ComplexVoltageComponentsFigureBuilder:
    """GridSpec builder for F5 complex voltage-spectrum components."""

    figsize: tuple[float, float] = (12.0, 8.0)

    def build(
        self,
        frequency_hz: np.ndarray,
        spectrum_v: np.ndarray,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(2, 2, hspace=0.25, wspace=0.22)

        ax_real = figure.add_subplot(grid[0, 0])
        ax_imag = figure.add_subplot(grid[0, 1], sharex=ax_real)
        ax_mag = figure.add_subplot(grid[1, 0], sharex=ax_real)
        ax_phase = figure.add_subplot(grid[1, 1], sharex=ax_real)

        plot_voltage_spectrum(ax_real, frequency_hz, spectrum_v, component="real", title="Real")
        plot_voltage_spectrum(ax_imag, frequency_hz, spectrum_v, component="imag", title="Imag")
        plot_voltage_spectrum(ax_mag, frequency_hz, spectrum_v, component="magnitude", title="Magnitude")
        plot_voltage_spectrum(ax_phase, frequency_hz, spectrum_v, component="phase", title="Phase")
        figure.suptitle("Complex Voltage Spectrum Components")
        return figure, {"real": ax_real, "imag": ax_imag, "magnitude": ax_mag, "phase": ax_phase}


@dataclass
class VoltagePowerComparisonFigureBuilder:
    """GridSpec builder for F6 voltage vs power spectrum comparison."""

    figsize: tuple[float, float] = (10.0, 7.5)

    def build(
        self,
        frequency_hz: np.ndarray,
        voltage_spectrum_v: np.ndarray,
        power_spectrum_v2: np.ndarray,
        *,
        voltage_component: str = "magnitude",
        power_db: bool = False,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.08)
        ax_voltage = figure.add_subplot(grid[0, 0])
        ax_power = figure.add_subplot(grid[1, 0], sharex=ax_voltage)

        plot_voltage_spectrum(
            ax_voltage,
            frequency_hz,
            voltage_spectrum_v,
            component=voltage_component,  # type: ignore[arg-type]
            title="Voltage Spectrum",
        )
        plot_power_spectrum(
            ax_power,
            frequency_hz,
            power_spectrum_v2,
            db=power_db,
            title="Power Spectrum",
        )
        return figure, {"voltage": ax_voltage, "power": ax_power}


@dataclass
class LeakageComparisonFigureBuilder:
    """GridSpec builder for F10 bin-centered vs off-bin leakage (standalone; use LeakageAndResolutionFigureBuilder for combined figure)."""

    figsize: tuple[float, float] = (10.0, 5.0)

    def build(
        self,
        frequency_hz: np.ndarray,
        bin_centered_power_v2: np.ndarray,
        off_bin_power_v2: np.ndarray,
        *,
        db: bool = True,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_power_spectrum_comparison(
            ax,
            frequency_hz,
            bin_centered_power_v2,
            off_bin_power_v2,
            label_a="Bin-centered",
            label_b="Off-bin",
            db=db,
            title="Spectral Leakage Comparison",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class ResolutionFigureBuilder:
    """GridSpec builder for resolution vs N (standalone; use LeakageAndResolutionFigureBuilder for combined F10)."""

    figsize: tuple[float, float] = (9.0, 5.5)

    def build(
        self,
        n_samples: np.ndarray,
        measured_delta_f_hz: np.ndarray,
        *,
        sample_rate_hz: float | np.ndarray | None = None,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)

        plot_resolution_vs_n(
            ax,
            n_samples,
            measured_delta_f_hz,
            sample_rate_hz=sample_rate_hz,
            title="Frequency Resolution vs N",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class LeakageAndResolutionFigureBuilder:
    """GridSpec builder for F10: combined spectral leakage and resolution side-by-side."""

    figsize: tuple[float, float] = (14.0, 5.0)

    def build(
        self,
        # Leakage parameters
        leakage_frequency_hz: np.ndarray,
        bin_centered_power_v2: np.ndarray,
        off_bin_power_v2: np.ndarray,
        # Resolution parameters
        n_samples: np.ndarray,
        measured_delta_f_hz: np.ndarray,
        *,
        db: bool = True,
        sample_rate_hz: float | np.ndarray | None = None,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(1, 2, wspace=0.25)
        ax_leakage = figure.add_subplot(grid[0, 0])
        ax_resolution = figure.add_subplot(grid[0, 1])

        # Left: Spectral leakage comparison
        plot_power_spectrum_comparison(
            ax_leakage,
            leakage_frequency_hz,
            bin_centered_power_v2,
            off_bin_power_v2,
            label_a="Bin-centered",
            label_b="Off-bin",
            db=db,
            title="Spectral Leakage Comparison",
        )
        ax_leakage.legend(loc="best")

        # Right: Frequency resolution vs N
        plot_resolution_vs_n(
            ax_resolution,
            n_samples,
            measured_delta_f_hz,
            sample_rate_hz=sample_rate_hz,
            title="Frequency Resolution vs N",
        )
        ax_resolution.legend(loc="best")

        return figure, {"leakage": ax_leakage, "resolution": ax_resolution}


@dataclass
class MultiWindowSpectrumFigureBuilder:
    """GridSpec builder for F11 multi-window spectral views."""

    figsize: tuple[float, float] = (12, 7)

    def build(
        self,
        window_spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
        *,
        ncols: int = 2,
        db: bool = True,
    ) -> tuple[Figure, dict[str, Axes]]:
        if not window_spectra:
            raise ValueError("window_spectra cannot be empty.")
        if ncols < 1:
            raise ValueError("ncols must be >= 1.")

        labels = list(window_spectra.keys())
        n_plots = len(labels)
        nrows = int(math.ceil(n_plots / ncols))

        figure = plt.figure(figsize=self.figsize, dpi=300)
        grid = figure.add_gridspec(nrows, ncols, hspace=0.3, wspace=0.2)
        axes_by_window: dict[str, Axes] = {}

        for idx, label in enumerate(labels):
            row, col = divmod(idx, ncols)
            ax = figure.add_subplot(grid[row, col])
            freq, power = window_spectra[label]
            plot_windowed_spectra(
                ax,
                freq,
                {label: power},
                db=db,
                title=f"Window: {label}",
            )
            ax.legend(loc="best")
            axes_by_window[label] = ax

        # Turn off any unused panels for cleaner layout.
        for idx in range(n_plots, nrows * ncols):
            row, col = divmod(idx, ncols)
            ax_empty = figure.add_subplot(grid[row, col])
            ax_empty.axis("off")

        figure.suptitle("Multi-Window Spectra")
        return figure, axes_by_window


@dataclass
class NoiseHistogramFigureBuilder:
    """GridSpec builder for noise histogram + Gaussian fit (standalone; use NoiseHistogramAndRadiometerFigureBuilder for combined F12)."""

    figsize: tuple[float, float] = (8.5, 5.0)

    def build(
        self,
        samples: np.ndarray,
        *,
        bins: int = 60,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_histogram_with_gaussian(ax, samples, bins=bins, title="Noise Histogram + Gaussian Fit")
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class RadiometerFigureBuilder:
    """GridSpec builder for radiometer scaling (standalone; use NoiseHistogramAndRadiometerFigureBuilder for combined F12)."""

    figsize: tuple[float, float] = (9.0, 5.5)

    def build(
        self,
        n_avg: np.ndarray,
        sigma: np.ndarray,
        *,
        fit_result: dict[str, float] | None = None,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)

        plot_radiometer_fit(ax, n_avg, sigma, fit_result=fit_result, title="Radiometer Scaling")
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class NoiseHistogramAndRadiometerFigureBuilder:
    """GridSpec builder for F13: combined noise histogram and radiometer scaling side-by-side."""

    figsize: tuple[float, float] = (14.0, 5.0)

    def build(
        self,
        # Histogram parameters
        samples: np.ndarray,
        # Radiometer parameters
        n_avg: np.ndarray,
        sigma: np.ndarray,
        *,
        bins: int = 60,
        fit_result: dict[str, float] | None = None,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(1, 2, wspace=0.25)
        ax_histogram = figure.add_subplot(grid[0, 0])
        ax_radiometer = figure.add_subplot(grid[0, 1])

        # Left: Noise histogram + Gaussian fit
        plot_histogram_with_gaussian(
            ax_histogram,
            samples,
            bins=bins,
            title="Noise Histogram + Gaussian Fit",
        )
        ax_histogram.legend(loc="best")

        # Right: Radiometer scaling
        plot_radiometer_fit(
            ax_radiometer,
            n_avg,
            sigma,
            fit_result=fit_result,
            title="Radiometer Scaling",
        )
        ax_radiometer.legend(loc="best")

        return figure, {"histogram": ax_histogram, "radiometer": ax_radiometer}


@dataclass
class ACFSpectrumConsistencyFigureBuilder:
    """GridSpec builder for F11 ACF/spectrum consistency checks (used in E4 for single-tone demonstration)."""

    figsize: tuple[float, float] = (10.0, 7.0)

    def build(
        self,
        lag_s: np.ndarray,
        autocorrelation_values: np.ndarray,
        frequency_hz: np.ndarray,
        power_v2: np.ndarray,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(2, 1, hspace=0.15)
        ax_acf = figure.add_subplot(grid[0, 0])
        ax_psd = figure.add_subplot(grid[1, 0])

        plot_autocorrelation(ax_acf, lag_s, autocorrelation_values, component="real", title="Autocorrelation")
        plot_power_spectrum(ax_psd, frequency_hz, power_v2, db=True, title="Power Spectrum")
        return figure, {"acf": ax_acf, "spectrum": ax_psd}


@dataclass
class DSBOutputSpectrumFigureBuilder:
    """GridSpec builder for F14 DSB output spectrum + line markers."""

    figsize: tuple[float, float] = (10.0, 5.5)

    def build(
        self,
        frequency_hz: np.ndarray,
        power_v2: np.ndarray,
        *,
        expected_lines_hz: np.ndarray | None = None,
        annotate_top_n: int = 0,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)

        plot_spur_survey(
            ax,
            frequency_hz,
            power_v2,
            expected_lines_hz=expected_lines_hz,
            annotate_top_n=annotate_top_n,
            title="DSB Output Spectrum",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class FilteredWaveformFigureBuilder:
    """GridSpec builder for F15 inverse-transformed filtered waveform."""

    figsize: tuple[float, float] = (10.0, 4.5)

    def build(
        self,
        time_s: np.ndarray,
        original_voltage_v: np.ndarray,
        reconstructed_voltage_v: np.ndarray,
        *,
        sample_slice: SampleSlice = slice(0, 300),
    ) -> tuple[Figure, dict[str, Axes]]:
        """Build figure from a readable time slice (default first 300 samples)."""

        time_values, original_values, reconstructed_values = _apply_sample_slice(
            time_s, original_voltage_v, reconstructed_voltage_v, sample_slice
        )
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_waveform_comparison(
            ax,
            time_values,
            original_values,
            reconstructed_values,
            title="Filtered Waveform Reconstruction",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class SpurSurveyFigureBuilder:
    """GridSpec builder for F16 harmonic spur survey."""

    figsize: tuple[float, float] = (10.0, 5.5)

    def build(
        self,
        frequency_hz: np.ndarray,
        power_v2: np.ndarray,
        *,
        expected_lines_hz: np.ndarray | None = None,
        annotate_top_n: int = 8,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_spur_survey(
            ax,
            frequency_hz,
            power_v2,
            expected_lines_hz=expected_lines_hz,
            annotate_top_n=annotate_top_n,
            title="Harmonic Spur Survey",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class SSBIQBehaviorFigureBuilder:
    """GridSpec builder for F17 SSB IQ trajectory behavior."""

    figsize: tuple[float, float] = (11.0, 5.0)

    def build(
        self,
        i_upper_v: np.ndarray,
        q_upper_v: np.ndarray,
        i_lower_v: np.ndarray,
        q_lower_v: np.ndarray,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        grid = figure.add_gridspec(1, 2, wspace=0.25)
        ax_upper = figure.add_subplot(grid[0, 0])
        ax_lower = figure.add_subplot(grid[0, 1], sharex=ax_upper, sharey=ax_upper)

        plot_iq_phase_trajectory(
            ax_upper,
            i_upper_v,
            q_upper_v,
            title="Upper Sideband IQ",
            label="USB",
        )
        plot_iq_phase_trajectory(
            ax_lower,
            i_lower_v,
            q_lower_v,
            title="Lower Sideband IQ",
            label="LSB",
        )
        ax_upper.legend(loc="best")
        ax_lower.legend(loc="best")
        return figure, {"upper": ax_upper, "lower": ax_lower}


@dataclass
class RevertedDSBComparisonFigureBuilder:
    """GridSpec builder for F18 reverted-DSB comparison."""

    figsize: tuple[float, float] = (10.0, 5.0)

    def build(
        self,
        frequency_hz: np.ndarray,
        ssb_power_v2: np.ndarray,
        reverted_dsb_power_v2: np.ndarray,
        *,
        db: bool = True,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_power_spectrum_comparison(
            ax,
            frequency_hz,
            ssb_power_v2,
            reverted_dsb_power_v2,
            label_a="SSB",
            label_b="Reverted DSB",
            db=db,
            title="Reverted-DSB Comparison",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


@dataclass
class R820TComparisonFigureBuilder:
    """GridSpec builder for F19 external vs R820T internal comparison."""

    figsize: tuple[float, float] = (10.0, 5.0)

    def build(
        self,
        frequency_hz: np.ndarray,
        external_power_v2: np.ndarray,
        r820t_power_v2: np.ndarray,
        *,
        db: bool = True,
    ) -> tuple[Figure, dict[str, Axes]]:
        figure = plt.figure(figsize=self.figsize)
        ax = figure.add_subplot(1, 1, 1)
        plot_power_spectrum_comparison(
            ax,
            frequency_hz,
            external_power_v2,
            r820t_power_v2,
            label_a="External Mixer",
            label_b="R820T Internal",
            db=db,
            title="R820T vs External Mixer",
        )
        ax.legend(loc="best")
        return figure, {"main": ax}


def _normalize_sample_slice(sample_slice: SampleSlice, n_samples: int) -> slice:
    if sample_slice is None:
        return slice(0, n_samples)
    if isinstance(sample_slice, tuple):
        if len(sample_slice) != 2:
            raise ValueError("sample_slice tuple must be (start, stop).")
        start, stop = int(sample_slice[0]), int(sample_slice[1])
        return slice(start, stop)
    return sample_slice


def _apply_sample_slice(
    time_s: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    sample_slice: SampleSlice,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_values = np.asarray(time_s)
    first_values = np.asarray(first)
    second_values = np.asarray(second)
    if time_values.ndim != 1 or first_values.ndim != 1 or second_values.ndim != 1:
        raise ValueError("Time-domain inputs must be 1D arrays.")
    if not (time_values.size == first_values.size == second_values.size):
        raise ValueError("Time-domain inputs must have the same length.")

    selection = _normalize_sample_slice(sample_slice, time_values.size)
    sliced_time = time_values[selection]
    sliced_first = first_values[selection]
    sliced_second = second_values[selection]
    if sliced_time.size == 0:
        raise ValueError("sample_slice selects zero samples.")
    return sliced_time, sliced_first, sliced_second


__all__ = [
    "ACFSpectrumConsistencyFigureBuilder",
    "AliasMapFigureBuilder",
    "BandpassFigureBuilder",
    "ComplexVoltageComponentsFigureBuilder",
    "DSBOutputSpectrumFigureBuilder",
    "FilteredWaveformFigureBuilder",
    "LeakageAndResolutionFigureBuilder",
    "LeakageComparisonFigureBuilder",
    "MultiWindowSpectrumFigureBuilder",
    "NoiseHistogramAndRadiometerFigureBuilder",
    "NoiseHistogramFigureBuilder",
    "R820TComparisonFigureBuilder",
    "RadiometerFigureBuilder",
    "ResolutionFigureBuilder",
    "RevertedDSBComparisonFigureBuilder",
    "SSBIQBehaviorFigureBuilder",
    "SetupDiagramFigureBuilder",
    "SpurSurveyFigureBuilder",
    "TimeDomainComparisonFigureBuilder",
    "VoltagePowerComparisonFigureBuilder",
]
