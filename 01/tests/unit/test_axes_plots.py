"""Unit tests for plotting.axes_plots."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ugradio_lab1.plotting.axes_plots import (
    plot_alias_map,
    plot_bandpass_curves,
    plot_histogram_with_gaussian,
    plot_iq_scatter,
    plot_iq_phase_trajectory,
    plot_power_spectrum,
    plot_power_spectrum_comparison,
    plot_radiometer_fit,
    plot_resolution_vs_n,
    plot_spur_survey,
    plot_time_series_comparison,
    plot_time_series,
    plot_voltage_spectrum,
    plot_waveform_comparison,
    plot_windowed_spectra,
)


def test_plot_time_series_sets_labels_and_data() -> None:
    figure, axis = plt.subplots()
    time = np.linspace(0.0, 1.0, 100)
    voltage = np.sin(2.0 * np.pi * 5.0 * time)

    line = plot_time_series(axis, time, voltage, label="signal")

    assert line.get_xdata().size == 100
    assert axis.get_xlabel() == "Time (s)"
    assert axis.get_ylabel() == "Voltage (V)"
    assert line.get_label() == "signal"
    plt.close(figure)


def test_plot_voltage_spectrum_phase_component() -> None:
    figure, axis = plt.subplots()
    frequency = np.array([-1.0, 0.0, 1.0])
    spectrum = np.array([1.0 + 1.0j, 2.0 + 0.0j, 1.0 - 1.0j])

    line = plot_voltage_spectrum(axis, frequency, spectrum, component="phase")

    assert np.allclose(line.get_ydata(), np.angle(spectrum))
    assert axis.get_ylabel() == "Phase (rad)"
    plt.close(figure)


def test_plot_power_spectrum_db_conversion() -> None:
    figure, axis = plt.subplots()
    frequency = np.array([-1.0, 0.0, 1.0])
    power = np.array([1.0, 10.0, 100.0])

    line = plot_power_spectrum(axis, frequency, power, db=True, grid=False)

    assert np.allclose(line.get_ydata(), np.array([0.0, 10.0, 20.0]))
    assert axis.get_ylabel() == "Power (dB re 1 V^2)"
    plt.close(figure)


def test_plot_histogram_with_gaussian_returns_artists() -> None:
    figure, axis = plt.subplots()
    rng = np.random.default_rng(seed=4)
    samples = rng.normal(loc=0.2, scale=1.7, size=2000)

    artists = plot_histogram_with_gaussian(axis, samples, bins=40, annotate_stats=True)

    assert artists["hist"] is not None
    assert artists["gaussian"] is not None
    assert artists["stats_text"] is not None
    assert artists["gaussian"].get_xdata().size == 512
    plt.close(figure)


def test_plot_iq_scatter_sets_equal_aspect() -> None:
    figure, axis = plt.subplots()
    i_samples = np.array([0.0, 1.0, -1.0, 0.5])
    q_samples = np.array([1.0, 0.0, -1.0, -0.5])

    scatter = plot_iq_scatter(axis, i_samples, q_samples, label="iq")

    assert scatter.get_offsets().shape[0] == 4
    assert axis.get_xlabel() == "In-phase I (V)"
    assert axis.get_ylabel() == "Quadrature Q (V)"
    assert axis.get_aspect() in ("equal", 1.0)
    plt.close(figure)


def test_plot_alias_map_with_residual_axis() -> None:
    figure, (main_ax, residual_ax) = plt.subplots(2, 1, sharex=True)
    f_true = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 1.9, 3.1])
    measured = np.array([1.0, 2.0, 3.0])

    artists = plot_alias_map(
        main_ax,
        f_true,
        measured,
        predicted_alias_hz=predicted,
        residual_ax=residual_ax,
    )

    assert artists["measured"] is not None
    assert artists["predicted"] is not None
    assert artists["residual"] is not None
    plt.close(figure)


def test_plot_bandpass_curves_multiple_modes() -> None:
    figure, axis = plt.subplots()
    freq = np.array([1.0, 2.0, 3.0])
    curves = {"default": np.array([0.0, -1.0, -3.0]), "fir": np.array([0.0, -0.8, -2.2])}

    lines = plot_bandpass_curves(axis, freq, curves)

    assert len(lines) == 2
    assert axis.get_ylabel() == "Gain (dB)"
    plt.close(figure)


def test_plot_resolution_vs_n_with_theory() -> None:
    figure, axis = plt.subplots()
    n = np.array([128, 256, 512], dtype=float)
    measured = np.array([8.0, 4.2, 2.1], dtype=float)

    artists = plot_resolution_vs_n(axis, n, measured, sample_rate_hz=1024.0)

    assert artists["measured"] is not None
    assert artists["theory"] is not None
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "log"
    plt.close(figure)


def test_plot_radiometer_fit_outputs_fit_artist() -> None:
    figure, axis = plt.subplots()
    n = np.array([1, 2, 4, 8, 16], dtype=float)
    sigma = 1.0 / np.sqrt(n)

    artists = plot_radiometer_fit(axis, n, sigma)

    assert artists["data"] is not None
    assert artists["fit"] is not None
    plt.close(figure)


def test_plot_spur_survey_expected_line_markers() -> None:
    figure, axis = plt.subplots()
    freq = np.linspace(0.0, 10.0, 200)
    power = np.exp(-0.5 * ((freq - 4.0) / 0.2) ** 2)

    artists = plot_spur_survey(axis, freq, power, expected_lines_hz=np.array([4.0]), annotate_top_n=2)

    assert artists["spectrum"] is not None
    assert len(artists["expected_lines"]) == 1
    assert len(artists["annotations"]) == 2
    plt.close(figure)


def test_plot_iq_phase_trajectory_marks_start_end() -> None:
    figure, axis = plt.subplots()
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    i = np.cos(t)
    q = np.sin(t)

    artists = plot_iq_phase_trajectory(axis, i, q)

    assert artists["trajectory"] is not None
    assert artists["start"] is not None
    assert artists["end"] is not None
    plt.close(figure)


def test_plot_time_series_comparison_with_clip_bounds() -> None:
    figure, axis = plt.subplots()
    t = np.linspace(0.0, 1.0, 64)
    good = np.sin(2 * np.pi * 4.0 * t)
    bad = 1.5 * good

    artists = plot_time_series_comparison(axis, t, good, bad, clip_limits_v=(-1.0, 1.0))

    assert artists["reference"] is not None
    assert artists["comparison"] is not None
    assert len(artists["clip_bounds"]) == 2
    plt.close(figure)


def test_plot_windowed_spectra_multiple_lines() -> None:
    figure, axis = plt.subplots()
    freq = np.array([-1.0, 0.0, 1.0])
    lines = plot_windowed_spectra(
        axis,
        freq,
        {"boxcar": np.array([1.0, 2.0, 1.0]), "hann": np.array([0.8, 1.9, 0.8])},
    )
    assert len(lines) == 2
    plt.close(figure)


def test_plot_waveform_comparison_returns_lines() -> None:
    figure, axis = plt.subplots()
    t = np.linspace(0.0, 1.0, 100)
    original = np.sin(2 * np.pi * 5.0 * t)
    reconstructed = 0.95 * original

    artists = plot_waveform_comparison(axis, t, original, reconstructed)
    assert artists["original"] is not None
    assert artists["reconstructed"] is not None
    plt.close(figure)


def test_plot_power_spectrum_comparison_overlays_two_curves() -> None:
    figure, axis = plt.subplots()
    freq = np.array([-1.0, 0.0, 1.0])
    artists = plot_power_spectrum_comparison(
        axis,
        freq,
        np.array([1.0, 3.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
    )
    assert artists["a"] is not None
    assert artists["b"] is not None
    plt.close(figure)
