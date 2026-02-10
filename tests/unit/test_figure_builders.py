"""Unit tests for plotting.figure_builders."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from ugradio_lab1.plotting.figure_builders import (
    ACFSpectrumConsistencyFigureBuilder,
    AliasMapFigureBuilder,
    BandpassFigureBuilder,
    ComplexVoltageComponentsFigureBuilder,
    DSBOutputSpectrumFigureBuilder,
    FilteredWaveformFigureBuilder,
    LeakageComparisonFigureBuilder,
    MultiWindowSpectrumFigureBuilder,
    NoiseHistogramFigureBuilder,
    R820TComparisonFigureBuilder,
    RadiometerFigureBuilder,
    ResolutionFigureBuilder,
    RevertedDSBComparisonFigureBuilder,
    SSBIQBehaviorFigureBuilder,
    SetupDiagramFigureBuilder,
    SpurSurveyFigureBuilder,
    TimeDomainComparisonFigureBuilder,
    VoltagePowerComparisonFigureBuilder,
)


def test_alias_map_figure_builder_returns_axes() -> None:
    builder = AliasMapFigureBuilder()
    true_freq = np.array([1.0, 2.0, 3.0])
    measured = np.array([1.1, 2.0, 2.9])
    predicted = np.array([1.0, 2.0, 3.0])

    fig, axes = builder.build(true_freq, measured, predicted_alias_hz=predicted)
    assert "main" in axes and "residual" in axes
    fig.clf()


def test_bandpass_figure_builder_returns_main_axis() -> None:
    builder = BandpassFigureBuilder()
    freq = np.array([1.0, 2.0, 3.0])
    curves = {"default": np.array([0.0, -1.0, -3.0])}

    fig, axes = builder.build(freq, curves)
    assert "main" in axes
    fig.clf()


def test_voltage_power_comparison_builder_two_axes() -> None:
    builder = VoltagePowerComparisonFigureBuilder()
    freq = np.array([-1.0, 0.0, 1.0])
    voltage = np.array([1.0 + 0j, 2.0 + 0j, 1.0 + 0j])
    power = np.array([1.0, 4.0, 1.0])

    fig, axes = builder.build(freq, voltage, power)
    assert "voltage" in axes and "power" in axes
    fig.clf()


def test_resolution_builder_outputs_main_axis() -> None:
    builder = ResolutionFigureBuilder()
    fig, axes = builder.build(np.array([128, 256]), np.array([8.0, 4.0]), sample_rate_hz=1024.0)
    assert "main" in axes
    fig.clf()


def test_radiometer_builder_outputs_main_axis() -> None:
    builder = RadiometerFigureBuilder()
    n = np.array([1, 2, 4, 8], dtype=float)
    sigma = 1.0 / np.sqrt(n)
    fig, axes = builder.build(n, sigma)
    assert "main" in axes
    fig.clf()


def test_dsb_builder_outputs_main_axis() -> None:
    builder = DSBOutputSpectrumFigureBuilder()
    freq = np.linspace(0.0, 10.0, 100)
    power = np.exp(-0.5 * ((freq - 4.0) / 0.3) ** 2)
    fig, axes = builder.build(freq, power, expected_lines_hz=np.array([4.0]))
    assert "main" in axes
    fig.clf()


def test_ssb_iq_builder_outputs_two_axes() -> None:
    builder = SSBIQBehaviorFigureBuilder()
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    i_u, q_u = np.cos(t), np.sin(t)
    i_l, q_l = np.cos(t), -np.sin(t)

    fig, axes = builder.build(i_u, q_u, i_l, q_l)
    assert "upper" in axes and "lower" in axes
    fig.clf()


def test_setup_diagram_builder_outputs_axes() -> None:
    builder = SetupDiagramFigureBuilder()
    image = np.random.default_rng(seed=1).normal(size=(32, 32))
    fig, axes = builder.build(
        image,
        signal_chain=["SigGen -> Mixer -> SDR"],
        annotations=["default mode"],
    )
    assert "image" in axes and "notes" in axes
    fig.clf()


def test_time_domain_comparison_builder_outputs_main_axis() -> None:
    builder = TimeDomainComparisonFigureBuilder()
    t = np.linspace(0.0, 1.0, 128)
    good = np.sin(2.0 * np.pi * 5.0 * t)
    bad = 1.2 * good
    fig, axes = builder.build(t, good, bad, clip_limits_v=(-1.0, 1.0))
    assert "main" in axes
    fig.clf()


def test_complex_voltage_components_builder_outputs_four_axes() -> None:
    builder = ComplexVoltageComponentsFigureBuilder()
    freq = np.array([-1.0, 0.0, 1.0])
    spectrum = np.array([1.0 + 1j, 2.0 + 0j, 1.0 - 1j])
    fig, axes = builder.build(freq, spectrum)
    assert {"real", "imag", "magnitude", "phase"}.issubset(axes.keys())
    fig.clf()


def test_leakage_comparison_builder_outputs_main_axis() -> None:
    builder = LeakageComparisonFigureBuilder()
    freq = np.linspace(-5.0, 5.0, 256)
    centered = np.exp(-0.5 * (freq / 0.2) ** 2)
    offbin = np.exp(-0.5 * ((freq - 0.05) / 0.2) ** 2) + 0.02
    fig, axes = builder.build(freq, centered, offbin)
    assert "main" in axes
    fig.clf()


def test_multi_window_spectrum_builder_outputs_named_axes() -> None:
    builder = MultiWindowSpectrumFigureBuilder()
    freq = np.array([-1.0, 0.0, 1.0])
    window_spectra = {
        "boxcar": (freq, np.array([1.0, 2.0, 1.0])),
        "hann": (freq, np.array([0.8, 1.9, 0.8])),
        "hamming": (freq, np.array([0.85, 1.95, 0.85])),
    }
    fig, axes = builder.build(window_spectra, ncols=2)
    assert set(axes.keys()) == {"boxcar", "hann", "hamming"}
    fig.clf()


def test_noise_histogram_builder_outputs_main_axis() -> None:
    builder = NoiseHistogramFigureBuilder()
    rng = np.random.default_rng(seed=5)
    samples = rng.normal(size=1000)
    fig, axes = builder.build(samples)
    assert "main" in axes
    fig.clf()


def test_acf_spectrum_consistency_builder_outputs_two_axes() -> None:
    builder = ACFSpectrumConsistencyFigureBuilder()
    lag = np.array([-1.0, 0.0, 1.0])
    acf = np.array([0.1, 1.0, 0.1])
    freq = np.array([-1.0, 0.0, 1.0])
    power = np.array([1.0, 2.0, 1.0])
    fig, axes = builder.build(lag, acf, freq, power)
    assert "acf" in axes and "spectrum" in axes
    fig.clf()


def test_filtered_waveform_builder_outputs_main_axis() -> None:
    builder = FilteredWaveformFigureBuilder()
    t = np.linspace(0.0, 1.0, 128)
    original = np.sin(2 * np.pi * 3.0 * t)
    reconstructed = 0.9 * original
    fig, axes = builder.build(t, original, reconstructed)
    assert "main" in axes
    fig.clf()


def test_spur_survey_builder_outputs_main_axis() -> None:
    builder = SpurSurveyFigureBuilder()
    freq = np.linspace(0.0, 10.0, 256)
    power = np.exp(-0.5 * ((freq - 3.0) / 0.3) ** 2)
    fig, axes = builder.build(freq, power, expected_lines_hz=np.array([3.0]))
    assert "main" in axes
    fig.clf()


def test_reverted_dsb_comparison_builder_outputs_main_axis() -> None:
    builder = RevertedDSBComparisonFigureBuilder()
    freq = np.linspace(-2.0, 2.0, 256)
    ssb = np.exp(-0.5 * ((freq - 0.5) / 0.1) ** 2)
    dsb = np.exp(-0.5 * ((freq - 0.5) / 0.1) ** 2) + np.exp(-0.5 * ((freq + 0.5) / 0.1) ** 2)
    fig, axes = builder.build(freq, ssb, dsb)
    assert "main" in axes
    fig.clf()


def test_r820t_comparison_builder_outputs_main_axis() -> None:
    builder = R820TComparisonFigureBuilder()
    freq = np.linspace(-2.0, 2.0, 256)
    external = np.exp(-0.5 * ((freq - 0.7) / 0.12) ** 2)
    internal = 0.9 * external
    fig, axes = builder.build(freq, external, internal)
    assert "main" in axes
    fig.clf()
