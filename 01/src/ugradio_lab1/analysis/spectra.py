"""Voltage/power spectrum and autocorrelation analysis routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

from ugradio_lab1.utils.validation import as_1d_array, as_2d_array

WindowSpec = Literal["boxcar", "hann", "hamming", "blackman"] | np.ndarray | None
SpectrumScaling = Literal["raw", "amplitude"]
PowerScaling = Literal["power", "density"]
CircularCorrelationNormalization = Literal["none", "biased", "coeff"]
FFTBackend = Literal["numpy", "ugradio"]


@dataclass(frozen=True)
class AveragedPowerSpectrum:
    """Container for block-averaged power-spectrum statistics."""

    frequency_hz: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    num_blocks: int


def voltage_spectrum(
    voltage: np.ndarray,
    *,
    sample_rate_hz: float,
    window: WindowSpec = None,
    detrend: bool = False,
    scaling: SpectrumScaling = "amplitude",
    center: bool = True,
    fft_backend: FFTBackend = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a two-sided complex voltage spectrum.

    By default, the returned spectrum is amplitude-scaled so a bin-centered
    complex sinusoid with amplitude ``A`` has peak magnitude ``A``.
    """

    samples = as_1d_array(voltage, "voltage")
    fs = _validate_sample_rate(sample_rate_hz)

    prepared = samples.astype(np.complex128, copy=True)
    if detrend:
        prepared -= np.mean(prepared)

    window_values, coherent_gain, _ = _resolve_window(window, prepared.size)
    prepared *= window_values

    frequency_hz, spectrum = _fft_with_backend(prepared, sample_rate_hz=fs, fft_backend=fft_backend)
    if scaling == "raw":
        pass
    elif scaling == "amplitude":
        if np.isclose(coherent_gain, 0.0):
            raise ValueError("window coherent gain is zero; amplitude scaling is undefined.")
        spectrum = spectrum / (prepared.size * coherent_gain)
    else:
        raise ValueError(f"Unsupported scaling: {scaling}")

    if center:
        frequency_hz = np.fft.fftshift(frequency_hz)
        spectrum = np.fft.fftshift(spectrum)
    return frequency_hz, spectrum


def power_spectrum(
    voltage: np.ndarray,
    *,
    sample_rate_hz: float,
    window: WindowSpec = None,
    detrend: bool = False,
    scaling: PowerScaling = "power",
    center: bool = True,
    fft_backend: FFTBackend = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a two-sided power spectrum.

    ``scaling="power"`` returns per-bin power in V^2.
    ``scaling="density"`` returns PSD in V^2/Hz.
    """

    samples = as_1d_array(voltage, "voltage")
    fs = _validate_sample_rate(sample_rate_hz)

    prepared = samples.astype(np.complex128, copy=True)
    if detrend:
        prepared -= np.mean(prepared)

    window_values, _, power_gain = _resolve_window(window, prepared.size)
    prepared *= window_values

    frequency_hz, raw_spectrum = _fft_with_backend(
        prepared, sample_rate_hz=fs, fft_backend=fft_backend
    )
    magnitude_squared = np.abs(raw_spectrum) ** 2
    n = prepared.size
    if scaling == "power":
        power = magnitude_squared / (n**2 * power_gain)
    elif scaling == "density":
        power = magnitude_squared / (fs * n * power_gain)
    else:
        raise ValueError(f"Unsupported scaling: {scaling}")

    if center:
        frequency_hz = np.fft.fftshift(frequency_hz)
        power = np.fft.fftshift(power)
    return frequency_hz, power.real


def average_power_spectrum(
    voltage_blocks: np.ndarray,
    *,
    sample_rate_hz: float,
    window: WindowSpec = None,
    detrend: bool = True,
    scaling: PowerScaling = "power",
    center: bool = True,
    fft_backend: FFTBackend = "numpy",
) -> AveragedPowerSpectrum:
    """Compute average power-spectrum statistics from pre-blocked SDR data.

    ``voltage_blocks`` must be a 2D array of shape ``(n_blocks, n_samples)``.
    """

    blocks = as_2d_array(voltage_blocks, "voltage_blocks")
    if blocks.shape[0] < 1:
        raise ValueError("voltage_blocks must include at least one block.")
    if blocks.shape[1] < 1:
        raise ValueError("Each block in voltage_blocks must contain at least one sample.")

    fs = _validate_sample_rate(sample_rate_hz)

    power_blocks: list[np.ndarray] = []
    frequency_hz: np.ndarray | None = None

    for block in blocks:
        current_frequency, current_power = power_spectrum(
            block,
            sample_rate_hz=fs,
            window=window,
            detrend=detrend,
            scaling=scaling,
            center=center,
            fft_backend=fft_backend,
        )
        if frequency_hz is None:
            frequency_hz = current_frequency
        power_blocks.append(current_power)

    stacked = np.vstack(power_blocks)
    ddof = 1 if stacked.shape[0] > 1 else 0
    return AveragedPowerSpectrum(
        frequency_hz=frequency_hz if frequency_hz is not None else np.array([], dtype=float),
        mean=np.mean(stacked, axis=0),
        std=np.std(stacked, axis=0, ddof=ddof),
        num_blocks=stacked.shape[0],
    )


def autocorrelation(
    voltage: np.ndarray,
    *,
    sample_rate_hz: float,
    detrend: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full-lag, biased-normalized autocorrelation from voltage samples."""

    samples = as_1d_array(voltage, "voltage")
    fs = _validate_sample_rate(sample_rate_hz)

    prepared = samples.astype(np.complex128, copy=True)
    if detrend:
        prepared -= np.mean(prepared)

    correlation = signal.correlate(prepared, prepared, mode="full", method="fft")
    lag_samples = signal.correlation_lags(prepared.size, prepared.size, mode="full")
    correlation = correlation / prepared.size
    correlation = _maybe_real(correlation, source=samples)
    lag_seconds = lag_samples / fs
    return lag_seconds, correlation


def autocorrelation_from_power_spectrum(
    power_spectrum_v2: np.ndarray,
    *,
    sample_rate_hz: float,
    centered: bool = True,
    normalize: CircularCorrelationNormalization = "biased",
    fft_backend: FFTBackend = "numpy",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute circular autocorrelation from a two-sided power spectrum.

    This function assumes ``power_spectrum_v2`` was generated with
    :func:`power_spectrum` and ``scaling="power"``.
    """

    power = as_1d_array(power_spectrum_v2, "power_spectrum_v2", dtype=float)
    fs = _validate_sample_rate(sample_rate_hz)

    n = power.size
    backend = _resolve_fft_backend(fft_backend)
    power_scaled = power * (n**2)

    if backend == "numpy":
        power_unshifted = np.fft.ifftshift(power_scaled) if centered else power_scaled
        raw_correlation = np.fft.ifft(power_unshifted)
    else:
        dft_module = _import_ugradio_dft()
        centered_spectrum = power_scaled if centered else np.fft.fftshift(power_scaled)
        _, shifted_correlation = dft_module.idft(centered_spectrum, vsamp=fs)
        raw_correlation = np.fft.ifftshift(np.asarray(shifted_correlation, dtype=np.complex128))

    if normalize == "none":
        correlation = raw_correlation
    elif normalize == "biased":
        correlation = raw_correlation / n
    elif normalize == "coeff":
        zero_lag = raw_correlation[0]
        if np.isclose(np.abs(zero_lag), 0.0):
            raise ValueError("Cannot apply coeff normalization when zero-lag power is zero.")
        correlation = raw_correlation / zero_lag
    else:
        raise ValueError(f"Unsupported normalization: {normalize}")

    lag_samples = np.arange(n, dtype=int)
    if centered:
        lag_samples = np.fft.fftshift(np.arange(-n // 2, n - n // 2, dtype=int))
        correlation = np.fft.fftshift(correlation)

    correlation = _maybe_real(correlation, source=power)
    lag_seconds = lag_samples / fs
    return lag_seconds, correlation


def _validate_sample_rate(sample_rate_hz: float) -> float:
    rate = float(sample_rate_hz)
    if rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    return rate


def _resolve_fft_backend(fft_backend: str) -> FFTBackend:
    backend = fft_backend.lower()
    if backend not in {"numpy", "ugradio"}:
        raise ValueError("fft_backend must be 'numpy' or 'ugradio'.")
    return backend  # type: ignore[return-value]


def _import_ugradio_dft():
    try:
        import ugradio.dft as dft_module
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "fft_backend='ugradio' requires the 'ugradio' package to be installed and importable."
        ) from exc
    return dft_module


def _fft_with_backend(
    samples: np.ndarray,
    *,
    sample_rate_hz: float,
    fft_backend: FFTBackend,
) -> tuple[np.ndarray, np.ndarray]:
    n = samples.size
    backend = _resolve_fft_backend(fft_backend)
    if backend == "numpy":
        frequency_hz = np.fft.fftfreq(n, d=1.0 / sample_rate_hz)
        spectrum = np.fft.fft(samples)
        return frequency_hz, spectrum

    dft_module = _import_ugradio_dft()
    frequency_centered, spectrum_centered = dft_module.dft(samples, vsamp=sample_rate_hz)
    frequency_hz = np.fft.ifftshift(np.asarray(frequency_centered, dtype=float))
    spectrum = np.fft.ifftshift(np.asarray(spectrum_centered, dtype=np.complex128))
    return frequency_hz, spectrum


def _resolve_window(window: WindowSpec, n_samples: int) -> tuple[np.ndarray, float, float]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if window is None:
        values = np.ones(n_samples, dtype=float)
    elif isinstance(window, str):
        if window == "boxcar":
            return np.ones(n_samples, dtype=float), 1.0, 1.0
        available = {
            "hann": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
        }
        if window not in available:
            raise ValueError(
                f"Unsupported window {window!r}. "
                "Use one of: boxcar, hann, hamming, blackman."
            )
        values = available[window](n_samples)
    else:
        values = as_1d_array(window, "window", dtype=float)
        if values.size != n_samples:
            raise ValueError("window must have the same length as voltage.")

    if not np.all(np.isfinite(values)):
        raise ValueError("window contains non-finite values.")

    coherent_gain = float(np.mean(values))
    power_gain = float(np.mean(values**2))
    if power_gain <= 0.0:
        raise ValueError("window power gain must be positive.")
    return values, coherent_gain, power_gain


def _maybe_real(values: np.ndarray, *, source: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(source):
        return values
    if np.allclose(np.imag(values), 0.0, rtol=1e-12, atol=1e-12):
        return np.real(values)
    return values


__all__ = [
    "AveragedPowerSpectrum",
    "autocorrelation",
    "autocorrelation_from_power_spectrum",
    "average_power_spectrum",
    "power_spectrum",
    "voltage_spectrum",
]
