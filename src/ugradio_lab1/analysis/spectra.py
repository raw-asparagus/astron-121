from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal
# import ugradio.dft as dft

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

    Parameters
    ----------
    voltage : array_like
        1-D array of time-domain voltage samples (real or complex).
    sample_rate_hz : float
        Sampling rate in Hz.  Must be positive.
    window : str, array_like, or None
        Window function applied before the FFT.  Accepts ``"boxcar"``,
        ``"hann"``, ``"hamming"``, ``"blackman"``, a custom 1-D array,
        or ``None`` (equivalent to boxcar).
    detrend : bool
        If True, subtract the mean before windowing.
    scaling : ``"raw"`` or ``"amplitude"``
        ``"amplitude"`` divides by ``N * coherent_gain`` so a bin-centered
        sinusoid of amplitude *A* produces a peak of magnitude *A*.
        ``"raw"`` returns the unscaled FFT output.
    center : bool
        If True, apply ``fftshift`` so the DC bin is in the center.
    fft_backend : ``"numpy"`` or ``"ugradio"``
        Which FFT implementation to use.

    Returns
    -------
    frequency_hz : ndarray
        Frequency axis in Hz (length ``N``).
    spectrum : ndarray
        Complex voltage spectrum (length ``N``).

    Raises
    ------
    ValueError
        If *sample_rate_hz* is non-positive, *scaling* is unrecognised,
        or the window coherent gain is zero under amplitude scaling.
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

    Parameters
    ----------
    voltage : array_like
        1-D array of time-domain voltage samples (real or complex).
    sample_rate_hz : float
        Sampling rate in Hz.
    window : str, array_like, or None
        Window function (see :func:`voltage_spectrum`).
    detrend : bool
        Subtract the mean before windowing.
    scaling : ``"power"`` or ``"density"``
        ``"power"`` normalises by ``N^2 * power_gain`` (units V^2 per bin).
        ``"density"`` normalises by ``fs * N * power_gain`` (units V^2/Hz).
    center : bool
        Apply ``fftshift`` so DC is centered.
    fft_backend : ``"numpy"`` or ``"ugradio"``
        FFT implementation.

    Returns
    -------
    frequency_hz : ndarray
        Frequency axis in Hz.
    power : ndarray
        Real-valued power spectrum.
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
        power = magnitude_squared / (n ** 2 * power_gain)
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
    """Compute block-averaged power-spectrum statistics from SDR capture data.

    Each row of ``voltage_blocks`` is treated as an independent capture block.
    The per-block power spectra are stacked, then the mean and standard
    deviation are computed across blocks.

    Parameters
    ----------
    voltage_blocks : array_like
        2-D array of shape ``(n_blocks, n_samples)``.
    sample_rate_hz : float
        Sampling rate in Hz.
    window : str, array_like, or None
        Window function applied to each block (see :func:`voltage_spectrum`).
    detrend : bool
        Subtract the per-block mean before windowing.
    scaling : ``"power"`` or ``"density"``
        Power normalisation (see :func:`power_spectrum`).
    center : bool
        Apply ``fftshift`` so DC is centered.
    fft_backend : ``"numpy"`` or ``"ugradio"``
        FFT implementation.

    Returns
    -------
    AveragedPowerSpectrum
        Dataclass with fields ``frequency_hz``, ``mean``, ``std``,
        and ``num_blocks``.

    Raises
    ------
    ValueError
        If *voltage_blocks* has fewer than 1 block or fewer than 1 sample
        per block.
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
    """Compute full-lag, biased-normalised autocorrelation via ``scipy.signal.correlate``.

    The biased estimator divides by ``N`` (not ``N - |lag|``), which keeps the
    result non-negative-definite and is consistent with Wiener–Khinchin via the
    power spectrum.

    Parameters
    ----------
    voltage : array_like
        1-D voltage samples (real or complex).
    sample_rate_hz : float
        Sampling rate in Hz (used to convert lag indices to seconds).
    detrend : bool
        Subtract the mean before correlating.

    Returns
    -------
    lag_seconds : ndarray
        Lag axis in seconds, from ``-(N-1)/fs`` to ``+(N-1)/fs``.
    correlation : ndarray
        Autocorrelation values (real if input is real).
    """

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

    Applies the Wiener–Khinchin theorem: IFFT of the power spectrum yields the
    circular autocorrelation.  The input should be generated by
    :func:`power_spectrum` with ``scaling="power"``.

    Parameters
    ----------
    power_spectrum_v2 : array_like
        1-D power spectrum in V^2 (length ``N``).
    sample_rate_hz : float
        Sampling rate in Hz.
    centered : bool
        Whether the input spectrum is ``fftshift``-ed (DC in center).
    normalize : ``"none"``, ``"biased"``, or ``"coeff"``
        ``"none"`` returns the raw IFFT.
        ``"biased"`` divides by ``N``.
        ``"coeff"`` normalises so the zero-lag value is 1.
    fft_backend : ``"numpy"`` or ``"ugradio"``
        FFT implementation.

    Returns
    -------
    lag_seconds : ndarray
        Lag axis in seconds.
    correlation : ndarray
        Circular autocorrelation values.

    Raises
    ------
    ValueError
        If *normalize* is ``"coeff"`` and the zero-lag power is zero.
    """

    power = as_1d_array(power_spectrum_v2, "power_spectrum_v2", dtype=float)
    fs = _validate_sample_rate(sample_rate_hz)

    n = power.size
    power_scaled = power * (n ** 2)

    raw_correlation = _ifft_with_backend(
        power_scaled, sample_rate_hz=fs, centered=centered, fft_backend=fft_backend,
    )

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
    """Cast to float and raise if non-positive."""
    rate = float(sample_rate_hz)
    if rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    return rate


def _resolve_fft_backend(fft_backend: str) -> FFTBackend:
    """Validate and normalise the FFT backend string; guard ugradio availability."""
    backend = fft_backend.lower()
    if backend not in {"numpy", "ugradio"}:
        raise ValueError("fft_backend must be 'numpy' or 'ugradio'.")
    if backend == "ugradio" and dft is None:
        raise ImportError(
            "fft_backend='ugradio' requires the 'ugradio' package to be installed."
        )
    return backend


def _fft_with_backend(
        samples: np.ndarray,
        *,
        sample_rate_hz: float,
        fft_backend: FFTBackend,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the forward FFT, returning ``(frequency_hz, spectrum)`` in standard order."""
    n = samples.size
    backend = _resolve_fft_backend(fft_backend)
    if backend == "numpy":
        frequency_hz = np.fft.fftfreq(n, d=1.0 / sample_rate_hz)
        spectrum = np.fft.fft(samples)
        return frequency_hz, spectrum

    frequency_centered, spectrum_centered = dft.dft(samples, vsamp=sample_rate_hz)
    frequency_hz = np.fft.ifftshift(np.asarray(frequency_centered, dtype=float))
    spectrum = np.fft.ifftshift(np.asarray(spectrum_centered, dtype=np.complex128))
    return frequency_hz, spectrum


def _ifft_with_backend(
        spectrum: np.ndarray,
        *,
        sample_rate_hz: float,
        centered: bool,
        fft_backend: FFTBackend,
) -> np.ndarray:
    """Compute the IFFT, returning the result in standard (non-shifted) order."""
    backend = _resolve_fft_backend(fft_backend)
    if backend == "numpy":
        unshifted = np.fft.ifftshift(spectrum) if centered else spectrum
        return np.fft.ifft(unshifted)

    centered_spectrum = spectrum if centered else np.fft.fftshift(spectrum)
    _, shifted_result = dft.idft(centered_spectrum, vsamp=sample_rate_hz)
    return np.fft.ifftshift(np.asarray(shifted_result, dtype=np.complex128))


def _resolve_window(window: WindowSpec, n_samples: int) -> tuple[np.ndarray, float, float]:
    """Return ``(window_values, coherent_gain, power_gain)`` for the given window spec."""
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
    power_gain = float(np.mean(values ** 2))
    if power_gain <= 0.0:
        raise ValueError("window power gain must be positive.")
    return values, coherent_gain, power_gain


def _maybe_real(values: np.ndarray, *, source: np.ndarray) -> np.ndarray:
    """Strip negligible imaginary parts when the original input was real-valued."""
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
