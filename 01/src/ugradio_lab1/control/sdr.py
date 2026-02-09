"""SDR control wrappers and acquisition helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import signal
import threading
import time
from typing import Any, Callable

import numpy as np

from ugradio_lab1.utils.validation import as_2d_array


class SDRCaptureError(RuntimeError):
    """Raised when SDR acquisition fails after retries."""


class SDRCaptureTimeoutError(TimeoutError):
    """Raised when one SDR capture call exceeds configured timeout."""


@dataclass(frozen=True)
class SDRCaptureConfig:
    """Capture settings for one direct-sampling acquisition."""

    sample_rate_hz: float
    device_index: int = 0
    direct: bool = True
    gain: float = 0.0
    fir_coeffs: np.ndarray | None = None
    nsamples: int = 2048
    nblocks: int = 11
    stale_blocks: int = 1
    timeout_s: float = 10.0
    max_retries: int = 3
    retry_sleep_s: float = 0.25


@dataclass(frozen=True)
class ADCSummary:
    """ADC-level summary metrics for a capture."""

    mean_block_rms: float
    adc_max: int
    adc_min: int
    is_clipped: bool
    passes_guard: bool


@dataclass(frozen=True)
class SDRCaptureResult:
    """One capture result: de-staled blocks + ADC summary + sample-rate logs."""

    blocks: np.ndarray
    summary: ADCSummary
    requested_sample_rate_hz: float
    actual_sample_rate_hz: float


def alias_hack_fir_coeffs() -> np.ndarray:
    """Return FIR coefficients from the lab manual aliasing footnote."""

    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2047], dtype=int)


def acquire_sdr_capture(
    config: SDRCaptureConfig,
    *,
    sdr_factory: Callable[..., Any] | None = None,
) -> SDRCaptureResult:
    """Capture blocks from SDR with retry and timeout.

    Notes
    -----
    ``nblocks`` is requested from the SDR directly. The first ``stale_blocks``
    blocks are discarded before metrics are computed and before data is returned.
    For E1 this is ``nblocks=11`` and ``stale_blocks=1``, leaving 10 clean blocks.
    """

    if config.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if config.nsamples <= 0:
        raise ValueError("nsamples must be positive.")
    if config.nblocks <= 0:
        raise ValueError("nblocks must be positive.")
    if config.stale_blocks < 0:
        raise ValueError("stale_blocks cannot be negative.")
    if config.nblocks <= config.stale_blocks:
        raise ValueError("nblocks must be larger than stale_blocks.")
    if config.max_retries < 1:
        raise ValueError("max_retries must be >= 1.")

    constructor = _resolve_sdr_factory(sdr_factory)
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            return _capture_once(config, constructor=constructor)
        except Exception as error:
            last_error = error
            if attempt >= config.max_retries:
                break
            time.sleep(config.retry_sleep_s)
    raise SDRCaptureError(
        f"SDR capture failed after {config.max_retries} attempts."
    ) from last_error


def summarize_adc_blocks(blocks: np.ndarray) -> ADCSummary:
    """Compute ADC guard metrics from a ``(n_blocks, n_samples)`` array."""

    validated = as_2d_array(np.asarray(blocks), "blocks", dtype=float)
    per_block_rms = np.sqrt(np.mean(np.square(validated), axis=1))
    mean_block_rms = float(np.mean(per_block_rms))

    adc_max = int(np.max(validated))
    adc_min = int(np.min(validated))
    is_clipped = bool(adc_max >= 127 or adc_min <= -128)
    passes_guard = bool(mean_block_rms >= 10.0 and not is_clipped)
    return ADCSummary(
        mean_block_rms=mean_block_rms,
        adc_max=adc_max,
        adc_min=adc_min,
        is_clipped=is_clipped,
        passes_guard=passes_guard,
    )


def _capture_once(
    config: SDRCaptureConfig,
    *,
    constructor: Callable[..., Any],
) -> SDRCaptureResult:
    radio = constructor(
        device_index=config.device_index,
        direct=config.direct,
        center_freq=0.0,
        sample_rate=float(config.sample_rate_hz),
        gain=float(config.gain),
        fir_coeffs=config.fir_coeffs,
    )
    try:
        actual_sample_rate_hz = _read_actual_sample_rate(radio, fallback=config.sample_rate_hz)
        with _timeout_guard(config.timeout_s):
            raw_blocks = radio.capture_data(nsamples=config.nsamples, nblocks=config.nblocks)
    finally:
        close = getattr(radio, "close", None)
        if callable(close):
            close()

    blocks = np.asarray(raw_blocks)
    if blocks.ndim != 2:
        raise ValueError(
            "Expected 2D direct-sampling data with shape (n_blocks, n_samples); "
            f"received shape {blocks.shape!r}."
        )

    validated = as_2d_array(blocks, "raw_blocks")
    clean_blocks = np.asarray(validated[config.stale_blocks:, :], dtype=np.int8)
    summary = summarize_adc_blocks(clean_blocks)
    return SDRCaptureResult(
        blocks=clean_blocks,
        summary=summary,
        requested_sample_rate_hz=float(config.sample_rate_hz),
        actual_sample_rate_hz=float(actual_sample_rate_hz),
    )


def _resolve_sdr_factory(sdr_factory: Callable[..., Any] | None) -> Callable[..., Any]:
    if sdr_factory is not None:
        return sdr_factory
    try:
        import ugradio.sdr as ugradio_sdr
    except Exception as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "Unable to import ugradio.sdr; provide sdr_factory explicitly or install ugradio."
        ) from error
    return ugradio_sdr.SDR


def _read_actual_sample_rate(radio: Any, *, fallback: float) -> float:
    getter = getattr(radio, "get_sample_rate", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:
            return float(fallback)
    return float(fallback)


@contextmanager
def _timeout_guard(timeout_s: float):
    if timeout_s <= 0.0:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        # SIGALRM only works on the main thread; callers still get retries.
        yield
        return

    def _handler(_signum, _frame):
        raise SDRCaptureTimeoutError(f"Capture exceeded {timeout_s:.3f} s timeout.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


__all__ = [
    "ADCSummary",
    "SDRCaptureConfig",
    "SDRCaptureError",
    "SDRCaptureResult",
    "SDRCaptureTimeoutError",
    "acquire_sdr_capture",
    "alias_hack_fir_coeffs",
    "summarize_adc_blocks",
]
