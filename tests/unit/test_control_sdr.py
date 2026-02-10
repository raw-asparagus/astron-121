"""Unit tests for control.sdr helpers."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.control.sdr import SDRCaptureConfig, acquire_sdr_capture, summarize_adc_blocks


class _FakeRadio:
    def __init__(self, *, sample_rate: float, **_kwargs) -> None:
        self._sample_rate = sample_rate

    def capture_data(self, *, nsamples: int, nblocks: int):
        values = np.arange(nblocks * nsamples, dtype=np.int16).reshape(nblocks, nsamples)
        values = np.mod(values, 50) - 25
        return values.astype(np.int8)

    def get_sample_rate(self) -> float:
        return self._sample_rate - 123.0

    def close(self) -> None:
        return None


def _fake_factory(**kwargs):
    return _FakeRadio(**kwargs)


def test_acquire_sdr_capture_drops_stale_first_block() -> None:
    config = SDRCaptureConfig(
        sample_rate_hz=1.0e6,
        nsamples=16,
        nblocks=6,
        stale_blocks=1,
    )
    result = acquire_sdr_capture(config, sdr_factory=_fake_factory)
    assert result.blocks.shape == (5, 16)
    assert result.requested_sample_rate_hz == 1.0e6
    assert result.actual_sample_rate_hz == 1.0e6 - 123.0


def test_summarize_adc_blocks_reports_guard_metrics() -> None:
    blocks = np.array([[10, -10, 10, -10], [5, -5, 5, -5]], dtype=np.int8)
    summary = summarize_adc_blocks(blocks)
    assert summary.adc_max == 10
    assert summary.adc_min == -10
    assert summary.is_clipped is False
    assert summary.passes_guard is False  # mean RMS < 10
