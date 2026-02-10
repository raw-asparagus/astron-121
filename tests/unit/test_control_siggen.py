"""Unit tests for control.siggen timeout handling."""

from __future__ import annotations

import pytest

from ugradio_lab1.control.siggen import N9310AUSBTMC, SigGenIOError, SigGenRetryPolicy


class _FakeSigGen(N9310AUSBTMC):
    def __init__(self, responses, *, timeout_s: float = 1.0):
        super().__init__(
            device_path="/tmp/fake-usbtmc0",
            retry=SigGenRetryPolicy(
                timeout_s=timeout_s,
                max_retries=1,
                retry_sleep_s=0.0,
                settle_time_s=0.0,
                query_poll_s=0.0,
            ),
        )
        self._responses = list(responses)
        self.writes: list[str] = []

    def _write_once(self, command: str) -> None:
        self.writes.append(command)

    def _read_once(self) -> str:
        if len(self._responses) == 0:
            raise TimeoutError("[Errno 110] Connection timed out")
        current = self._responses.pop(0)
        if isinstance(current, Exception):
            raise current
        return str(current)


def test_get_freq_tolerates_transient_timeouts_until_response() -> None:
    siggen = _FakeSigGen([TimeoutError(), TimeoutError(), "Agilent,N9310A,123,1.0"], timeout_s=2.0)
    response = siggen.get_freq()
    assert response == "Agilent,N9310A,123,1.0"
    assert siggen.writes == [":FREQuency:CW?"]


def test_get_freq_raises_siggenioerror_if_timeout_budget_exhausted() -> None:
    siggen = _FakeSigGen([TimeoutError(), TimeoutError(), TimeoutError()], timeout_s=0.0)
    with pytest.raises(SigGenIOError, match="failed after 1 attempts"):
        siggen.get_freq()
