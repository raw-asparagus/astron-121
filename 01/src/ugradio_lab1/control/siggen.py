"""Signal-generator control wrappers for N9310A over direct USBTMC."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
from pathlib import Path
import time
from typing import Callable, TypeVar

_T = TypeVar("_T")


class SigGenIOError(RuntimeError):
    """Raised when the signal generator cannot be controlled reliably."""


@dataclass(frozen=True)
class SigGenRetryPolicy:
    """Retry and timeout controls for USBTMC operations."""

    timeout_s: float = 10.0
    max_retries: int = 3
    retry_sleep_s: float = 0.25
    settle_time_s: float = 1.0
    query_poll_s: float = 0.02


@dataclass
class N9310AUSBTMC:
    """Minimal N9310A USBTMC interface."""

    device_path: str | Path = "/dev/usbtmc0"
    retry: SigGenRetryPolicy = field(default_factory=SigGenRetryPolicy)

    def __post_init__(self) -> None:
        self.device_path = Path(self.device_path)

    def set_freq_mhz(self, mhz: float) -> None:
        """Set CW frequency in MHz."""

        if mhz <= 0.0:
            raise ValueError("mhz must be positive.")
        self._with_retries(
            lambda: self._write_once(f":FREQuency:CW {float(mhz):.9f} MHz"),
            description=f"write frequency {mhz!r} MHz",
        )
        time.sleep(self.retry.settle_time_s)

    def get_freq(self) -> str:
        """Query CW frequency."""

        return self._query_with_retries(":FREQuency:CW?")

    def set_ampl_dbm(self, power_dbm: float) -> None:
        """Set CW output power in dBm."""

        self._with_retries(
            lambda: self._write_once(f":AMPLitude:CW {float(power_dbm):.3f} dBm"),
            description=f"write amplitude {power_dbm!r} dBm",
        )
        time.sleep(self.retry.settle_time_s)

    def get_ampl(self) -> str:
        """Query CW output power."""

        return self._query_with_retries(":AMPLitude:CW?")

    def rf_on(self) -> None:
        """Enable RF output."""

        self._with_retries(
            lambda: self._write_once(":RFOutput:STATe ON"),
            description="write RF output ON",
        )
        time.sleep(self.retry.settle_time_s)

    def rf_off(self) -> None:
        """Disable RF output."""

        self._with_retries(
            lambda: self._write_once(":RFOutput:STATe OFF"),
            description="write RF output OFF",
        )
        time.sleep(self.retry.settle_time_s)

    def rf_state(self) -> str:
        """Query RF output state."""

        return self._query_with_retries(":RFOutput:STATe?")

    def _query_with_retries(self, command: str) -> str:
        def _operation() -> str:
            self._write_once(command)
            deadline = time.monotonic() + self.retry.timeout_s
            while True:
                try:
                    response = self._read_once()
                except Exception as error:
                    if not _is_transient_read_timeout(error):
                        raise
                    response = ""
                if response:
                    return response
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for response to {command!r}.")
                time.sleep(self.retry.query_poll_s)

        return self._with_retries(_operation, description=f"query {command!r}")

    def _write_once(self, command: str) -> None:
        with self.device_path.open("w", buffering=1) as device:
            device.write(command.rstrip("\n") + "\n")
            device.flush()

    def _read_once(self) -> str:
        with self.device_path.open("r") as device:
            return device.read().strip()

    def _with_retries(self, operation: Callable[[], _T], *, description: str) -> _T:
        last_error: Exception | None = None
        for attempt in range(1, self.retry.max_retries + 1):
            try:
                return operation()
            except Exception as error:  # pragma: no cover - exercised via caller behavior
                last_error = error
                if attempt >= self.retry.max_retries:
                    break
                time.sleep(self.retry.retry_sleep_s)
        raise SigGenIOError(
            f"Signal-generator {description} failed after {self.retry.max_retries} attempts."
        ) from last_error


def _is_transient_read_timeout(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, OSError) and getattr(error, "errno", None) == errno.ETIMEDOUT:
        return True
    return False


__all__ = [
    "N9310AUSBTMC",
    "SigGenIOError",
    "SigGenRetryPolicy",
]
