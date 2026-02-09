"""Signal-generator control wrappers for N9310A over direct USBTMC."""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
from pathlib import Path
import re
import time
from typing import Callable, TypeVar

_T = TypeVar("_T")
_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


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
    """Minimal N9310A USBTMC interface.

    The class intentionally keeps the command surface small and explicit so
    acquisition logic can verify every set/query transaction.
    """

    device_path: str | Path = "/dev/usbtmc0"
    retry: SigGenRetryPolicy = field(default_factory=SigGenRetryPolicy)

    def __post_init__(self) -> None:
        self.device_path = Path(self.device_path)

    def identify(self) -> str:
        """Return ``*IDN?`` response."""

        return self.query("*IDN?")

    def validate_identity(self, expected_substring: str = "N9310A") -> str:
        """Query ``*IDN?`` and verify model substring."""

        response = self.identify()
        if expected_substring.upper() not in response.upper():
            raise SigGenIOError(
                f"Unexpected signal generator identity: {response!r}. "
                f"Expected to contain {expected_substring!r}."
            )
        return response

    def set_frequency_hz(self, frequency_hz: float) -> None:
        """Set CW frequency in Hz."""

        if frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be positive.")
        self.write(f":FREQuency:CW {float(frequency_hz):.9f} Hz")
        time.sleep(self.retry.settle_time_s)

    def get_frequency_hz(self) -> float:
        """Query CW frequency in Hz."""

        response = self.query(":FREQuency:CW?")
        return _parse_first_float(response, name="frequency")

    def set_frequency_hz_verified(
        self,
        frequency_hz: float,
        *,
        tolerance_hz: float = 1.0,
    ) -> float:
        """Set frequency and verify by query-back."""

        self.set_frequency_hz(frequency_hz)
        measured = self.get_frequency_hz()
        if abs(measured - float(frequency_hz)) > float(tolerance_hz):
            raise SigGenIOError(
                "Frequency query-back mismatch: "
                f"requested={frequency_hz:.6f} Hz measured={measured:.6f} Hz."
            )
        return measured

    def set_power_dbm(self, power_dbm: float) -> None:
        """Set CW output power in dBm."""

        self.write(f":AMPLitude:CW {float(power_dbm):.3f} dBm")
        time.sleep(self.retry.settle_time_s)

    def get_power_dbm(self) -> float:
        """Query CW output power in dBm."""

        response = self.query(":AMPLitude:CW?")
        return _parse_first_float(response, name="power")

    def set_power_dbm_verified(
        self,
        power_dbm: float,
        *,
        tolerance_dbm: float = 0.05,
    ) -> float:
        """Set power and verify by query-back."""

        self.set_power_dbm(power_dbm)
        measured = self.get_power_dbm()
        if abs(measured - float(power_dbm)) > float(tolerance_dbm):
            raise SigGenIOError(
                "Power query-back mismatch: "
                f"requested={power_dbm:.3f} dBm measured={measured:.3f} dBm."
            )
        return measured

    def set_rf_output(self, enabled: bool) -> None:
        """Enable/disable RF output."""

        self.write(":RFOutput:STATe ON" if enabled else ":RFOutput:STATe OFF")
        time.sleep(self.retry.settle_time_s)

    def get_rf_output(self) -> bool:
        """Query RF output state."""

        response = self.query(":RFOutput:STATe?").strip().upper()
        if response.startswith(("1", "ON")):
            return True
        if response.startswith(("0", "OFF")):
            return False
        raise SigGenIOError(f"Unexpected RF output response: {response!r}.")

    def set_rf_output_verified(self, enabled: bool) -> bool:
        """Set RF output and verify by query-back."""

        self.set_rf_output(enabled)
        measured = self.get_rf_output()
        if measured != bool(enabled):
            raise SigGenIOError(
                f"RF output query-back mismatch: requested={enabled!r} measured={measured!r}."
            )
        return measured

    def write(self, command: str) -> None:
        """Write one command with retries."""

        self._with_retries(
            lambda: self._write_once(command),
            description=f"write {command!r}",
        )

    def query(self, command: str) -> str:
        """Write query command and wait for non-empty response."""

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


def _parse_first_float(response: str, *, name: str) -> float:
    match = _FLOAT_PATTERN.search(response)
    if match is None:
        raise SigGenIOError(f"Unable to parse {name} value from response: {response!r}.")
    return float(match.group(0))


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
