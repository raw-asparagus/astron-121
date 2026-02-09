"""Planning helpers for E4 leakage and resolution physical captures."""

from __future__ import annotations


def leakage_tone_from_center(
    *,
    sample_rate_hz: float,
    n_samples: int,
    center_frequency_hz: float,
    bin_offset: float,
    bin_index: int | None = None,
) -> tuple[float, int, float]:
    """Return leakage tone frequency and effective bin assignment.

    The tone frequency is computed by:

    ``f_tone = (k + epsilon) * (f_s / N)``

    where:
    - ``k`` is either the provided ``bin_index`` or the nearest bin to
      ``center_frequency_hz`` for each run.
    - ``epsilon`` is ``bin_offset``.
    """

    fs = float(sample_rate_hz)
    n = int(n_samples)
    center = float(center_frequency_hz)
    epsilon = float(bin_offset)
    if fs <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if n < 2:
        raise ValueError("n_samples must be >= 2.")
    if center <= 0.0:
        raise ValueError("center_frequency_hz must be positive.")
    if epsilon < 0.0:
        raise ValueError("bin_offset must be >= 0.")

    delta_f_hz = fs / float(n)
    max_bin = (n // 2) - 1
    if max_bin < 1:
        raise ValueError("n_samples is too small to support positive in-band bin indices.")
    max_bin_for_offset = int((n / 2.0) - epsilon - 1e-12)
    max_bin_for_offset = min(max_bin_for_offset, max_bin)
    if max_bin_for_offset < 1:
        raise ValueError("bin_offset is too large for the requested n_samples.")

    if bin_index is None:
        k = int(round(center / delta_f_hz))
        k = max(1, min(k, max_bin_for_offset))
    else:
        k = int(bin_index)

    if k < 1 or k > max_bin_for_offset:
        raise ValueError(
            f"bin_index must satisfy 1 <= k <= {max_bin_for_offset} for N={n} and epsilon={epsilon}."
        )

    tone_frequency_hz = (float(k) + epsilon) * delta_f_hz
    if tone_frequency_hz <= 0.0:
        raise ValueError("Computed leakage tone must be positive.")
    if tone_frequency_hz >= fs / 2.0:
        raise ValueError("Computed leakage tone reached/exceeded Nyquist; reduce bin_offset or bin_index.")
    return float(tone_frequency_hz), int(k), float(k * delta_f_hz)


def resolution_tones_from_center(
    *,
    center_frequency_hz: float,
    delta_f_hz: float,
) -> tuple[float, float]:
    """Return two-tone frequencies around a center frequency.

    ``f1 = f_center - delta_f/2`` and ``f2 = f_center + delta_f/2``.
    """

    center = float(center_frequency_hz)
    delta = float(delta_f_hz)
    if center <= 0.0:
        raise ValueError("center_frequency_hz must be positive.")
    if delta <= 0.0:
        raise ValueError("delta_f_hz must be positive for resolution runs.")

    lower = center - (delta / 2.0)
    upper = center + (delta / 2.0)
    if lower <= 0.0:
        raise ValueError("center_frequency_hz is too low for the requested delta_f_hz.")
    return float(lower), float(upper)


__all__ = [
    "leakage_tone_from_center",
    "resolution_tones_from_center",
]
