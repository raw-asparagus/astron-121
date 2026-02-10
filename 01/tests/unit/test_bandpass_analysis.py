"""Unit tests for analysis.bandpass."""

from __future__ import annotations

import numpy as np

from ugradio_lab1.analysis.experiments import bandpass_curve


def test_bandpass_curve_gain_reference_default() -> None:
    freq = np.array([1.0, 2.0, 3.0])
    amp = np.array([2.0, 1.0, 0.5])

    curve = bandpass_curve(freq, amp)

    assert np.isclose(curve["gain_linear"].max(), 1.0)
    assert np.isclose(curve["gain_db"].iloc[0], 0.0)
