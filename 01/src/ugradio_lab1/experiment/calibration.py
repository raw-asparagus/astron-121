"""Calibration pre-check figure for §4 (cell 20)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ugradio_lab1.plotting.figure_builders import TimeDomainComparisonFigureBuilder


def plot_calibration_precheck(
    time_s: np.ndarray,
    good_voltage_v: np.ndarray,
    bad_voltage_v: np.ndarray,
    *,
    clip_limits_v: tuple[float, float] | None = None,
    sample_slice: slice | tuple[int, int] | None = slice(0, 100),
) -> tuple[Figure, dict[str, Axes]]:
    """Thin wrapper around TimeDomainComparisonFigureBuilder for F3."""

    builder = TimeDomainComparisonFigureBuilder()
    return builder.build(
        time_s,
        good_voltage_v,
        bad_voltage_v,
        clip_limits_v=clip_limits_v,
        sample_slice=sample_slice,
    )


__all__ = ["plot_calibration_precheck"]
