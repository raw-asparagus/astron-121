"""Centralized plotting style configuration."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl


LAB_RC_PARAMS: dict[str, object] = {
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 180,
    "axes.prop_cycle": mpl.cycler(
        color=["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    ),
}


def get_lab_rc_params(*, overrides: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return style parameters for Lab 1 plots."""

    merged = dict(LAB_RC_PARAMS)
    if overrides:
        merged.update(dict(overrides))
    return merged


def apply_lab_style(*, overrides: Mapping[str, object] | None = None) -> None:
    """Apply Lab 1 style globally via Matplotlib rcParams."""

    mpl.rcParams.update(get_lab_rc_params(overrides=overrides))


@contextmanager
def lab_style_context(*, overrides: Mapping[str, object] | None = None) -> Iterator[None]:
    """Temporarily apply Lab 1 plot style in a context manager."""

    with mpl.rc_context(get_lab_rc_params(overrides=overrides)):
        yield


__all__ = ["LAB_RC_PARAMS", "apply_lab_style", "get_lab_rc_params", "lab_style_context"]
