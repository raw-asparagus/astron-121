"""Unit tests for plotting.style."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")

from ugradio_lab1.plotting.style import get_lab_rc_params, lab_style_context


def test_get_lab_rc_params_applies_overrides() -> None:
    rc = get_lab_rc_params(overrides={"axes.grid": False})
    assert rc["axes.grid"] is False


def test_lab_style_context_temporarily_sets_params() -> None:
    original = matplotlib.rcParams["axes.grid"]
    with lab_style_context(overrides={"axes.grid": (not original)}):
        assert matplotlib.rcParams["axes.grid"] == (not original)
    assert matplotlib.rcParams["axes.grid"] == original
