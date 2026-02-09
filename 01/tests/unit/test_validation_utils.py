"""Unit tests for shared validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ugradio_lab1.utils.validation import as_1d_array, as_2d_array


def test_as_1d_array_accepts_valid_input() -> None:
    values = as_1d_array([1.0, 2.0, 3.0], "values", dtype=float)
    assert values.ndim == 1
    assert values.shape == (3,)


def test_as_1d_array_rejects_non_1d() -> None:
    with pytest.raises(ValueError, match="must be a 1D array"):
        as_1d_array(np.zeros((2, 2)), "values")


def test_as_2d_array_accepts_valid_input() -> None:
    values = as_2d_array(np.zeros((3, 4)), "values")
    assert values.ndim == 2
    assert values.shape == (3, 4)


def test_as_2d_array_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="must be a 2D array"):
        as_2d_array(np.zeros(4), "values")
