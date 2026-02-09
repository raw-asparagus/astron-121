"""Input validation helpers shared across the package."""

from __future__ import annotations

import numpy as np


def as_1d_array(values: np.ndarray, name: str, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Return a validated non-empty 1D NumPy array."""

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    return array


def as_2d_array(values: np.ndarray, name: str, *, dtype: np.dtype | None = None) -> np.ndarray:
    """Return a validated non-empty 2D NumPy array."""

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (n_blocks, n_samples).")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot have empty dimensions.")
    return array


__all__ = ["as_1d_array", "as_2d_array"]
