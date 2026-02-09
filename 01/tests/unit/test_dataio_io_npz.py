"""Unit tests for dataio.io_npz."""

from __future__ import annotations

import numpy as np
import pytest

from ugradio_lab1.dataio.io_npz import load_npz_dataset, save_npz_dataset


def test_save_and_load_npz_dataset_roundtrip(tmp_path) -> None:
    path = tmp_path / "run.npz"
    arrays = {"time_s": np.array([0.0, 1.0]), "voltage_v": np.array([1.0, -1.0])}
    metadata = {"run_id": "r001", "experiment": "E1"}

    save_npz_dataset(path, arrays, metadata=metadata)
    loaded_arrays, loaded_metadata = load_npz_dataset(path)

    assert set(loaded_arrays.keys()) == {"time_s", "voltage_v"}
    assert np.allclose(loaded_arrays["voltage_v"], arrays["voltage_v"])
    assert loaded_metadata["run_id"] == "r001"


def test_save_npz_dataset_respects_overwrite_flag(tmp_path) -> None:
    path = tmp_path / "run.npz"
    save_npz_dataset(path, {"x": np.array([1.0])})

    with pytest.raises(FileExistsError):
        save_npz_dataset(path, {"x": np.array([2.0])}, overwrite=False)
