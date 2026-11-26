"""Tests for the synthetic Verasonics workspaces in ``tests/fixtures``."""

import h5py
import numpy as np
import pytest

from tests.fixtures.verasonics_workspace import (
    CellArray,
    StructArray,
    VerasonicsWorkspace,
    check,
    write_workspace,
)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"n_buffers": 2, "n_frames": [2, 3], "n_image_buffers": 1},
        {"rf_storage": "bin", "n_buffers": 2, "n_image_buffers": 0, "save_raw_version": None},
        {"n_tx": 1, "sample_mode": "BS100BW"},
    ],
)
def test_generated_workspace_matches_schema(tmp_path, settings):
    """Generated workspaces store every variable as the schema says, like real files."""
    mat = VerasonicsWorkspace(**settings).write(tmp_path / "raw_data.mat")
    problems, _ = check(mat)
    assert problems == []
    with open(mat, "rb") as file:
        assert file.read(10) == b"MATLAB 7.3"


def test_check_reports_storage_that_differs_from_matlab(tmp_path):
    """The schema check catches storage MATLAB does not produce."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        f.create_dataset("RcvData", data=np.zeros((1, 2, 4), np.int16))  # not a cell
        version = f.create_dataset("save_raw_version", data=np.ones((1, 1)))  # not a char
        version.attrs["MATLAB_class"] = np.bytes_("double")
    problems, missing = check(mat)
    assert any("RcvData: expected a cell" in p for p in problems)
    assert any("save_raw_version: double" in p and "expected char" in p for p in problems)
    assert "Trans" in missing


def test_write_workspace_stores_values_as_matlab_does(tmp_path):
    """Struct arrays, 1x1 structs, cells and vectors follow the schema."""
    mat = tmp_path / "raw_data.mat"
    write_workspace(
        mat,
        {
            "Receive": StructArray({"bufnum": [1, 2], "sampleMode": ["NS200BW", "BS100BW"]}),
            "TW": StructArray({"type": ["parametric"]}),
            "Trans": {"units": "mm", "ConnectorES": np.arange(3)},
            "NonzeroRFcolumns": CellArray([np.array([True, False])]),
            "SeqControl/argument": None,
        },
    )
    with h5py.File(mat, "r") as f:
        assert f["Receive/bufnum"].shape == (2, 1)
        assert h5py.check_dtype(ref=f["Receive/bufnum"].dtype) is not None
        assert f["TW/type"].attrs["MATLAB_class"] == b"char"  # a 1x1 struct stores fields
        assert f["Trans/ConnectorES"].shape == (1, 3)
        mask = f[f["NonzeroRFcolumns"][0, 0]]
        assert mask.shape == (2, 1) and mask.attrs["MATLAB_class"] == b"logical"
        assert f["SeqControl/argument"].attrs["MATLAB_empty"] == 1
        fields = {bytes(name).decode() for name in f["Trans"].attrs["MATLAB_fields"]}
        assert fields == {"units", "ConnectorES"}


@pytest.mark.parametrize(
    "variables, error, match",
    [
        ({"Trans/madeUpField": 1.0}, KeyError, "not in the Verasonics workspace SCHEMA"),
        ({"RF_rows": np.ones((1, 2))}, ValueError, "column vector"),
        ({"RcvData": np.zeros((1, 2, 4), np.int16)}, TypeError, "MATLAB cell"),
        ({"Trans": {"units": 3}}, TypeError, "char array"),
        ({"Trans": {"frequency": None}}, ValueError, "never stored as an empty"),
    ],
)
def test_write_workspace_rejects_what_matlab_does_not_store(tmp_path, variables, error, match):
    with pytest.raises(error, match=match):
        write_workspace(tmp_path / "raw_data.mat", variables)
