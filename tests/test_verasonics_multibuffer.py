"""Standalone tests for the Verasonics converter's multi-buffer RF reading.

These build tiny synthetic MATLAB-v7.3-style HDF5 workspaces in a ``tmp_path``
(no external data) and exercise the two per-buffer RF storage forms the
converter supports:

* Several RF buffers stored **in** the ``.mat`` as a ``RcvData`` cell array.
* RF stored in **external** ``RF_data_{k}.bin`` files next to a metadata ``.mat``
  (with the per-buffer ``RF_rows``/``RF_cols``/``RF_frames`` dimensions and the
  ``NonzeroRFcolumns`` channel masks).

Both are read through :meth:`VerasonicsFile.read_raw_buffer_array`, which returns
one buffer as ``(n_frames, n_channels, n_samples)`` regardless of storage form.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from zea.data.convert.verasonics import VerasonicsFile

_REF_DTYPE = h5py.special_dtype(ref=h5py.Reference)


def _write_cell_array(f, name, arrays):
    """Write ``arrays`` as a MATLAB-style cell array (a dataset of HDF5 refs).

    Args:
        f (h5py.File): Open file to write into.
        name (str): Name of the cell-array dataset.
        arrays (list[np.ndarray]): One array per cell.
    """
    refs_group = f.require_group("#refs#")
    dset = f.create_dataset(name, shape=(len(arrays), 1), dtype=_REF_DTYPE)
    for i, arr in enumerate(arrays):
        target = refs_group.create_dataset(f"{name}_{i}", data=arr)
        dset[i, 0] = target.ref


def _write_rcvdata_workspace(path, buffers, bufnums):
    """Write a synthetic workspace with an in-``.mat`` ``RcvData`` cell array.

    Args:
        path (str | Path): Output ``.mat`` (HDF5) path.
        buffers (list[np.ndarray]): One ``(n_frames, n_channels, n_samples)``
            int16 array per RF buffer.
        bufnums (list[int]): 1-based buffer number for each ``Receive`` entry.
    """
    with h5py.File(path, "w") as f:
        _write_cell_array(f, "RcvData", [b.astype(np.int16) for b in buffers])
        receive = f.create_group("Receive")
        receive.create_dataset("bufnum", data=np.asarray(bufnums, np.float64).reshape(-1, 1))


def _write_binary_workspace(path, full_buffers, masks, saved):
    """Write a synthetic workspace whose RF lives in external ``RF_data_{k}.bin``.

    Args:
        path (Path): Output ``.mat`` (HDF5) path; the ``.bin`` files are written
            next to it.
        full_buffers (list[np.ndarray]): Full ``(n_frames, n_channels, n_samples)``
            int16 array per buffer (zero on non-saved channels).
        masks (list[np.ndarray]): Boolean saved-channel mask per buffer.
        saved (list[bool]): Whether each buffer's ``.bin`` is actually written
            (a buffer may be defined but not saved for a measurement).
    """
    path = str(path)
    directory = Path(path).parent
    n_buffers = len(full_buffers)
    rf_rows, rf_cols, rf_frames = [], [], []
    for k in range(n_buffers):
        full = full_buffers[k]
        mask = masks[k]
        n_frames, _, n_samples = full.shape
        rf_rows.append(n_samples)  # fast-time samples (RcvData rows)
        rf_cols.append(int(mask.sum()))  # saved channels
        rf_frames.append(n_frames)
        if saved[k]:
            # (frames, cols, rows) -> MATLAB (rows, cols, frames), column-major bytes.
            saved_ch = full[:, mask, :]
            matlab = np.transpose(saved_ch, (2, 1, 0)).astype("<i2")
            matlab.flatten(order="F").tofile(directory / f"RF_data_{k + 1}.bin")

    with h5py.File(path, "w") as f:
        f.create_dataset("RF_rows", data=np.asarray(rf_rows, np.float64).reshape(-1, 1))
        f.create_dataset("RF_cols", data=np.asarray(rf_cols, np.float64).reshape(-1, 1))
        f.create_dataset("RF_frames", data=np.asarray(rf_frames, np.float64).reshape(-1, 1))
        _write_cell_array(f, "NonzeroRFcolumns", [m.astype(np.uint8).reshape(1, -1) for m in masks])


def test_multibuffer_rcvdata_reads_each_buffer(tmp_path):
    """A ``.mat`` with a multi-cell ``RcvData`` exposes and reads each RF buffer."""
    rng = np.random.default_rng(0)
    n_frames, n_channels, n_samples = 2, 6, 8
    buffers = [
        rng.integers(-2000, 2000, (n_frames, n_channels, n_samples), dtype=np.int16),
        rng.integers(-2000, 2000, (n_frames, n_channels, n_samples), dtype=np.int16),
    ]
    # Receives: first two belong to buffer 1, last three to buffer 2 (1-based).
    bufnums = [1, 1, 2, 2, 2]
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, buffers, bufnums)

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.has_rcvdata is True
        assert vf.available_buffers() == [0, 1]

        # Receive -> buffer mapping (0-based).
        np.testing.assert_array_equal(vf.read_receive_bufnums(), [0, 0, 1, 1, 1])
        np.testing.assert_array_equal(vf.receive_indices_for_buffer(0), [0, 1])
        np.testing.assert_array_equal(vf.receive_indices_for_buffer(1), [2, 3, 4])

        # Each buffer reads back exactly, and the buffers are not swapped.
        np.testing.assert_array_equal(vf.read_raw_buffer_array(0), buffers[0])
        np.testing.assert_array_equal(vf.read_raw_buffer_array(1), buffers[1])
        assert not np.array_equal(buffers[0], buffers[1])


def test_binary_buffer_reconstruction(tmp_path):
    """External ``RF_data_{k}.bin`` files reconstruct the full-channel RF buffers."""
    rng = np.random.default_rng(1)
    n_frames, n_channels, n_samples = 3, 6, 5
    masks = [
        np.array([1, 1, 0, 0, 1, 0], dtype=bool),  # 3 saved channels
        np.array([0, 1, 1, 1, 0, 1], dtype=bool),  # 4 saved channels
    ]
    full_buffers = []
    for mask in masks:
        full = np.zeros((n_frames, n_channels, n_samples), dtype=np.int16)
        full[:, mask, :] = rng.integers(
            -2000, 2000, (n_frames, int(mask.sum()), n_samples), dtype=np.int16
        )
        full_buffers.append(full)

    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, full_buffers, masks, saved=[True, True])

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.has_rcvdata is False
        assert vf.available_buffers() == [0, 1]
        for k in range(2):
            reconstructed = vf.read_raw_buffer_array(k)
            assert reconstructed.shape == (n_frames, n_channels, n_samples)
            assert reconstructed.dtype == np.int16
            # Saved channels match; non-saved channels are zero-filled.
            np.testing.assert_array_equal(reconstructed, full_buffers[k])
            np.testing.assert_array_equal(vf.read_nonzero_rf_columns(k), masks[k])


def test_binary_only_saved_buffers_are_available(tmp_path):
    """A buffer defined in the dimensions but without a ``.bin`` is not listed."""
    n_frames, n_channels, n_samples = 2, 4, 4
    masks = [np.ones(n_channels, bool), np.ones(n_channels, bool), np.ones(n_channels, bool)]
    full = [np.zeros((n_frames, n_channels, n_samples), np.int16) for _ in range(3)]
    for f in full:
        f[...] = 7
    mat = tmp_path / "Parameters.mat"
    # Three buffers are described, but only buffers 1 and 3 are actually saved.
    _write_binary_workspace(mat, full, masks, saved=[True, False, True])

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.available_buffers() == [0, 2]
        with pytest.raises(FileNotFoundError):
            vf.read_raw_buffer_array(1)


def test_binary_dimension_mismatch_raises(tmp_path):
    """A ``.bin`` whose size disagrees with the stored dimensions is rejected."""
    n_frames, n_channels, n_samples = 2, 4, 4
    mask = np.ones(n_channels, bool)
    full = [np.full((n_frames, n_channels, n_samples), 3, np.int16)]
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, full, [mask], saved=[True])

    # Corrupt the binary so its length no longer matches RF_rows*RF_cols*RF_frames.
    binary = tmp_path / "RF_data_1.bin"
    truncated = np.fromfile(binary, dtype="<i2")[:-3]
    truncated.tofile(binary)

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="inconsistent"):
            vf.read_raw_buffer_array(0)
