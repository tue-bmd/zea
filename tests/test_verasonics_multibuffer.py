"""Standalone tests for the Verasonics converter's multi-buffer RF reading.

These build tiny synthetic MATLAB-v7.3-style HDF5 workspaces in a ``tmp_path``
(no external data) and exercise the two per-buffer RF storage forms the
converter supports:

* Several RF buffers stored **in** the ``.mat`` as a ``RcvData`` cell array.
* RF stored in **external** ``RF_data_{k}.bin`` files next to a metadata ``.mat``
  (see :mod:`zea.data.convert._verasonics_bin` for the layout).

Both are read through :meth:`VerasonicsFile.read_raw_buffer_array`, which returns
one buffer as ``(n_frames, n_channels, n_samples)`` regardless of storage form.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest

from zea.data.convert.verasonics import VerasonicsFile, _validate_convert_config

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
        assert vf.available_buffers() == [0, 1]
        for k in range(2):
            reconstructed = vf.read_raw_buffer_array(k)
            assert reconstructed.shape == (n_frames, n_channels, n_samples)
            assert reconstructed.dtype == np.int16
            # Saved channels match; non-saved channels are zero-filled.
            np.testing.assert_array_equal(reconstructed, full_buffers[k])


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


@pytest.mark.parametrize("num_frames, expected_order", [(3, [2, 0, 1]), (5, [0, 1, 2])])
def test_binary_frame_count_and_order(tmp_path, num_frames, expected_order):
    """A binary buffer uses RF_frames, and lastFrame only when every frame was saved."""
    n_frames, n_channels, n_samples = 3, 2, 4
    full = [np.zeros((n_frames, n_channels, n_samples), np.int16)]
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, full, [np.ones(n_channels, bool)], saved=[True])
    with h5py.File(mat, "a") as f:
        rcv_buffer = f.create_group("Resource/RcvBuffer")
        rcv_buffer.create_dataset("numFrames", data=np.array([[num_frames]], np.float64))
        rcv_buffer.create_dataset("lastFrame", data=np.array([[2]], np.float64))

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.get_frame_count(0) == n_frames
        np.testing.assert_array_equal(vf.get_raw_data_order(0), expected_order)


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


@pytest.mark.parametrize("frames", ["0-1", [0], np.arange(2), ("all", "0")])
def test_frame_selection_with_several_buffers_is_not_implemented(tmp_path, frames):
    """Frame selection is ambiguous when buffers hold different numbers of frames."""
    buffers = [np.zeros((2, 2, 4), np.int16), np.zeros((3, 2, 4), np.int16)]
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, buffers, bufnums=[1, 2])

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(NotImplementedError):
            vf.read_verasonics_file(frames=frames)
        # Selecting a single buffer lifts the restriction (it then fails later on the
        # missing metadata of this synthetic file, not on the frame selection).
        with pytest.raises(Exception) as excinfo:
            vf.read_verasonics_file(frames=frames, buffers=[0])
        assert not isinstance(excinfo.value, NotImplementedError)


def test_convert_yaml_selects_buffers(tmp_path):
    """'buffers' in convert.yaml picks the buffer, so its frame selection is allowed."""
    buffers = [np.zeros((2, 2, 4), np.int16), np.zeros((3, 2, 4), np.int16)]
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, buffers, bufnums=[1, 2])
    (tmp_path / "convert.yaml").write_text(
        "files:\n  - name: raw_data.mat\n    frames: 0-1\n    buffers: [1]\n"
    )

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.load_convert_config()["buffers"] == [1]
        # Fails later on the missing metadata of this synthetic file.
        with pytest.raises(Exception) as excinfo:
            vf.read_verasonics_file()
        assert not isinstance(excinfo.value, NotImplementedError)
        assert "RF buffer 1 (RcvData{2})" in "".join(getattr(excinfo.value, "__notes__", []))


@pytest.mark.parametrize("buffers", [0, [-1], ["0"], [True]])
def test_convert_yaml_rejects_invalid_buffers(buffers):
    """'buffers' in convert.yaml must be a list of non-negative ints."""
    config = {"files": [{"name": "raw_data.mat", "buffers": buffers}]}
    with pytest.raises(ValueError, match="'buffers'"):
        _validate_convert_config(config)


@pytest.mark.parametrize("frames", ["all", ["all"], ("all",)])
def test_all_frames_with_several_buffers_is_allowed(tmp_path, frames):
    """Every spelling of 'all' passes the multi-buffer frame-selection check."""
    buffers = [np.zeros((2, 2, 4), np.int16), np.zeros((3, 2, 4), np.int16)]
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, buffers, bufnums=[1, 2])

    with VerasonicsFile(str(mat), "r") as vf:
        # Fails later on the missing metadata of this synthetic file.
        with pytest.raises(Exception) as excinfo:
            vf.read_verasonics_file(frames=frames)
        assert not isinstance(excinfo.value, NotImplementedError)


def test_unused_buffer_is_skipped(tmp_path):
    """An RcvData cell that no Receive entry points to is not a buffer to convert."""
    buffers = [np.zeros((1, 2, 4), np.int16), np.zeros((1, 2, 4), np.int16)]
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, buffers, bufnums=[1, 1])

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.available_buffers() == [0]
        with pytest.raises(ValueError, match="no Receive"):
            vf.read_verasonics_file(buffers=[1])


def test_unknown_buffer_is_rejected(tmp_path):
    """Requesting an RF buffer that is not in the file raises."""
    mat = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(mat, [np.zeros((1, 2, 4), np.int16)], bufnums=[1])

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match=r"not in this file.*RF buffer 0 \(RcvData\{1\}\)"):
            vf.read_verasonics_file(buffers=[1])
        with pytest.raises(ValueError, match="No RF buffers"):
            vf.read_verasonics_file(buffers=[])
        with pytest.raises(ValueError, match="No Receive entries"):
            vf.receive_indices_for_buffer(3)


def _matlab_value(value):
    """MATLAB storage of a struct field value: chars as a uint16 column, numbers as (1, 1)."""
    if isinstance(value, str):
        return np.array([ord(c) for c in value], dtype=np.uint16).reshape(-1, 1)
    return np.atleast_2d(np.asarray(value, dtype=np.float64))


def _write_struct_field(f, group, field, values):
    """Write a struct-array field as an ``(N, 1)`` dataset of refs, one per entry."""
    _write_cell_array(f, f"{group}/{field}", [_matlab_value(v) for v in values])


def _write_struct(f, group, fields):
    """Write a struct array given as ``{field: [value per entry]}``."""
    for field, values in fields.items():
        _write_struct_field(f, group, field, values)


def _write_trans(f, n_elements, connector=None, probe_type=0):
    trans = f.create_group("Trans")
    trans.create_dataset("units", data=_matlab_value("mm"))
    trans.create_dataset("ElementPos", data=np.zeros((5, n_elements)))
    if probe_type is not None:
        trans.create_dataset("type", data=_matlab_value(probe_type))
    if connector is not None:
        trans.create_dataset("ConnectorES", data=np.asarray(connector, np.float64).reshape(1, -1))


def _write_two_buffer_receive(f, sample_mode="NS200BW"):
    """Three receives: one in buffer 1 (TGC 2), two in buffer 2 (TGC 1)."""
    _write_struct(
        f,
        "Receive",
        {
            "bufnum": [1, 2, 2],
            "startSample": [1, 1, 5],
            "endSample": [4, 4, 8],
            "sampleMode": [sample_mode] * 3,
            "demodFrequency": [5.0] * 3,
            "decimSampleRate": [20.0] * 3,
            "TGC": [2, 1, 1],
        },
    )


def test_dereference_index_out_of_range(tmp_path):
    """Indexing past a reference dataset, or past 0 of a plain dataset, raises."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        _write_cell_array(f, "Cells", [np.ones((1, 1)), 2 * np.ones((1, 1))])
        f.create_dataset("Plain", data=np.ones((2, 2)))

    with VerasonicsFile(str(mat), "r") as vf:
        assert [float(c[0, 0]) for c in vf.dereference_all(vf["Cells"])] == [1.0, 2.0]
        assert vf.dereference_all(vf["Plain"], func=lambda d: d.shape) == [(2, 2)]
        for index in (-1, 2):
            with pytest.raises(IndexError, match="out of range"):
                vf.dereference_index(vf["Cells"], index)
        with pytest.raises(IndexError, match="only index 0"):
            vf.dereference_index(vf["Plain"], 1)


def test_subset_indices_spreads_and_keeps_ends():
    """The constancy spot-check uses an evenly spread subset including both ends."""
    picks = VerasonicsFile._subset_indices(np.arange(100, 200), 5)
    np.testing.assert_array_equal(picks, [100, 125, 150, 174, 199])
    np.testing.assert_array_equal(VerasonicsFile._subset_indices(np.arange(3), 0), [0, 1, 2])


def test_bufnums_fall_back_to_buffer_zero(tmp_path):
    """Without ``Receive.bufnum`` every receive maps to buffer 0."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        _write_struct_field(f, "Receive", "startDepth", [0, 0, 0])

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.read_receive_bufnums(), [0, 0, 0])

    with h5py.File(mat, "a") as f:
        del f["Receive"]
        f.create_group("Receive")
    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(KeyError, match="number of Receive entries"):
            vf.read_receive_bufnums()


def test_receive_metadata_per_buffer(tmp_path):
    """Receive-derived metadata is read per buffer and for all buffers together."""
    mat = tmp_path / "raw_data.mat"
    tgc_waveforms = [np.full((10, 1), 0.0), np.full((10, 1), 1023.0)]
    with h5py.File(mat, "w") as f:
        _write_two_buffer_receive(f)
        _write_cell_array(f, "TGC/Waveform", tgc_waveforms)

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.start_samples, [1, 1, 5])
        np.testing.assert_array_equal(vf.end_samples, [4, 4, 8])
        np.testing.assert_array_equal(vf.get_end_samples(1), [4, 8])
        assert vf.n_ax == 4
        assert vf.sample_mode == 200
        assert not vf.is_baseband_mode
        assert vf.sampling_frequency == 20e6
        assert vf.demodulation_frequency == 5e6
        assert vf.demodulation_frequency == 5e6  # served from the cache
        assert vf.get_demodulation_frequency(1) == 5e6

        # Each buffer uses the TGC waveform its receives reference (1-based).
        np.testing.assert_allclose(vf.get_tgc_gain_curve(0), 100.0)  # 40 dB
        np.testing.assert_allclose(vf.get_tgc_gain_curve(1), 1.0)  # 0 dB
        # Over all buffers the receives disagree; the first referenced TGC is used.
        np.testing.assert_allclose(vf.tgc_gain_curve, 1.0)

        vf._clear_caches()
        assert "_scalar_field_cache" not in vf.__dict__
        assert "_constant_field_cache" not in vf.__dict__


def test_tgc_index_out_of_range_uses_first_waveform(tmp_path):
    """A ``Receive.TGC`` beyond the stored waveforms falls back to the first one."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        _write_struct(f, "Receive", {"bufnum": [1], "TGC": [5]})
        _write_cell_array(f, "TGC/Waveform", [np.full((4, 1), 7.0), np.full((4, 1), 9.0)])

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf._read_tgc_waveform(0), [7.0] * 4)


def _write_event_workspace(path, modes, framenums, seq_control_refs=True):
    """Events ``i`` transmit ``TX(i)`` and receive into ``Receive(i)`` (buffer 1).

    A trailing event transmits without a receive. Every event's SeqControl points
    to a ``timeToNextAcq`` of 10 us.
    """
    n = len(modes)
    with h5py.File(path, "w") as f:
        _write_struct(
            f,
            "Receive",
            {"bufnum": [1] * n, "mode": list(modes), "framenum": list(framenums)},
        )
        _write_struct(f, "Event", {"tx": list(range(1, n + 1)) + [1], "rcv": [*range(1, n + 1), 0]})
        if seq_control_refs:
            _write_struct_field(f, "Event", "seqControl", [[0, 1]] * (n + 1))
        else:
            # A single event's seqControl stored as a plain index vector.
            f["Event"].create_dataset("seqControl", data=np.array([[0.0], [1.0]]))
        _write_struct(f, "SeqControl", {"command": ["timeToNextAcq"], "argument": [10.0]})
        rcv_buffer = f.create_group("Resource/RcvBuffer")
        rcv_buffer.create_dataset("numFrames", data=_matlab_value(max(framenums)))


def test_transmit_events_with_plain_seq_control(tmp_path):
    """A single event with a plain (non-reference) seqControl vector is parsed."""
    mat = tmp_path / "raw_data.mat"
    _write_event_workspace(mat, modes=[0], framenums=[1], seq_control_refs=False)

    with VerasonicsFile(str(mat), "r") as vf:
        tx_order, rcv_order, time_to_next_acq = vf.read_transmit_events()
    np.testing.assert_array_equal(tx_order, [0])
    np.testing.assert_array_equal(rcv_order, [0])
    np.testing.assert_allclose(time_to_next_acq, [[10e-6]])


def test_transmit_events_non_uniform_frames_drop_timing(tmp_path):
    """Timing that does not split evenly into frames of transmits is dropped."""
    mat = tmp_path / "raw_data.mat"
    # Frame 1 has two transmits, frame 2 only one.
    _write_event_workspace(mat, modes=[0, 0, 0], framenums=[1, 1, 2])

    with VerasonicsFile(str(mat), "r") as vf:
        tx_order, _, time_to_next_acq = vf.read_transmit_events()
    np.testing.assert_array_equal(tx_order, [0, 1])
    assert time_to_next_acq is None


def test_transmit_events_accumulate_drop_timing(tmp_path):
    """With ``allow_accumulate`` accumulate-mode transmits and the timing are dropped."""
    mat = tmp_path / "raw_data.mat"
    _write_event_workspace(mat, modes=[0, 1], framenums=[1, 1])

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="accumulate mode"):
            vf.read_transmit_events()
        tx_order, rcv_order, time_to_next_acq = vf.read_transmit_events(allow_accumulate=True)
    np.testing.assert_array_equal(tx_order, [0])
    np.testing.assert_array_equal(rcv_order, [0])
    assert time_to_next_acq is None


def _write_raw_data_workspace(path, rcv_data, connector=None, n_elements=None):
    """Single-buffer workspace with two receives of 4 samples each."""
    n_elements = rcv_data.shape[1] if n_elements is None else n_elements
    with h5py.File(path, "w") as f:
        f.create_dataset("RcvData", data=rcv_data)
        _write_struct(
            f,
            "Receive",
            {
                "bufnum": [1, 1],
                "startSample": [1, 5],
                "endSample": [4, 8],
                "sampleMode": ["NS200BW"] * 2,
            },
        )
        _write_trans(f, n_elements, connector=connector)
        rcv_buffer = f.create_group("Resource/RcvBuffer")
        rcv_buffer.create_dataset("numFrames", data=_matlab_value(rcv_data.shape[0]))
        rcv_buffer.create_dataset("lastFrame", data=_matlab_value(1))


def test_read_raw_data_layout(tmp_path):
    """Raw data is reordered from the circular buffer and split per transmit."""
    rcv_data = np.arange(2 * 3 * 8, dtype=np.int16).reshape(2, 3, 8)
    mat = tmp_path / "raw_data.mat"
    _write_raw_data_workspace(mat, rcv_data)

    with VerasonicsFile(str(mat), "r") as vf:
        raw = vf.read_raw_data()
    # lastFrame=1 (1-based) means frame 1 (0-based) is the oldest.
    expected = rcv_data[[1, 0]].reshape(2, 3, 2, 4).transpose(0, 2, 3, 1)[..., None]
    np.testing.assert_array_equal(raw, expected)


def test_read_raw_data_connector_out_of_range(tmp_path):
    """A connector map pointing past the acquired channels is rejected."""
    mat = tmp_path / "raw_data.mat"
    _write_raw_data_workspace(mat, np.zeros((1, 4, 8), np.int16), connector=[1, 9], n_elements=2)

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="Cannot map acquired channels"):
            vf.read_raw_data()


def test_missing_rcvdata_and_image_buffer(tmp_path):
    """Missing RF data and out-of-range image buffers raise clear errors."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        f.create_dataset("ImgDataP", data=np.zeros((1, 2, 2)))

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(KeyError, match="'RcvData' not found"):
            vf.read_raw_buffer_array(0)
        with pytest.raises(KeyError, match="'RcvData' not found"):
            vf.read_verasonics_file()
        with pytest.raises(ValueError, match="image_buffer_index=1 is out of range"):
            vf.read_image_data_p(image_buffer_index=1)


@pytest.mark.parametrize("probe_type", [None, 99])
def test_probe_type_falls_back_to_unknown(tmp_path, probe_type):
    """A missing or unrecognized ``Trans.type`` gives probe type 'unknown'."""
    mat = tmp_path / "raw_data.mat"
    with h5py.File(mat, "w") as f:
        _write_trans(f, 2, probe_type=probe_type)

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.probe.type == "unknown"
        assert vf.probe.connector is None


def test_binary_buffer_without_channel_mask(tmp_path):
    """Without ``NonzeroRFcolumns`` the binary is taken to hold every channel."""
    full = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, [full], [np.ones(3, bool)], saved=[True])
    with h5py.File(mat, "a") as f:
        del f["NonzeroRFcolumns"]

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.read_raw_buffer_array(0), full)


def test_binary_inconsistent_metadata_raises(tmp_path):
    """A channel mask or dimensions that disagree with the binary are rejected."""
    full = np.zeros((2, 3, 4), np.int16)
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, [full], [np.ones(3, bool)], saved=[True])
    with h5py.File(mat, "a") as f:
        # Mark only 2 of the 3 saved channels.
        f["#refs#/NonzeroRFcolumns_0"][...] = np.array([[1, 0, 1]], np.uint8)

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="NonzeroRFcolumns"):
            vf.read_raw_buffer_array(0)

    # A second .bin without a matching RF_rows entry.
    full.tofile(tmp_path / "RF_data_2.bin")
    with h5py.File(mat, "a") as f:
        del f["RF_frames"]
        f.create_dataset("RF_frames", data=np.array([[2.0], [2.0]]))
    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(IndexError, match="out of range for 'RF_rows'"):
            vf.read_raw_buffer_array(1)


@pytest.mark.parametrize("answer, overwritten", [("n", False), ("na", False), ("y", True)])
def test_convert_existing_output(tmp_path, monkeypatch, answer, overwritten):
    """An existing output file is skipped or overwritten depending on the answer."""
    from zea.data.convert import verasonics

    src = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(src, [np.zeros((1, 2, 4), np.int16)], bufnums=[1])
    dst = tmp_path / "raw_data.hdf5"
    dst.write_text("existing")

    converted = []
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    monkeypatch.setattr(
        verasonics, "_zea_from_verasonics_workspace", lambda src, dst, **_: converted.append(dst)
    )
    args = argparse.Namespace(src=str(src), dst=str(dst), frames=["all"], allow_accumulate=False)
    args.hf_repo_id = None
    verasonics.convert_verasonics(args)

    assert dst.exists() != overwritten
    assert converted == ([dst] if overwritten else [])


def test_convert_forwards_buffers(tmp_path, monkeypatch):
    """The CLI's --buffers reaches the conversion."""
    from zea.data.convert import verasonics

    src = tmp_path / "raw_data.mat"
    _write_rcvdata_workspace(src, [np.zeros((1, 2, 4), np.int16)], bufnums=[1])
    dst = tmp_path / "raw_data.hdf5"

    calls = []
    monkeypatch.setattr(
        verasonics, "_zea_from_verasonics_workspace", lambda src, dst, **kw: calls.append(kw)
    )
    args = argparse.Namespace(
        src=str(src), dst=str(dst), frames=None, allow_accumulate=False, buffers=[0]
    )
    args.hf_repo_id = None
    verasonics.convert_verasonics(args)

    assert [kw["buffers"] for kw in calls] == [[0]]
