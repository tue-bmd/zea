"""Standalone tests for the Verasonics converter's multi-buffer RF reading.

These build tiny synthetic MATLAB ``-v7.3`` workspaces in a ``tmp_path`` with
:mod:`tests.fixtures.verasonics_workspace`, which stores every variable the way its
schema (checked against real Verasonics files) says. They exercise the two per-buffer RF
storage forms the converter supports:

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

from tests.fixtures.verasonics_workspace import (
    CellArray,
    StructArray,
    VerasonicsWorkspace,
    write_bin,
    write_workspace,
)
from zea.data.convert.verasonics import (
    VerasonicsFile,
    _time_to_next_acq_per_frame,
    _validate_convert_config,
)
from zea.data.file import File


def _write_rcvdata_workspace(path, buffers, bufnums):
    """Write a synthetic workspace with an in-``.mat`` ``RcvData`` cell array.

    Args:
        path (str | Path): Output ``.mat`` path.
        buffers (list[np.ndarray]): One ``(n_frames, n_channels, n_samples)``
            int16 array per RF buffer.
        bufnums (list[int]): 1-based buffer number for each ``Receive`` entry.
    """
    write_workspace(
        path, {"RcvData": CellArray(buffers), "Receive": StructArray({"bufnum": list(bufnums)})}
    )


def _write_binary_workspace(path, full_buffers, masks, saved, variables=None):
    """Write a synthetic workspace whose RF lives in external ``RF_data_{k}.bin``.

    Args:
        path (Path): Output ``.mat`` path; the ``.bin`` files are written next to it.
        full_buffers (list[np.ndarray]): Full ``(n_frames, n_channels, n_samples)``
            int16 array per buffer (zero on non-saved channels).
        masks (list[np.ndarray] | None): Boolean saved-channel mask per buffer, or
            ``None`` to leave ``NonzeroRFcolumns`` out.
        saved (list[bool]): Whether each buffer's ``.bin`` is actually written
            (a buffer may be defined but not saved for a measurement).
        variables (dict, optional): Further workspace variables to write.
    """
    channel_masks = masks or [np.ones(full.shape[1], bool) for full in full_buffers]
    for k, (full, mask) in enumerate(zip(full_buffers, channel_masks)):
        if saved[k]:
            write_bin(Path(path).parent / f"RF_data_{k + 1}.bin", full[:, mask, :])
    workspace = {
        "RF_rows": [full.shape[2] for full in full_buffers],  # fast-time samples
        "RF_cols": [int(mask.sum()) for mask in channel_masks],  # saved channels
        "RF_frames": [full.shape[0] for full in full_buffers],
    }
    if masks is not None:
        workspace["NonzeroRFcolumns"] = CellArray(masks)
    write_workspace(path, {**workspace, **(variables or {})})


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
    rcv_buffer = {"numFrames": num_frames, "lastFrame": 2}
    _write_binary_workspace(
        mat, full, [np.ones(n_channels, bool)], [True], {"Resource/RcvBuffer": rcv_buffer}
    )

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.get_frame_count(0) == n_frames
        np.testing.assert_array_equal(vf.get_raw_data_order(0), expected_order)


def test_binary_buffer_without_rcvbuffer_entry_keeps_order(tmp_path):
    """A second .bin with only one RcvBuffer struct is assumed to be in order."""
    full = [np.zeros((3, 2, 4), np.int16), np.zeros((2, 2, 4), np.int16)]
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(
        mat, full, [np.ones(2, bool)] * 2, [True, True], {"Resource/RcvBuffer": {"numFrames": 3}}
    )

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.get_raw_data_order(1), [0, 1])


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
        with pytest.raises(ValueError, match=r"\[0\] \(0-based\) are selected more than once"):
            vf.read_verasonics_file(buffers=[0, 0])
        with pytest.raises(ValueError, match="No Receive entries"):
            vf.receive_indices_for_buffer(3)


def _trans(n_elements, connector=None, probe_type=0):
    """A 1x1 ``Trans`` struct with ``n_elements`` elements."""
    trans = {"units": "mm", "ElementPos": np.zeros((4, n_elements))}
    if probe_type is not None:
        trans["type"] = probe_type
    if connector is not None:
        trans["ConnectorES"] = np.asarray(connector)
    return trans


def _two_buffer_receive(sample_mode="NS200BW"):
    """Three receives: one in buffer 1 (TGC 2), two in buffer 2 (TGC 1)."""
    return StructArray(
        {
            "bufnum": [1, 2, 2],
            "startSample": [1, 1, 5],
            "endSample": [4, 4, 8],
            "sampleMode": [sample_mode] * 3,
            "demodFrequency": [5.0] * 3,
            "decimSampleRate": [20.0] * 3,
            "TGC": [2, 1, 1],
        }
    )


def test_dereference_index_out_of_range(tmp_path):
    """Indexing past a reference dataset, or past 0 of a plain dataset, raises."""
    mat = tmp_path / "raw_data.mat"
    images = [np.ones((1, 1, 1, 1)), 2 * np.ones((1, 1, 1, 1))]
    write_workspace(mat, {"ImgDataP": CellArray(images), "Trans": {"ElementPos": np.ones((4, 2))}})

    with VerasonicsFile(str(mat), "r") as vf:
        assert [float(c[0, 0, 0, 0]) for c in vf.dereference_all(vf["ImgDataP"])] == [1.0, 2.0]
        element_pos = vf["Trans"]["ElementPos"]
        assert vf.dereference_all(element_pos, func=lambda d: d.shape) == [(4, 2)]
        for index in (-1, 2):
            with pytest.raises(IndexError, match="out of range"):
                vf.dereference_index(vf["ImgDataP"], index)
        with pytest.raises(IndexError, match="only index 0"):
            vf.dereference_index(element_pos, 1)


def test_missing_file_is_reported_as_missing(tmp_path):
    """A path that does not exist is not reported as a non-HDF5 (-v7.3) file."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        VerasonicsFile(str(tmp_path / "typo.mat"), "r")


def test_constant_receive_field_checks_every_receive(tmp_path):
    """A single differing receive in the middle of a buffer is detected."""
    n = 101
    modes = ["NS200BW"] * n
    modes[37] = "BS100BW"
    mat = tmp_path / "raw_data.mat"
    write_workspace(mat, {"Receive": StructArray({"bufnum": [1] * n, "sampleMode": modes})})

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(AssertionError, match="Multiple sample modes"):
            vf.get_sample_mode(0)


@pytest.mark.parametrize(
    "rates, quad_decims, message",
    [
        ([20.0, 20.0, 40.0, 20.0], [1.0] * 4, "Multiple decimation sample rates"),
        ([20.0] * 4, [1.0, 2.0, 1.0, 1.0], "Multiple quadrature decimation factors"),
    ],
)
def test_sampling_rate_is_checked_on_every_receive(tmp_path, rates, quad_decims, message):
    """A buffer that mixes decimSampleRate or quadDecim values is rejected."""
    mat = tmp_path / "raw_data.mat"
    receive = {
        "bufnum": [1] * 4,
        "sampleMode": ["NS200BW"] * 4,
        "decimSampleRate": rates,
        "quadDecim": quad_decims,
    }
    write_workspace(mat, {"Receive": StructArray(receive)})
    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(AssertionError, match=message):
            vf.get_sampling_frequency(0)


def test_bufnums_fall_back_to_buffer_zero(tmp_path):
    """Without ``Receive.bufnum`` every receive maps to buffer 0."""
    mat = tmp_path / "raw_data.mat"
    write_workspace(mat, {"Receive": StructArray({"startDepth": [0, 0, 0]})})

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
    tgc_waveforms = [np.full(10, 0.0), np.full(10, 1023.0)]
    write_workspace(
        mat, {"Receive": _two_buffer_receive(), "TGC": StructArray({"Waveform": tgc_waveforms})}
    )

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


def test_buffer_with_several_tgc_waveforms_is_rejected(tmp_path):
    """One buffer whose receives use different TGC waveforms cannot be one track."""
    mat = tmp_path / "raw_data.mat"
    waveforms = [np.full(4, 7.0), np.full(4, 9.0)]
    write_workspace(
        mat,
        {
            "Receive": StructArray({"bufnum": [1, 1], "TGC": [1, 2]}),
            "TGC": StructArray({"Waveform": waveforms}),
        },
    )

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match=r"RF buffer 0 \(RcvData\{1\}\).*\[1, 2\]"):
            vf._read_tgc_waveform(0)


def test_tgc_index_out_of_range_uses_first_waveform(tmp_path):
    """A ``Receive.TGC`` beyond the stored waveforms falls back to the first one."""
    mat = tmp_path / "raw_data.mat"
    waveforms = [np.full(4, 7.0), np.full(4, 9.0)]
    write_workspace(
        mat,
        {
            "Receive": StructArray({"bufnum": [1], "TGC": [5]}),
            "TGC": StructArray({"Waveform": waveforms}),
        },
    )

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf._read_tgc_waveform(0), [7.0] * 4)


def _write_event_workspace(path, modes, framenums, trailing_event=True, ttna_events=None):
    """Events ``i`` transmit ``TX(i)`` and receive into ``Receive(i)`` (buffer 1).

    With ``trailing_event`` a last event transmits without a receive. The events in
    ``ttna_events`` (default: all) have a SeqControl ``timeToNextAcq`` of 10 us, the
    others none.
    """
    n = len(modes)
    tx, rcv = list(range(1, n + 1)), list(range(1, n + 1))
    if trailing_event:
        tx, rcv = [*tx, 1], [*rcv, 0]
    seq_controls = [
        [0, 1] if ttna_events is None or i in ttna_events else [0, 0] for i in range(len(tx))
    ]
    write_workspace(
        path,
        {
            "Receive": StructArray(
                {"bufnum": [1] * n, "mode": list(modes), "framenum": list(framenums)}
            ),
            "Event": StructArray({"tx": tx, "rcv": rcv, "seqControl": seq_controls}),
            "SeqControl": StructArray({"command": ["timeToNextAcq"], "argument": [10.0]}),
            "Resource/RcvBuffer": {"numFrames": max(framenums)},
        },
    )


def test_transmit_events_with_plain_seq_control(tmp_path):
    """A 1x1 Event struct, whose seqControl is a plain (non-reference) vector, is parsed."""
    mat = tmp_path / "raw_data.mat"
    _write_event_workspace(mat, modes=[0], framenums=[1], trailing_event=False)

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


def test_transmit_events_timing_per_frame_dropped(tmp_path):
    """Timing only on the last event of each frame cannot be mapped and is dropped."""
    mat = tmp_path / "raw_data.mat"
    _write_event_workspace(mat, modes=[0, 0, 0, 0], framenums=[1, 1, 2, 2], ttna_events=[1, 3])

    with VerasonicsFile(str(mat), "r") as vf:
        tx_order, _, time_to_next_acq = vf.read_transmit_events()
    np.testing.assert_array_equal(tx_order, [0, 1])
    assert time_to_next_acq is None


def test_sample_mode_single_receive_plain_dataset(tmp_path):
    """A single Receive stores sampleMode as a plain char dataset, not a reference."""
    mat = tmp_path / "raw_data.mat"
    write_workspace(mat, {"Receive": StructArray({"sampleMode": ["BS100BW"]})})

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.get_sample_mode() == 100


def test_time_to_next_acq_per_frame():
    """Per-event timing is reshaped to (n_frames, n_tx) and the frames are selected."""
    values = np.arange(6.0)
    timing, problem = _time_to_next_acq_per_frame(values, 6, 2, np.array([0, 2]))
    assert problem is None
    np.testing.assert_array_equal(timing, [[0, 1], [4, 5]])


@pytest.mark.parametrize(
    "n_values, n_events, n_tx, frame_indices, reason",
    [
        (0, 4, 2, [0, 1], "found 0 timeToNextAcq values"),  # no timeToNextAcq at all
        (2, 4, 2, [0, 1], "expected exactly one per event"),  # one per frame
        (3, 3, 2, [0], "do not split into frames"),
        (0, 0, 0, [0], "do not split into frames"),  # no events in the buffer
        (4, 4, 2, [0, 2], "fewer than the requested frames"),
    ],
)
def test_time_to_next_acq_per_frame_dropped(n_values, n_events, n_tx, frame_indices, reason):
    """Timing that does not map one-to-one onto the requested transmits is dropped."""
    timing, problem = _time_to_next_acq_per_frame(
        np.zeros(n_values), n_events, n_tx, np.array(frame_indices)
    )
    assert timing is None
    assert reason in problem


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


def _write_raw_data_workspace(path, rcv_data, connector=None, n_elements=None, variables=None):
    """Single-buffer workspace with two receives of 4 samples each."""
    n_elements = rcv_data.shape[1] if n_elements is None else n_elements
    receive = {
        "bufnum": [1, 1],
        "startSample": [1, 5],
        "endSample": [4, 8],
        "sampleMode": ["NS200BW"] * 2,
    }
    workspace = {
        "RcvData": CellArray([rcv_data]),
        "Receive": StructArray(receive),
        "Trans": _trans(n_elements, connector=connector),
        "Resource/RcvBuffer": {"numFrames": rcv_data.shape[0], "lastFrame": 1},
    }
    write_workspace(path, {**workspace, **(variables or {})})


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


def test_read_raw_data_without_last_frame_keeps_order(tmp_path):
    """Without ``RcvBuffer.lastFrame``, RcvData frames are assumed to be in order."""
    rcv_data = np.arange(2 * 3 * 8, dtype=np.int16).reshape(2, 3, 8)
    mat = tmp_path / "raw_data.mat"
    _write_raw_data_workspace(
        mat, rcv_data, variables={"Resource/RcvBuffer": {"numFrames": rcv_data.shape[0]}}
    )

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.get_raw_data_order(0), [0, 1])
        raw = vf.read_raw_data()
    expected = rcv_data.reshape(2, 3, 2, 4).transpose(0, 2, 3, 1)[..., None]
    np.testing.assert_array_equal(raw, expected)


@pytest.mark.parametrize(
    "save_raw_version, connector_applied, permutation",
    [
        (True, None, [2, 0, 1]),  # our saver never reorders
        (False, False, [2, 0, 1]),
        (False, True, [0, 1, 2]),
    ],
)
def test_read_raw_data_connector(tmp_path, save_raw_version, connector_applied, permutation):
    """The connector is applied unless the raw data was already reordered with it."""
    rcv_data = np.arange(3 * 8, dtype=np.int16).reshape(1, 3, 8)
    mat = tmp_path / "raw_data.mat"
    variables = {"save_raw_version": "1.0"} if save_raw_version else None
    _write_raw_data_workspace(mat, rcv_data, connector=[3, 1, 2], variables=variables)

    with VerasonicsFile(str(mat), "r") as vf:
        raw = vf.read_raw_data(connector_applied=connector_applied)
    expected = rcv_data[:, permutation].reshape(1, 3, 2, 4).transpose(0, 2, 3, 1)[..., None]
    np.testing.assert_array_equal(raw, expected)


def test_read_raw_data_connector_ambiguous(tmp_path):
    """Without save_raw_version and with equal channel and element counts, the user must choose."""
    mat = tmp_path / "raw_data.mat"
    _write_raw_data_workspace(mat, np.zeros((1, 3, 8), np.int16), connector=[3, 1, 2])

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="connector_applied"):
            vf.read_raw_data()


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
    write_workspace(mat, {"ImgDataP": CellArray([np.zeros((1, 1, 2, 2))])})

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(KeyError, match="'RcvData' not found"):
            vf.read_raw_buffer_array(0)
        with pytest.raises(KeyError, match="'RcvData' not found"):
            vf.read_verasonics_file()
        with pytest.raises(ValueError, match="image_buffer_index=1 is out of range"):
            vf.read_image_data_p(image_buffer_index=1)


def test_image_buffer_without_first_and_last_frame_keeps_order(tmp_path):
    """Without ``ImageBuffer.firstFrame``/``lastFrame``, ImgDataP frames stay in order."""
    image = np.arange(3 * 1 * 2 * 2, dtype=float).reshape(3, 1, 2, 2)
    mat = tmp_path / "raw_data.mat"
    write_workspace(
        mat,
        {"ImgDataP": CellArray([image]), "Resource/ImageBuffer": {"numFrames": 3}},
    )

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.get_image_data_p_frame_order(0), [0, 1, 2])


@pytest.mark.parametrize("probe_type", [None, 99])
def test_probe_type_none_when_missing_or_unknown(tmp_path, probe_type):
    """A missing or unrecognized ``Trans.type`` gives probe type ``None``."""
    mat = tmp_path / "raw_data.mat"
    write_workspace(mat, {"Trans": _trans(2, probe_type=probe_type)})

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.probe.type is None
        assert vf.probe.connector is None


def test_binary_buffer_without_channel_mask(tmp_path):
    """Without ``NonzeroRFcolumns`` the binary is taken to hold every channel."""
    full = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, [full], None, saved=[True])

    with VerasonicsFile(str(mat), "r") as vf:
        np.testing.assert_array_equal(vf.read_raw_buffer_array(0), full)


def test_binary_inconsistent_metadata_raises(tmp_path):
    """A channel mask or dimensions that disagree with the binary are rejected."""
    full = np.zeros((2, 3, 4), np.int16)
    mat = tmp_path / "Parameters.mat"
    _write_binary_workspace(mat, [full], [np.ones(3, bool)], saved=[True])
    # Mark only 2 of the 3 saved channels.
    write_workspace(
        mat,
        {
            "RF_rows": [4],
            "RF_cols": [3],
            "RF_frames": [2],
            "NonzeroRFcolumns": CellArray([np.array([True, False, True])]),
        },
    )

    with VerasonicsFile(str(mat), "r") as vf:
        with pytest.raises(ValueError, match="NonzeroRFcolumns"):
            vf.read_raw_buffer_array(0)

    # A second .bin without a matching RF_rows entry.
    write_bin(tmp_path / "RF_data_2.bin", full)
    write_workspace(mat, {"RF_rows": [4], "RF_cols": [3], "RF_frames": [2, 2]})
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


@pytest.mark.parametrize("rf_storage", ["rcvdata", "bin"])
def test_full_workspace_reads_back_each_buffer(tmp_path, rf_storage):
    """Every buffer's RF reads back per transmit and element, with consistent metadata."""
    ws = VerasonicsWorkspace(rf_storage=rf_storage, n_buffers=2, n_frames=[2, 3])
    mat = ws.write(tmp_path / "raw_data.mat")

    with VerasonicsFile(str(mat), "r") as vf:
        assert vf.available_buffers() == [0, 1]
        for k, rf in enumerate(ws.rf):
            raw = vf.read_raw_data(buffer_index=k)
            # (frames, channels, tx * ax) -> (frames, tx, ax, elements, 1)
            elements = rf[:, ws.connector - 1, :]
            expected = elements.reshape(len(rf), ws.n_elements, ws.n_tx, ws.n_ax)
            np.testing.assert_array_equal(raw, expected.transpose(0, 2, 3, 1)[..., None])

            scan = vf.read_scan(buffer_index=k)
            assert scan["t0_delays"].shape == (ws.n_tx, ws.n_elements)
            assert np.all(np.isinf(scan["focus_distances"]))  # plane waves
            np.testing.assert_allclose(scan["time_to_next_transmit"], 100e-6)
            assert scan["time_to_next_transmit"].shape == (len(rf), ws.n_tx)
            assert scan["sampling_frequency"] == 20e6


def test_full_workspace_converts_to_one_track_per_buffer(tmp_path):
    """A two-buffer workspace converts to a valid two-track zea file."""
    mat = VerasonicsWorkspace(n_buffers=2).write(tmp_path / "raw_data.mat")
    with VerasonicsFile(str(mat), "r") as vf:
        vf.to_zea(tmp_path / "raw_data.hdf5")
    with File(tmp_path / "raw_data.hdf5", "r") as f:
        f.validate()
        assert f.track_labels == ["rf_buffer_0", "rf_buffer_1"]
