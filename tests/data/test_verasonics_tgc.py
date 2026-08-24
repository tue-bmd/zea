"""Tests for reading time-gain-compensation curves from Verasonics files."""

import h5py
import numpy as np
import pytest

from zea.data.convert.verasonics import VerasonicsFile

N_AX = 128
N_RCV = 4
SAMPLING_FREQUENCY = 20e6


def _write_references(file, path, arrays):
    """Write arrays the way MATLAB stores a struct array field with more than one element."""
    references = file.require_group("#refs#")
    dataset = file.create_dataset(path, shape=(len(arrays), 1), dtype=h5py.ref_dtype)
    for i, array in enumerate(arrays):
        element = references.create_dataset(f"{path}_{i}".replace("/", "_"), data=array)
        dataset[i, 0] = element.ref


def _column(value):
    """A scalar the way MATLAB stores it: a 1x1 matrix, which h5py reads as a column."""
    return np.array([[value]], dtype=np.float64)


def _encode_string(string):
    """A string the way MATLAB stores it: one column of character codes."""
    return np.array([[ord(character)] for character in string], dtype=np.uint16)


def _make_verasonics_file(path, tgc_waveforms, tgc_selections):
    """Write a minimal Verasonics file holding only what the TGC readers need.

    Args:
        path (Path): Path to write the file to.
        tgc_waveforms (list): Raw gain curve of every TGC struct in the file.
        tgc_selections (list or None): 1-based TGC struct selected by every receive event,
            or None to leave ``Receive.TGC`` out of the file entirely.
    """
    with h5py.File(path, "w") as file:
        _write_references(
            file, "Receive/decimSampleRate", [_column(SAMPLING_FREQUENCY / 1e6)] * N_RCV
        )
        _write_references(file, "Receive/startSample", [_column(1)] * N_RCV)
        _write_references(file, "Receive/endSample", [_column(N_AX)] * N_RCV)
        _write_references(file, "Receive/sampleMode", [_encode_string("NS200BW")] * N_RCV)

        if tgc_selections is not None:
            _write_references(file, "Receive/TGC", [_column(s) for s in tgc_selections])

        waveforms = [np.asarray(waveform, dtype=np.float64)[:, None] for waveform in tgc_waveforms]
        if len(waveforms) == 1:
            # MATLAB stores a single-element struct array as a regular dataset
            file.create_dataset("TGC/Waveform", data=waveforms[0])
        else:
            _write_references(file, "TGC/Waveform", waveforms)

    return path


def _expected_curve(waveform):
    return VerasonicsFile.compute_tgc_gain_curve(
        np.asarray(waveform, dtype=np.float64), N_AX, SAMPLING_FREQUENCY
    )


# Two distinct gain curves, plus a copy of the first one to check that equal curves
# stored in different structs are recognised as equal
RAMP = np.linspace(0, 1023, 64)
STEEP_RAMP = np.linspace(512, 1023, 64)


@pytest.fixture(name="rcv_order")
def fixture_rcv_order():
    """Receives in event order; deliberately not the order of the Receive struct."""
    return [2, 0, 3, 1]


def test_missing_receive_tgc_reads_the_single_curve(tmp_path, rcv_order):
    """Files that leave Receive.TGC unset have a single gain curve to use."""
    path = _make_verasonics_file(tmp_path / "no_receive_tgc.mat", [RAMP], tgc_selections=None)

    with VerasonicsFile(path) as file:
        assert np.array_equal(file.read_tgc_selection(rcv_order), np.ones(len(rcv_order)))
        np.testing.assert_allclose(file.read_tgc_gain_curves(rcv_order), _expected_curve(RAMP))


def test_selection_follows_the_receive_order(tmp_path, rcv_order):
    """The selected struct is read per receive event, in the order of the events."""
    path = _make_verasonics_file(
        tmp_path / "selection.mat", [RAMP, STEEP_RAMP], tgc_selections=[1, 2, 2, 1]
    )

    with VerasonicsFile(path) as file:
        # rcv_order picks receives 2, 0, 3, 1 out of selections [1, 2, 2, 1]
        assert np.array_equal(file.read_tgc_selection(rcv_order), [2, 1, 1, 2])


def test_single_selected_curve_is_stored_once(tmp_path, rcv_order):
    """Transmits that all select the same struct share one gain curve."""
    path = _make_verasonics_file(
        tmp_path / "one_selected.mat", [RAMP, STEEP_RAMP], tgc_selections=[2] * N_RCV
    )

    with VerasonicsFile(path) as file:
        np.testing.assert_allclose(
            file.read_tgc_gain_curves(rcv_order), _expected_curve(STEEP_RAMP)
        )


def test_equal_curves_in_different_structs_are_stored_once(tmp_path, rcv_order):
    """Different structs holding the same gain curve are not stored per transmit."""
    path = _make_verasonics_file(
        tmp_path / "equal_curves.mat", [RAMP, RAMP.copy()], tgc_selections=[1, 2, 1, 2]
    )

    with VerasonicsFile(path) as file:
        np.testing.assert_allclose(file.read_tgc_gain_curves(rcv_order), _expected_curve(RAMP))


def test_differing_curves_are_stored_per_transmit(tmp_path, rcv_order):
    """Transmits compensated with different curves each keep their own curve."""
    path = _make_verasonics_file(
        tmp_path / "differing_curves.mat", [RAMP, STEEP_RAMP], tgc_selections=[1, 2, 2, 1]
    )

    with VerasonicsFile(path) as file:
        gain_curves = file.read_tgc_gain_curves(rcv_order)

    assert gain_curves.shape == (len(rcv_order), N_AX)
    # rcv_order picks receives 2, 0, 3, 1, which select structs 2, 1, 1, 2
    expected = [STEEP_RAMP, RAMP, RAMP, STEEP_RAMP]
    for gain_curve, waveform in zip(gain_curves, expected):
        np.testing.assert_allclose(gain_curve, _expected_curve(waveform))
