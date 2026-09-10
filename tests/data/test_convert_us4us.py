"""Tests for the us4us (ARRUS + gui4us) conversion script.

The recordings us4us systems produce are pickled ``arrus`` objects. Rather than
downloading one, these tests build a small synthetic recording with stand-in
``arrus`` classes (same module names, same attributes), which keeps the tests
fast, offline and able to cover the error paths of the converter. The
end-to-end test against a real us4us recording lives in
``test_conversion_scripts.py`` (``test_conversion_script[us4us]``).
"""

import pickle
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from zea.data.convert.us4us import (
    Us4usConversionError,
    convert_us4us,
    convert_us4us_file,
    load_us4us_pickle,
    parse_mapping,
)
from zea.data.file import File

N_FRAMES = 2
N_TX = 8
N_EL = 64
N_RX = 32
N_AX_RAW = 256
N_AX_BF = 128
N_Z, N_X = 40, 24
SOUND_SPEED = 1500.0
SAMPLING_FREQUENCY = 65e6
BF_SAMPLING_FREQUENCY = SAMPLING_FREQUENCY / 4
CENTER_FREQUENCY = 6e6
TX_FOCUS = 0.02
PRI = 2e-4
PITCH = 3e-4

_FAKE_ARRUS_MODULES = (
    "arrus",
    "arrus.metadata",
    "arrus.devices",
    "arrus.devices.probe",
    "arrus.ops",
    "arrus.ops.us4r",
    "arrus.ops.imaging",
    "arrus.medium",
)


class _ArrusLike:
    """Stand-in for an ``arrus`` object: keeps whatever attributes it is given."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


@pytest.fixture(scope="module", autouse=True)
def fake_arrus_modules():
    """Register fake ``arrus.*`` modules so the synthetic recording can be pickled.

    ``pickle`` resolves a class through ``sys.modules[cls.__module__]``, so the
    stand-in classes must live in modules named like the real ARRUS ones. The
    converter never imports ``arrus``; it only matches on those module names.
    """
    created = {}
    for name in _FAKE_ARRUS_MODULES:
        module = types.ModuleType(name)
        created[name] = module
        sys.modules[name] = module
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(sys.modules[parent], child, module)
    yield created
    for name in reversed(_FAKE_ARRUS_MODULES):
        sys.modules.pop(name, None)


def _arrus_class(module: str, class_name: str, /) -> type:
    """Return a stand-in ARRUS class, registered once on its (fake) module.

    ``pickle`` checks that ``sys.modules[module].<name>`` *is* the class of the
    object being pickled, so the classes have to be created only once.
    """
    cls = getattr(sys.modules[module], class_name, None)
    if cls is None:
        cls = type(class_name, (_ArrusLike,), {"__module__": module})
        setattr(sys.modules[module], class_name, cls)
    return cls


def _arrus(module: str, class_name: str, /, **attributes):
    """Instantiate a stand-in ARRUS object."""
    return _arrus_class(module, class_name)(**attributes)


def make_probe_model(n_el: int = N_EL, **overrides):
    """A single-axis linear probe model, as ARRUS reports it."""
    attributes = {
        "n_elements": n_el,
        "pitch": PITCH,
        "curvature_radius": 0.0,
        "element_pos_x": (np.arange(n_el) - (n_el - 1) / 2) * PITCH,
        "element_pos_z": np.zeros(n_el),
        "model_id": _arrus(
            "arrus.devices.probe", "ProbeModelId", name="l7-4", manufacturer="us4us"
        ),
        "lens": _arrus("arrus.devices.probe", "Lens", thickness=1e-4, speed_of_sound=1000.0),
    }
    attributes.update(overrides)
    return _arrus("arrus.devices.probe", "ProbeModel", **attributes)


def make_ops(n_tx: int = N_TX, n_el: int = N_EL, n_rx: int = N_RX):
    """A classic scan-line (LIN) sequence: a sliding aperture across the probe.

    The first transmit sits at the probe edge, so its receive aperture is padded
    on the left -- which is exactly the layout the raw-data remapping has to undo.
    """
    ops = []
    for i in range(n_tx):
        aperture = np.zeros(n_el, dtype=bool)
        if i == 0:
            aperture[: n_rx // 2] = True  # edge transmit: half the aperture is padding
            padding = (n_rx // 2, 0)
        else:
            start = min(i * n_rx // 2, n_el - n_rx)
            aperture[start : start + n_rx] = True
            padding = (0, 0)
        n_active = int(aperture.sum())
        tx = _arrus(
            "arrus.ops.us4r",
            "Tx",
            aperture=aperture,
            delays=np.linspace(0, 1e-6, n_active),
            excitation=_arrus(
                "arrus.ops.us4r", "Pulse", center_frequency=CENTER_FREQUENCY, n_periods=2
            ),
        )
        rx = _arrus(
            "arrus.ops.us4r",
            "Rx",
            aperture=aperture,
            padding=padding,
            sample_range=(0, N_AX_RAW),
            time_range=(0.0, N_AX_RAW / SAMPLING_FREQUENCY),
        )
        ops.append(_arrus("arrus.ops.us4r", "TxRx", tx=tx, rx=rx, pri=PRI))
    return ops


def make_context(probe_model=None, ops=None, probes=None):
    """A ``FrameAcquisitionContext`` holding the probe, sequence and medium."""
    ops = make_ops() if ops is None else ops
    if probes is None:
        model = make_probe_model() if probe_model is None else probe_model
        probes = [_arrus("arrus.devices.probe", "ProbeDTO", model=model)]
    return _arrus(
        "arrus.metadata",
        "FrameAcquisitionContext",
        device=_arrus("arrus.devices.probe", "UltrasoundDeviceDTO", probe=probes),
        sequence=_arrus(
            "arrus.ops.imaging",
            "LinSequence",
            tx_focus=TX_FOCUS,
            angles=0.0,
            speed_of_sound=SOUND_SPEED,
        ),
        raw_sequence=_arrus("arrus.ops.us4r", "TxRxSequence", ops=ops),
        medium=_arrus("arrus.medium", "MediumDTO", name="water", speed_of_sound=SOUND_SPEED),
    )


def make_metadata(context, sampling_frequency, spacing=None, version="0.13.0"):
    """A ``ConstMetadata`` entry describing one pipeline output."""
    return _arrus(
        "arrus.metadata",
        "ConstMetadata",
        _context=context,
        _data_char=_arrus(
            "arrus.metadata",
            "EchoDataDescription",
            sampling_frequency=sampling_frequency,
            spacing=spacing,
        ),
        version=version,
    )


def make_recording(context=None, n_frames: int = N_FRAMES, seed: int = 0):
    """Build a three-output us4us recording: image, beamformed IQ and channel data.

    Returns:
        tuple: ``(payload, frames)`` where ``payload`` is the dict gui4us pickles
        and ``frames`` the raw per-frame arrays, for use in assertions.
    """
    rng = np.random.default_rng(seed)
    context = make_context() if context is None else context

    z_grid = np.linspace(0.005, 0.045, N_Z)
    x_grid = np.linspace(-0.01, 0.01, N_X)
    spacing = _arrus("arrus.metadata", "Grid", coordinates=(z_grid, x_grid))

    frames = []
    for _ in range(n_frames):
        # ARRUS B-mode output: log-compressed but not normalized (so > 0 dB).
        image = rng.uniform(-20.0, 60.0, size=(N_Z, N_X)).astype(np.float32)
        beamformed = (
            rng.normal(size=(1, N_TX, N_AX_BF)) + 1j * rng.normal(size=(1, N_TX, N_AX_BF))
        ).astype(np.complex64)
        raw = rng.integers(-2048, 2048, size=(1, N_TX, N_AX_RAW, N_RX), dtype=np.int16)
        frames.append((image, beamformed, raw))

    payload = {
        "data": frames,
        "metadata": (
            make_metadata(context, BF_SAMPLING_FREQUENCY, spacing=spacing),
            make_metadata(context, BF_SAMPLING_FREQUENCY),
            make_metadata(context, SAMPLING_FREQUENCY),
        ),
    }
    return payload, frames


def write_pickle(path, payload):
    """Pickle ``payload`` to ``path`` and return the path."""
    with open(path, "wb") as file:
        pickle.dump(payload, file)
    return path


@pytest.fixture
def recording(tmp_path):
    """A pickled synthetic us4us recording plus its source frames."""
    payload, frames = make_recording()
    return write_pickle(tmp_path / "recording.pkl", payload), frames


# ---------------------------------------------------------------------------
# Mapping parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mapping, expected",
    [
        (None, {0: "image"}),
        (["0:image"], {0: "image"}),
        (["1:raw_data", "0:image"], {1: "raw_data", 0: "image"}),
        ('{"0": "image", "2": "raw_data"}', {0: "image", 2: "raw_data"}),
        ({0: "image"}, {0: "image"}),
    ],
)
def test_parse_mapping_accepts_cli_and_json_forms(mapping, expected):
    assert parse_mapping(mapping) == expected


@pytest.mark.parametrize(
    "mapping, message",
    [
        (["image"], "Expected '<output index>:<data type>'"),
        (["-1:image"], "non-negative"),
        (["0:image", "0:raw_data"], "more than once"),
        (["0:image", "1:image"], "multiple pipeline outputs"),
        (["0:doppler"], "Unsupported zea data type"),
        (["a:image"], "must be integers"),
        ("{not json}", "Invalid JSON"),
        ('["image"]', "must be an object"),
    ],
)
def test_parse_mapping_rejects_invalid_specs(mapping, message):
    with pytest.raises(Us4usConversionError, match=message):
        parse_mapping(mapping)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def test_convert_writes_all_outputs_to_one_track(recording, tmp_path):
    """All mapped outputs land in a single, valid zea file with sensible metadata."""
    src, frames = recording
    dst = convert_us4us_file(
        src, tmp_path / "out.hdf5", ["0:image", "1:beamformed_data", "2:raw_data"]
    )

    with File(dst, "r") as file:
        file.validate()
        data = file.tracks[0].data
        assert set(data.keys()) >= {"image", "beamformed_data", "raw_data"}

        image = data.image.values[:]
        assert image.shape == (N_FRAMES, N_Z, N_X)
        assert image.max() <= 0, "zea images must be in dB with a maximum of 0"
        # The converter only shifts the image, so differences are preserved.
        np.testing.assert_allclose(
            image[0] - image[0].max(), frames[0][0] - frames[0][0].max(), atol=1e-4
        )
        # Coordinates come from the ARRUS grid spacing (x, y, z) in metres.
        coordinates = data.image.coordinates[:]
        assert coordinates.shape == (N_Z, N_X, 3)
        np.testing.assert_allclose(coordinates[:, 0, 2], np.linspace(0.005, 0.045, N_Z), rtol=1e-5)
        np.testing.assert_allclose(coordinates[0, :, 0], np.linspace(-0.01, 0.01, N_X), rtol=1e-5)

        beamformed = data.beamformed_data.values[:]
        assert beamformed.shape == (N_FRAMES, N_AX_BF, N_TX, 2), "expected (frames, z, x, IQ)"
        np.testing.assert_allclose(beamformed[0, :, :, 0], frames[0][1][0].T.real, rtol=1e-5)
        np.testing.assert_allclose(beamformed[0, :, :, 1], frames[0][1][0].T.imag, rtol=1e-5)
        assert list(data.beamformed_data.labels[:]) == ["I", "Q"]
        # Without an ARRUS grid, coordinates are derived per scan line.
        bf_coordinates = data.beamformed_data.coordinates[:]
        assert bf_coordinates.shape == (N_AX_BF, N_TX, 3)
        depth_step = SOUND_SPEED / (2 * BF_SAMPLING_FREQUENCY)
        np.testing.assert_allclose(bf_coordinates[1, 0, 2] - bf_coordinates[0, 0, 2], depth_step)

        raw = data.raw_data[:]
        assert raw.shape == (N_FRAMES, N_TX, N_AX_RAW, N_EL, 1)

        scan = file.tracks[0].scan
        assert scan.t0_delays.shape == (N_TX, N_EL)
        assert np.all(scan.t0_delays >= 0)
        np.testing.assert_allclose(scan.focus_distances, np.full(N_TX, TX_FOCUS), rtol=1e-6)
        np.testing.assert_allclose(scan.sound_speed, SOUND_SPEED)
        np.testing.assert_allclose(scan.center_frequency, CENTER_FREQUENCY)
        np.testing.assert_allclose(scan.sampling_frequency, SAMPLING_FREQUENCY)
        assert scan.time_to_next_transmit.shape == (N_FRAMES, N_TX)

        probe = file.probe
        assert probe.probe_geometry.shape == (N_EL, 3)
        np.testing.assert_allclose(probe.element_width, PITCH, rtol=1e-6)
        assert probe.name == "l7-4"
        assert probe.type == "linear"


def test_raw_data_is_scattered_back_onto_the_full_aperture(recording, tmp_path):
    """Receive-aperture padding is undone: samples land on the elements that recorded them."""
    src, frames = recording
    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["2:raw_data"])

    with File(dst, "r") as file:
        raw = file.tracks[0].data.raw_data[:]

    source_frame = frames[0][2][0]  # (n_tx, n_ax, n_rx)
    # Transmit 0 sits at the probe edge: 16 channels of padding, then elements 0-15.
    np.testing.assert_array_equal(raw[0, 0, :, : N_RX // 2, 0], source_frame[0, :, N_RX // 2 :])
    assert np.all(raw[0, 0, :, N_RX // 2 :, 0] == 0), "inactive elements must stay zero"
    # Transmit 1 receives on elements 16-47 without padding.
    first = N_RX // 2
    np.testing.assert_array_equal(raw[0, 1, :, first : first + N_RX, 0], source_frame[1])
    assert np.all(raw[0, 1, :, :first, 0] == 0)
    assert np.all(raw[0, 1, :, first + N_RX :, 0] == 0)


def test_image_without_a_grid_is_stored_as_depth_by_transmit(tmp_path):
    """An un-scan-converted B-mode (one column per transmit) is transposed to (z, x)."""
    rng = np.random.default_rng(1)
    payload, _ = make_recording()
    payload["metadata"][0]._data_char.spacing = None
    for i, frame in enumerate(payload["data"]):
        scanline_image = rng.uniform(-60.0, 0.0, size=(N_TX, N_AX_BF)).astype(np.float32)
        payload["data"][i] = (scanline_image,) + frame[1:]
    src = write_pickle(tmp_path / "scanlines.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])
    with File(dst, "r") as file:
        file.validate()
        image = file.tracks[0].data.image
        assert image.values.shape == (N_FRAMES, N_AX_BF, N_TX)
        assert image.coordinates.shape == (N_AX_BF, N_TX, 3)


def test_separate_tracks_writes_one_track_per_output(recording, tmp_path):
    src, _ = recording
    dst = convert_us4us_file(
        src, tmp_path / "out.hdf5", ["0:image", "2:raw_data"], separate_tracks=True
    )
    with File(dst, "r") as file:
        file.validate()
        assert len(file.tracks) == 2
        assert "image" in file.tracks[0].data
        assert "raw_data" in file.tracks[1].data


def test_convert_directory_of_recordings(tmp_path):
    """A source directory converts every recording into ``<dst>/<name>.hdf5``."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for name in ("first", "second"):
        write_pickle(src / f"{name}.pkl", make_recording()[0])

    convert_us4us(SimpleNamespace(src=src, dst=dst, mapping=["0:image"]))

    assert sorted(path.name for path in dst.glob("*.hdf5")) == ["first.hdf5", "second.hdf5"]


def test_existing_output_is_only_replaced_with_overwrite(recording, tmp_path):
    src, _ = recording
    dst = tmp_path / "out.hdf5"
    convert_us4us_file(src, dst, ["0:image"])
    with pytest.raises(FileExistsError):
        convert_us4us_file(src, dst, ["0:image"])
    convert_us4us_file(src, dst, ["0:image"], overwrite=True)


def test_mapping_beyond_the_available_outputs_is_rejected(recording, tmp_path):
    src, _ = recording
    with pytest.raises(Us4usConversionError, match=r"only has 3 output"):
        convert_us4us_file(src, tmp_path / "out.hdf5", ["7:image"])


def test_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Source path not found"):
        convert_us4us(SimpleNamespace(src=tmp_path / "nope.pkl", dst=tmp_path / "out.hdf5"))


def test_directory_without_recordings_raises(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="No .pkl files"):
        convert_us4us(SimpleNamespace(src=src, dst=tmp_path / "dst"))


# ---------------------------------------------------------------------------
# Recordings this converter cannot (yet) handle: the error must say why
# ---------------------------------------------------------------------------
def test_pickle_without_data_key_is_rejected(tmp_path):
    src = write_pickle(tmp_path / "bad.pkl", {"metadata": ()})
    with pytest.raises(Us4usConversionError, match="no 'data' key"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_pickle_that_is_not_a_recording_is_rejected(tmp_path):
    src = write_pickle(tmp_path / "bad.pkl", "just a string")
    with pytest.raises(Us4usConversionError, match="expected a dict with 'data' and 'metadata'"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_recording_without_metadata_points_at_the_metadata_option(tmp_path):
    """A bare list of frames (data and metadata saved separately) explains the fix."""
    _, frames = make_recording()
    src = write_pickle(tmp_path / "data.pkl", frames)
    with pytest.raises(Us4usConversionError, match=r"--metadata"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


@pytest.mark.parametrize("sidecar", [True, False])
def test_metadata_from_a_separate_file(tmp_path, sidecar):
    """Data-only recordings convert when the metadata pickle is supplied."""
    payload, _ = make_recording()
    src = write_pickle(tmp_path / "data.pkl", payload["data"])
    metadata_name = "data_metadata.pkl" if sidecar else "elsewhere.pkl"
    metadata_path = write_pickle(tmp_path / metadata_name, {"metadata": payload["metadata"]})

    dst = convert_us4us_file(
        src,
        tmp_path / "out.hdf5",
        ["0:image"],
        metadata_path=None if sidecar else metadata_path,
    )
    with File(dst, "r") as file:
        file.validate()
        assert file.tracks[0].data.image.values.shape == (N_FRAMES, N_Z, N_X)


def test_data_and_metadata_stored_as_a_pair(tmp_path):
    payload, _ = make_recording()
    src = write_pickle(tmp_path / "pair.pkl", [payload["data"], payload["metadata"]])
    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])
    assert dst.exists()


def test_recording_from_an_older_arrus_is_rejected_with_a_hint(tmp_path):
    """Metadata without a TX/RX sequence names the ARRUS version that introduced it."""
    payload, _ = make_recording()
    context = payload["metadata"][0]._context
    context.raw_sequence = _arrus("arrus.ops.us4r", "TxRxSequence", ops=[])
    src = write_pickle(tmp_path / "old.pkl", payload)
    with pytest.raises(Us4usConversionError, match="older than 0.12.0"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_multi_probe_recording_is_rejected(tmp_path):
    """RCA / matrix acquisitions use several probes, which zea's scan cannot describe."""
    probes = [
        _arrus("arrus.devices.probe", "ProbeDTO", model=make_probe_model()),
        _arrus("arrus.devices.probe", "ProbeDTO", model=make_probe_model()),
    ]
    payload, _ = make_recording(context=make_context(probes=probes))
    src = write_pickle(tmp_path / "rca.pkl", payload)
    with pytest.raises(Us4usConversionError, match="row-column addressed"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_transmit_that_addresses_another_element_grid_is_rejected(tmp_path):
    """A sequence whose apertures do not match the probe model cannot be converted."""
    context = make_context(probe_model=make_probe_model(n_el=N_EL), ops=make_ops(n_el=2 * N_EL))
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "mismatch.pkl", payload)
    with pytest.raises(Us4usConversionError, match="while the probe model reports"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_frames_with_changing_shapes_are_rejected(tmp_path):
    payload, _ = make_recording()
    payload["data"][1] = (payload["data"][1][0][:, :-1],) + payload["data"][1][1:]
    src = write_pickle(tmp_path / "ragged.pkl", payload)
    with pytest.raises(Us4usConversionError, match="changes shape between frames"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_unknown_grid_spacing_is_ignored(tmp_path):
    """A spacing that does not match the data shape is dropped, not written."""
    payload, _ = make_recording()
    payload["metadata"][0]._data_char.spacing = _arrus(
        "arrus.metadata", "Grid", coordinates=(np.arange(3), np.arange(5))
    )
    src = write_pickle(tmp_path / "odd_spacing.pkl", payload)
    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])
    with File(dst, "r") as file:
        assert "coordinates" not in file.tracks[0].data.image


# ---------------------------------------------------------------------------
# Pickle trust boundary
# ---------------------------------------------------------------------------
def test_loader_refuses_classes_outside_the_allowlist(tmp_path):
    """Only arrus/numpy/collections globals may be resolved while unpickling."""
    src = write_pickle(tmp_path / "hostile.pkl", {"data": [], "metadata": (datetime.now(),)})
    with pytest.raises(pickle.UnpicklingError, match="not in the us4us converter allowlist"):
        load_us4us_pickle(src)


def test_loader_keeps_arrus_attributes_without_arrus_installed(recording):
    """ARRUS objects come back as attribute-preserving stubs, named after their class."""
    src, _ = recording
    payload = load_us4us_pickle(src)
    context = payload["metadata"][0]._context
    assert type(context).arrus_class == "arrus.metadata.FrameAcquisitionContext"
    assert len(context.raw_sequence.ops) == N_TX
    np.testing.assert_allclose(context.medium.speed_of_sound, SOUND_SPEED)
