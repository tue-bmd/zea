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
from collections import OrderedDict, deque
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from zea.data.convert import us4us
from zea.data.convert.us4us import (
    Us4usConversionError,
    _arrus_stub_for,
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
        """Store the given attributes, as the pickled ARRUS object would carry them."""
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


def make_single_output_recording(
    frame_array, *, context=None, spacing=None, sampling_frequency=None
):
    """A recording with one pipeline output, for exercising one conversion path.

    Args:
        frame_array: The array of a single frame; repeated across ``N_FRAMES``.
        context: ARRUS context to attach, defaults to :func:`make_context`.
        spacing: Optional ARRUS grid attached to the output's data description.
        sampling_frequency: Sampling frequency of the output.
    """
    context = make_context() if context is None else context
    metadata = make_metadata(context, sampling_frequency or BF_SAMPLING_FREQUENCY, spacing=spacing)
    return {"data": [(frame_array.copy(),) for _ in range(N_FRAMES)], "metadata": (metadata,)}


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
    """Both the CLI entries and a JSON object normalize to the same mapping."""
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
    """A malformed mapping is reported before any file is read."""
    with pytest.raises(Us4usConversionError, match=message):
        parse_mapping(mapping)


@pytest.mark.parametrize("mapping", [[], [""], ()])
def test_parse_mapping_without_entries_uses_the_default(mapping):
    """An unset --mapping converts pipeline output 0 as a B-mode image."""
    assert parse_mapping(mapping) == {0: "image"}


def test_parse_mapping_rejects_an_empty_json_object():
    """An explicit but empty mapping would convert nothing at all."""
    with pytest.raises(Us4usConversionError, match="at least one output"):
        parse_mapping("{}")


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
    """--separate-tracks keeps each mapped output in a track of its own."""
    src, _ = recording
    dst = convert_us4us_file(
        src, tmp_path / "out.hdf5", ["0:image", "2:raw_data"], separate_tracks=True
    )
    with File(dst, "r") as file:
        file.validate()
        assert len(file.tracks) == 2
        assert "image" in file.tracks[0].data
        assert "raw_data" in file.tracks[1].data


@pytest.mark.parametrize(
    "data_type, expected_shape, expected_labels",
    [
        # A real-valued beamformed output is RF, so it gets a single channel.
        ("beamformed_data", (N_FRAMES, N_AX_BF, N_TX, 1), ["RF"]),
        # Envelope data has no channel axis at all.
        ("envelope_data", (N_FRAMES, N_AX_BF, N_TX), None),
    ],
)
def test_real_valued_outputs(tmp_path, data_type, expected_shape, expected_labels):
    """A pipeline that already detected the envelope emits real, not IQ, samples."""
    rng = np.random.default_rng(2)
    frame = rng.uniform(0, 1, size=(1, N_TX, N_AX_BF)).astype(np.float32)
    src = write_pickle(tmp_path / "real.pkl", make_single_output_recording(frame))

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", [f"0:{data_type}"])

    with File(dst, "r") as file:
        file.validate()
        stored = getattr(file.tracks[0].data, data_type)
        assert stored.values.shape == expected_shape
        np.testing.assert_allclose(np.squeeze(stored.values[0]), frame[0].T, rtol=1e-6)
        if expected_labels is not None:
            assert list(stored.labels[:]) == expected_labels


def test_complex_envelope_output_is_stored_as_magnitude(tmp_path):
    """An envelope taken from IQ samples is stored as its magnitude."""
    frame = np.array([[[3 + 4j]]], dtype=np.complex64) * np.ones((1, N_TX, N_AX_BF))
    src = write_pickle(tmp_path / "complex.pkl", make_single_output_recording(frame))

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:envelope_data"])

    with File(dst, "r") as file:
        np.testing.assert_allclose(file.tracks[0].data.envelope_data.values[:], 5.0, rtol=1e-6)


def test_uint8_image_is_stored_unchanged(tmp_path):
    """A display-ready image is already 0-255, so zea stores it as uint8 as it is."""
    rng = np.random.default_rng(3)
    frame = rng.integers(0, 256, size=(N_Z, N_X), dtype=np.uint8)
    src = write_pickle(tmp_path / "uint8.pkl", make_single_output_recording(frame))

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        file.validate()
        image = file.tracks[0].data.image.values[:]
        assert image.dtype == np.uint8
        np.testing.assert_array_equal(image[0], frame)


def test_image_nans_are_stored_as_negative_infinity(tmp_path):
    """ARRUS marks pixels outside the scan region with NaN; in dB that is -inf."""
    frame = np.zeros((N_Z, N_X), dtype=np.float32)
    frame[0, 0] = np.nan
    src = write_pickle(tmp_path / "nan.pkl", make_single_output_recording(frame))

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        file.validate()
        image = file.tracks[0].data.image.values[:]
        assert np.isneginf(image[0, 0, 0])
        assert np.isfinite(image[0, 1, 1])


@pytest.mark.parametrize(
    "dtype, expected_n_ch",
    [(np.int16, 1), (np.int32, 1), (np.complex64, 2)],
)
def test_channel_data_dtypes(tmp_path, dtype, expected_n_ch):
    """us4us channel data can be RF (one channel) or already demodulated IQ (two)."""
    frame = np.ones((1, N_TX, N_AX_RAW, N_EL), dtype=dtype)
    src = write_pickle(tmp_path / "raw.pkl", make_single_output_recording(frame))

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:raw_data"])

    with File(dst, "r") as file:
        file.validate()
        raw = file.tracks[0].data.raw_data[:]
        assert raw.shape == (N_FRAMES, N_TX, N_AX_RAW, N_EL, expected_n_ch)
        assert raw.dtype == (np.int16 if dtype == np.int16 else np.float32)


def test_channel_data_with_an_unexpected_shape_is_rejected(tmp_path):
    """Channel data that is not (n_tx, n_ax, n_rx) per frame cannot be laid out."""
    frame = np.zeros((1, N_TX, N_AX_RAW), dtype=np.int16)
    src = write_pickle(tmp_path / "flat.pkl", make_single_output_recording(frame))

    with pytest.raises(Us4usConversionError, match="Expected ARRUS channel data"):
        convert_us4us_file(src, tmp_path / "out.hdf5", ["0:raw_data"])


def test_convert_directory_of_recordings(tmp_path):
    """A source directory converts every recording into ``<dst>/<name>.hdf5``."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for name in ("first", "second"):
        write_pickle(src / f"{name}.pkl", make_recording()[0])

    convert_us4us(SimpleNamespace(src=src, dst=dst, mapping=["0:image"]))

    assert sorted(path.name for path in dst.glob("*.hdf5")) == ["first.hdf5", "second.hdf5"]


def test_single_recording_into_an_existing_directory(recording, tmp_path):
    """Pointing at a directory names the output after the recording."""
    src, _ = recording
    dst = tmp_path / "converted"
    dst.mkdir()

    convert_us4us(SimpleNamespace(src=src, dst=dst, mapping=["0:image"]))

    assert (dst / f"{src.stem}.hdf5").exists()


def test_metadata_option_is_rejected_for_a_directory_of_recordings(tmp_path):
    """One metadata file cannot describe several recordings at once."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for name in ("first", "second"):
        write_pickle(src / f"{name}.pkl", make_recording()[0])

    with pytest.raises(Us4usConversionError, match="--metadata applies to a single recording"):
        convert_us4us(
            SimpleNamespace(src=src, dst=dst, mapping=["0:image"], metadata=tmp_path / "m.pkl")
        )


def test_existing_output_is_only_replaced_with_overwrite(recording, tmp_path):
    """A converted file is never silently replaced by a second run."""
    src, _ = recording
    dst = tmp_path / "out.hdf5"
    convert_us4us_file(src, dst, ["0:image"])
    with pytest.raises(FileExistsError):
        convert_us4us_file(src, dst, ["0:image"])
    convert_us4us_file(src, dst, ["0:image"], overwrite=True)


def test_mapping_beyond_the_available_outputs_is_rejected(recording, tmp_path):
    """Mapping an output index the recording does not have names the valid range."""
    src, _ = recording
    with pytest.raises(Us4usConversionError, match=r"only has 3 output"):
        convert_us4us_file(src, tmp_path / "out.hdf5", ["7:image"])


def test_missing_source_raises(tmp_path):
    """A source path that does not exist fails before anything is written."""
    with pytest.raises(FileNotFoundError, match="Source path not found"):
        convert_us4us(SimpleNamespace(src=tmp_path / "nope.pkl", dst=tmp_path / "out.hdf5"))


def test_directory_without_recordings_raises(tmp_path):
    """An empty source directory is reported rather than silently doing nothing."""
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="No .pkl files"):
        convert_us4us(SimpleNamespace(src=src, dst=tmp_path / "dst"))


# ---------------------------------------------------------------------------
# hf:// sources (the download itself is faked; the real one is covered by
# test_conversion_scripts.py)
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Serve a ``zeahub/pytest`` repo out of ``tmp_path`` instead of the Hub."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def resolve(hf_path, **_):
        """Local stand-in for the download: map hf://zeahub/pytest/<sub> onto the repo."""
        _, subpath = us4us._hf_parse_path(hf_path)
        target = repo / subpath
        if not target.exists():
            raise FileNotFoundError(f"{subpath} not found in zeahub/pytest")
        return target

    monkeypatch.setattr(us4us, "_hf_resolve_path", resolve)
    monkeypatch.setattr(
        us4us,
        "_hf_list_files",
        lambda repo_id, **_: [str(f.relative_to(repo)) for f in repo.rglob("*") if f.is_file()],
    )
    return repo


def test_hf_source_is_downloaded_before_conversion(fake_hub, tmp_path):
    """``hf://`` survives argument parsing and is resolved, not treated as a local path.

    ``Path("hf://...")`` collapses the double slash to ``hf:/``, so the CLI has to keep
    the source as a string all the way to the converter.
    """
    payload, _ = make_recording()
    write_pickle(fake_hub / "us4us_recording.pkl", payload)

    convert_us4us(
        SimpleNamespace(
            src="hf://zeahub/pytest/us4us_recording.pkl",
            dst=tmp_path / "out.hdf5",
            mapping=["0:image"],
        )
    )

    with File(tmp_path / "out.hdf5", "r") as file:
        file.validate()


def test_hf_source_that_does_not_exist_names_the_repo_and_file(fake_hub, tmp_path):
    """A missing hf:// source is reported against the repo, not as a mangled local path.

    ``Path("hf://zeahub/pytest/nope.pkl")`` used to reach the existence check as
    ``hf:/zeahub/pytest/nope.pkl``, which said nothing about the Hub.
    """
    with pytest.raises(FileNotFoundError, match="nope.pkl not found in zeahub/pytest"):
        convert_us4us(SimpleNamespace(src="hf://zeahub/pytest/nope.pkl", dst=tmp_path / "out.hdf5"))


def test_hf_sidecar_metadata_is_found_in_the_repo(fake_hub, tmp_path):
    """Resolving an hf:// file downloads only that file, so the sidecar comes from the repo."""
    payload, _ = make_recording()
    (fake_hub / "us4us").mkdir()
    write_pickle(fake_hub / "us4us" / "recording.pkl", payload["data"])
    write_pickle(fake_hub / "us4us" / "recording_metadata.pkl", {"metadata": payload["metadata"]})

    convert_us4us_file("hf://zeahub/pytest/us4us/recording.pkl", tmp_path / "out.hdf5", ["0:image"])

    with File(tmp_path / "out.hdf5", "r") as file:
        assert file.tracks[0].data.image.values.shape == (N_FRAMES, N_Z, N_X)


def test_hf_metadata_argument_is_resolved(fake_hub, tmp_path):
    """--metadata takes an hf:// path too, and need not sit next to the recording."""
    payload, _ = make_recording()
    write_pickle(fake_hub / "data.pkl", payload["data"])
    write_pickle(fake_hub / "elsewhere.pkl", {"metadata": payload["metadata"]})

    convert_us4us(
        SimpleNamespace(
            src="hf://zeahub/pytest/data.pkl",
            dst=tmp_path / "out.hdf5",
            mapping=["0:image"],
            metadata="hf://zeahub/pytest/elsewhere.pkl",
        )
    )

    assert (tmp_path / "out.hdf5").exists()


# ---------------------------------------------------------------------------
# ARRUS metadata the converter has to cope with
# ---------------------------------------------------------------------------
def test_speed_of_sound_falls_back_to_a_default(tmp_path, caplog):
    """A recording that reports no speed of sound still converts, with a warning."""
    context = make_context()
    context.medium = _arrus("arrus.medium", "MediumDTO", name="unknown")
    context.sequence = _arrus("arrus.ops.imaging", "LinSequence", tx_focus=TX_FOCUS, angles=0.0)
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "no_sos.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    assert "does not report a speed of sound" in caplog.text
    with File(dst, "r") as file:
        np.testing.assert_allclose(file.tracks[0].scan.sound_speed, 1540.0)


def test_probe_geometry_keeps_elevation_positions(tmp_path):
    """A probe model that reports y positions must not be flattened onto y = 0."""
    y_positions = np.linspace(-1e-3, 1e-3, N_EL)
    probe_model = make_probe_model(element_pos_y=y_positions)
    payload, _ = make_recording(context=make_context(probe_model=probe_model))
    src = write_pickle(tmp_path / "elevation.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        np.testing.assert_allclose(file.probe.probe_geometry[:, 1], y_positions, rtol=1e-5)


def test_transmit_apodization_is_taken_from_the_sequence(tmp_path):
    """When ARRUS reports a TX apodization, it is stored instead of a flat aperture."""
    ops = make_ops()
    apodizations = {}
    for i, op in enumerate(ops):
        n_active = int(np.asarray(op.tx.aperture).sum())
        op.tx.apodization = np.linspace(0.2, 1.0, n_active).astype(np.float32)
        apodizations[i] = op.tx.apodization
    payload, _ = make_recording(context=make_context(ops=ops))
    src = write_pickle(tmp_path / "apodized.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        stored = file.tracks[0].scan.tx_apodizations[:]
    active = np.asarray(ops[0].tx.aperture, dtype=bool)
    np.testing.assert_allclose(stored[0][active], apodizations[0], rtol=1e-6)


def test_initial_times_fall_back_to_the_receive_time_range(tmp_path):
    """Without a sample range, the receive time range gives the acquisition start."""
    ops = make_ops()
    for op in ops:
        op.rx.sample_range = None
        op.rx.time_range = (3e-6, 5e-5)
    payload, _ = make_recording(context=make_context(ops=ops))
    src = write_pickle(tmp_path / "time_range.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        np.testing.assert_allclose(file.tracks[0].scan.initial_times[:], 3e-6, rtol=1e-5)


def test_focus_and_angles_come_from_each_transmit(tmp_path):
    """A bare TxRxSequence carries focus and angle per transmit, not on the sequence."""
    ops = make_ops()
    angles = np.linspace(-0.2, 0.2, len(ops)).astype(np.float32)
    for op, angle in zip(ops, angles):
        op.tx.focus = 0.03
        op.tx.angle = float(angle)
    context = make_context(ops=ops)
    context.sequence = _arrus("arrus.ops.us4r", "TxRxSequence", ops=ops)  # no tx_focus/angles
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "per_tx.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        scan = file.tracks[0].scan
        np.testing.assert_allclose(scan.focus_distances[:], np.full(N_TX, 0.03), rtol=1e-6)
        np.testing.assert_allclose(scan.polar_angles[:], angles, rtol=1e-5)


def test_transmits_without_a_focus_are_treated_as_plane_waves(tmp_path):
    """Raw delays without a focus mean the transmit was not focused."""
    ops = make_ops()
    context = make_context(ops=ops)
    context.sequence = _arrus("arrus.ops.us4r", "TxRxSequence", ops=ops)
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "plane.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        scan = file.tracks[0].scan
        assert np.all(np.isinf(scan.focus_distances[:]))
        assert np.all(scan.polar_angles[:] == 0)


def test_one_metadata_entry_serves_a_single_output(tmp_path):
    """gui4us may pickle a lone ConstMetadata rather than a one-entry sequence."""
    payload, _ = make_recording()
    payload["metadata"] = payload["metadata"][0]
    payload["data"] = [frame[:1] for frame in payload["data"]]
    src = write_pickle(tmp_path / "single_meta.pkl", payload)

    assert convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"]).exists()


def test_missing_metadata_entries_reuse_the_last_one(tmp_path, caplog):
    """More outputs than metadata entries is worth a warning, not a failure."""
    payload, _ = make_recording()
    payload["metadata"] = payload["metadata"][:1]
    src = write_pickle(tmp_path / "short_meta.pkl", payload)

    convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image", "1:beamformed_data"])

    assert "only 1 metadata entries" in caplog.text


def test_the_metadata_file_wins_over_embedded_metadata(tmp_path, caplog):
    """--metadata is an explicit choice, so it overrides what the recording carries."""
    payload, _ = make_recording()
    src = write_pickle(tmp_path / "recording.pkl", payload)
    other = make_context(probe_model=make_probe_model(n_el=N_EL, pitch=4e-4))
    metadata_path = write_pickle(
        tmp_path / "metadata.pkl", {"metadata": (make_metadata(other, SAMPLING_FREQUENCY),)}
    )

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"], metadata_path=metadata_path)

    assert "instead of the embedded one" in caplog.text
    with File(dst, "r") as file:
        np.testing.assert_allclose(file.probe.element_width, 4e-4, rtol=1e-6)


# ---------------------------------------------------------------------------
# Recordings this converter cannot (yet) handle: the error must say why
# ---------------------------------------------------------------------------
def test_pickle_without_data_key_is_rejected(tmp_path):
    """A capture dict without frames names the key that is missing."""
    src = write_pickle(tmp_path / "bad.pkl", {"metadata": ()})
    with pytest.raises(Us4usConversionError, match="no 'data' key"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_pickle_that_is_not_a_recording_is_rejected(tmp_path):
    """A pickle of something else entirely says what was expected instead."""
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
    """A [data, metadata] two-element pickle is understood as a recording."""
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
    with pytest.raises(Us4usConversionError, match="did not store it before 0.12.0"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_a_single_rca_probe_is_rejected(tmp_path):
    """An RCA acquisition may report one probe; its model still gives it away."""
    rca_model = _arrus("arrus.devices.probe", "ProbeModelRca", n_elements=N_EL)
    probes = [_arrus("arrus.devices.probe", "ProbeDTO", model=rca_model)]
    payload, _ = make_recording(context=make_context(probes=probes))
    src = write_pickle(tmp_path / "rca_single.pkl", payload)

    with pytest.raises(Us4usConversionError, match="row-column"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_scan_line_coordinates_need_a_sampling_frequency(tmp_path):
    """Without an output sampling frequency there is no depth axis to place pixels on."""
    rng = np.random.default_rng(5)
    frame = rng.normal(size=(1, N_TX, N_AX_BF)).astype(np.complex64)
    payload = make_single_output_recording(frame)
    payload["metadata"][0]._data_char.sampling_frequency = None
    src = write_pickle(tmp_path / "no_fs.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:beamformed_data"])

    with File(dst, "r") as file:
        assert "coordinates" not in file.tracks[0].data.beamformed_data


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


@pytest.mark.parametrize(
    "payload, message",
    [
        # A dict is expected; a sequence of something else is not a recording.
        ([{"a": 1}, {"b": 2}], "expected a dict with 'data' and 'metadata'"),
        ({"data": [], "metadata": ()}, "non-empty list of frames"),
        ({"data": [(np.zeros(2),)], "metadata": {"a": 1}}, "expected one ``ConstMetadata``"),
        ({"data": [(np.zeros(2),)], "metadata": ()}, "expected one ``ConstMetadata``"),
    ],
)
def test_malformed_recordings_are_rejected(tmp_path, payload, message):
    """Whatever the pickle holds, the error says what was expected and what was found."""
    src = write_pickle(tmp_path / "bad.pkl", payload)
    with pytest.raises(Us4usConversionError, match=message):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_frames_holding_different_output_counts_are_rejected(tmp_path):
    """Every frame must carry the same pipeline outputs."""
    payload, _ = make_recording()
    payload["data"][1] = payload["data"][1][:2]
    src = write_pickle(tmp_path / "uneven.pkl", payload)
    with pytest.raises(Us4usConversionError, match="same number of pipeline outputs"):
        convert_us4us_file(src, tmp_path / "out.hdf5")


@pytest.mark.parametrize(
    "break_context, message",
    [
        (lambda context: setattr(context, "_context", None), "no acquisition context"),
        (lambda context: setattr(context.raw_sequence.ops[0], "tx", None), "no 'tx' operation"),
        (lambda context: setattr(context.raw_sequence.ops[0], "rx", None), "no 'rx' operation"),
        (lambda context: setattr(context.device, "probe", None), "context.device.probe is missing"),
        (lambda context: setattr(context.device, "probe", []), "context.device.probe is empty"),
        (lambda context: setattr(context.device.probe[0], "model", None), "no probe model"),
        (
            lambda context: setattr(context.device.probe[0].model, "element_pos_x", None),
            "has no 'element_pos_x'",
        ),
    ],
)
def test_incomplete_arrus_metadata_names_what_is_missing(tmp_path, break_context, message):
    """Each structure the conversion relies on is reported by name when absent."""
    payload, _ = make_recording()
    for entry in payload["metadata"]:
        # The context is shared between the entries; the metadata wrappers are not.
        break_context(entry if message == "no acquisition context" else entry._context)
    src = write_pickle(tmp_path / "incomplete.pkl", payload)

    with pytest.raises(Us4usConversionError, match=message):
        convert_us4us_file(src, tmp_path / "out.hdf5")


def test_a_single_probe_outside_a_list_is_accepted(tmp_path):
    """ARRUS reports one probe as a bare DTO in some versions, a list in others."""
    context = make_context()
    context.device.probe = context.device.probe[0]
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "bare_probe.pkl", payload)

    assert convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"]).exists()


def test_an_untested_arrus_version_warns_but_converts(tmp_path, caplog):
    """A newer ARRUS is converted best-effort rather than refused."""
    payload, _ = make_recording()
    for entry in payload["metadata"]:
        entry.version = "0.15.0"
    src = write_pickle(tmp_path / "newer.pkl", payload)

    convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    assert "outside the tested releases" in caplog.text


def test_an_untested_arrus_version_is_named_in_conversion_errors(tmp_path):
    """A failure on a newer ARRUS says the version is unsupported, not just what is missing.

    The attribute that happens to be missing is the symptom; the ARRUS release that
    moved it is the cause, and that is what the user has to act on.
    """
    payload, _ = make_recording()
    for entry in payload["metadata"]:
        entry.version = "0.15.0"
        entry._context.device.probe[0].model.element_pos_x = None
    src = write_pickle(tmp_path / "newer_broken.pkl", payload)

    with pytest.raises(Us4usConversionError, match=r"ARRUS 0\.15\.0, which this converter"):
        convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])


def test_missing_optional_arrus_metadata_is_reported_together(tmp_path, caplog):
    """Fields that silently fall back are named in one warning, with the ARRUS version.

    Each of these has a default, so the conversion succeeds either way; the warning is
    the only sign that an ARRUS release moved something the converter reads.
    """
    payload, _ = make_recording()
    for entry in payload["metadata"]:
        entry._context.raw_sequence.ops[0].pri = None
        entry._context.device.probe[0].model.pitch = None
    src = write_pickle(tmp_path / "sparse_metadata.pkl", payload)

    convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    assert "ops[0].pri (falls back for scan.time_to_next_transmit)" in caplog.text
    assert "probe.model.pitch (falls back for probe.element_width)" in caplog.text
    assert "reports ARRUS 0.13.0, which this converter supports" in caplog.text


def test_complete_arrus_metadata_reports_no_divergence(recording, tmp_path, caplog):
    """The healthy recording must not warn, or the warning means nothing when it fires."""
    src, _ = recording

    convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    assert "does not carry every field" not in caplog.text


def test_grid_spacing_with_too_few_axes_is_left_out(tmp_path):
    """A grid that describes fewer axes than the data cannot place its pixels."""
    payload, _ = make_recording()
    payload["metadata"][0]._data_char.spacing = _arrus(
        "arrus.metadata", "Grid", coordinates=(np.linspace(0, 0.04, N_Z),)
    )
    src = write_pickle(tmp_path / "odd_grid.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        assert "coordinates" not in file.tracks[0].data.image


def test_volumetric_output_is_converted_without_coordinates(tmp_path):
    """Only 2-D grids are mapped; a volume converts, but keeps no coordinate grid."""
    rng = np.random.default_rng(4)
    frame = rng.uniform(-60, 0, size=(N_Z, N_X, 2)).astype(np.float32)
    spacing = _arrus(
        "arrus.metadata",
        "Grid",
        coordinates=(np.linspace(0, 0.04, N_Z), np.linspace(-0.01, 0.01, N_X), np.zeros(2)),
    )
    src = write_pickle(
        tmp_path / "volume.pkl", make_single_output_recording(frame, spacing=spacing)
    )

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        file.validate()
        assert file.tracks[0].data.image.values.shape == (N_FRAMES, N_Z, N_X, 2)
        assert "coordinates" not in file.tracks[0].data.image


def test_angles_that_match_neither_one_nor_every_transmit_fall_back(tmp_path):
    """A sequence angle list of the wrong length is per-transmit data we cannot trust."""
    context = make_context()
    context.sequence.angles = [0.1, 0.2, 0.3]  # N_TX is 8
    payload, _ = make_recording(context=context)
    src = write_pickle(tmp_path / "odd_angles.pkl", payload)

    dst = convert_us4us_file(src, tmp_path / "out.hdf5", ["0:image"])

    with File(dst, "r") as file:
        # Falls back to the per-transmit values, which this sequence does not set.
        assert np.all(file.tracks[0].scan.polar_angles[:] == 0)


def test_frames_with_changing_shapes_are_rejected(tmp_path):
    """zea files need one shape for all frames, so a ragged recording is refused."""
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
    """Only arrus classes and the listed data constructors may be resolved."""
    src = write_pickle(tmp_path / "hostile.pkl", {"data": [], "metadata": (datetime.now(),)})
    with pytest.raises(pickle.UnpicklingError, match="Refusing to load datetime.datetime"):
        load_us4us_pickle(src)


def _pickle_that_calls(module: str, name: str, argument: str) -> bytes:
    """Hand-build a pickle whose REDUCE opcode calls ``module.name(argument)``.

    Written byte-wise rather than through ``pickle.dumps`` so the test never has
    to import the module it is trying to smuggle in.
    """
    encoded = argument.encode()
    return (
        b"\x80\x02"  # protocol 2
        + f"c{module}\n{name}\n".encode()  # GLOBAL module name
        + b"("  # MARK
        + b"X"
        + len(encoded).to_bytes(4, "little")
        + encoded  # BINUNICODE argument
        + b"t"  # TUPLE
        + b"R"  # REDUCE: call the global with it
        + b"."  # STOP
    )


@pytest.mark.parametrize(
    "module, name",
    [
        # Shell execution reachable from inside numpy itself, which is why the
        # allowlist is by (module, name) and not by module.
        ("numpy.distutils.exec_command", "exec_command"),
        ("os", "system"),
        ("subprocess", "check_output"),
        ("builtins", "eval"),
    ],
)
def test_loader_refuses_globals_that_execute(tmp_path, module, name):
    """``find_class`` must refuse before a REDUCE opcode can call the global."""
    src = tmp_path / "hostile.pkl"
    src.write_bytes(_pickle_that_calls(module, name, "echo pwned"))

    with pytest.raises(pickle.UnpicklingError, match=f"Refusing to load {module}.{name}"):
        load_us4us_pickle(src)


@pytest.mark.parametrize("protocol", [2, 4, 5])
def test_loader_reads_the_numpy_payloads_a_recording_holds(tmp_path, protocol):
    """The allowlist must still cover everything a real recording pickles."""
    payload = {
        "arrays": [
            np.zeros((2, 3), dtype=np.int16),
            np.zeros(4, dtype=np.complex64),
            np.array(["I", "Q"], dtype=np.str_),
            np.zeros((3, 4), dtype=np.float32, order="F"),
            np.zeros(3, dtype=bool),
            np.array([b"raw"]),
        ],
        "scalars": [np.float32(1.5), np.int64(3), np.float64(np.inf)],
        "dtype": np.dtype("float32"),
        "frames": deque([np.zeros(2, dtype=np.int16)]),
        "ordered": OrderedDict(sampling_frequency=np.float32(65e6)),
    }
    src = tmp_path / "payloads.pkl"
    with open(src, "wb") as file:
        pickle.dump(payload, file, protocol=protocol)

    loaded = load_us4us_pickle(src)

    assert loaded["arrays"][0].shape == (2, 3)
    assert loaded["arrays"][1].dtype == np.complex64
    assert list(loaded["arrays"][2]) == ["I", "Q"]
    assert loaded["scalars"][0] == np.float32(1.5)
    assert loaded["dtype"] == np.dtype("float32")
    assert len(loaded["frames"]) == 1
    assert loaded["ordered"]["sampling_frequency"] == np.float32(65e6)


def test_arrus_stubs_restore_state_and_name_their_class():
    """Stubs stand in for arrus objects: they keep the state and say what they replace."""
    stub_class = _arrus_stub_for("arrus.metadata", "ConstMetadata")
    stub = stub_class("positional", keyword=1)

    assert stub.arrus_class == "arrus.metadata.ConstMetadata"
    assert repr(stub) == "<arrus.metadata.ConstMetadata stub>"
    # Frozen dataclasses restore __dict__; classes with __slots__ pass a second mapping.
    stub.__setstate__(({"sampling_frequency": 65e6}, {"version": "0.13.0"}))
    assert stub.sampling_frequency == 65e6
    assert stub.version == "0.13.0"


def test_loader_keeps_arrus_attributes_without_arrus_installed(recording):
    """ARRUS objects come back as attribute-preserving stubs, named after their class."""
    src, _ = recording
    payload = load_us4us_pickle(src)
    context = payload["metadata"][0]._context
    assert type(context).arrus_class == "arrus.metadata.FrameAcquisitionContext"
    assert len(context.raw_sequence.ops) == N_TX
    np.testing.assert_allclose(context.medium.speed_of_sound, SOUND_SPEED)
