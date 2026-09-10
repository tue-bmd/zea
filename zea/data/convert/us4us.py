"""Convert us4us (ARRUS + gui4us) recordings to the zea format.

`us4us <https://us4us.eu>`__ systems (us4R, us4R-lite) are programmed with the
`ARRUS <https://github.com/us4useu/arrus>`__ Python API, usually through
`gui4us <https://github.com/us4useu/gui4us>`__. A captured acquisition is stored
as a Python *pickle* holding the raw pipeline outputs plus the ARRUS metadata
that describes the probe, the TX/RX sequence and the medium:

.. code-block:: text

    recording.pkl
    ├── "data":     list of N frames, each a tuple of M numpy arrays
    │               (one array per pipeline output, e.g. image, beamformed IQ,
    │               raw channel data)
    └── "metadata": sequence of M ``arrus.metadata.ConstMetadata`` objects,
                    one per pipeline output

This module maps those outputs onto zea data types and writes a single zea
HDF5 file per recording. ``arrus`` does **not** need to be installed: every
``arrus`` class in the pickle is replaced by a lightweight stub that keeps the
original attribute values (see :func:`load_us4us_pickle`).

Usage
-----

.. code-block:: shell

    # Single file, default mapping (pipeline output 0 is a B-mode image)
    zea convert us4us recording.pkl recording.hdf5

    # Map several pipeline outputs at once
    zea convert us4us recording.pkl recording.hdf5 \\
        --mapping 0:image 1:beamformed_data 2:raw_data

    # Whole directory: every <name>.pkl becomes <dst>/<name>.hdf5
    zea convert us4us ./recordings ./converted --mapping 0:image

``--mapping`` entries are ``<pipeline output index>:<zea data type>`` pairs;
a JSON object (``--mapping '{"0": "image", "1": "raw_data"}'``) is accepted as
well. Supported zea data types are listed in :data:`SUPPORTED_DATA_TYPES`.

ARRUS metadata used
-------------------

The ARRUS acquisition context is mapped onto the zea ``scan`` and ``probe`` groups
as follows, with ``ops`` short for ``context.raw_sequence.ops`` (one entry per
transmit) and ``model`` for ``context.device.probe[0].model``:

- ``model.element_pos_x`` / ``element_pos_y`` / ``element_pos_z`` -> ``probe.probe_geometry``
- ``model.pitch`` -> ``probe.element_width`` (ARRUS does not report the element
  width; the pitch approximates it, ignoring the kerf)
- ``model.curvature_radius`` -> ``probe.type``, ``model.model_id`` -> ``probe.name``,
  ``model.lens`` -> ``probe.lens_thickness`` / ``probe.lens_sound_speed``
- ``ops[i].tx.excitation.center_frequency`` -> ``scan.center_frequency`` (and
  ``scan.demodulation_frequency``, ``probe.probe_center_frequency``)
- ``ops[i].tx.delays`` + ``ops[i].tx.aperture`` -> ``scan.t0_delays``, shifted so
  the first active element of each transmit fires at ``t = 0``
- ``ops[i].tx.aperture`` -> ``scan.tx_apodizations`` and, through the element
  positions, ``scan.transmit_origins`` (the centre of the active aperture)
- ``ops[i].rx.sample_range`` (or ``rx.time_range``) -> ``scan.initial_times``
- ``ops[i].pri`` -> ``scan.time_to_next_transmit``
- ``context.sequence.tx_focus``, or ``ops[i].tx.focus`` for a bare ``TxRxSequence``
  -> ``scan.focus_distances``
- ``context.sequence.angles``, or ``ops[i].tx.angle`` for a bare ``TxRxSequence``
  -> ``scan.polar_angles``
- ``metadata.data_description.sampling_frequency`` -> ``scan.sampling_frequency``
- ``metadata.data_description.spacing.coordinates`` -> per-pixel ``coordinates``
- ``context.medium.speed_of_sound`` -> ``scan.sound_speed``

.. note::
    Tested against ARRUS 0.12.x – 0.14.x recordings (gui4us 0.3.x). Files
    written by other versions are converted on a best-effort basis: the
    converter validates the structures it needs and reports precisely which one
    is missing rather than guessing.

.. warning::
    **Limitations.** This converter targets *basic, standard* us4us
    acquisitions with a single, single-axis array probe (linear, convex, phased
    or ring). Not supported:

    - **RCA / multi-probe / matrix acquisitions.** Recordings whose context
      holds more than one probe are rejected with an explicit error, because
      zea's ``probe_geometry`` and ``t0_delays`` assume a single element grid.
    - **Recordings without embedded metadata.** Some setups save the frame data
      and the ARRUS metadata as two separate pickles. Point ``--metadata`` at
      the metadata pickle in that case (a sibling ``*_metadata.pkl`` next to the
      data file is picked up automatically).
    - **Per-frame changing sequences.** The TX/RX sequence of the first
      metadata entry is assumed to hold for every frame.
    - **Custom pipeline outputs.** Only the data types in
      :data:`SUPPORTED_DATA_TYPES` are understood; anything else (Doppler,
      segmentation, elastography, …) has no us4us layout convention and must be
      converted with a bespoke script using :meth:`zea.File.create`.
    - **Image dynamic range.** ARRUS B-mode images are log-compressed but not
      normalized, while zea expects float images in dB with a maximum of 0, so
      float images are shifted by their global maximum (``uint8`` images are
      stored unchanged).
    - **Coordinates.** Per-pixel coordinates are stored when ARRUS reports a
      grid (``data_description.spacing``). An output without a grid whose first
      axis has one entry per transmit is taken to be organized per scan line: it
      is transposed to ``(z, x) = (depth, transmit)`` and its coordinates are
      derived from the transmit origins and the depth axis, assuming a constant
      speed of sound.
"""

import json
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from zea import log
from zea.data.file import File
from zea.data.spec import DEFAULT_COMPRESSION

__all__ = [
    "SUPPORTED_DATA_TYPES",
    "DEFAULT_MAPPING",
    "Us4usConversionError",
    "convert_us4us",
    "convert_us4us_file",
    "load_us4us_pickle",
    "parse_mapping",
]

#: zea data types this converter knows how to build from a us4us pipeline output.
#: Deliberately narrower than :class:`zea.data.spec.DataSpec`: a data type is only
#: listed here once :func:`_prepare_output` has an explicit branch for its us4us
#: layout, so that unsupported types fail loudly instead of silently writing
#: arrays that downstream readers cannot interpret.
SUPPORTED_DATA_TYPES = ("raw_data", "beamformed_data", "envelope_data", "image")

#: Mapping used when ``--mapping`` is not given: pipeline output 0 is a B-mode image.
DEFAULT_MAPPING = {0: "image"}

#: Default speed of sound (m/s) used for depth axes when the recording does not
#: report one.
DEFAULT_SOUND_SPEED = 1540.0

# ARRUS releases this converter was developed and tested against. Recordings that
# report another version are still converted, but with a warning.
_TESTED_ARRUS_VERSION_RE = re.compile(r"^0\.(12|13|14)\.")

_METADATA_SUFFIXES = ("_metadata.pkl", ".metadata.pkl", "_meta.pkl")


class Us4usConversionError(ValueError):
    """Raised when a pickle cannot be interpreted as a us4us recording.

    Subclasses :class:`ValueError` so callers can keep catching that.
    """


# ---------------------------------------------------------------------------
# Mapping parsing
# ---------------------------------------------------------------------------
def parse_mapping(mapping) -> dict:
    """Normalize a ``--mapping`` value into ``{output_index: zea_data_type}``.

    Accepts the CLI form (a list of ``"<index>:<data_type>"`` strings), a JSON
    object string (``'{"0": "image"}'``), or an already-parsed dict.

    Args:
        mapping: Mapping specification, or ``None`` for :data:`DEFAULT_MAPPING`.

    Returns:
        dict: Pipeline output index (``int``) to zea data type (``str``).

    Raises:
        Us4usConversionError: If the specification is malformed, has duplicate
            indices, or names a data type outside :data:`SUPPORTED_DATA_TYPES`.
    """
    if mapping is None:
        return dict(DEFAULT_MAPPING)

    if isinstance(mapping, Mapping):
        raw_items = list(mapping.items())
    else:
        tokens = [mapping] if isinstance(mapping, str) else [str(token) for token in mapping]
        tokens = [token for token in tokens if token.strip()]
        if not tokens:
            return dict(DEFAULT_MAPPING)
        joined = " ".join(tokens).strip()
        if joined.startswith(("{", "[")):
            try:
                parsed = json.loads(joined)
            except json.JSONDecodeError as exc:
                raise Us4usConversionError(
                    f"Invalid JSON in --mapping: {joined!r} ({exc}). Expected either "
                    'a JSON object such as \'{"0": "image"}\' or entries like 0:image.'
                ) from exc
            if not isinstance(parsed, dict):
                raise Us4usConversionError(
                    f"--mapping JSON must be an object, got {type(parsed).__name__}."
                )
            raw_items = list(parsed.items())
        else:
            raw_items = []
            for token in tokens:
                if token.count(":") != 1:
                    raise Us4usConversionError(
                        f"Invalid --mapping entry {token!r}. Expected "
                        "'<output index>:<data type>', e.g. 0:image."
                    )
                index, _, data_type = token.partition(":")
                raw_items.append((index, data_type))

    result: dict = {}
    for raw_index, data_type in raw_items:
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise Us4usConversionError(
                f"--mapping indices must be integers, got {raw_index!r}."
            ) from exc
        if index < 0:
            raise Us4usConversionError(
                f"--mapping indices must be non-negative pipeline output indices, "
                f"got {raw_index!r}."
            )
        if index in result:
            raise Us4usConversionError(
                f"--mapping assigns pipeline output {index} more than once "
                f"({result[index]!r} and {data_type!r})."
            )
        if not isinstance(data_type, str) or data_type not in SUPPORTED_DATA_TYPES:
            raise Us4usConversionError(
                f"Unsupported zea data type {data_type!r} in --mapping. The us4us "
                f"converter supports: {', '.join(SUPPORTED_DATA_TYPES)}."
            )
        result[index] = data_type

    if not result:
        raise Us4usConversionError("--mapping is empty; at least one output must be mapped.")

    duplicates = {
        data_type for data_type in result.values() if list(result.values()).count(data_type) > 1
    }
    if duplicates:
        raise Us4usConversionError(
            f"--mapping assigns the data type(s) {sorted(duplicates)} to multiple pipeline "
            "outputs. A zea file holds one dataset per data type, so pick a different "
            "data type (e.g. 'beamformed_data' vs 'image') for one of them."
        )
    return result


# ---------------------------------------------------------------------------
# Allowlisted unpickler
#
# TRUST BOUNDARY: pickle is not a safe format for untrusted input -- every
# GLOBAL / STACK_GLOBAL opcode resolves a class that can run code through
# __reduce__ / __setstate__ / __new__. To keep this converter usable without the
# third-party ``arrus`` package installed, and to stop a hostile .pkl from
# reaching e.g. ``os.system``, ``find_class`` is restricted to an allowlist:
#
#   * ``arrus.*`` globals are replaced by a per-class stub;
#   * ``numpy.*`` and ``collections.*`` globals resolve normally (arrays,
#     dtypes, scalars, deques);
#   * everything else raises ``pickle.UnpicklingError``.
# ---------------------------------------------------------------------------
_ALLOWED_PICKLE_MODULE_PREFIXES = ("numpy", "collections")


class _ArrusStub:
    """Attribute-preserving placeholder for an ``arrus`` class.

    Instances keep whatever attributes the pickle restores, so the converter can
    read ARRUS metadata without importing ``arrus``. Subclasses are created per
    original class (see :func:`_arrus_stub_for`) so error messages can name the
    class that was actually stored.
    """

    #: Fully qualified name of the ARRUS class this stub stands in for.
    arrus_class = "arrus"

    def __init__(self, *args, **kwargs):
        # Enum-like classes are reconstructed by calling the class; keep the
        # arguments around so diagnostics can show them.
        self._stub_args = args
        self._stub_kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, tuple) and len(state) == 2:
            state, slots = state
            for key, value in (slots or {}).items():
                setattr(self, key, value)
        if state:
            self.__dict__.update(state)

    def __repr__(self):
        return f"<{self.arrus_class} stub>"


_arrus_stub_cache: dict = {}


def _arrus_stub_for(module: str, name: str) -> type:
    """Return (and cache) a :class:`_ArrusStub` subclass named after ``module.name``."""
    key = (module, name)
    if key not in _arrus_stub_cache:
        _arrus_stub_cache[key] = type(
            name.replace(".", "_"),
            (_ArrusStub,),
            {"__module__": module, "arrus_class": f"{module}.{name}"},
        )
    return _arrus_stub_cache[key]


class _ArrusUnpickler(pickle.Unpickler):
    """Allowlisted unpickler for us4us pickle files.

    Swaps ``arrus.*`` classes for stubs, resolves ``numpy.*``/``collections.*``
    globals normally, and refuses everything else so payloads referencing e.g.
    ``os.system`` or ``subprocess.Popen`` cannot execute.
    """

    def find_class(self, module, name):
        if module == "arrus" or module.startswith("arrus."):
            return _arrus_stub_for(module, name)
        if any(
            module == prefix or module.startswith(prefix + ".")
            for prefix in _ALLOWED_PICKLE_MODULE_PREFIXES
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Refusing to load class {module}.{name} from a us4us pickle: the module is "
            "not in the us4us converter allowlist (arrus, numpy, collections). Only "
            "trusted us4us .pkl files should be processed by this loader."
        )


def load_us4us_pickle(path):
    """Load a us4us pickle with an allowlisted unpickler.

    Args:
        path: Path to the ``.pkl`` file.

    Returns:
        The unpickled object, with every ``arrus`` instance replaced by an
        attribute-preserving stub.

    .. warning::
        Pickle is not a safe format for arbitrary input. Even with the allowlist
        enforced by the loader, only run this on ``.pkl`` files produced by
        us4us / ARRUS acquisition setups you control -- never on files received
        from untrusted third parties.
    """
    with open(path, "rb") as file:
        return _ArrusUnpickler(file).load()


# ---------------------------------------------------------------------------
# Payload normalization and validation
# ---------------------------------------------------------------------------
def _describe(obj) -> str:
    """Short, human-readable description of an unpickled object, for error messages."""
    if isinstance(obj, _ArrusStub):
        return obj.arrus_class
    if isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape}"
    if isinstance(obj, Mapping):
        return f"dict(keys={sorted(str(key) for key in obj)})"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}(len={len(obj)})"
    return type(obj).__name__


def _is_sequence(obj) -> bool:
    """True for indexable, sized sequences (list/tuple/deque), excluding str/bytes."""
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray))


def _is_frame(obj) -> bool:
    """True if ``obj`` looks like one acquired frame (an array or tuple of arrays)."""
    if isinstance(obj, np.ndarray):
        return True
    return _is_sequence(obj) and len(obj) > 0 and all(isinstance(a, np.ndarray) for a in obj)


def _is_frame_list(obj) -> bool:
    """True if ``obj`` looks like a non-empty list of frames."""
    return _is_sequence(obj) and len(obj) > 0 and _is_frame(obj[0])


def _is_metadata(obj) -> bool:
    """True if ``obj`` looks like an ARRUS ``ConstMetadata`` (it carries a context)."""
    return _get_context(obj) is not None


def _is_metadata_list(obj) -> bool:
    """True if ``obj`` looks like a non-empty sequence of ARRUS metadata objects."""
    return _is_sequence(obj) and len(obj) > 0 and _is_metadata(obj[0])


def _get_context(metadata_entry):
    """Return the ARRUS ``FrameAcquisitionContext`` of a metadata entry, or None."""
    return getattr(metadata_entry, "_context", None) or getattr(metadata_entry, "context", None)


def _get_data_description(metadata_entry):
    """Return the ARRUS data description of a metadata entry, or None."""
    return getattr(metadata_entry, "_data_char", None) or getattr(
        metadata_entry, "data_description", None
    )


def _find_sidecar_metadata(src: Path):
    """Return a sibling ``*_metadata.pkl`` file for ``src``, if one exists."""
    candidates = [src.with_name(src.stem + suffix) for suffix in _METADATA_SUFFIXES]
    candidates.append(src.parent / "metadata.pkl")
    for candidate in candidates:
        if candidate.exists() and candidate != src:
            return candidate
    return None


def normalize_us4us_payload(payload, metadata=None, *, source=None):
    """Bring a loaded us4us pickle into ``(frames, metadata)`` form.

    Handles the layouts we have seen in the wild:

    1. ``{"data": [...], "metadata": (...)}`` -- gui4us 0.3.x, the standard form.
    2. ``{"data": [...]}`` plus a separately pickled metadata file.
    3. ``[data, metadata]`` -- a two-element sequence.
    4. A bare list of frames, with the metadata in a separate file.

    Args:
        payload: The object returned by :func:`load_us4us_pickle`.
        metadata: Metadata loaded from a separate file, if any. Takes precedence
            over metadata embedded in ``payload``.
        source: Path of the source file, used in error messages.

    Returns:
        tuple: ``(frames, metadata)`` where ``frames`` is a list of per-frame
        tuples of arrays and ``metadata`` is a sequence with one entry per
        pipeline output.

    Raises:
        Us4usConversionError: If the payload does not look like a us4us
            recording, or if no metadata could be found.
    """
    where = f" in {source}" if source else ""
    embedded_metadata = None

    if isinstance(payload, Mapping):
        if "data" not in payload:
            raise Us4usConversionError(
                f"Not a us4us recording{where}: the pickled dict has keys "
                f"{sorted(str(key) for key in payload)} but no 'data' key holding the "
                "acquired frames."
            )
        frames = payload["data"]
        embedded_metadata = payload.get("metadata")
    elif _is_sequence(payload):
        if len(payload) == 2 and _is_frame_list(payload[0]) and _is_metadata_list(payload[1]):
            frames, embedded_metadata = payload[0], payload[1]
        elif _is_frame_list(payload):
            frames = payload
        else:
            raise Us4usConversionError(
                f"Not a us4us recording{where}: expected a dict with 'data' and 'metadata' "
                f"keys, or a list of frames, but got {_describe(payload)} whose first item "
                f"is {_describe(payload[0]) if len(payload) else 'nothing (empty)'}."
            )
    else:
        raise Us4usConversionError(
            f"Not a us4us recording{where}: expected a dict with 'data' and 'metadata' keys "
            f"(gui4us capture), got {_describe(payload)}."
        )

    if not _is_frame_list(frames):
        raise Us4usConversionError(
            f"Not a us4us recording{where}: 'data' must be a non-empty list of frames "
            "(each frame an array or a tuple of arrays, one per pipeline output), got "
            f"{_describe(frames)}."
        )

    if metadata is None:
        metadata = embedded_metadata
    elif embedded_metadata is not None:
        log.info("Using metadata from the separate metadata file instead of the embedded one.")

    if metadata is None:
        raise Us4usConversionError(
            f"No ARRUS metadata found{where}. Some us4us setups store the frame data and "
            "the metadata as two separate pickles; pass the metadata file with "
            "--metadata <file.pkl> (a sibling '<name>_metadata.pkl' is picked up "
            "automatically)."
        )

    if _is_metadata(metadata):  # a single ConstMetadata rather than one per output
        metadata = [metadata]
    if not _is_sequence(metadata) or len(metadata) == 0:
        raise Us4usConversionError(
            f"Invalid ARRUS metadata{where}: expected one ``ConstMetadata`` per pipeline "
            f"output (a list or tuple), got {_describe(metadata)}."
        )

    # Normalize every frame to a tuple of arrays, one entry per pipeline output.
    frames = [(frame,) if isinstance(frame, np.ndarray) else tuple(frame) for frame in frames]
    n_outputs = len(frames[0])
    if any(len(frame) != n_outputs for frame in frames):
        raise Us4usConversionError(
            f"Inconsistent frames{where}: every frame must hold the same number of pipeline "
            f"outputs, got {sorted({len(frame) for frame in frames})}."
        )
    if len(metadata) < n_outputs:
        log.warning(
            f"us4us recording has {n_outputs} pipeline outputs but only {len(metadata)} "
            "metadata entries; the last metadata entry is reused for the remaining outputs."
        )
        metadata = list(metadata) + [metadata[-1]] * (n_outputs - len(metadata))

    return frames, metadata


def _validate_metadata_entry(metadata_entry, *, source=None) -> None:
    """Check that a metadata entry carries everything the converter needs.

    Raises:
        Us4usConversionError: If the ARRUS context, sequence or probe model that
            the conversion relies on is missing.
    """
    where = f" in {source}" if source else ""
    context = _get_context(metadata_entry)
    if context is None:
        raise Us4usConversionError(
            f"Invalid ARRUS metadata{where}: {_describe(metadata_entry)} has no acquisition "
            "context. Expected an arrus.metadata.ConstMetadata with a 'context' attribute "
            "(ARRUS 0.12.0 or newer)."
        )

    version = getattr(metadata_entry, "version", None) or getattr(metadata_entry, "_version", None)
    if isinstance(version, str) and not _TESTED_ARRUS_VERSION_RE.match(version):
        log.warning(
            f"us4us recording reports ARRUS version {version}, outside the tested range "
            "0.12.x - 0.14.x. Conversion continues, but please double-check the result."
        )

    raw_sequence = getattr(context, "raw_sequence", None)
    ops = getattr(raw_sequence, "ops", None) if raw_sequence is not None else None
    if not ops:
        raise Us4usConversionError(
            f"Invalid ARRUS metadata{where}: context.raw_sequence.ops is missing or empty, so "
            "the TX/RX sequence cannot be converted. This usually means the recording was "
            "made with an ARRUS version older than 0.12.0."
        )
    for attribute in ("tx", "rx"):
        if getattr(ops[0], attribute, None) is None:
            raise Us4usConversionError(
                f"Invalid ARRUS metadata{where}: context.raw_sequence.ops[0] has no "
                f"'{attribute}' operation ({_describe(ops[0])})."
            )

    probes = _get_probes(context, source=source)
    model = getattr(probes[0], "model", None)
    if model is None:
        raise Us4usConversionError(
            f"Invalid ARRUS metadata{where}: context.device.probe[0] has no probe model "
            f"({_describe(probes[0])})."
        )
    for attribute in ("element_pos_x", "element_pos_z", "n_elements"):
        if getattr(model, attribute, None) is None:
            raise Us4usConversionError(
                f"Invalid ARRUS metadata{where}: probe model {_describe(model)} has no "
                f"'{attribute}', so the probe geometry cannot be reconstructed."
            )


def _get_probes(context, *, source=None) -> list:
    """Return the probe list of an ARRUS context, rejecting multi-probe setups.

    Raises:
        Us4usConversionError: If no probe is present, or if the recording uses
            more than one probe (RCA / matrix acquisitions).
    """
    where = f" in {source}" if source else ""
    device = getattr(context, "device", None)
    probes = getattr(device, "probe", None) if device is not None else None
    if probes is None:
        raise Us4usConversionError(
            f"Invalid ARRUS metadata{where}: context.device.probe is missing, so the probe "
            "geometry cannot be reconstructed."
        )
    if not _is_sequence(probes):
        probes = [probes]
    if len(probes) == 0:
        raise Us4usConversionError(f"Invalid ARRUS metadata{where}: context.device.probe is empty.")
    if len(probes) > 1:
        raise Us4usConversionError(
            f"Unsupported us4us recording{where}: the acquisition uses {len(probes)} probes "
            "(row-column addressed, matrix or dual-probe setup). This converter only "
            "supports single, single-axis array probes (linear / convex / phased / ring), "
            "because zea's probe_geometry and t0_delays describe one element grid. Convert "
            "such recordings with a bespoke script using zea.File.create."
        )
    model_name = _describe(getattr(probes[0], "model", probes[0]))
    if "rca" in model_name.lower():
        raise Us4usConversionError(
            f"Unsupported us4us recording{where}: probe model {model_name} is row-column "
            "addressed (RCA), which this converter does not support."
        )
    return list(probes)


# ---------------------------------------------------------------------------
# Probe extraction
# ---------------------------------------------------------------------------
def _extract_probe_dict(context) -> dict:
    """Build a :class:`~zea.data.spec.ProbeSpec`-compatible dict from an ARRUS context."""
    model = _get_probes(context)[0].model
    n_el = int(model.n_elements)

    x_pos = np.asarray(model.element_pos_x, dtype=np.float32).ravel()
    z_pos = np.asarray(model.element_pos_z, dtype=np.float32).ravel()
    y_source = getattr(model, "element_pos_y", None)
    if y_source is not None:
        y_pos = np.asarray(y_source, dtype=np.float32).ravel()
    else:
        y_pos = np.zeros(n_el, dtype=np.float32)
    probe_dict = {"probe_geometry": np.stack([x_pos, y_pos, z_pos], axis=1)}

    pitch = getattr(model, "pitch", None)
    if pitch:
        # ARRUS does not report the element width; the pitch is the closest
        # available approximation (it ignores the kerf between elements).
        probe_dict["element_width"] = np.float32(pitch)

    curvature_radius = getattr(model, "curvature_radius", None)
    if curvature_radius is not None:
        probe_dict["type"] = "linear" if float(curvature_radius) == 0.0 else "curved"

    model_id = getattr(model, "model_id", None)
    probe_dict["name"] = str(getattr(model_id, "name", None) or "us4us_generic")

    center_frequency = _tx_center_frequency(context)
    if center_frequency:
        probe_dict["probe_center_frequency"] = np.float32(center_frequency)

    lens = getattr(model, "lens", None)
    if lens is not None:
        thickness = getattr(lens, "thickness", None)
        sound_speed = getattr(lens, "speed_of_sound", None)
        if thickness and float(thickness) > 0:
            probe_dict["lens_thickness"] = np.float32(thickness)
        if sound_speed and float(sound_speed) > 0:
            probe_dict["lens_sound_speed"] = np.float32(sound_speed)

    return probe_dict


def _tx_center_frequency(context) -> float:
    """Center frequency of the transmitted pulse, in Hz (0.0 when unavailable)."""
    ops = context.raw_sequence.ops
    excitation = getattr(ops[0].tx, "excitation", None)
    center_frequency = getattr(excitation, "center_frequency", None)
    return float(center_frequency) if center_frequency else 0.0


# ---------------------------------------------------------------------------
# TX/RX sequence -> scan
# ---------------------------------------------------------------------------
def _sound_speed(context) -> float:
    """Speed of sound of the acquisition in m/s, falling back to a default."""
    medium = getattr(context, "medium", None)
    for source in (medium, getattr(context, "sequence", None)):
        sound_speed = getattr(source, "speed_of_sound", None)
        if sound_speed:
            return float(sound_speed)
    log.warning(
        "us4us recording does not report a speed of sound; "
        f"falling back to {DEFAULT_SOUND_SPEED} m/s."
    )
    return DEFAULT_SOUND_SPEED


def _per_transmit(value, n_tx: int) -> "np.ndarray | None":
    """Broadcast a scalar or per-transmit sequence to a ``(n_tx,)`` float32 array."""
    if value is None:
        return None
    array = np.atleast_1d(np.asarray(value, dtype=np.float32)).ravel()
    if array.size == n_tx:
        return array
    if array.size == 1:
        return np.full(n_tx, float(array[0]), dtype=np.float32)
    return None


def _extract_scan_dict(context, data_description, n_frames: int) -> dict:
    """Build a :class:`~zea.data.spec.ScanSpec`-compatible dict from an ARRUS context."""
    ops = context.raw_sequence.ops
    sequence = getattr(context, "sequence", None)
    n_tx = len(ops)

    model = _get_probes(context)[0].model
    n_el = int(model.n_elements)
    x_pos = np.asarray(model.element_pos_x, dtype=np.float64).ravel()
    z_pos = np.asarray(model.element_pos_z, dtype=np.float64).ravel()

    sampling_frequency = np.float32(data_description.sampling_frequency)
    center_frequency = np.float32(_tx_center_frequency(context))

    # t0_delays and tx_apodizations, both (n_tx, n_el)
    t0_delays = np.zeros((n_tx, n_el), dtype=np.float64)
    tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)
    for i, op in enumerate(ops):
        aperture = np.asarray(op.tx.aperture, dtype=bool)
        if aperture.size != n_el:
            raise Us4usConversionError(
                f"Unsupported us4us recording: transmit {i} addresses {aperture.size} "
                f"elements while the probe model reports {n_el}. This converter only "
                "supports sequences that address a single probe."
            )
        t0_delays[i, aperture] = np.asarray(op.tx.delays, dtype=np.float64)
        apodization = getattr(op.tx, "apodization", None)
        if apodization is None:
            tx_apodizations[i, aperture] = 1.0
        else:
            tx_apodizations[i, aperture] = np.asarray(apodization, dtype=np.float32).ravel()
        # Shift each transmit so its first active element fires at t = 0, then clip
        # to suppress floating-point underflow below zero (zea forbids negative delays).
        if np.any(aperture):
            t0_delays[i] -= t0_delays[i, aperture].min()
    t0_delays = np.clip(t0_delays, 0.0, None).astype(np.float32)

    # initial_times (n_tx,): time between the first element firing and the first sample.
    initial_times = np.zeros(n_tx, dtype=np.float32)
    for i, op in enumerate(ops):
        sample_range = getattr(op.rx, "sample_range", None)
        time_range = getattr(op.rx, "time_range", None)
        if sample_range is not None:
            initial_times[i] = np.float32(sample_range[0] / float(sampling_frequency))
        elif time_range is not None:
            initial_times[i] = np.float32(time_range[0])

    # focus_distances (n_tx,): a SimpleTxRxSequence (Lin/Pwi/Sta) carries `tx_focus` on
    # the sequence, a bare TxRxSequence stores `focus` per op (None when raw delays were
    # given, which we report as a plane wave).
    focus_distances = _per_transmit(getattr(sequence, "tx_focus", None), n_tx)
    if focus_distances is None:
        focus_distances = np.array(
            [
                float(op.tx.focus) if getattr(op.tx, "focus", None) is not None else np.inf
                for op in ops
            ],
            dtype=np.float32,
        )

    # transmit_origins (n_tx, 3): centre of the active TX aperture.
    transmit_origins = np.zeros((n_tx, 3), dtype=np.float32)
    for i, op in enumerate(ops):
        aperture = np.asarray(op.tx.aperture, dtype=bool)
        if np.any(aperture):
            transmit_origins[i, 0] = float(x_pos[aperture].mean())
            transmit_origins[i, 2] = float(z_pos[aperture].mean())

    # polar_angles (n_tx,): on the sequence for a SimpleTxRxSequence, per op otherwise.
    polar_angles = _per_transmit(getattr(sequence, "angles", None), n_tx)
    if polar_angles is None:
        polar_angles = np.array(
            [
                float(op.tx.angle) if getattr(op.tx, "angle", None) is not None else 0.0
                for op in ops
            ],
            dtype=np.float32,
        )

    pris = np.array([float(getattr(op, "pri", 0.0) or 0.0) for op in ops], dtype=np.float32)

    scan_dict = {
        "sampling_frequency": sampling_frequency,
        "center_frequency": center_frequency,
        "demodulation_frequency": center_frequency,
        "initial_times": initial_times,
        "t0_delays": t0_delays,
        "tx_apodizations": tx_apodizations,
        "focus_distances": focus_distances,
        "transmit_origins": transmit_origins,
        "polar_angles": polar_angles,
        "time_to_next_transmit": np.tile(pris[np.newaxis, :], (n_frames, 1)),
        "sound_speed": np.float32(_sound_speed(context)),
    }
    return scan_dict


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------
def _grid_coordinates(data_description, spatial_shape) -> "np.ndarray | None":
    """Build a ``(*spatial_shape, 3)`` coordinate grid from ARRUS grid spacing.

    ARRUS attaches a ``spacing`` with one coordinate vector per output axis to
    pipeline steps that reconstruct onto a grid (scan conversion, LRI/HRI
    reconstruction). Returns ``None`` when no usable grid is present.
    """
    spacing = getattr(data_description, "spacing", None)
    coordinates = getattr(spacing, "coordinates", None) if spacing is not None else None
    if coordinates is None or len(coordinates) < len(spatial_shape):
        return None
    axes = [
        np.asarray(axis, dtype=np.float32).ravel() for axis in coordinates[: len(spatial_shape)]
    ]
    if tuple(axis.size for axis in axes) != tuple(spatial_shape):
        log.warning(
            f"Ignoring ARRUS grid spacing: axis lengths {[axis.size for axis in axes]} do not "
            f"match the data shape {tuple(spatial_shape)}."
        )
        return None
    if len(axes) != 2:
        # Only 2-D grids (depth, lateral) are converted; 3-D volumes would need
        # a documented axis convention we cannot verify.
        return None
    z_grid, x_grid = np.meshgrid(axes[0], axes[1], indexing="ij")
    return np.stack([x_grid, np.zeros_like(x_grid), z_grid], axis=-1).astype(np.float32)


def _scanline_coordinates(scan_dict, n_ax: int, sampling_frequency) -> "np.ndarray | None":
    """Build coordinates for data that is still organized one column per transmit.

    Each column is the depth axis of one transmit, starting at
    ``initial_times[i] * c / 2`` and advancing by ``c / (2 * fs)`` per sample,
    along the beam direction given by ``polar_angles``.
    """
    if not sampling_frequency or sampling_frequency <= 0:
        return None
    sound_speed = float(scan_dict["sound_speed"])
    origins = np.asarray(scan_dict["transmit_origins"], dtype=np.float32)
    angles = np.asarray(scan_dict["polar_angles"], dtype=np.float32)
    initial_times = np.asarray(scan_dict["initial_times"], dtype=np.float32)

    # (n_ax, n_tx) depth along the beam for every transmit.
    sample_depths = np.arange(n_ax, dtype=np.float32)[:, None] * (
        sound_speed / (2.0 * float(sampling_frequency))
    )
    depths = sample_depths + (initial_times[None, :] * sound_speed / 2.0)

    x_coords = origins[None, :, 0] + depths * np.sin(angles)[None, :]
    z_coords = origins[None, :, 2] + depths * np.cos(angles)[None, :]
    return np.stack([x_coords, np.zeros_like(x_coords), z_coords], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline output -> zea data
# ---------------------------------------------------------------------------
def _stack_frames(frames, output_idx: int) -> np.ndarray:
    """Stack one pipeline output across frames, dropping a leading singleton axis."""
    arrays = [frame[output_idx] for frame in frames]
    if arrays[0].ndim > 1 and arrays[0].shape[0] == 1:
        arrays = [array[0] for array in arrays]
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        raise Us4usConversionError(
            f"Pipeline output {output_idx} changes shape between frames ({sorted(shapes)}); "
            "zea files require one consistent shape for all frames."
        )
    return np.stack(arrays, axis=0)


def _map_payload(values, coordinates) -> dict:
    """Wrap map values and (optional) coordinates in the dict zea expects."""
    payload = {"values": values}
    if coordinates is not None:
        payload["coordinates"] = coordinates
    return payload


def _prepare_raw_data(stacked: np.ndarray, ops) -> np.ndarray:
    """Reshape ARRUS channel data to zea ``(n_frames, n_tx, n_ax, n_el, n_ch)``.

    ARRUS stores only the elements of the receive aperture, ordered as
    ``[left padding][active elements][right padding]``. They are scattered back
    onto the full probe aperture using ``rx.aperture`` and ``rx.padding``.
    """
    if stacked.ndim != 4:
        raise Us4usConversionError(
            f"Expected ARRUS channel data of shape (n_tx, n_ax, n_rx) per frame, got "
            f"{stacked.shape[1:]}. Map this output to another data type, or convert it "
            "with a bespoke script."
        )
    n_el = (
        len(np.asarray(ops[0].rx.aperture))
        if getattr(ops[0].rx, "aperture", None) is not None
        else None
    )
    if n_el is not None and stacked.shape[-1] != n_el:
        full = np.zeros((*stacked.shape[:3], n_el), dtype=stacked.dtype)
        for i, op in enumerate(ops):
            padding = getattr(op.rx, "padding", (0, 0)) or (0, 0)
            left_pad = int(padding[0])
            active = np.where(np.asarray(op.rx.aperture, dtype=bool))[0]
            full[:, i][:, :, active] = stacked[:, i, :, left_pad : left_pad + active.size]
        stacked = full

    if np.iscomplexobj(stacked):  # IQ channel data -> two channels
        return np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32)
    if stacked.dtype != np.int16:
        stacked = stacked.astype(np.float32)
    return stacked[..., np.newaxis]


def _prepare_image(stacked: np.ndarray, coordinates) -> dict:
    """Convert an ARRUS B-mode output to a zea ``image``."""
    if stacked.dtype == np.uint8:
        return _map_payload(stacked, coordinates)
    values = stacked.astype(np.float32)
    if np.isnan(values).any():
        # ARRUS marks pixels outside the scan-converted region with NaN; zea images
        # must be finite or -inf, and "no echo" is -inf on a dB scale.
        log.warning("us4us image contains NaN values (outside the scan region); storing as -inf.")
        values = np.where(np.isnan(values), -np.inf, values)
    max_value = float(np.max(values[np.isfinite(values)], initial=0.0))
    if max_value > 0:
        # ARRUS images are log-compressed but not normalized; zea expects dB <= 0.
        values = values - max_value
    return _map_payload(values, coordinates)


def _prepare_output(frames, output_idx: int, data_type: str, metadata_entry, context, scan_dict):
    """Stack and reshape one pipeline output into the layout zea expects.

    Args:
        frames: List of per-frame tuples of arrays.
        output_idx: Index of the pipeline output within each frame tuple.
        data_type: Target zea data type (one of :data:`SUPPORTED_DATA_TYPES`).
        metadata_entry: ARRUS metadata belonging to this pipeline output.
        context: ARRUS acquisition context (shared by all outputs).
        scan_dict: Scan parameters, used to derive scan-line coordinates.

    Returns:
        A numpy array (``raw_data``) or a dict with ``values`` and optionally
        ``coordinates`` / ``labels`` (all other data types).
    """
    stacked = _stack_frames(frames, output_idx)
    data_description = _get_data_description(metadata_entry)

    if data_type == "raw_data":
        return _prepare_raw_data(stacked, context.raw_sequence.ops)

    spatial = stacked.shape[1:]
    coordinates = _grid_coordinates(data_description, spatial)
    n_tx = len(context.raw_sequence.ops)
    # A beamformed/envelope output that is not on a grid is organized one column
    # per transmit: (n_tx, n_ax) per frame. Transpose it to (n_ax, n_tx) = (z, x)
    # and derive coordinates from the transmit origins.
    is_scanline = coordinates is None and len(spatial) == 2 and spatial[0] == n_tx
    if is_scanline:
        stacked = np.swapaxes(stacked, 1, 2)
        sampling_frequency = getattr(data_description, "sampling_frequency", None)
        coordinates = _scanline_coordinates(scan_dict, stacked.shape[1], sampling_frequency)

    if data_type == "image":
        return _prepare_image(stacked, coordinates)

    if data_type == "beamformed_data":
        if np.iscomplexobj(stacked):
            values = np.stack([stacked.real, stacked.imag], axis=-1).astype(np.float32)
            labels = np.array(["I", "Q"], dtype=np.str_)
        else:
            values = stacked.astype(np.float32)[..., np.newaxis]
            labels = np.array(["RF"], dtype=np.str_)
        payload = _map_payload(values, coordinates)
        payload["labels"] = labels
        return payload

    # envelope_data: magnitude of the (possibly complex) input.
    values = np.abs(stacked) if np.iscomplexobj(stacked) else stacked
    return _map_payload(values.astype(np.float32), coordinates)


def _pick_scan_metadata(metadata, mapping: dict):
    """Return the metadata entry best suited for extracting scan parameters.

    Prefers the entry mapped to ``raw_data`` (its sampling frequency is the ADC
    rate, which ``initial_times`` are expressed in), then the entry with the
    highest sampling frequency, then the first entry.
    """
    for output_idx, data_type in mapping.items():
        if data_type == "raw_data":
            return metadata[output_idx]

    def sampling_frequency(entry):
        return float(getattr(_get_data_description(entry), "sampling_frequency", 0) or 0)

    return max(metadata, key=sampling_frequency, default=metadata[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def convert_us4us_file(
    src,
    dst,
    mapping=None,
    *,
    metadata_path=None,
    overwrite: bool = False,
    separate_tracks: bool = False,
) -> Path:
    """Convert a single us4us ``.pkl`` recording to a zea HDF5 file.

    Args:
        src: Source ``.pkl`` file.
        dst: Destination ``.hdf5`` file.
        mapping: Pipeline output index to zea data type, in any form accepted by
            :func:`parse_mapping`. Defaults to :data:`DEFAULT_MAPPING`.
        metadata_path: Optional pickle holding the ARRUS metadata, for
            recordings that store data and metadata separately. When omitted, a
            sibling ``<name>_metadata.pkl`` is used if present.
        overwrite: Replace ``dst`` if it already exists.
        separate_tracks: Write every mapped output to its own zea track instead
            of storing them side by side in a single track.

    Returns:
        Path: The written HDF5 file.

    Raises:
        Us4usConversionError: If the pickle is not a us4us recording, or uses a
            layout this converter does not support.
    """
    src, dst = Path(src), Path(dst)
    mapping = parse_mapping(mapping)

    log.info(f"Loading us4us pickle: {log.yellow(src)}")
    payload = load_us4us_pickle(src)

    if metadata_path is None:
        metadata_path = _find_sidecar_metadata(src)
        if metadata_path is not None:
            log.info(f"Found metadata file next to the recording: {log.yellow(metadata_path)}")
    metadata_payload = load_us4us_pickle(metadata_path) if metadata_path is not None else None
    if isinstance(metadata_payload, Mapping):
        metadata_payload = metadata_payload.get("metadata", metadata_payload)

    frames, metadata = normalize_us4us_payload(payload, metadata_payload, source=src)
    n_frames, n_outputs = len(frames), len(frames[0])
    log.info(f"Frames: {n_frames}, pipeline outputs: {n_outputs}, mapping: {mapping}")

    out_of_range = [index for index in mapping if index >= n_outputs]
    if out_of_range:
        raise Us4usConversionError(
            f"--mapping refers to pipeline output(s) {out_of_range}, but {src.name} only has "
            f"{n_outputs} output(s) per frame (valid indices: 0-{n_outputs - 1})."
        )

    scan_metadata = _pick_scan_metadata(metadata, mapping)
    _validate_metadata_entry(scan_metadata, source=src)
    context = _get_context(scan_metadata)

    probe_dict = _extract_probe_dict(context)
    scan_dict = _extract_scan_dict(context, _get_data_description(scan_metadata), n_frames)

    outputs = {}
    for output_idx, data_type in mapping.items():
        log.info(f"  output[{output_idx}] -> {data_type}")
        outputs[data_type] = _prepare_output(
            frames, output_idx, data_type, metadata[output_idx], context, scan_dict
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Writing zea file: {log.yellow(dst)}")
    description = "us4us (ARRUS + gui4us) recording converted to the zea format"
    if separate_tracks:
        File.create(
            path=dst,
            tracks=[
                {"data": {data_type: values}, "scan": scan_dict, "label": data_type}
                for data_type, values in outputs.items()
            ],
            probe=probe_dict,
            description=description,
            us_machine="us4R",
            compression=DEFAULT_COMPRESSION,
            overwrite=overwrite,
        )
    else:
        File.create(
            path=dst,
            data=outputs,
            scan=scan_dict,
            probe=probe_dict,
            description=description,
            us_machine="us4R",
            compression=DEFAULT_COMPRESSION,
            overwrite=overwrite,
        )

    log.success(f"Converted {log.yellow(src)} -> {log.yellow(dst)}")
    return dst


def convert_us4us(args) -> None:
    """Convert one or more us4us (ARRUS + gui4us) pickle recordings to the zea format.

    Args:
        args: Object with the attributes:

            - ``src``: Source ``.pkl`` file, or a directory holding ``*.pkl`` files.
            - ``dst``: Destination ``.hdf5`` file when ``src`` is a file, or the
              destination directory when ``src`` is a directory (each
              ``<name>.pkl`` becomes ``<dst>/<name>.hdf5``).
            - ``mapping`` (optional): Pipeline output index to zea data type, in
              any form accepted by :func:`parse_mapping`.
            - ``metadata`` (optional): Pickle holding the ARRUS metadata for
              recordings that store data and metadata separately.
            - ``overwrite`` (optional): Replace existing destination files.
            - ``separate_tracks`` (optional): Write each mapped output to its own
              zea track.

    Raises:
        FileNotFoundError: If ``src`` does not exist, or is a directory without
            any ``.pkl`` files.
        Us4usConversionError: If a recording cannot be interpreted.
    """
    src = Path(args.src)
    dst = Path(args.dst)
    mapping = parse_mapping(getattr(args, "mapping", None))
    metadata_path = getattr(args, "metadata", None)
    overwrite = bool(getattr(args, "overwrite", False))
    separate_tracks = bool(getattr(args, "separate_tracks", False))

    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    if src.is_dir():
        pkl_files = sorted(
            path
            for path in src.glob("*.pkl")
            if not any(path.name.endswith(suffix) for suffix in _METADATA_SUFFIXES)
        )
        if not pkl_files:
            raise FileNotFoundError(f"No .pkl files found in directory: {src}")
        if metadata_path is not None and len(pkl_files) > 1:
            raise Us4usConversionError(
                "--metadata applies to a single recording, but "
                f"{len(pkl_files)} .pkl files were found in {src}."
            )
        dst.mkdir(parents=True, exist_ok=True)
        file_pairs = [(path, dst / f"{path.stem}.hdf5") for path in pkl_files]
    else:
        dst_file = dst / f"{src.stem}.hdf5" if dst.is_dir() else dst
        file_pairs = [(src, dst_file)]

    for src_pkl, dst_hdf5 in file_pairs:
        convert_us4us_file(
            src_pkl,
            dst_hdf5,
            mapping,
            metadata_path=metadata_path,
            overwrite=overwrite,
            separate_tracks=separate_tracks,
        )
