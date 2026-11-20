"""Synthetic Verasonics workspaces for tests.

A Verasonics workspace is a MATLAB ``-v7.3`` ``.mat`` file: HDF5 with MATLAB's storage
conventions. :data:`SCHEMA` says how MATLAB stores each workspace variable the converter
reads, and every file the tests write goes through it:

* :class:`VerasonicsWorkspace` builds a complete, consistent workspace (probe, plane-wave
  transmits, receives, events, RF and image buffers) from a few settings, with its RF
  either in ``RcvData`` or in ``RF_data_{k}.bin`` files.
* :func:`write_workspace` writes just the given variables, for small targeted tests.

A variable or field that is not in :data:`SCHEMA` is rejected. The schema was checked
against real workspaces (a single-buffer ``RcvData`` workspace and two six-buffer
``.bin`` workspaces); check a new real file with::

    python -m tests.fixtures.verasonics_workspace check SRC.mat [SRC.mat ...]

which reports every variable the file stores differently from :data:`SCHEMA`. To cover
a new edge case, add it to the schema and check it against a real file that has it.

MATLAB ``-v7.3`` storage rules, as found in those files:

* Axes are reversed, so a MATLAB ``rows x cols x frames`` array is h5py
  ``(frames, cols, rows)``. Values passed here are in h5py order.
* A char array is ``uint16 (n, 1)`` with ``MATLAB_class=char`` and
  ``MATLAB_int_decode=2``; a logical is ``uint8`` with ``MATLAB_int_decode=1``.
* An empty array is stored as its dims, ``uint64 (2,)``, with ``MATLAB_empty=1``.
* A cell is a dataset of references into ``#refs#`` with ``MATLAB_class=cell``, even
  with a single cell.
* A struct is a group with ``MATLAB_class=struct`` and ``MATLAB_fields``. A 1x1 struct
  stores its fields as plain datasets, a longer struct array as ``(n, 1)`` datasets of
  references without a class.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# --------------------------------------------------------------------------- schema


@dataclass(frozen=True)
class Array:
    """How one MATLAB array (a variable, struct field or cell element) is stored.

    Args:
        matlab_class (str): ``double``, ``char``, ``logical`` or ``int16``.
        layout (str): ``scalar``, ``column`` or ``row`` (h5py ``(n, 1)`` / ``(1, n)``),
            or ``2d``, ``3d``, ``4d`` for that many dimensions. A char is a column.
        empty (str, optional): The ``MATLAB_class`` an empty value (``None``) is stored
            with, or None if the value is never empty.
    """

    matlab_class: str
    layout: str = "scalar"
    empty: str | None = None


@dataclass(frozen=True)
class Cell:
    """A cell array whose cells are stored as ``element``, laid out along h5py ``axis``."""

    element: Array
    axis: int


@dataclass(frozen=True)
class Struct:
    """A struct (array) with the given fields, each an :class:`Array` or a nested struct."""

    fields: dict


_DTYPES = {"double": "<f8", "char": "<u2", "logical": "|u1", "int16": "<i2"}
_INT_DECODE = {"char": 2, "logical": 1}

DOUBLE = Array("double")
CHAR = Array("char", "column")
COLUMN = Array("double", "column")
ROW = Array("double", "row")

SCHEMA = {
    "Trans": Struct(
        {
            "name": CHAR,
            "units": CHAR,
            "type": DOUBLE,
            "frequency": DOUBLE,
            "elementWidth": DOUBLE,
            "lensCorrection": DOUBLE,
            "Bandwidth": COLUMN,
            "ElementPos": Array("double", "2d"),
            "ConnectorES": ROW,
        }
    ),
    "TX": Struct(
        {
            "waveform": DOUBLE,
            "Origin": COLUMN,
            "focus": DOUBLE,
            "Steer": COLUMN,
            "Apod": COLUMN,
            "Delay": COLUMN,
        }
    ),
    "TW": Struct(
        {
            "type": CHAR,
            "Parameters": Array("double", "column", empty="canonical empty"),
            "Wvfm1Wy": ROW,
            "Wvfm2Wy": ROW,
        }
    ),
    "Receive": Struct(
        {
            "bufnum": DOUBLE,
            "framenum": DOUBLE,
            "acqNum": DOUBLE,
            "mode": DOUBLE,
            "startSample": DOUBLE,
            "endSample": DOUBLE,
            "startDepth": DOUBLE,
            "endDepth": DOUBLE,
            "sampleMode": CHAR,
            "demodFrequency": DOUBLE,
            "decimSampleRate": DOUBLE,
            "quadDecim": DOUBLE,
            "TGC": DOUBLE,
        }
    ),
    "Event": Struct({"tx": DOUBLE, "rcv": DOUBLE, "seqControl": COLUMN}),
    "SeqControl": Struct(
        {
            "command": CHAR,
            "argument": Array("double", empty="canonical empty"),
        }
    ),
    "TGC": Struct({"Waveform": COLUMN}),
    "Resource": Struct(
        {
            "Parameters": Struct({"speedOfSound": DOUBLE}),
            "RcvBuffer": Struct({"numFrames": DOUBLE, "lastFrame": DOUBLE}),
            "ImageBuffer": Struct({"numFrames": DOUBLE, "firstFrame": DOUBLE, "lastFrame": DOUBLE}),
        }
    ),
    # A captured single RcvData cell does not show the axis; ImgDataP, which VSX creates
    # the same way, is a MATLAB column (h5py (1, n)), so that is assumed.
    "RcvData": Cell(Array("int16", "3d"), axis=1),
    "ImgDataP": Cell(Array("double", "4d"), axis=1),
    "save_raw_version": CHAR,
    # Written by the .bin saving script (see zea.data.convert._verasonics_bin).
    "RF_rows": COLUMN,
    "RF_cols": COLUMN,
    "RF_frames": COLUMN,
    "NonzeroRFcolumns": Cell(Array("logical", "column"), axis=0),
}


class StructArray(dict):
    """Value of a MATLAB struct array, given as ``{field: [value per element]}``.

    A plain ``dict`` is a 1x1 struct, given as ``{field: value}``.
    """

    def __len__(self):
        lengths = {len(values) for values in self.values()}
        if len(lengths) != 1:
            raise ValueError(f"Struct array fields have different lengths: {sorted(lengths)}")
        return lengths.pop()


class CellArray(list):
    """Value of a MATLAB cell array: one value per cell."""


def _spec(path):
    """The schema entry for a ``/``-separated variable path."""
    spec = Struct(SCHEMA)
    for part in path.split("/"):
        if not isinstance(spec, Struct) or part not in spec.fields:
            raise KeyError(
                f"{path!r} is not in the Verasonics workspace SCHEMA. Add it, and check it "
                "against a real workspace (see tests/fixtures/verasonics_workspace.py), "
                "instead of guessing how MATLAB stores it."
            )
        spec = spec.fields[part]
    return spec


# --------------------------------------------------------------------------- writing


def _shaped(data, spec, path):
    """Lay out ``data`` the way ``spec`` stores it."""
    layout = spec.layout
    if layout == "scalar":
        if data.size != 1:
            raise ValueError(f"{path} is a scalar, got shape {data.shape}.")
        return data.reshape(1, 1)
    if layout in ("column", "row"):
        if data.ndim == 2 and data.size > 1:
            given = "column" if data.shape[1] == 1 else "row" if data.shape[0] == 1 else None
            if given != layout:
                raise ValueError(f"{path} is stored as a {layout} vector, got shape {data.shape}.")
        elif data.ndim > 2:
            raise ValueError(f"{path} is a {layout} vector, got shape {data.shape}.")
        return data.reshape(-1, 1) if layout == "column" else data.reshape(1, -1)
    ndim = int(layout[0])
    if data.ndim != ndim:
        raise ValueError(f"{path} has {ndim} dimensions, got shape {data.shape}.")
    return data


def _encode(value, spec, path):
    """Encode one MATLAB array as ``(data, attrs)``."""
    if value is None:
        if spec.empty is None:
            raise ValueError(f"{path} is never stored as an empty array.")
        return np.zeros(2, np.uint64), {"MATLAB_class": spec.empty, "MATLAB_empty": 1}
    attrs = {"MATLAB_class": spec.matlab_class}
    if spec.matlab_class in _INT_DECODE:
        attrs["MATLAB_int_decode"] = _INT_DECODE[spec.matlab_class]
    if spec.matlab_class == "char":
        if not isinstance(value, str):
            raise TypeError(f"{path} is a char array; pass a str, got {value!r}.")
        return np.array([ord(c) for c in value], np.uint16).reshape(-1, 1), attrs
    if isinstance(value, str):
        raise TypeError(f"{path} is a {spec.matlab_class} array, got the string {value!r}.")
    data = np.asarray(value).astype(_DTYPES[spec.matlab_class])
    return _shaped(data, spec, path), attrs


class _Writer:
    def __init__(self, f):
        self.f = f
        self.refs = f.require_group("#refs#")
        self.n_refs = 0

    def array(self, group, name, value, spec, path):
        data, attrs = _encode(value, spec, path)
        if name in group:
            del group[name]
        dset = group.create_dataset(name, data=data)
        for key, attr in attrs.items():
            if isinstance(attr, str):
                dset.attrs[key] = np.bytes_(attr)
            elif key == "MATLAB_int_decode":
                dset.attrs[key] = np.int32(attr)
            else:
                dset.attrs[key] = np.uint8(attr)
        return dset

    def references(self, group, name, values, spec, path, shape, matlab_class=None):
        targets = []
        for value in values:
            target = f"{self.n_refs:x}"
            self.n_refs += 1
            targets.append(self.array(self.refs, target, value, spec, path).ref)
        if name in group:
            del group[name]
        dset = group.create_dataset(name, shape=shape, dtype=h5py.ref_dtype)
        dset[...] = np.array(targets, dtype=h5py.ref_dtype).reshape(shape)
        if matlab_class is not None:
            dset.attrs["MATLAB_class"] = np.bytes_(matlab_class)

    def struct_group(self, group, name):
        if name not in group:
            group.create_group(name).attrs["MATLAB_class"] = np.bytes_("struct")
        return group[name]

    def write(self, group, name, value, spec, path):
        if isinstance(spec, Cell):
            if not isinstance(value, CellArray):
                raise TypeError(f"{path} is a MATLAB cell; pass a CellArray([...]).")
            shape = (len(value), 1) if spec.axis == 0 else (1, len(value))
            self.references(group, name, value, spec.element, path, shape, "cell")
        elif isinstance(spec, Struct):
            if isinstance(value, StructArray):
                if name in group:
                    del group[name]  # a struct array replaces the whole struct
                n = len(value)
                child = self.struct_group(group, name)
                for field_name, values in value.items():
                    field_path = f"{path}/{field_name}"
                    field_spec = _spec(field_path)
                    if not isinstance(field_spec, Array):
                        raise TypeError(f"{field_path}: nested struct arrays are unsupported.")
                    if n == 1:
                        self.array(child, field_name, values[0], field_spec, field_path)
                    else:
                        self.references(child, field_name, values, field_spec, field_path, (n, 1))
            elif isinstance(value, dict):
                child = self.struct_group(group, name)
                for field_name, field_value in value.items():
                    field_path = f"{path}/{field_name}"
                    self.write(child, field_name, field_value, _spec(field_path), field_path)
            else:
                raise TypeError(f"{path} is a MATLAB struct; pass a dict or a StructArray.")
        else:
            self.array(group, name, value, spec, path)

    def finish(self):
        # MATLAB_fields lists a struct's fields, as an array of vlen char arrays.
        vlen = h5py.vlen_dtype(np.dtype("S1"))

        def visit(_name, obj):
            if isinstance(obj, h5py.Group) and obj.attrs.get("MATLAB_class") == b"struct":
                fields = np.empty(len(obj), dtype=object)
                fields[:] = [np.array(list(k), dtype="S1") for k in obj]
                obj.attrs.create("MATLAB_fields", fields, dtype=vlen)

        self.f.visititems(visit)


def _write_userblock(path):
    header = b"MATLAB 7.3 MAT-file, Platform: zea tests, Created by: zea test fixtures"
    header = header.ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM"
    with open(path, "r+b") as file:
        file.write(header)


def write_workspace(path, variables):
    """Write a MATLAB ``-v7.3`` workspace holding just ``variables``.

    Args:
        path (str | Path): Output ``.mat`` path.
        variables (dict): ``{path: value}``, where a path is a variable or a field such
            as ``"Resource/RcvBuffer"``. A value is a number, array, string or ``None``
            (empty), a :class:`CellArray`, a :class:`StructArray`, or a ``dict`` of
            fields for a 1x1 struct. Arrays are in h5py order (MATLAB's axes reversed);
            a 1-D array becomes the row or column :data:`SCHEMA` says.

    Returns:
        Path: ``path``.
    """
    with h5py.File(path, "w", userblock_size=512) as f:
        writer = _Writer(f)
        for name, value in variables.items():
            *parents, leaf = name.split("/")
            group = f
            for i, parent in enumerate(parents):
                _spec("/".join(parents[: i + 1]))
                group = writer.struct_group(group, parent)
            writer.write(group, leaf, value, _spec(name), name)
        writer.finish()
    _write_userblock(path)
    return Path(path)


def write_bin(path, data):
    """Write one RF buffer as ``RF_data_{k}.bin``: int16 little-endian, MATLAB column-major.

    Args:
        path (str | Path): Output ``.bin`` path.
        data (np.ndarray): ``(n_frames, n_saved_channels, n_samples)``, h5py order.
    """
    np.asarray(data).astype("<i2").tofile(path)


# --------------------------------------------------------------------------- workspace


@dataclass
class VerasonicsWorkspace:
    """A complete synthetic Verasonics workspace for a plane-wave sequence.

    Every RF buffer acquires ``n_frames`` frames of ``n_tx`` plane waves, one ``Receive``
    and one acquisition ``Event`` per transmit, followed by a ``transferToHost`` event per
    frame. Each receive stores ``n_ax`` samples, so a frame holds ``n_tx * n_ax`` samples
    per channel. ``Trans.ConnectorES`` maps the ``n_elements`` elements to some of the
    ``n_channels`` hardware channels.

    Change :attr:`variables` (MATLAB-level values, see :func:`write_workspace`) or
    :attr:`rf` before :meth:`write` to build an edge case.

    Args:
        rf_storage (str): ``"rcvdata"`` to store the RF in the ``.mat`` as ``RcvData``,
            or ``"bin"`` for ``RF_data_{k}.bin`` files next to it.
        n_buffers (int): Number of RF buffers.
        n_frames (int | list[int]): Frames per buffer.
        n_tx (int): Plane-wave transmits per frame.
        n_elements (int): Probe elements.
        n_channels (int): Hardware channels of the RF buffers.
        n_ax (int): Samples per receive.
        sample_mode (str): ``Receive.sampleMode``.
        n_image_buffers (int): ``ImgDataP`` image buffers (0 for none).
        save_raw_version (str | None): ``save_raw_version`` of our saving script, or
            None to leave it out (as in the ``.bin`` workspaces).
        seed (int): Seed for the RF and image data.
    """

    rf_storage: str = "rcvdata"
    n_buffers: int = 1
    n_frames: int | list = 2
    n_tx: int = 3
    n_elements: int = 16
    n_channels: int = 32
    n_ax: int = 64
    sample_mode: str = "NS200BW"
    n_image_buffers: int = 2
    save_raw_version: str | None = "1.0"
    seed: int = 0
    variables: dict = field(init=False)
    rf: list = field(init=False)

    center_frequency_mhz = 5.0
    sound_speed = 1540.0
    pitch_mm = 0.3

    def __post_init__(self):
        if self.rf_storage not in ("rcvdata", "bin"):
            raise ValueError(f"rf_storage must be 'rcvdata' or 'bin', got {self.rf_storage!r}.")
        n_frames = self.n_frames
        self.frames_per_buffer = (
            list(n_frames) if isinstance(n_frames, (list, tuple)) else [n_frames] * self.n_buffers
        )
        rng = np.random.default_rng(self.seed)
        n_samples = self.n_tx * self.n_ax
        self.rf = [
            rng.integers(-2000, 2000, (n, self.n_channels, n_samples), dtype=np.int16)
            for n in self.frames_per_buffer
        ]
        self.variables = {
            "Trans": self._trans(),
            "Resource": self._resource(),
            "TW": {
                "type": "parametric",
                "Parameters": [self.center_frequency_mhz, 0.67, 2, 1],
                "Wvfm1Wy": np.sin(np.linspace(0, 4 * np.pi, 50)),
                "Wvfm2Wy": np.sin(np.linspace(0, 8 * np.pi, 100)),
            },
            "TX": self._tx(),
            "TGC": {"Waveform": np.linspace(0, 1023, 512)},
            **self._receives_and_events(),
            "SeqControl": StructArray(
                {"command": ["timeToNextAcq", "transferToHost"], "argument": [100.0, None]}
            ),
        }
        if self.rf_storage == "rcvdata":
            self.variables["RcvData"] = CellArray(self.rf)
        else:
            self.variables.update(
                {
                    "RF_rows": [n_samples] * self.n_buffers,
                    "RF_cols": [self.n_elements] * self.n_buffers,
                    "RF_frames": self.frames_per_buffer,
                    "NonzeroRFcolumns": CellArray([self.channel_mask] * self.n_buffers),
                }
            )
        if self.n_image_buffers:
            self.variables["ImgDataP"] = CellArray(
                [rng.random((3, 1, 20, 16)) for _ in range(self.n_image_buffers)]
            )
        if self.save_raw_version is not None:
            self.variables["save_raw_version"] = self.save_raw_version

    @property
    def connector(self):
        """1-based hardware channel of each element (``Trans.ConnectorES``)."""
        return np.arange(self.n_elements) * (self.n_channels // self.n_elements) + 1

    @property
    def channel_mask(self):
        """Hardware channels connected to an element (``NonzeroRFcolumns``)."""
        mask = np.zeros(self.n_channels, bool)
        mask[self.connector - 1] = True
        return mask

    def _trans(self):
        x = (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.pitch_mm
        element_pos = np.zeros((4, self.n_elements))  # x, y, z, angle per element
        element_pos[0] = x
        return {
            "name": "custom",
            "units": "mm",
            "type": 0,
            "frequency": self.center_frequency_mhz,
            "elementWidth": 0.25,
            "lensCorrection": 0.5,
            "Bandwidth": [3.5, 6.5],
            "ElementPos": element_pos,
            "ConnectorES": self.connector,
        }

    def _resource(self):
        resource = {
            "Parameters": {"speedOfSound": self.sound_speed},
            "RcvBuffer": StructArray(
                {
                    "numFrames": self.frames_per_buffer,
                    "lastFrame": self.frames_per_buffer,  # the buffer was filled in order
                }
            ),
        }
        if self.n_image_buffers:
            resource["ImageBuffer"] = StructArray(
                {
                    "numFrames": [3] * self.n_image_buffers,
                    "firstFrame": [1] * self.n_image_buffers,
                    "lastFrame": [3] * self.n_image_buffers,
                }
            )
        return resource

    def _tx(self):
        wavelength_mm = self.sound_speed / self.center_frequency_mhz / 1e3
        x_wl = (np.arange(self.n_elements) - (self.n_elements - 1) / 2) * self.pitch_mm
        x_wl = x_wl / wavelength_mm
        angles = np.linspace(-0.2, 0.2, self.n_tx) if self.n_tx > 1 else np.zeros(1)
        delays = [x_wl * np.sin(a) - (x_wl * np.sin(a)).min() for a in angles]
        n = self.n_tx
        return StructArray(
            {
                "waveform": [1] * n,
                "Origin": [np.zeros(3)] * n,
                "focus": [0.0] * n,
                "Steer": [np.array([a, 0.0]) for a in angles],
                "Apod": [np.ones(self.n_elements)] * n,
                "Delay": delays,
            }
        )

    def _receives_and_events(self):
        receive = {name: [] for name in SCHEMA["Receive"].fields}
        event = {"tx": [], "rcv": [], "seqControl": []}
        for buffer, n_frames in enumerate(self.frames_per_buffer):
            for frame in range(n_frames):
                for tx in range(self.n_tx):
                    values = {
                        "bufnum": buffer + 1,
                        "framenum": frame + 1,
                        "acqNum": tx + 1,
                        "mode": 0,
                        "startSample": tx * self.n_ax + 1,
                        "endSample": (tx + 1) * self.n_ax,
                        "startDepth": 0.0,
                        "endDepth": 100.0,
                        "sampleMode": self.sample_mode,
                        "demodFrequency": self.center_frequency_mhz,
                        "decimSampleRate": 4 * self.center_frequency_mhz,
                        "quadDecim": 1.0,
                        "TGC": 1,
                    }
                    for name, value in values.items():
                        receive[name].append(value)
                    event["tx"].append(tx + 1)
                    event["rcv"].append(len(receive["bufnum"]))
                    event["seqControl"].append([1])  # timeToNextAcq
                event["tx"].append(0)
                event["rcv"].append(0)
                event["seqControl"].append([2])  # transferToHost
        return {"Receive": StructArray(receive), "Event": StructArray(event)}

    def write(self, path, saved_buffers=None):
        """Write the workspace to ``path``, and with ``bin`` storage its ``.bin`` files.

        Args:
            path (str | Path): Output ``.mat`` path.
            saved_buffers (list[int], optional): 0-based buffers whose ``.bin`` is
                written (``bin`` storage only). Defaults to all.

        Returns:
            Path: ``path``.
        """
        path = Path(path)
        if self.rf_storage == "bin":
            for k, rf in enumerate(self.rf):
                if saved_buffers is None or k in saved_buffers:
                    write_bin(path.parent / f"RF_data_{k + 1}.bin", rf[:, self.channel_mask, :])
        return write_workspace(path, self.variables)


# --------------------------------------------------------------------------- checking


def _check_array(obj, spec, path):
    """Problems with how the dataset ``obj`` stores the array ``spec``."""
    if not isinstance(obj, h5py.Dataset) or h5py.check_dtype(ref=obj.dtype) is not None:
        return [f"{path}: expected a plain {spec.matlab_class} dataset"]
    matlab_class = obj.attrs.get("MATLAB_class", b"").decode()
    if obj.attrs.get("MATLAB_empty"):
        if matlab_class != spec.empty:
            return [f"{path}: empty with MATLAB_class={matlab_class!r}, expected {spec.empty!r}"]
        return []
    if matlab_class != spec.matlab_class or obj.dtype != np.dtype(_DTYPES[spec.matlab_class]):
        return [
            f"{path}: {matlab_class} {obj.dtype}, expected {spec.matlab_class} "
            f"{np.dtype(_DTYPES[spec.matlab_class])}"
        ]
    shape = obj.shape
    layout = spec.layout
    ok = {
        "scalar": shape == (1, 1),
        "column": len(shape) == 2 and shape[1] == 1,
        "row": len(shape) == 2 and shape[0] == 1,
    }.get(layout, len(shape) == (int(layout[0]) if layout[0].isdigit() else -1))
    return [] if ok else [f"{path}: shape {shape} is not a {layout}"]


def _check(f, obj, spec, path, n_struct=None):
    if isinstance(spec, Struct):
        if not isinstance(obj, h5py.Group) or obj.attrs.get("MATLAB_class") != b"struct":
            return [f"{path}: expected a struct group"]
        # Fields of a struct array are references without a class; a cell field is not.
        sizes = {
            obj[k].size if h5py.check_dtype(ref=obj[k].dtype) is not None else 1
            for k in obj
            if isinstance(obj[k], h5py.Dataset) and obj[k].attrs.get("MATLAB_class") != b"cell"
        }
        if len(sizes) > 1:
            return [f"{path}: fields hold different numbers of elements {sorted(sizes)}"]
        n = sizes.pop() if sizes else 1
        problems = []
        for name, field_spec in spec.fields.items():
            if name in obj:
                problems += _check(f, obj[name], field_spec, f"{path}/{name}", n)
        return problems
    if isinstance(spec, Cell):
        if obj.attrs.get("MATLAB_class") != b"cell" or h5py.check_dtype(ref=obj.dtype) is None:
            return [f"{path}: expected a cell (references with MATLAB_class=cell)"]
        if 1 not in obj.shape or (obj.size > 1 and obj.shape[1 - spec.axis] != 1):
            return [f"{path}: cell of shape {obj.shape} is not laid out along axis {spec.axis}"]
        problems = []
        for i, ref in enumerate(obj[()].reshape(-1)):
            problems += _check_array(f[ref], spec.element, f"{path}{{{i + 1}}}")
        return problems
    if n_struct is not None and n_struct > 1:
        if h5py.check_dtype(ref=obj.dtype) is None:
            return [f"{path}: field of a struct array is not stored as references"]
        # Elements with the same storage are only checked once; reading every attribute
        # of a 30000-element struct array takes a while.
        problems, seen = [], set()
        for i, ref in enumerate(obj[()].reshape(-1)):
            element = f[ref]
            key = (element.shape, element.dtype.str, tuple(sorted(element.attrs.items())))
            if key not in seen:
                seen.add(key)
                problems += _check_array(element, spec, f"{path}({i + 1})")
        return problems
    return _check_array(obj, spec, path)


def check(path):
    """Problems with how a real workspace stores the :data:`SCHEMA` variables.

    Args:
        path (str | Path): A MATLAB ``-v7.3`` ``.mat`` file.

    Returns:
        tuple: ``(problems, missing)``, lists of messages for variables stored
        differently from :data:`SCHEMA` and of schema variables the file lacks.
    """
    with h5py.File(path, "r") as f:
        problems, missing = [], []
        for name, spec in SCHEMA.items():
            if name in f:
                problems += _check(f, f[name], spec, name)
            else:
                missing.append(name)
        return problems, missing


def main():
    parser = argparse.ArgumentParser(description="Check real workspaces against SCHEMA.")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="check .mat files against SCHEMA")
    check_parser.add_argument("src", nargs="+", help="MATLAB -v7.3 .mat files")
    args = parser.parse_args()
    failed = False
    for src in args.src:
        problems, missing = check(src)
        print(f"{src}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
        for problem in problems:
            print(f"  {problem}")
        if missing:
            print(f"  not in this file: {', '.join(missing)}")
        failed |= bool(problems)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
