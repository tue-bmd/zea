"""zea data file (HDF5)."""

import enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import h5py
import numpy as np

import zea
from zea import log
from zea.data.legacy_file import legacy_data, legacy_probe, legacy_scan
from zea.data.spec import (
    DEFAULT_COMPRESSION,
    DataSpec,
    FileSpec,
    MetadataSpec,
    MetricsSpec,
    ProbeSpec,
    ScanSpec,
)
from zea.internal.checks import _DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.internal.core import DataTypes
from zea.internal.preset_utils import HF_PREFIX, _hf_resolve_path
from zea.internal.utils import deprecated

if TYPE_CHECKING:
    from zea.probes import Probe
    from zea.scan import Parameters


class _StringDataset:
    """Thin wrapper around an h5py string Dataset that auto-decodes bytes on read.

    h5py returns variable-length or fixed-length string datasets as ``bytes``
    objects.  This wrapper intercepts ``__getitem__`` and converts any bytes
    elements to ``str`` so callers always receive a ``numpy.ndarray`` with
    ``dtype=numpy.str_``.
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: h5py.Dataset):
        self._ds = ds

    def __getitem__(self, key):
        data = self._ds[key]
        if isinstance(data, np.ndarray):
            decoded = np.array(
                [v.decode() if isinstance(v, bytes) else str(v) for v in data.flat],
                dtype=np.str_,
            ).reshape(data.shape)
            return decoded
        return data.decode() if isinstance(data, bytes) else str(data)

    def __getattr__(self, name: str):
        return getattr(self._ds, name)

    def __len__(self):
        return len(self._ds)

    def __repr__(self):
        return f"<StringDataset shape={self._ds.shape} dtype=str>"


class _GroupProxy:
    """Lazy proxy for an h5py.Group that exposes children as attributes.

    Datasets are returned as-is (h5py.Dataset supports slicing without
    loading everything into RAM).  Sub-groups are wrapped in another
    ``GroupProxy`` so the dot-access pattern works recursively::

        with File(path) as f:
            # returns h5py.Dataset – no data loaded yet
            f.data.raw_data
            # slicing triggers the actual read, just like plain h5py
            f.data.raw_data[:, :n_tx]
            # nested groups work too
            f.data.image.values[0]

    String datasets are automatically wrapped in :class:`_StringDataset` so
    slicing always returns ``numpy.ndarray`` with ``dtype=numpy.str_`` rather
    than raw ``bytes``.
    """

    __slots__ = ("_group",)

    def __init__(self, group: h5py.Group):
        self._group = group

    def __getattr__(self, name: str):
        try:
            child = self._group[name]
        except KeyError:
            raise AttributeError(
                f"No key '{name}' in group '{self._group.name}'. "
                f"Available keys: {list(self._group.keys())}"
            )
        if isinstance(child, h5py.Group):
            return _GroupProxy(child)
        if isinstance(child, h5py.Dataset) and h5py.check_string_dtype(child.dtype):
            return _StringDataset(child)
        return child  # h5py.Dataset – supports slicing natively

    def __dir__(self):
        return list(self._group.keys())

    def __repr__(self):
        return repr(self._group)

    def keys(self):
        """Return the keys of the underlying group."""
        return self._group.keys()

    def __contains__(self, key):
        return key in self._group

    def __iter__(self):
        return iter(self._group)


def assert_key(file: h5py.File, key: str):
    """Asserts key is in a h5py.File."""
    if key not in file.keys():
        raise KeyError(f"{key} not found in file")


class Track:
    """A single acquisition track within a :class:`File`.

    Provides the same ``.data``, ``.scan`` and ``.load_parameters()`` interface
    as :class:`File` but scoped to one ``tracks/track_N`` group.  Obtain
    instances through :attr:`File.tracks` rather than constructing this class
    directly.

    Example::

        with File("multi_track.hdf5") as f:
            for track in f.tracks:
                raw = track.data.raw_data[:]
                parameters = track.load_parameters()
    """

    __slots__ = ("_index", "_group", "_timestamps", "_label", "_probe")

    def __init__(
        self,
        index: int,
        group: "h5py.Group",
        timestamps: "np.ndarray | None" = None,
        label: "str | None" = None,
        probe: "dict | None" = None,
    ):
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_timestamps", timestamps)
        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_probe", probe)

    @property
    def data(self) -> _GroupProxy:
        """Lazy proxy for this track's ``data`` group."""
        if "data" not in self._group:
            raise KeyError(
                f"Track {self._index} has no 'data' group. "
                f"Available keys: {list(self._group.keys())}"
            )
        return _GroupProxy(self._group["data"])

    @property
    def scan(self) -> "ScanSpec":
        """Return the validated :class:`~zea.data.spec.ScanSpec` for this track.

        This is the bare scan group as a spec object.  For a full, derivable
        parameter object (merged probe + scan) use :meth:`load_parameters`.
        """
        if "scan" not in self._group:
            raise KeyError(
                f"Track {self._index} has no 'scan' group. "
                f"Available keys: {list(self._group.keys())}"
            )
        scan_dict = load_dict_from_hdf5_group(self._group["scan"])

        return ScanSpec(**scan_dict)

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return _shape_from_data_group(
            self._group["data"],
            index=0,
            name="n_frames",
            requires_raw=False,
        )

    @property
    def n_tx(self) -> int:
        """Number of transmit events."""
        return _shape_from_data_group(
            self._group["data"],
            index=1,
            name="n_tx",
            requires_raw=True,
        )

    @property
    def n_ax(self) -> int:
        """Number of axial samples."""
        return _shape_from_data_group(
            self._group["data"],
            index=2,
            name="n_ax",
            requires_raw=True,
        )

    @property
    def n_el(self) -> int:
        """Number of elements."""
        return _shape_from_data_group(
            self._group["data"],
            index=3,
            name="n_el",
            requires_raw=True,
        )

    def load_parameters(self, **overrides) -> "Parameters":
        """Load this track's parameters (merged probe + scan) as :class:`~zea.Parameters`.

        Each track shares the same probe but has its own scan, so the returned
        object has the same shape as :meth:`File.load_parameters` for a
        single-track file.

        Args:
            **overrides: Override any parameter.

        Returns:
            Parameters: Initialised parameters object for this track.
        """
        from zea.scan import Parameters

        if "scan" not in self._group:
            raise KeyError(
                f"Track {self._index} has no 'scan' group. "
                f"Available keys: {list(self._group.keys())}"
            )

        scan = self.scan
        scan_dict = scan.to_dict()
        other_dict = {
            "n_ax": self.n_ax,
            "n_el": scan.n_el,
            "n_tx": scan.n_tx,
        }
        merged_dict = {**(self._probe or {}), **scan_dict, **other_dict, **overrides}

        return Parameters(**merged_dict)

    @property
    def label(self) -> "str | None":
        """Human-readable name for this track (e.g. ``'focused'`` or ``'planewave'``).

        Returns ``None`` for single-track files or legacy files written without a label.
        Use :attr:`File.track_labels` to print all labels in acquisition order and
        :meth:`File.get_track` to retrieve a track by name.
        """
        return self._label

    @property
    def timestamps(self) -> "np.ndarray | None":
        """Global transmit timestamps for this track, shape ``(n_frames, n_tx)``.

        Timestamps are pre-computed when the :class:`Track` is created via
        :attr:`File.tracks`.  Returns ``None`` if the file has no
        ``track_schedule`` or any track is missing ``time_to_next_transmit``.
        """
        return self._timestamps

    def __repr__(self) -> str:
        label_part = f' "{self._label}"' if self._label is not None else ""
        keys = list(self._group.get("data", {}).keys())
        return f"<Track[{self._index}]{label_part} data={keys}>"


def load_dict_from_hdf5_group(group: "h5py.Group") -> dict:
    """Recursively load the contents of an HDF5 group into a plain dict.

    Datasets are returned as numpy arrays or scalars; nested groups are
    converted recursively.  String datasets are decoded to ``np.str_``.

    Args:
        group: An open :class:`h5py.Group` (or :class:`h5py.File`).

    Returns:
        dict: Nested dictionary mirroring the group structure.
    """
    ans = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            if h5py.check_string_dtype(item.dtype) is not None:
                val = item.asstr()[()]
                if isinstance(val, np.ndarray) and val.dtype == object:
                    val = val.astype(np.str_)
                ans[key] = val
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            ans[key] = load_dict_from_hdf5_group(item)
    return ans


def _get_data_array_shape(data_group: "h5py.Group") -> "tuple | None":
    """Return the shape one of the data arrays in *data_group*.

    Checks flat datasets first (e.g., ``raw_data``).
    Then looks for the first spatial map group (e.g., ``image``) containing a ``values``
    dataset and returns its shape.

    Returns ``None`` if neither is found.
    """
    flat_datasets = [k for k in data_group if isinstance(data_group[k], h5py.Dataset)]
    spatial_map_groups = [k for k in data_group if isinstance(data_group[k], h5py.Group)]

    has_raw_data = False

    # first check for one of the recognized data arrays
    for key in _DATA_TYPES:
        if key in flat_datasets:
            if key == "raw_data":
                has_raw_data = True
            return data_group[key].shape, has_raw_data
        if key in spatial_map_groups:
            group = data_group[key]
            if "values" in group and isinstance(group["values"], h5py.Dataset):
                return group["values"].shape, has_raw_data

    # if none of the arrays are found under their expected names,
    # look for any other dataset
    for key in flat_datasets:
        return data_group[key].shape, has_raw_data

    for key in spatial_map_groups:
        group = data_group[key]
        if "values" in group and isinstance(group["values"], h5py.Dataset):
            return group["values"].shape, has_raw_data

    return None, has_raw_data


def _shape_from_data_group(
    data_group: "h5py.Group", index: int, name: str = "", requires_raw: bool = False
) -> int:
    shape, has_raw_data = _get_data_array_shape(data_group)
    if requires_raw and not has_raw_data:
        raise TypeError(f"`{name}` is only available if the file contains a `raw_data` dataset")

    if shape is None:
        raise TypeError(
            f"Cannot determine `{name}`, no recognized data arrays found in the data group."
        ) from None
    return shape[index]


def _compute_all_track_timestamps(
    schedule: "np.ndarray",
    tracks: "list[Track]",
) -> "list[np.ndarray | None]":
    """Compute and return timestamps for every track given a track schedule.

    Walks the schedule once, keeping a single scalar cumulative timestamp and
    a per-track ``[frame_idx, tx_idx]`` counter to index into each track's
    ``time_to_next_transmit`` array.  The schedule must cover exactly
    ``sum(n_frames_t * n_tx_t)`` events across all tracks.

    Args:
        schedule: ``int32`` array mapping each global transmit event to a track
            index, shape ``(n_total_tx,)``.
        tracks: :class:`Track` list (without timestamps yet assigned).

    Returns:
        list: One ``np.ndarray`` of shape ``(n_frames_t, n_tx_t)`` per track,
        or a list of ``None`` values if timestamps cannot be computed.
    """
    n_tracks = len(tracks)
    t2nts: list[np.ndarray] = []

    # pre-load time_to_next_transmit for each track,
    # validating that it's present and has the right shape
    for track in tracks:
        t2nt = track.scan.time_to_next_transmit
        if t2nt is None:
            log.warning(
                f"Track {track._index} has no 'time_to_next_transmit';"
                " cannot compute track timestamps."
            )
            return [None] * n_tracks
        t2nts.append(np.asarray(t2nt, dtype=np.float32))

    n_frames_per_track = [t2nt.shape[0] for t2nt in t2nts]
    n_tx_per_frame_per_track = [t2nt.shape[1] for t2nt in t2nts]

    # results will be stored here as we walk the schedule
    timestamp_matrices_per_track = [np.zeros_like(t2nt) for t2nt in t2nts]
    # counters to keep track of where we are in each track's
    # timestamp matrix.
    track_counters = [[0, 0] for _ in tracks]  # [frame_idx, tx_idx] per track

    cumulative_timestamp = 0.0

    # walk through the schedule, filling in the timestamp matrices as we go
    for track_idx in schedule:
        frame_idx, tx_idx = track_counters[track_idx]
        timestamp_matrices_per_track[track_idx][frame_idx, tx_idx] = cumulative_timestamp
        cumulative_timestamp += float(t2nts[track_idx][frame_idx, tx_idx])

        # update the counters keeping track of where we are in this track's timestamp matrix
        tx_idx += 1
        if tx_idx >= n_tx_per_frame_per_track[track_idx]:
            tx_idx = 0
            frame_idx += 1
        track_counters[track_idx] = [frame_idx, tx_idx]

    for i, (frame_idx, _) in enumerate(track_counters):
        assert frame_idx == n_frames_per_track[i], (
            f"There was a mismatch between the track_schedule and the number of frames and "
            f"transmits in track {i}. "
            f"Please ensure that the track_schedule correctly maps to the global number of "
            f"transmit events."
        )

    return timestamp_matrices_per_track


def _parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split(".")[:3] if p.isdigit())


def _warn_if_legacy_file(file: "File") -> None:
    """Warn if *file* has no zea_version or was written before v0.1.0."""
    version = file.attrs.get("zea_version", None)
    if version is None or _parse_version(version) < (0, 1, 0):
        legacy_version = version if version is not None else "<0.1.0"
        log.warning_once(
            f"This ``zea.File`` '{file.filename}' was created with a legacy version of "
            f"zea ({legacy_version}), while you are using zea v{zea.__version__}. "
            "It may behave in unexpected ways. Install an earlier version of zea<0.1.0 for full "
            "compatibility or re-save the file with zea v0.1.0 or later (e.g. via File.create).",
            key=file.filename,
        )


def _warn_custom_keys(data: dict, metadata: dict):
    """Warn about custom keys in data/metadata dicts when saving."""
    custom_maps = [k for k in data if k not in DataSpec.SCHEMA]
    if custom_maps:
        supported = ", ".join(k for k, v in DataSpec.SCHEMA.items() if "spec" in v)
        log.warning(
            f"Custom spatial map key(s) added to 'data': {', '.join(sorted(custom_maps))}. "
            "These are validated as generic Map specs. "
            "If your data matches an existing type, prefer one of the supported "
            f"spatial maps: {supported}."
        )
    custom_signals = [k for k in metadata if k not in MetadataSpec.SCHEMA]
    if custom_signals:
        supported = ", ".join(k for k, v in MetadataSpec.SCHEMA.items() if "spec" in v)
        log.warning(
            f"Custom signal key(s) added to 'metadata': {', '.join(sorted(custom_signals))}. "
            "These are validated as generic SignalND specs. "
            "If your signal matches an existing type, prefer one of the supported "
            f"signal fields: {supported}."
        )


class File(h5py.File):
    """h5py.File in zea format."""

    def __init__(self, name, mode="r", *args, **kwargs):
        """Initialize the file.

        Args:
            name (str, Path, HFPath): The path to the file.
                Can be a string or a Path object. Additionally can be a string with
                the prefix 'hf://', in which case it will be resolved to a
                huggingface path.
            mode (str, optional): The mode to open the file in. Defaults to "r".
            revision (str, optional): HuggingFace revision (branch, tag, or commit hash)
                to download from. Only used when ``name`` starts with ``hf://``.
                Defaults to ``"main"``. Example: ``revision="v0.1.0"``.
            repo_type (str, optional): HuggingFace repository type. Only used when
                ``name`` starts with ``hf://``. Defaults to ``"dataset"``.
            cache_dir (str or Path, optional): Local cache directory for downloaded
                HuggingFace files. Only used when ``name`` starts with ``hf://``.
            *args: Additional arguments to pass to h5py.File.
            **kwargs: Additional keyword arguments to pass to h5py.File.
        """
        # First check if the file is an HDF5 file
        assert str(name).endswith(".hdf5") or str(name).endswith(".h5"), (
            "File must be an HDF5 file with .hdf5 or .h5 extension."
        )

        # Extract HF-only kwargs so they never reach h5py
        hf_kwargs = {}
        for key in ("revision", "repo_type", "cache_dir"):
            if key in kwargs:
                hf_kwargs[key] = kwargs.pop(key)

        # Resolve huggingface path
        if str(name).startswith(HF_PREFIX):
            name = _hf_resolve_path(str(name), **hf_kwargs)

        # Disable locking for read mode by default
        if "locking" not in kwargs and mode == "r":
            # If the file is opened in read mode, disable locking
            kwargs["locking"] = False

        # Initialize the h5py.File
        super().__init__(name, mode, *args, **kwargs)

        # Warn when opening an existing file that pre-dates zea v0.1.0
        if mode in ("r", "r+"):
            _warn_if_legacy_file(self)

    def __contains__(self, key):
        """Check whether *key* exists in the file.

        Extends the h5py default to also match legacy short-form keys
        (``"scan"``, ``"data"``) against the tracks layout
        (``tracks/track_0/scan``, ``tracks/track_0/data``), including
        sub-paths like ``"data/segmentation"``.
        """
        if super().__contains__(key):
            return True
        # Handle both "data" and "data/..." paths — only remap for single-track files.
        parts = key.split("/", 1)
        if parts[0] in ("scan", "data"):
            if super().__contains__("tracks"):
                if len(self["tracks"]) == 1:
                    remapped = f"tracks/track_0/{key}"
                    return super().__contains__(remapped)
                else:
                    log.warning(
                        f"Multiple tracks found; Try accessing '{key}' on a specific track instead."
                    )
            return False
        return False

    def __getitem__(self, key):
        """Open an object in the file.

        Extends the h5py default to redirect ``"data"`` and ``"scan"`` (and
        sub-paths like ``"data/segmentation"``) to the tracks layout for
        single-track new-format files.  Multi-track files raise :exc:`AttributeError`.
        """
        parts = key.split("/", 1)
        if parts[0] in ("data", "scan") and not super().__contains__(key):
            n = self._n_tracks
            if n > 1:
                raise AttributeError(
                    f"This file has {n} tracks; use file.tracks to access each one."
                )
            if n == 1:
                return super().__getitem__(f"tracks/track_0/{key}")
        return super().__getitem__(key)

    @property
    def path(self):
        """Return the path of the file."""
        return Path(self.filename)

    @property
    def zea_version(self) -> str | None:
        """Return the zea version that wrote this file, or ``None`` for legacy files.

        Files created with zea v0.1.0 and later store a ``zea_version``
        root attribute.  Files written before zea v0.1.0 return ``None``.
        """
        return self.attrs.get("zea_version", None)

    @property
    def _n_tracks(self) -> int:
        """Return the number of tracks stored in this file.

        Returns 0 for files with neither a ``tracks/`` group nor a root-level
        ``data/`` group, 1 for flat-layout files (no tracks group), and the actual
        track count for files written with the multi-track layout.
        """
        if not self.id.valid:
            raise ValueError(
                "File is closed. Use 'with File(...) as f:' or call f.close() "
                "explicitly after you're done."
            )
        if "tracks" not in self:
            return 1 if (super().__contains__("data") or super().__contains__("scan")) else 0
        tracks_group = self["tracks"]
        count = 0
        while f"track_{count}" in tracks_group:
            count += 1
        return count

    @property
    def tracks(self) -> "list[Track]":
        """Return a list of :class:`Track` objects, one per track.

        Each track exposes ``.data`` (a :class:`GroupProxy`), ``.scan`` (a
        :class:`~zea.data.spec.ScanSpec`) and ``.load_parameters()`` (a
        :class:`~zea.Parameters` factory method) for that specific track.

        Raises:
            AttributeError: For flat-layout files that have no
                ``tracks/`` group — use :attr:`data` and :meth:`scan`
                directly for those.

        Example::

            with File("multi_track.hdf5") as f:
                for track in f.tracks:
                    raw = track.data.raw_data[:]
                    parameters = track.load_parameters()
        """
        if "tracks" not in self:
            raise AttributeError(
                "This file uses the flat layout (no 'tracks' group). "
                "Access data and parameters directly with file.data and file.load_parameters()."
            )
        tracks_group = self["tracks"]
        # Load file-level probe once so every track's scan can supplement its
        # scan parameters with probe_geometry, element_width, etc.
        probe_dict: "dict | None" = None
        if super().__contains__("probe"):
            probe_dict = load_dict_from_hdf5_group(self["probe"])
        tracks: list[Track] = []
        i = 0
        while f"track_{i}" in tracks_group:
            track_group = tracks_group[f"track_{i}"]
            label = None
            if "label" in track_group:
                raw = track_group["label"][()]
                label = raw.decode() if isinstance(raw, bytes) else str(raw)
            tracks.append(Track(i, track_group, label=label, probe=probe_dict))
            i += 1

        schedule = self.track_schedule
        if schedule is not None:
            all_timestamps = _compute_all_track_timestamps(schedule, tracks)
            for track, ts in zip(tracks, all_timestamps):
                object.__setattr__(track, "_timestamps", ts)
        else:
            log.warning_once(
                "`track_schedule` was not found in the file; cannot compute track timestamps."
            )

        return tracks

    @property
    def track_labels(self) -> "list[str | None]":
        """Labels of all tracks in acquisition order.

        Returns a list with one entry per track.  Each entry is the label
        string stored on that track, or ``None`` for unlabelled tracks (e.g.
        single-track or legacy files).  The list order matches
        :attr:`tracks`, so unpacking ``f.tracks`` in the same order as
        ``f.track_labels`` is always safe.

        Example::

            with File("acquisition.hdf5") as f:
                print(f.track_labels)  # ['focused', 'planewave']
                focused, planewave = f.tracks  # safe — same order
        """
        if "tracks" not in self:
            return []
        tracks_group = self["tracks"]
        labels = []
        i = 0
        while f"track_{i}" in tracks_group:
            tg = tracks_group[f"track_{i}"]
            if "label" in tg:
                raw = tg["label"][()]
                labels.append(raw.decode() if isinstance(raw, bytes) else str(raw))
            else:
                labels.append(None)
            i += 1
        return labels

    def get_track(self, label: str) -> "Track":
        """Return the track with the given label.

        Args:
            label: The exact label string assigned to the desired track.

        Returns:
            Track: The matching :class:`Track` object.

        Raises:
            KeyError: If no track with that label exists, with a message
                listing the available labels so the error is self-diagnosing.

        Example::

            with File("acquisition.hdf5") as f:
                focused = f.get_track("focused")
                raw = focused.data.raw_data[:]
        """
        for track in self.tracks:
            if track.label == label:
                return track
        available = [t.label for t in self.tracks]
        raise KeyError(
            f"No track with label {label!r}. Available labels (in acquisition order): {available}"
        )

    @property
    def track_schedule(self) -> "np.ndarray | None":
        """Track index for each global transmit event, shape ``(n_total_tx,)``.

        Returns an ``int32`` array that maps every transmit event (in
        acquisition order) to the track it belongs to, or ``None`` if no
        ``track_schedule`` dataset was stored in this file.

        Example::

            with File("multi_track.hdf5") as f:
                sched = f.track_schedule  # e.g. array([0, 1, 0, 1, ...])
        """
        if "track_schedule" not in self:
            return None
        return self["track_schedule"][()].astype(np.int32)

    @property
    def _scan_h5_group(self) -> "h5py.Group | None":
        """Return the HDF5 scan group for single-track or flat-layout files.

        Track format (single track): ``tracks/track_0/scan/``
        Flat layout (no tracks group): ``scan/`` at root
        Returns ``None`` when neither is present.

        Raises:
            AttributeError: For multi-track files — use ``file.tracks[i]`` instead.
        """
        n = self._n_tracks
        if n > 1:
            raise AttributeError(
                f"This file has {n} tracks. "
                "Use file.tracks[i].scan to access a specific track's scan parameters."
            )
        if "tracks" in self:
            track0 = self["tracks"].get("track_0")
            if track0 is not None and "scan" in track0:
                return track0["scan"]
        if super().__contains__("scan"):
            return self["scan"]
        return None

    @classmethod
    def create(
        cls,
        path,
        data: dict | None = None,
        scan: dict | None = None,
        tracks: list | None = None,
        track_schedule: "np.ndarray | None" = None,
        metadata: dict | None = None,
        metrics: dict | None = None,
        probe_name: str | None = None,
        probe: "ProbeSpec | dict | None" = None,
        us_machine: str | None = None,
        description: str | None = None,
        compression: str = DEFAULT_COMPRESSION,
        chunk_frames: bool = False,
        overwrite: bool = False,
    ):
        """Create a new zea HDF5 file from data, scan, and optional metadata.

        All inputs are validated against the :class:`~zea.data.spec.FileSpec`
        schema (dtypes, shapes, dimension consistency) **before** anything is
        written to disk.

        For single-track files, supply ``data`` and ``scan``.  For multi-track
        files, supply ``tracks`` (a list of dicts with ``"data"`` and ``"scan"``
        keys, or :class:`~zea.data.spec.TrackSpec` objects) and optionally
        ``track_schedule``.

        Args:
            path: Destination file path.
            data: Data dict accepted by :class:`~zea.data.spec.DataSpec`.
                Mutually exclusive with ``tracks``.
            scan: Scan-parameter dict accepted by :class:`~zea.data.spec.ScanSpec`.
                Mutually exclusive with ``tracks``.
            tracks: List of track dicts (each with ``"data"`` and ``"scan"``
                keys) accepted by :class:`~zea.data.spec.TrackSpec` objects.
                Mutually exclusive with ``data``/``scan``.
            track_schedule: Optional int32 array of length ``n_total_tx``
                indicating which track each global transmit belongs to.
                Only used with ``tracks``.
            metadata: Optional metadata dict accepted by
                :class:`~zea.data.spec.MetadataSpec`.
            metrics: Optional metrics dict accepted by
                :class:`~zea.data.spec.MetricsSpec`.
            probe_name: Removed — use ``probe={'name': ...}`` instead.
            probe: Probe specification as a :class:`~zea.probes.Probe` object or a
                plain dict accepted by :class:`~zea.data.spec.ProbeSpec`.
            us_machine: Name of the ultrasound machine.
            description: Free-text description of the acquisition.
            compression: HDF5 compression filter (default ``"lzf"``).
            chunk_frames: If *True*, use frame-wise chunking for all datasets containing
                a "frames" dimension. Dataset will be stored with HDF5 chunking enabled,
                using a single frame (a single slice along the first dimension) per chunk.
            overwrite: If *False* (default), raise if the file exists.

        Returns:
            File: An open read-only :class:`File` handle.

        Single-track example:

        .. doctest::

            >>> import os, tempfile
            >>> import numpy as np
            >>> from zea import File

            >>> n_frames, n_tx, n_ax, n_el = 2, 4, 64, 8
            >>> raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
            >>> probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
            >>> scan = {
            ...     "sampling_frequency": np.float32(40e6),
            ...     "center_frequency": np.float32(5e6),
            ...     "demodulation_frequency": np.float32(5e6),
            ...     "initial_times": np.zeros(n_tx, dtype=np.float32),
            ...     "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
            ...     "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
            ...     "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
            ...     "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
            ...     "polar_angles": np.zeros(n_tx, dtype=np.float32),
            ...     "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
            ... }

            >>> _, path = tempfile.mkstemp(suffix=".hdf5")
            >>> File.create(
            ...     path,
            ...     data={"raw_data": raw},
            ...     scan=scan,
            ...     probe={"name": "verasonics_l11_4v", "probe_geometry": probe_geometry},
            ...     overwrite=True,
            ... )
            >>> os.unlink(path)
        """
        if tracks is not None and (data is not None or scan is not None):
            raise ValueError("Provide either 'tracks' or 'data'/'scan', not both.")

        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")

        if probe_name is not None:
            raise TypeError(
                "probe_name is no longer supported. "
                "Use probe={'name': ...} to specify the probe name."
            )

        if probe is not None and not isinstance(probe, (dict, ProbeSpec)):
            raise TypeError(f"probe must be a Probe object or a dict, got {type(probe).__name__}.")

        kwargs: dict = {}
        if tracks is not None:
            kwargs["tracks"] = tracks
            if track_schedule is not None:
                kwargs["track_schedule"] = track_schedule
        else:
            if data is None:
                raise ValueError("Either 'data' or 'tracks' must be provided.")
            kwargs["data"] = data
            if scan:
                kwargs["scan"] = scan

        if metadata is not None:
            kwargs["metadata"] = metadata
        if metrics is not None:
            kwargs["metrics"] = metrics
        if probe is not None:
            kwargs["probe"] = probe
        if us_machine is not None:
            kwargs["us_machine"] = us_machine
        if description is not None:
            kwargs["description"] = description

        _warn_custom_keys(kwargs.get("data", {}), kwargs.get("metadata", {}))
        spec = FileSpec(**kwargs)
        spec.save(
            str(path),
            compression=compression,
            chunk_frames=chunk_frames,
        )

    @property
    def data(self) -> _GroupProxy:
        """Lazy proxy for the ``data`` group of a single-track file.

        Supports both the new ``tracks/track_0/data/`` layout and the
        flat ``data/`` layout (files without a tracks group).

        Returns a :class:`GroupProxy` so individual datasets can be accessed
        as attributes without loading everything into RAM::

            with File(path) as f:
                f.data.raw_data[:, :n_tx]  # read a slice
                f.data.image.values[0]  # nested group access

        Raises:
            AttributeError: When the file contains more than one track.
                Use :attr:`tracks` to iterate over individual tracks.
        """
        n = self._n_tracks
        if n > 1:
            raise AttributeError(
                f"This file has {n} tracks. "
                "Use file.tracks to get a list of tracks and access each "
                "track's data individually: file.tracks[i].data"
            )
        # New-format single track
        if "tracks" in self:
            track0 = self["tracks"].get("track_0")
            if track0 is not None and "data" in track0:
                return _GroupProxy(track0["data"])
        # Flat layout (no tracks group): root-level data/ group
        if super().__contains__("data"):
            return _GroupProxy(self["data"])
        raise KeyError("No 'data' group found in this file.")

    @property
    def name(self):
        """Return the name of the file."""
        return self.path.name

    @property
    def stem(self):
        """Return the stem of the file."""
        return self.path.stem

    def _get_single_track_data_group(self) -> "h5py.Group":
        """Return the data group for single-track or flat-layout files."""
        if "tracks" in self:
            return self["tracks/track_0/data"]
        return self["data"]

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        if self._n_tracks > 1:
            raise AttributeError(
                f"This file has {self._n_tracks} tracks. Use file.tracks[i].n_frames."
            )
        return _shape_from_data_group(
            self._get_single_track_data_group(),
            index=0,
            name="n_frames",
            requires_raw=False,
        )

    @property
    def n_tx(self) -> int:
        """Number of transmit events."""
        if self._n_tracks > 1:
            raise AttributeError(f"This file has {self._n_tracks} tracks. Use file.tracks[i].n_tx.")
        return _shape_from_data_group(
            self._get_single_track_data_group(),
            index=1,
            name="n_tx",
            requires_raw=True,
        )

    @property
    def n_ax(self) -> int:
        """Number of axial samples."""
        if self._n_tracks > 1:
            raise AttributeError(f"This file has {self._n_tracks} tracks. Use file.tracks[i].n_ax.")
        return _shape_from_data_group(
            self._get_single_track_data_group(),
            index=2,
            name="n_ax",
            requires_raw=True,
        )

    @property
    def n_el(self) -> int:
        """Number of elements."""
        if self._n_tracks > 1:
            raise AttributeError(f"This file has {self._n_tracks} tracks. Use file.tracks[i].n_el.")
        return _shape_from_data_group(
            self._get_single_track_data_group(),
            index=3,
            name="n_el",
            requires_raw=True,
        )

    def shape(self, key) -> tuple:
        """Return shape of some key."""
        key = self.format_key(key)
        return self[key].shape

    def format_key(self, key):
        """Format the key to match the data type."""
        if isinstance(key, enum.Enum):
            key = key.value

        assert isinstance(key, str), f"Key must be a string, got {type(key)}. "

        # Return the key if it is already reachable (handles nested paths like
        # "tracks/track_0/data/raw_data", not just top-level keys).
        if key in self:
            return key

        # New-format: redirect bare or data/-prefixed keys to tracks/track_0/
        if "tracks" in self:
            track0 = self["tracks"].get("track_0")
            if track0 is not None and self._n_tracks == 1:
                bare = key.removeprefix("data/")
                data_grp = track0.get("data")
                if data_grp is not None and bare in data_grp:
                    return f"tracks/track_0/data/{bare}"

        # Flat layout: add 'data/' prefix if not present
        if "data/" not in key:
            key = "data/" + key

        available = list(self["data"].keys()) if super().__contains__("data") else list(self.keys())
        assert key in self, f"Key {key} not found in file. Available keys: {available}"

        return key

    def to_iterator(self, key):
        """Convert the data to an iterator over all frames."""
        key = self.format_key(key)
        for frame_idx in range(self.n_frames):
            yield self[key][frame_idx]

    @staticmethod
    def key_to_data_type(key):
        """Convert the key to a data type."""
        data_type = key.split("/")[-1]
        return data_type

    def load_transmits(self, key, selected_transmits):
        """Load raw_data or aligned_data for a given list of transmits.
        Args:
            key (str): The type of data to load. Options are 'raw_data' and 'aligned_data'.
            selected_transmits (list, np.ndarray): The transmits to load.
        """
        key = self.format_key(key)
        data_type = self.key_to_data_type(key)
        assert data_type in ["raw_data", "aligned_data"], (
            f"Cannot load transmits for {data_type}. Only raw_data and aligned_data are supported."
        )
        # First axis: all frames, second axis: selected transmits
        indices = (slice(None), np.array(selected_transmits))
        return self[key][indices]

    @deprecated(replacement="File.data.<key> with h5py slice indexing")
    def load_data(
        self,
        data_type,
        indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    ) -> np.ndarray:
        """Load data from the file.

        .. deprecated::
           Use ``file.data.<key>`` with standard h5py slice indexing instead::

               with File(path) as f:
                   raw = f.data.raw_data[:]  # all frames
                   raw = f.data.raw_data[0]  # first frame
                   raw = f.data.raw_data[0, [0, 2]]  # frame 0, transmits 0 and 2

        .. include:: ../common/file_indexing.rst

        Args:
            data_type (str): The type of data to load. Options are 'raw_data', 'aligned_data',
                'beamformed_data', 'envelope_data', 'image' and 'image_sc'.
            indices (optional): The indices to load. Defaults to ``None`` in
                which case all data is loaded.
        """
        key = self.format_key(data_type)
        if indices is None or (isinstance(indices, str) and indices == "all"):
            indices = slice(None)

        data = self[key]
        try:
            data = data[indices]
        except (OSError, IndexError) as exc:
            raise ValueError(
                f"Invalid indices {indices} for key {key}. {key} has shape {data.shape}."
            ) from exc

        return data

    @property
    def probe_name(self):
        """Reads the probe name from the data file and returns it."""
        # Priority: 'probe_name' attr → 'probe' group name → legacy 'probe' attr
        if "probe_name" in self.attrs:
            return self.attrs["probe_name"]
        # Check the structured probe group for a 'name' dataset.
        if "probe" in self and "name" in self["probe"]:
            return self["probe"]["name"].asstr()[()]
        if "probe" in self.attrs:
            return self.attrs["probe"]
        raise AttributeError(
            "Probe name not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )

    @property
    def us_machine(self):
        """Reads the ultrasound machine name from the data file and returns it."""
        assert "us_machine" in self.attrs, (
            "Ultrasound machine name not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        us_machine = self.attrs["us_machine"]
        return us_machine

    @property
    def description(self):
        """Reads the description from the data file and returns it."""
        assert "description" in self.attrs, (
            "Description not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        description = self.attrs["description"]
        return description

    def get_scan_parameters(self):
        """Returns a dictionary of parameters to initialize a scan
        object that comes with the file (stored inside datafile).

        If there are no scan parameters in the hdf5 file, returns
        an empty dictionary.

        Returns:
            dict: The scan parameters.
        """
        scan_group = self._scan_h5_group
        if scan_group is None:
            log.warning("Could not find scan parameters in file.")
            return {}

        scan_parameters = load_dict_from_hdf5_group(scan_group)

        return scan_parameters

    @property
    def scan(self) -> "ScanSpec | None":
        """Return the validated :class:`~zea.data.spec.ScanSpec` for this file.

        This is the bare scan group as a spec object.  For a full, derivable
        parameter object (merged probe + scan, with caching and derived
        properties) use :meth:`load_parameters`.

        Raises:
            AttributeError: When the file contains more than one track.
                Use :attr:`tracks` and access ``.scan`` on each track instead.

        .. doctest::

            >>> from zea import File
            >>> path = (
            ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
            ... )
            >>> with File(path, revision="v0.1.0", mode="r") as f:
            ...     scan = f.scan
            >>> type(scan).__name__
            'ScanSpec'
        """
        n = self._n_tracks
        if n > 1:
            raise AttributeError(
                f"This file has {n} tracks. "
                "Use file.tracks to get a list of tracks and access .scan on each: "
                "file.tracks[i].scan"
            )
        scan_dict = self.get_scan_parameters()

        if _is_legacy_file(self):
            scan_dict = legacy_scan(scan_dict)

        return ScanSpec(**scan_dict) if len(scan_dict) > 0 else None

    def load_parameters(self, **overrides) -> "Parameters":
        """Load the acquisition parameters (merged probe + scan) from the file.

        Reads both the ``scan`` and ``probe`` groups and merges them into a
        single :class:`~zea.Parameters` object that owns derivation,
        caching, and lazy loading of derived quantities. The probe and scan
        groups live at the same level and have non-overlapping field names, so
        merging is a plain dict union.

        Args:
            **overrides: Override any parameter from the file. Custom
                (non-spec) keys are stored as passthrough parameters.

        Returns:
            Parameters: The merged, derivable parameters object.

        Raises:
            AttributeError: When the file contains more than one track.
                Use :attr:`tracks` and call ``.load_parameters()`` on each track.

        .. doctest::

            >>> from zea import File
            >>> path = (
            ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
            ... )
            >>> with File(path) as f:
            ...     parameters = f.load_parameters()
            >>> type(parameters).__name__
            'Parameters'
        """
        from zea.scan import Parameters

        n = self._n_tracks
        if n > 1:
            raise AttributeError(
                f"This file has {n} tracks. "
                "Use file.tracks to get a list of tracks and call "
                "file.tracks[i].load_parameters() on each."
            )

        scan = self.scan
        probe = self.probe

        probe_dict = probe.get_parameters()
        if scan is not None:
            scan_dict = scan.to_dict()
            other_dict = {
                "n_ax": self.n_ax,
                "n_el": scan.n_el,
                "n_tx": scan.n_tx,
            }
        else:
            scan_dict = {}
            other_dict = {}
        merged_dict = {**probe_dict, **scan_dict, **other_dict, **overrides}

        # skip None values
        merged_dict = {k: v for k, v in merged_dict.items() if v is not None}

        return Parameters(**merged_dict)

    @property
    def probe(self) -> "Probe":
        """Returns a Probe object initialized with the parameters from the file.

        Returns:
            Probe: The probe object.

        Example:
            .. doctest::

                >>> from zea import File
                >>> path = (
                ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
                ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
                ... )
                >>> with File(path) as f:
                ...     probe = f.probe
                >>> probe.name
                'verasonics_l11_4v'
        """
        from zea.probes import Probe

        if "probe" in self.keys():
            probe_dict = self.recursively_load_dict_contents_from_group("probe")
        elif _is_legacy_file(self):
            scan_dict = self.get_scan_parameters()
            probe_dict = legacy_probe(scan_dict)
            probe_dict["name"] = self.probe_name
        else:
            # Image-only datasets carry no probe group; an empty Probe lets
            # load_parameters merge in nothing rather than failing.
            probe_dict = {}

        return Probe(**probe_dict)

    @property
    def metadata(self) -> MetadataSpec:
        """Return a validated :class:`~zea.data.spec.MetadataSpec` object from the file.

        Returns:
            MetadataSpec: The validated metadata spec.

        Raises:
            KeyError: If the file has no ``metadata`` group.

        Example:
            .. doctest::

                >>> from zea import File
                >>> path = (
                ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
                ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
                ... )
                >>> with File(path, revision="v0.1.0") as f:
                ...     meta = f.metadata
                ...     print(meta.subject.type)
                phantom
        """
        if "metadata" not in self:
            raise KeyError("No 'metadata' group in this file.")
        raw = load_dict_from_hdf5_group(self["metadata"])
        return MetadataSpec(**raw)

    @property
    def metrics(self) -> MetricsSpec:
        """Return a validated :class:`~zea.data.spec.MetricsSpec` object from the file.

        Returns:
            MetricsSpec: The validated metrics spec.

        Raises:
            KeyError: If the file has no ``metrics`` group.

        Example::

            >>> with File("my_file.hdf5") as f:  # doctest: +SKIP
            ...     met = f.metrics
            ...     print(met.coherence_factor.shape)
        """
        if "metrics" not in self:
            raise KeyError("No 'metrics' group in this file.")
        raw = load_dict_from_hdf5_group(self["metrics"])
        return MetricsSpec(**raw)

    def recursively_load_dict_contents_from_group(self, path: str) -> dict:
        """Load dict from contents of group.

        .. deprecated::
            Use the module-level :func:`load_dict_from_hdf5_group` function instead,
            passing an :class:`h5py.Group` directly.

        Args:
            path (str): path to group
        Returns:
            dict: dictionary with contents of group
        """
        return load_dict_from_hdf5_group(self[path])

    def has_key(self, key: str) -> bool:
        """Check if the file has a specific key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        try:
            key = self.format_key(key)
        except AssertionError:
            return False
        return True

    @classmethod
    def get_shape(cls, path: str, key: str) -> tuple:
        """Get the shape of a key in a file.

        Args:
            path (str): The path to the file.
            key (str): The key to get the shape of.

        Returns:
            tuple: The shape of the key.
        """
        with cls(path, mode="r") as file:
            return file.shape(key)

    def validate(self):
        """Lightweight structural validation — no array data is loaded into RAM.

        Checks that the file has a ``data`` group and that all keys within it
        are recognised zea data types.  For legacy files (before zea v0.1.0)
        a minimal key-name check is performed.  For files created with
        zea v0.1.0 and later (via :meth:`File.create`) the keys are checked
        against the :class:`~zea.data.spec.DataSpec` schema.

        Use :meth:`validate_spec` for a **full** validation that loads all data
        and checks dtypes, shapes, and cross-field dimension consistency.

        Returns:
            dict: ``{"status": "success"}`` on success.

        Raises:
            AssertionError: If the file is missing required groups or contains
                unrecognised data keys.
        """
        try:
            return validate_file(file=self)
        except Exception as e:
            log.error(f"File {self.path} is not a valid zea file.\n{e}\n")
            raise

    def _to_file_spec(self) -> FileSpec:
        """Load the whole file into a validated :class:`~zea.data.spec.FileSpec`.

        Unlike the lazy :attr:`data` / :attr:`scan` accessors, every dataset is
        read into memory here.  Both the multi-track ``tracks/track_N/`` layout
        and the flat ``data/`` + ``scan/`` layout (files without a tracks group) are supported.

        Returns:
            FileSpec: A fully validated spec object, with all arrays in RAM.
        """
        kwargs: dict = {"tracks": self._load_tracks(), "probe": self.probe}

        if self.track_schedule is not None:
            kwargs["track_schedule"] = self.track_schedule
        if "metadata" in self:
            kwargs["metadata"] = self.metadata
        if "metrics" in self:
            kwargs["metrics"] = self.metrics
        if "us_machine" in self.attrs:
            kwargs["us_machine"] = self.attrs["us_machine"]
        if "description" in self.attrs:
            kwargs["description"] = self.attrs["description"]

        return FileSpec(**kwargs)

    def _load_tracks(self) -> "list[dict]":
        """Read every track's ``data`` and ``scan`` fully into a list of dicts.

        Each dict is shaped for :class:`~zea.data.spec.TrackSpec`. Flat-layout
        files (no tracks group) are returned as a single, unlabelled track.
        """
        # Flat layout: one unlabelled track at the file root.
        if "tracks" not in self:
            track: dict = {}
            if super().__contains__("data"):
                data = load_dict_from_hdf5_group(self["data"])
                track["data"] = legacy_data(data) if _is_legacy_file(self) else data
            if self.scan is not None:
                track["scan"] = self.scan
            return [track]

        # Multi-track layout: tracks/track_N/{data,scan,label}.
        tracks = []
        for track in self.tracks:
            track_dict: dict = {"label": track.label}
            if "data" in track._group:
                track_dict["data"] = load_dict_from_hdf5_group(track._group["data"])
            if "scan" in track._group:
                track_dict["scan"] = track.scan
            tracks.append(track_dict)
        return tracks

    def validate_spec(self) -> FileSpec:
        """Full schema validation — loads all data into RAM.

        Reads every dataset in the file and runs dtype, shape, and
        cross-dimension consistency checks as defined by :class:`~zea.data.spec.FileSpec`.
        Use this to confirm a file is fully spec-compliant before sharing or
        processing it.

        For a fast, zero-IO structural check use :meth:`validate` instead.

        .. note::
            This method only works on files created with zea v0.1.0 and later.
            Files written before zea v0.1.0 should be re-saved through
            :meth:`File.create`.

        Returns:
            FileSpec: The fully validated spec object, with all data accessible
            as typed attributes (e.g. ``spec.data.raw_data``, ``spec.scan.n_tx``).

        Raises:
            TypeError, ValueError: If the file does not conform to the spec.

        .. doctest::

            >>> with File("my_file.hdf5") as f:  # doctest: +SKIP
            ...     spec = f.validate_spec()
            ...     print(spec.scan.n_tx)
        """
        return self._to_file_spec()

    def __repr__(self):
        name = Path(self.filename).name
        try:
            n = self._n_tracks
            if n > 1:
                labels = self.track_labels
                label_str = ", ".join(
                    f'"{label}"' if label else str(i) for i, label in enumerate(labels)
                )
                track_info = f"{n} tracks: {label_str}"
            else:
                track_info = f"{n} track"
        except Exception:
            track_info = ""
        mode_info = f"mode {self.mode}"
        return f'<File "{name}" ({mode_info}, {track_info})>'

    def copy_key(self, key: str, dst: "File"):
        """Copy a specific key to another file.

        Will always copy the attributes and the scan data if it exists. Will warn if the key is
        not in this file or if the key already exists in the destination file.

        Args:
            key (str): The key to copy.
            dst (File): The destination file to copy the key to.
        """
        key = self.format_key(key)

        # Copy the key if it does not already exist in the destination file
        if key in dst:
            log.warning(f"Skipping key '{key}' because it already exists in dst file {dst.path}.")
        elif key in self:
            self.copy(key, dst, name=key)
        else:
            log.warning(f"Key '{key}' not found in src file {self.path}. Skipping copy.")

        # Copy attributes from src to dst
        for attr_key, attr_value in self.attrs.items():
            dst[key].attrs[attr_key] = attr_value

        # Copy scan data if requested
        if "scan" in self and "scan" not in dst:
            # Use the actual HDF5 path (not our overridden key) for h5py.copy
            scan_path = self._scan_h5_group.name.lstrip("/")
            self.copy(scan_path, dst, name="scan")

    def summary(self):
        """Print the contents of the file."""
        _print_hdf5_attrs(self)


def load_file_all_data_types(
    path,
    indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    scan_kwargs: dict = None,
):
    """Loads a zea data files (h5py file).

    Returns all data types together with a parameters object containing the parameters
    of the acquisition. Probe information is available via ``parameters.to_probe_dict()``
    or ``File.probe``.

    Additionally, it can load a specific subset of frames / transmits.

    .. include:: ../common/file_indexing.rst

    Args:
        path (str, pathlike): The path to the hdf5 file.
        indices (optional): The indices to load. Defaults to None in
            which case all frames are loaded.
        scan_kwargs (Config, dict, optional): Additional keyword arguments
            to pass to :meth:`File.load_parameters`. These will override the
            parameters from the file if they are present. Defaults to None.

    Returns:
        (dict): A dictionary with all data types as keys and the corresponding data as values.
        (Parameters): A parameters object containing the parameters of the acquisition.
    """
    # Define the additional keyword parameters from the scan object
    if scan_kwargs is None:
        scan_kwargs = {}

    data_dict = {}

    # Data types stored as HDF5 groups (Map-based specs with values/coordinates)
    _GROUP_DATA_TYPES = {"aligned_data", "beamformed_data", "envelope_data", "image_sc", "image"}
    # Among _GROUP_DATA_TYPES, only aligned_data has a transmit (n_tx) axis as its 2nd dimension.
    # All others have spatial axes after n_frames, so a transmit-selection tuple index must not
    # be applied to them (it would mis-slice a spatial dimension instead of a transmit dimension).
    _GROUP_TYPES_WITH_TX_AXIS = {"aligned_data"}

    with File(path, mode="r") as file:
        for data_type in DataTypes:
            if not file.has_key(data_type.value):
                data_dict[data_type.value] = None
                continue

            # Load the desired frames from the file
            _key = file.format_key(data_type.value)
            _indices = indices if indices is not None else slice(None)
            item = file[_key]

            if isinstance(item, h5py.Group) and data_type.value in _GROUP_DATA_TYPES:
                # Map-based group: load all sub-datasets as a dict.
                # Compute per-dataset indices once: for non-TX types, a transmit-selection
                # tuple must not be applied to spatial dimensions.
                if (
                    isinstance(_indices, tuple)
                    and len(_indices) > 1
                    and data_type.value not in _GROUP_TYPES_WITH_TX_AXIS
                ):
                    indices_for_ds = (_indices[0],)
                else:
                    indices_for_ds = _indices

                group_dict = {}
                for sub_key in item.keys():
                    ds = item[sub_key]
                    if isinstance(ds, h5py.Dataset):
                        if sub_key == "values":
                            group_dict[sub_key] = ds[indices_for_ds]
                        elif sub_key == "coordinates":
                            # Coordinates may omit the leading frame axis (broadcast mode —
                            # one grid shared across all frames). Only apply frame indexing
                            # when the first dim matches the values dataset's first dim.
                            values_ds = item.get("values")
                            if values_ds is not None and ds.shape[0] == values_ds.shape[0]:
                                group_dict[sub_key] = ds[indices_for_ds]
                            else:
                                group_dict[sub_key] = ds[()]
                        elif h5py.check_string_dtype(ds.dtype) is not None:
                            val = ds.asstr()[()]
                            if isinstance(val, np.ndarray) and val.dtype == object:
                                val = val.astype(np.str_)
                            group_dict[sub_key] = val
                        else:
                            group_dict[sub_key] = ds[()]
                data_dict[data_type.value] = group_dict
            else:
                data_dict[data_type.value] = item[_indices]

        # extract transmits from indices
        # we only have to do this when the data has a n_tx dimension
        # in that case we also have update scan parameters to match
        # the number of selected transmits
        if isinstance(indices, tuple) and len(indices) > 1:
            scan_kwargs["selected_transmits"] = indices[1]

        parameters = file.load_parameters(**scan_kwargs)

        return data_dict, parameters


@deprecated(replacement="File(...) with file.load_parameters() and file.data.<type>[...]")
def load_file(
    path,
    data_type="raw_data",
    indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    scan_kwargs: dict = None,
) -> Tuple[np.ndarray, "Parameters"]:
    """Loads a zea data files (h5py file).

    Returns the data together with a parameters object containing the parameters
    of the acquisition. Probe information is available via ``parameters.to_probe_dict()``
    or ``File.probe``.

    Additionally, it can load a specific subset of frames / transmits.

    .. include:: ../common/file_indexing.rst

    Args:
        path (str, pathlike): The path to the hdf5 file.
        data_type (str, optional): The type of data to load. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        indices (optional): The indices to load. Defaults to None in
            which case all frames are loaded.
        scan_kwargs (Config, dict, optional): Additional keyword arguments
            to pass to :meth:`File.load_parameters`. These will override the
            parameters from the file if they are present. Defaults to None.

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_ax, n_el, n_ch).
        (Parameters): A parameters object containing the parameters of the acquisition.
    """
    # Define the additional keyword parameters from the scan object
    if scan_kwargs is None:
        scan_kwargs = {}

    with File(path, mode="r") as file:
        # Load the desired frames from the file
        _key = file.format_key(data_type)
        _indices = indices if indices is not None else slice(None)
        item = file[_key]
        if isinstance(item, h5py.Group):
            data = item["values"][_indices]
        else:
            data = item[_indices]

        # extract transmits from indices
        # we only have to do this when the data has a n_tx dimension
        # in that case we also have update scan parameters to match
        # the number of selected transmits
        if data_type in ["raw_data", "aligned_data"]:
            if isinstance(indices, tuple) and len(indices) > 1:
                scan_kwargs["selected_transmits"] = indices[1]

        parameters = file.load_parameters(**scan_kwargs)

        return data, parameters


def _print_hdf5_attrs(hdf5_obj, prefix=""):
    """Recursively prints all keys, attributes, and shapes in an HDF5 file.

    Args:
        hdf5_obj (h5py.File, h5py.Group, h5py.Dataset): HDF5 object to print.
        prefix (str, optional): Prefix to print before each line. This
            parameter is used in internal recursion and should not be supplied
            by the user.
    """
    assert isinstance(hdf5_obj, (h5py.File, h5py.Group, h5py.Dataset)), (
        "ERROR: hdf5_obj must be a File, Group, or Dataset object"
    )

    if isinstance(hdf5_obj, h5py.File):
        name = "root" if hdf5_obj.name == "/" else hdf5_obj.name
        print(prefix + name + "/")
        prefix += "    "
    elif isinstance(hdf5_obj, h5py.Dataset):
        shape_str = str(hdf5_obj.shape).replace(",)", ")")
        print(prefix + "├── " + hdf5_obj.name + " (shape=" + shape_str + ")")
        prefix += "│   "

    # Print all attributes
    for key, val in hdf5_obj.attrs.items():
        print(prefix + "├── " + key + ": " + str(val))

    # Recursively print all keys, attributes, and shapes in groups
    if isinstance(hdf5_obj, h5py.Group):
        for i, key in enumerate(hdf5_obj.keys()):
            is_last = i == len(hdf5_obj.keys()) - 1
            if is_last:
                marker = "└── "
                new_prefix = prefix + "    "
            else:
                marker = "├── "
                new_prefix = prefix + "│   "
            print(prefix + marker + key + "/")
            _print_hdf5_attrs(hdf5_obj[key], new_prefix)


def validate_file(path: str = None, file: File = None):
    """Validate the structure and data of a zea HDF5 file.

    For files created with zea v0.1.0 and later this runs the full
    :class:`~zea.data.spec.FileSpec` schema validation (dtypes, shapes, and
    dimension consistency).  Legacy files (before zea v0.1.0) are detected by the
    presence of scalar dataset ``scan/n_frames``; for those only a lightweight
    structural ``data`` group check is performed.

    Provide either *path* or *file*, but not both.

    Args:
        path (str | pathlike): Path to the HDF5 file.
        file (File): An already-open :class:`File` instance.

    Returns:
        dict: ``{"status": "success"}`` on success.

    Raises:
        AssertionError: If the file is missing the ``data`` group.
        TypeError, ValueError: If spec validation fails on files created with zea v0.1.0 and later.
    """
    assert (path is not None) ^ (file is not None), (
        "Provide either the path or the file, but not both."
    )

    if path is not None:
        with File(path, "r") as _file:
            _validate_file_impl(_file)
    else:
        _validate_file_impl(file)

    return {"status": "success"}


def _is_legacy_file(file: File) -> bool:
    """Return ``True`` when *file* pre-dates the dataspec format.

    Files created with zea v0.1.0 and later always store a
    ``zea_version`` root attribute.  Files that lack it were produced by
    the legacy data format path and are treated as legacy.
    """
    return "zea_version" not in file.attrs


def _validate_file_impl(file: File) -> None:
    """Lightweight structural validation — no array data is loaded.

    Checks that:
    - a ``data`` group is present — either at ``tracks/track_N/data``
      or at the root ``data`` group (legacy)
    - for legacy files, every key in ``data`` is a recognised zea data type
    - for files created with zea v0.1.0 and later, every key in ``data``
    is in :class:`~zea.data.spec.DataSpec`\'s schema
    """
    # Collect all data groups to validate
    data_groups: list[tuple[str, h5py.Group]] = []

    if super(File, file).__contains__("tracks"):
        # New multi-track format: tracks/track_N/data
        tracks_group = file["tracks"]
        for track_key in tracks_group.keys():
            track_grp = tracks_group[track_key]
            assert "data" in track_grp, f"Track group '{track_key}' is missing a 'data' subgroup."
            assert isinstance(track_grp["data"], h5py.Group), (
                f"'{track_key}/data' is not a group - this may not be a zea file."
            )
            data_groups.append((f"tracks/{track_key}/data", track_grp["data"]))
    elif super(File, file).__contains__("data"):
        # Legacy root-level data group
        assert isinstance(file["data"], h5py.Group), (
            "'data' is not a group - this may not be a zea file."
        )
        data_groups.append(("data", file["data"]))

    assert data_groups, (
        "'data' group not found in file. "
        "Expected either tracks/track_N/data or a root 'data' group."
    )

    for group_path, data_group in data_groups:
        if _is_legacy_file(file):
            # For legacy files: accepted keys are the flat _DATA_TYPES list.
            has_raw = any(k in data_group for k in _NON_IMAGE_DATA_TYPES)
            if has_raw:
                assert "scan" in file, "Legacy file is missing the 'scan' group."
            for key in data_group.keys():
                assert key in _DATA_TYPES, (
                    f"'{group_path}/{key}' is not a recognised zea data type."
                )
        else:
            # For new-format files: flat datasets must be known DataSpec keys.
            # HDF5 Groups are Map specs (either a named type or a custom map)
            # and are always accepted; validate() is a structural check only.
            known = set(DataSpec.SCHEMA.keys())
            known_flat = {k for k, v in DataSpec.SCHEMA.items() if "spec" not in v}
            for key in data_group.keys():
                if isinstance(data_group[key], h5py.Group):
                    # Named map or custom map — accepted without further checks here.
                    continue
                assert key in known_flat, (
                    f"'{group_path}/{key}' is not in the DataSpec schema. "
                    f"Known keys: {sorted(known)}"
                )
