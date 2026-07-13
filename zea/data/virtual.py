"""
zea.data.virtual
================

Virtual (Zarr) references for cloud-optimized reads of zea HDF5 files.

A *virtual reference* is a small JSON file holding the chunk manifest (byte
offset + length of every HDF5 chunk) of one or more zea files. Reading through
it goes straight to byte ranges over HTTP via Zarr + obstore: no HDF5 metadata
traversal on open, concurrent chunk fetches, and many files combined into one
logical array. Opening a virtual reference costs ~0 HTTP requests, where opening
the same files with h5py costs a handful of round trips *per file*.

Two sides:

- **Generation** (maintainer, run once per dataset): :func:`build_virtual_reference`
  reads each file's metadata (over HTTP for ``hf://`` inputs — the data itself is
  never downloaded) and writes a combined reference. Also available as
  ``zea data virtualize <input> <output>``.
- **Reading** (user): :func:`open_virtual_reference`, or
  :class:`~zea.data.datasets.Dataset` with ``lazy="virtual"``.

Files are combined along a new leading ``file`` axis, so a read looks like::

    virtual = open_virtual_reference("hf://zeahub/camus-sample/virtual/index.json")
    virtual["raw_data"][2, 0:3]  # file 2, frames 0-3 — one concurrent range read

Only numeric bulk arrays (``raw_data``, ``image/values``, …) live in a reference;
scan/probe parameters are not virtualized (they are vlen strings and scalars) —
open the file itself for those.

Requires the optional dependencies: ``pip install 'zea[virtual]'``.

Note:
    Chunks must be readable by Zarr, so files have to be written with a
    Zarr-decodable codec (the Blosc default since zea 0.1.3). Older ``lzf``
    files must be resaved first (``zea data resave``).
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, cast

import h5py
import numpy as np
import tqdm

from zea import log
from zea.data.file import File
from zea.internal.preset_utils import HF_PREFIX, _hf_parse_path, _hf_resolve_path

if TYPE_CHECKING:
    import xarray as xr
    import zarr

    from zea.parameters import Parameters

# Location of a published reference inside a dataset (HF repo or local folder). The
# parameter sidecar sits next to it, under the same ``virtual/`` prefix.
VIRTUAL_INDEX_PATH = "virtual/index.json"
_PARAMS_FILENAME = "params.json"
VIRTUAL_PARAMS_PATH = f"{VIRTUAL_INDEX_PATH.rsplit('/', 1)[0]}/{_PARAMS_FILENAME}"

# Host serving the chunk bytes for ``hf://`` datasets.
HF_HOST = "https://huggingface.co"

# The reference's root attributes hold this manifest: which files went into it,
# in which order, and where their arrays live in the Zarr hierarchy.
_MANIFEST_KEY = "zea_virtual"
_MANIFEST_VERSION = 1

# Axis (prepended by ``xarray.concat``) along which files are stacked.
_FILE_DIM = "file"

# Key read by default when indexing a reference without naming one.
_DEFAULT_KEY = "raw_data"

# HDF5 filter id of lzf (zea's default codec before 0.1.3). It has no Zarr codec.
_LZF_FILTER_ID = 32000

_INSTALL_HINT = (
    "The virtual read path needs extra dependencies. Install them with: pip install 'zea[virtual]'"
)


def _require_deps() -> None:
    """Raise a helpful ImportError when the ``zea[virtual]`` extra is missing."""
    try:
        import obstore  # noqa: F401
        import virtualizarr  # noqa: F401
        import xarray  # noqa: F401
        import zarr  # noqa: F401
    except ImportError as exc:
        raise ImportError(f"{_INSTALL_HINT} (missing '{exc.name}').") from exc


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _object_store_registry():
    """Registry resolving ``file://`` and ``https://huggingface.co`` chunk URLs."""
    from obstore.store import HTTPStore, LocalStore

    try:
        from obspec_utils.registry import ObjectStoreRegistry
    except ImportError:  # virtualizarr < 2.7 re-exported it itself
        from virtualizarr.registry import ObjectStoreRegistry

    client_options: dict[str, Any] = {}
    token = _hf_token()
    if token:
        # Private/gated repos: obstore has no notion of HF auth, so pass the token
        # as a bearer header on every range request.
        client_options["default_headers"] = {"Authorization": f"Bearer {token}"}

    return ObjectStoreRegistry(
        {
            "file://": LocalStore(),
            HF_HOST: HTTPStore.from_url(HF_HOST, client_options=client_options or None),
        }
    )


def _chunk_url(path: str, revision: str | None = None) -> str:
    """URL the chunk references of ``path`` should point at.

    ``hf://`` paths become the HuggingFace *resolve* URL (pinned to ``revision``,
    or ``main``); local paths become a ``file://`` URI.
    """
    if path.startswith(HF_PREFIX):
        repo_id, subpath = _hf_parse_path(path)
        if not subpath:
            raise ValueError(
                f"Cannot virtualize '{path}': expected an 'hf://' path to a single file."
            )
        return f"{HF_HOST}/datasets/{repo_id}/resolve/{revision or 'main'}/{subpath}"
    return Path(path).absolute().as_uri()


def _unfiltered_datasets(group: h5py.Group, path: str) -> List[str]:
    """Names of datasets in ``group`` that Zarr cannot decode chunk-for-chunk.

    When a filter does not shrink a chunk (incompressible data), HDF5 stores that
    chunk *raw* and records it in the chunk's filter mask. Zarr has no equivalent —
    it applies the codec to every chunk — so such a dataset would decode to garbage.
    Datasets like these are left out of the reference; they stay readable via h5py.

    Raises:
        ValueError: When a dataset is lzf-compressed. lzf has no Zarr codec at all,
            so nothing in the file can be virtualized until it is resaved.
    """
    unfiltered = []
    for name, obj in group.items():
        if not isinstance(obj, h5py.Dataset) or obj.chunks is None:
            continue
        dataset_id = obj.id
        plist = dataset_id.get_create_plist()
        filters = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
        if not filters:
            continue
        if _LZF_FILTER_ID in filters:
            raise ValueError(
                f"'{path}' is lzf-compressed, which Zarr cannot decode, so it cannot be "
                "virtualized. Resave it with the default Blosc codec first: "
                "zea data resave <input> <output>"
            )
        if any(
            dataset_id.get_chunk_info(i).filter_mask for i in range(dataset_id.get_num_chunks())
        ):
            unfiltered.append(name)
    return unfiltered


def _data_groups(file: File, path: str) -> "OrderedDict[str, tuple[str, List[str]]]":
    """Map each virtualizable HDF5 group of a file to ``(group_path, datasets_to_skip)``.

    Keys are relative to the file's data group: ``""`` is the data group itself
    (holding ``raw_data``), ``"image"`` the image subgroup, and so on. Groups without
    datasets of their own are skipped, as are datasets that Zarr cannot decode (see
    :func:`_unfiltered_datasets`). Only metadata is read, so over HTTP this costs a
    few range requests and no array data.
    """
    data_group = file._get_single_track_data_group()
    base = data_group.name.lstrip("/")

    groups: OrderedDict[str, tuple[str, List[str]]] = OrderedDict()

    def collect(rel: str, group: h5py.Group) -> None:
        datasets = [name for name, obj in group.items() if isinstance(obj, h5py.Dataset)]
        skip = _unfiltered_datasets(group, path)
        if skip:
            log.warning(
                f"Not virtualizing {', '.join(f'{rel}/{name}'.lstrip('/') for name in skip)} "
                f"of '{path}': some chunks were stored uncompressed by HDF5 (the data did "
                "not compress), which Zarr cannot decode. The rest of the file is "
                "virtualized; these arrays stay readable through zea.File."
            )
        if set(datasets) - set(skip):
            groups[rel] = (f"{base}/{rel}".rstrip("/"), skip)
        for name, obj in group.items():
            if isinstance(obj, h5py.Group):
                collect(f"{rel}/{name}".lstrip("/"), obj)

    collect("", data_group)

    if not groups:
        raise ValueError(f"No virtualizable datasets found in the data group of '{path}'.")
    return groups


def _encode(value: Any) -> Any:
    """Make a parameter value JSON-serializable, preserving its dtype."""
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.ravel().tolist(),
        }
    if isinstance(value, np.generic):
        return {"dtype": str(value.dtype), "data": value.item()}
    return value


def _decode(value: Any) -> Any:
    """Inverse of :func:`_encode`."""
    if isinstance(value, dict) and "dtype" in value:
        dtype = np.dtype(value["dtype"])
        if "shape" in value:
            return np.asarray(value["data"], dtype=dtype).reshape(value["shape"])
        return dtype.type(value["data"])
    return value


def _encode_parameters(file: File) -> dict:
    """Serializable scan + probe parameters of a file.

    Mirrors the merge in :meth:`zea.File.load_parameters`, so that
    ``Parameters(**decoded)`` reconstructs the same object. Parameters cannot be
    virtualized (vlen strings, scalars), hence this sidecar.
    """
    parameters = file.load_parameters()
    merged = {
        **parameters.to_probe_dict(),
        **parameters.to_scan_dict(),
        "n_ax": parameters.n_ax,
        "n_el": parameters.n_el,
        "n_tx": parameters.n_tx,
    }
    return {key: _encode(value) for key, value in merged.items()}


def _inspect_file(path: str, revision: str | None) -> tuple:
    """One open per file: its virtualizable groups and its (encoded) parameters."""
    kwargs = {"revision": revision} if revision is not None else {}
    with File(path, **kwargs) as file:
        return _data_groups(file, path), _encode_parameters(file)


def _virtual_datasets(
    path: str, groups: "OrderedDict[str, tuple[str, List[str]]]", registry, revision: str | None
) -> "OrderedDict[str, xr.Dataset]":
    """Virtualize every data group of one file, keyed by group path (see :func:`_data_groups`).

    The chunk manifest is built by parsing the file at its final URL, so the
    references already point where the reader will look — no path rewriting.
    """
    from virtualizarr.parsers import HDFParser

    url = _chunk_url(path, revision)
    datasets: OrderedDict[str, xr.Dataset] = OrderedDict()
    for rel, (hdf5_group, skip) in groups.items():
        try:
            store = HDFParser(group=hdf5_group, drop_variables=skip)(url, registry)
        except Exception as exc:
            if "codec not available" in str(exc):
                raise ValueError(
                    f"'{path}' uses an HDF5 codec that Zarr cannot decode ({exc}). "
                    "Files written before zea 0.1.3 use lzf; resave them with the "
                    "default Blosc codec first: zea data resave <input> <output>."
                ) from exc
            raise
        datasets[rel] = store.to_virtual_dataset()
    return datasets


def _signature(datasets: "OrderedDict[str, xr.Dataset]") -> Tuple:
    """Shape/dtype fingerprint of a file's arrays.

    Files are stacked along a new ``file`` axis, which requires every other axis
    to match exactly. Files with different signatures (e.g. a differing number of
    frames) therefore go into separate groups of the reference.
    """
    return tuple(
        sorted(
            (f"{rel}/{name}".lstrip("/"), tuple(var.shape), str(var.dtype))
            for rel, dataset in datasets.items()
            for name, var in dataset.variables.items()
        )
    )


def build_virtual_reference(
    paths: Sequence[str | Path] | str | Path,
    output_path: str | Path,
    revision: str | None = None,
    verbose: bool = True,
) -> Path:
    """Build a combined virtual (Zarr) reference for a set of zea files.

    Reads only the HDF5 *metadata* of each file — for ``hf://`` inputs over HTTP
    range requests, so nothing is downloaded — and writes a single kerchunk-style
    JSON reference. Files with identical array shapes are combined into one logical
    array with a leading ``file`` axis; files whose shapes differ (e.g. a differing
    number of frames) are placed in separate groups of the same reference.

    A ``params.json`` sidecar is written next to the reference, holding each file's
    scan and probe parameters (identical parameter sets are stored once). These cannot
    be virtualized — they are vlen strings and scalars — so they travel alongside, and
    are served by :meth:`VirtualReference.parameters`.

    This is maintainer tooling: run it once per dataset and publish both files next
    to the data (conventionally under ``virtual/`` in the repo), so that readers can
    use ``Dataset(..., lazy="virtual")``.

    Args:
        paths (str, Path or list): The file(s), folder(s) or ``hf://`` path(s) to
            virtualize. Accepts anything :class:`~zea.data.datasets.Dataset` accepts.
        output_path (str or Path): Path of the JSON reference to write. The parameter
            sidecar is written next to it as ``params.json``.
        revision (str, optional): HuggingFace revision (branch, tag or commit hash)
            to pin the chunk URLs to. Only used for ``hf://`` paths. Defaults to
            ``None`` (``main``). Pin to a commit hash for references that cannot go
            stale when the dataset is updated.
        verbose (bool, optional): Show a progress bar. Defaults to ``True``.

    Returns:
        Path: The path of the written reference.

    Raises:
        ValueError: When a file's codec is not Zarr-decodable (old ``lzf`` files —
            resave them with ``zea data resave`` first).
    """
    _require_deps()

    import xarray as xr

    from zea.data.datasets import Dataset

    with Dataset(
        paths, validate=False, lazy=True, revision=revision, _suggest_lazy=False
    ) as dataset:
        # Sorted, so that the file index of a reference is reproducible: folder walks
        # and HF repo listings come back in arbitrary order.
        file_paths = sorted(dataset.file_paths)

    registry = _object_store_registry()

    # Group files by array shape/dtype: only same-shaped files can be stacked.
    groups: OrderedDict[Tuple, list[tuple[str, OrderedDict[str, xr.Dataset]]]] = OrderedDict()
    parameters: dict[str, dict] = {}
    for path in tqdm.tqdm(
        file_paths,
        desc=f"Virtualizing {len(file_paths)} file(s)",
        disable=not verbose,
    ):
        file_groups, parameters[path] = _inspect_file(path, revision)
        datasets = _virtual_datasets(path, file_groups, registry, revision)
        groups.setdefault(_signature(datasets), []).append((path, datasets))

    refs: dict[str, Any] = {}
    manifest_groups = []
    for index, members in enumerate(groups.values()):
        group_name = f"files_{index}"
        arrays: dict[str, dict] = {}
        for rel in members[0][1]:
            combined = xr.concat([datasets[rel] for _, datasets in members], dim=_FILE_DIM)
            zarr_group = f"{group_name}/{rel}".rstrip("/")
            # Nest each group's references under its Zarr path; a kerchunk store is
            # a flat key->[url, offset, length] map, so prefixing builds the hierarchy.
            for key, value in combined.vz.to_kerchunk(format="dict")["refs"].items():
                refs[f"{zarr_group}/{key}"] = value
            for name, variable in combined.variables.items():
                arrays[f"{rel}/{name}".lstrip("/")] = {
                    "group": zarr_group,
                    "name": str(name),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                }
        manifest_groups.append(
            {"name": group_name, "files": [path for path, _ in members], "arrays": arrays}
        )

    refs[".zgroup"] = json.dumps({"zarr_format": 2})
    refs[".zattrs"] = json.dumps(
        {
            _MANIFEST_KEY: {
                "version": _MANIFEST_VERSION,
                "revision": revision,
                "groups": manifest_groups,
            }
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"version": 1, "refs": refs}), encoding="utf-8")

    params_path = output_path.parent / _PARAMS_FILENAME
    params_path.write_text(json.dumps(_pack_parameters(parameters)), encoding="utf-8")

    log.info(
        f"Wrote virtual reference for {len(file_paths)} file(s) in "
        f"{len(manifest_groups)} shape group(s) to {log.yellow(str(output_path))} "
        f"(parameters: {log.yellow(str(params_path))})"
    )
    return output_path


def _pack_parameters(parameters: dict[str, dict]) -> dict:
    """Deduplicate the per-file parameter dicts (a dataset usually shares a few sets).

    Scan parameters carry per-transmit arrays (``t0_delays`` is ``(n_tx, n_el)``), which
    would dwarf the reference itself if written out once per file.
    """
    unique: dict[str, int] = {}
    files: dict[str, int] = {}
    packed: list[dict] = []
    for path, encoded in parameters.items():
        key = json.dumps(encoded, sort_keys=True)
        if key not in unique:
            unique[key] = len(packed)
            packed.append(encoded)
        files[path] = unique[key]
    return {"version": _MANIFEST_VERSION, "parameters": packed, "files": files}


def _expand_index(index, ndim: int) -> tuple:
    """Normalize a NumPy-style index to a full-length tuple (resolving ``Ellipsis``)."""
    if not isinstance(index, tuple):
        index = (index,)
    if index.count(Ellipsis) > 1:
        raise IndexError("An index can only have a single ellipsis ('...').")
    if Ellipsis in index:
        at = index.index(Ellipsis)
        fill = ndim - (len(index) - 1)
        index = index[:at] + (slice(None),) * fill + index[at + 1 :]
    if len(index) > ndim:
        raise IndexError(f"Too many indices for array with {ndim} dimensions.")
    return index + (slice(None),) * (ndim - len(index))


def _read(array: "zarr.Array", index: tuple) -> np.ndarray:
    """Read ``index`` from a Zarr array, using orthogonal indexing when needed."""
    if any(isinstance(part, (list, np.ndarray)) for part in index):
        return cast(np.ndarray, array.oindex[index])
    return cast(np.ndarray, array[index])


class VirtualArray:
    """One virtualized data key (e.g. ``raw_data``) across all files of a reference.

    Index it as ``array[file_index, ...]``, where the remaining indices address the
    file's own axes (frames, transmits, …)::

        array[3]  # every frame of file 3
        array[3, 0:2]  # first two frames of file 3
        array[[0, 4], :, 0]  # transmit 0 of files 0 and 4

    Reads go straight to the chunk byte ranges (concurrently), so ask for as much
    as you need in a single index expression.
    """

    def __init__(self, reference: "VirtualReference", key: str):
        self.reference = reference
        self.key = key

    def __repr__(self):
        return f"VirtualArray(key='{self.key}', n_files={len(self.reference)})"

    @property
    def shape(self) -> tuple:
        """Shape of the combined array, including the leading ``file`` axis.

        Raises:
            AttributeError: When the reference holds several shape groups (the files
                do not share one shape). Use :meth:`group_shapes` instead.
        """
        shapes = self.group_shapes()
        if len(shapes) > 1:
            raise AttributeError(
                f"'{self.key}' has {len(shapes)} shape groups {shapes}: the files in this "
                "reference do not share a single shape. Use .group_shapes(), and index "
                "files that belong to the same group together."
            )
        return shapes[0]

    def group_shapes(self) -> list[tuple]:
        """Shape of the combined array in each shape group of the reference."""
        return [
            tuple(group["arrays"][self.key]["shape"])
            for group in self.reference._groups
            if self.key in group["arrays"]
        ]

    def __getitem__(self, index) -> np.ndarray:
        if not isinstance(index, tuple):
            index = (index,)
        if not index:
            raise IndexError("Index a virtual array as array[file_index, ...].")

        file_index, rest = index[0], index[1:]
        group_index, local = self.reference._locate(file_index)
        array = self.reference._zarr_array(group_index, self.key)
        return _read(array, (local, *_expand_index(rest, array.ndim - 1)))


class VirtualReference:
    """A combined virtual reference: many zea files as one logical, cloud-read array.

    Obtain one with :func:`open_virtual_reference` or from
    :attr:`zea.Dataset.virtual`. Index it by data key, then by file::

        reference["raw_data"][0, 0:4]  # first 4 frames of the first file
        reference[0, 0:4]  # same (defaults to the 'raw_data' key)

    Files keep the order they had at generation time; :attr:`file_paths` lists them.
    Files whose arrays differ in shape (e.g. a differing frame count) sit in separate
    *shape groups*: each is a contiguous block of file indices, and one index
    expression cannot span two of them (their shapes do not stack).
    """

    def __init__(
        self, refs_path: str | Path, revision: str | None = None, source: str | None = None
    ):
        _require_deps()

        self.path = Path(refs_path)
        self.revision = revision
        # Where the reference came from (an ``hf://`` path, or the local path): the
        # parameter sidecar is fetched from next to it, on first use.
        self.source = source or str(refs_path)
        self._parameters: dict | None = None

        with open(self.path, "r", encoding="utf-8") as handle:
            references = json.load(handle)
        try:
            manifest = json.loads(references["refs"][".zattrs"])[_MANIFEST_KEY]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"'{self.path}' is not a zea virtual reference (no '{_MANIFEST_KEY}' manifest). "
                "Generate one with: zea data virtualize <input> <output>"
            ) from exc

        if manifest["version"] != _MANIFEST_VERSION:
            raise ValueError(
                f"Virtual reference '{self.path}' has version {manifest['version']}, but this "
                f"zea version reads version {_MANIFEST_VERSION}. Regenerate it with "
                "`zea data virtualize`."
            )

        self._groups: list[dict] = manifest["groups"]
        # File index i lives in the group whose block of file indices contains i.
        self._offsets = np.cumsum([0] + [len(group["files"]) for group in self._groups])
        self._registry = _object_store_registry()
        self._arrays: dict[tuple[int, str], zarr.Array] = {}

    def __repr__(self):
        return f"VirtualReference(n_files={len(self)}, keys={self.keys()})"

    def __len__(self) -> int:
        return int(self._offsets[-1])

    @property
    def file_paths(self) -> List[str]:
        """The virtualized files, in the order of their file index."""
        return [path for group in self._groups for path in group["files"]]

    def keys(self) -> List[str]:
        """Data keys available in this reference (e.g. ``raw_data``, ``image/values``)."""
        keys: list[str] = []
        for group in self._groups:
            keys += [key for key in group["arrays"] if key not in keys]
        return keys

    @property
    def default_key(self) -> str:
        """Key used when indexing the reference without naming one."""
        keys = self.keys()
        return _DEFAULT_KEY if _DEFAULT_KEY in keys else keys[0]

    def index_of(self, path: str | Path) -> int:
        """File index of ``path`` (as recorded at generation time)."""
        try:
            return self.file_paths.index(str(path))
        except ValueError as exc:
            raise KeyError(f"'{path}' is not part of virtual reference '{self.path}'.") from exc

    def _load_parameters(self) -> dict:
        """Read (and cache) the parameter sidecar published next to the reference."""
        if self._parameters is not None:
            return self._parameters

        if self.source.startswith(HF_PREFIX):
            sidecar = f"{self.source.rsplit('/', 1)[0]}/{_PARAMS_FILENAME}"
            kwargs = {"revision": self.revision} if self.revision is not None else {}
            try:
                local_path = _hf_resolve_path(sidecar, **kwargs)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"No parameter sidecar at '{sidecar}'. It is written next to the reference "
                    "by `zea data virtualize`; regenerate and publish it, or read parameters "
                    "from the file itself with zea.File."
                ) from exc
        else:
            local_path = self.path.parent / _PARAMS_FILENAME
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"No parameter sidecar at '{local_path}'. It is written next to the "
                    "reference by `zea data virtualize`; regenerate it, or read parameters "
                    "from the file itself with zea.File."
                )

        with open(local_path, "r", encoding="utf-8") as handle:
            self._parameters = json.load(handle)
        return self._parameters

    def parameters(self, file_index: int) -> "Parameters":
        """Scan and probe parameters of one file, from the published sidecar.

        Parameters are not virtualized (they are vlen strings and scalars), so they are
        published next to the reference. This reconstructs the same object
        :meth:`zea.File.load_parameters` returns — without opening the HDF5 file.

        Args:
            file_index (int): Index of the file (see :attr:`file_paths`).

        Returns:
            Parameters: The merged scan + probe parameters of that file.
        """
        from zea.parameters import Parameters

        sidecar = self._load_parameters()
        path = self.file_paths[file_index]
        try:
            encoded = sidecar["parameters"][sidecar["files"][path]]
        except KeyError as exc:
            raise KeyError(
                f"'{path}' has no parameters in the sidecar next to '{self.path}'. "
                "The reference and its params.json are out of sync: regenerate both with "
                "`zea data virtualize`."
            ) from exc
        return Parameters(**{key: _decode(value) for key, value in encoded.items()})

    def _locate(self, file_index):
        """Map a file index (int, slice or list) to its group and index within it."""
        n_files = len(self)
        if isinstance(file_index, slice):
            indices = list(range(*file_index.indices(n_files)))
        elif isinstance(file_index, (list, np.ndarray)):
            indices = [int(i) for i in file_index]
        else:
            index = int(file_index)
            if index < 0:
                index += n_files
            if not 0 <= index < n_files:
                raise IndexError(f"File index {file_index} out of range for {n_files} file(s).")
            group_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
            return group_index, index - int(self._offsets[group_index])

        if not indices:
            raise IndexError("Empty file selection.")
        normalized = [i + n_files if i < 0 else i for i in indices]
        if any(not 0 <= i < n_files for i in normalized):
            raise IndexError(f"File selection {file_index} out of range for {n_files} file(s).")

        group_indices = {
            int(np.searchsorted(self._offsets, i, side="right") - 1) for i in normalized
        }
        if len(group_indices) > 1:
            raise IndexError(
                f"File selection {file_index} spans {len(group_indices)} shape groups, whose "
                "arrays have different shapes and cannot be stacked. Select files from a single "
                "group (see .groups()), or read them one by one."
            )
        group_index = group_indices.pop()
        offset = int(self._offsets[group_index])
        return group_index, [i - offset for i in normalized]

    def groups(self) -> List[dict]:
        """The shape groups: ``{"name", "files", "arrays"}`` per group of same-shaped files."""
        return self._groups

    def _zarr_array(self, group_index: int, key: str) -> "zarr.Array":
        """Open (and cache) the Zarr array for ``key`` in shape group ``group_index``."""
        cached = self._arrays.get((group_index, key))
        if cached is not None:
            return cached

        import zarr
        from virtualizarr.parsers import KerchunkJSONParser

        group = self._groups[group_index]
        if key not in group["arrays"]:
            if key in self.keys():
                raise KeyError(
                    f"'{key}' is not virtualized for the selected file(s) {group['files']}, "
                    "though other files in this reference do have it. It was left out at "
                    "generation time (e.g. its chunks are not Zarr-decodable); read those "
                    "files with zea.File instead."
                )
            raise KeyError(f"'{key}' is not in this reference. Available keys: {self.keys()}.")
        array_info = group["arrays"][key]
        store = KerchunkJSONParser(group=array_info["group"])(
            self.path.absolute().as_uri(), self._registry
        )
        array = cast("zarr.Array", zarr.open_group(store, mode="r")[array_info["name"]])
        self._arrays[(group_index, key)] = array
        return array

    def __getitem__(self, item):
        if isinstance(item, str):
            if item not in self.keys():
                raise KeyError(f"'{item}' is not in this reference. Available keys: {self.keys()}.")
            return VirtualArray(self, item)
        return VirtualArray(self, self.default_key)[item]


def open_virtual_reference(path: str | Path, revision: str | None = None) -> VirtualReference:
    """Open a virtual reference, downloading it first when it is an ``hf://`` path.

    Args:
        path (str or Path): Local path or ``hf://`` path of the JSON reference,
            e.g. ``hf://zeahub/camus-sample/virtual/index.json``.
        revision (str, optional): HuggingFace revision to fetch the reference from.
            Note that the chunk URLs inside the reference are pinned at generation
            time; this only selects which reference file is read.

    Returns:
        VirtualReference: The combined, cloud-readable view of the dataset.
    """
    _require_deps()

    path = str(path)
    if path.startswith(HF_PREFIX):
        kwargs = {"revision": revision} if revision is not None else {}
        try:
            local_path = _hf_resolve_path(path, **kwargs)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No virtual reference at '{path}'. Generate and publish one with: "
                "zea data virtualize <input> <output>"
            ) from exc
    else:
        local_path = Path(path)
        if not local_path.is_file():
            raise FileNotFoundError(
                f"No virtual reference at '{path}'. Generate one with: "
                "zea data virtualize <input> <output>"
            )

    return VirtualReference(local_path, revision=revision, source=path)
