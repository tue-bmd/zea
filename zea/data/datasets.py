"""
zea.data.datasets
===================

This module provides classes and utilities for loading, validating, and managing
ultrasound datasets stored in HDF5 format. It supports both local and Hugging Face
Hub datasets, and offers efficient file handle caching for large collections of files.

Main Classes
------------

- H5FileHandleCache: Caches open HDF5 file handles to optimize repeated access.
- Folder: Represents a group of HDF5 files in a directory, with optional validation.
- Dataset: Provides an iterable interface over multiple HDF5 files or folders, with
    support for directory-based splitting and validation.

Functions
---------

- split_files_by_directory: Splits files among directories according to specified ratios.
- count_samples_per_directory: Counts the number of files per directory.

Features
--------

- Validation of dataset integrity with flag files and error logging.
- Support for Hugging Face Hub datasets with local caching.
- Utilities for dataset splitting and sample counting.
- Example usage provided in the module's main block.

"""

import functools
import multiprocessing
import os
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Sequence

import numpy as np
import tqdm

from zea import log
from zea.data.file import File, validate_file
from zea.datapaths import format_data_path
from zea.internal.cache import cache_output
from zea.internal.core import hash_elements
from zea.internal.preset_utils import (
    HF_DATASETS_DIR,
    HF_PREFIX,
    _hf_content_id,
    _hf_list_files,
    _hf_list_h5_files,
    _hf_parse_path,
    _hf_resolve_path,
)
from zea.internal.utils import reduce_to_signature
from zea.io_lib import search_file_tree
from zea.tools.hf import HFPath

if TYPE_CHECKING:
    # ``Self`` is in ``typing`` only from 3.11; import lazily to keep the
    # 3.10 floor runtime-clean.
    from typing_extensions import Self

FILE_HANDLE_CACHE_CAPACITY = 128
# Below this many files the sweeps stay serial; sized for network opens
_PARALLEL_MIN_FILES = 32
# How many invalid files a validation error spells out before summarising the rest.
_MAX_LISTED_INVALID_FILES = 10
# Seconds a per-file sweep may take before it draws a progress bar.
_PROGRESS_DELAY_S = 1.0
FILE_TYPES = [".hdf5", ".h5"]


def EXISTS(value: Any) -> bool:
    """Condition for :func:`compile_file_filter` dict filters: field is present.

    Use as the condition for a dotted path to keep only files where that field is
    present (i.e. resolves to a non-``None`` value).  For example
    ``{"metadata.subject.fat_percentage": EXISTS}`` keeps only files that record a
    subject fat percentage.

    It is a plain callable (so it survives module reloads and needs no special
    casing), evaluated like any other value-level predicate.
    """
    return value is not None


def _resolve_dotted_path(obj: Any, path: str) -> Any:
    """Resolve a dotted attribute ``path`` on ``obj``, tolerating missing fields.

    Walks each segment of ``path`` via :func:`getattr`.  Returns ``None`` as soon
    as any segment is missing or already ``None`` (e.g. ``f.metadata`` raising
    ``KeyError`` when there is no metadata group, or ``f.scan`` returning ``None``).
    """
    for part in path.split("."):
        if obj is None:
            return None
        try:
            obj = getattr(obj, part)
        except (AttributeError, KeyError):
            return None
    return obj


def _compile_dict_filter(spec: dict) -> Callable[["File"], bool]:
    """Compile a dotted-path dict into a ``File -> bool`` predicate.

    Each ``path -> condition`` entry is ANDed together. ``condition`` may be:

    - a callable (e.g. :func:`EXISTS`, or a value-level lambda) — keep when
      ``condition(value)`` is truthy. A ``None`` value (missing field) never
      matches.
    - any other value — keep when the resolved value equals it.
    """
    conditions = list(spec.items())

    def predicate(file: "File") -> bool:
        for path, condition in conditions:
            value = _resolve_dotted_path(file, path)
            if callable(condition):
                if value is None or not condition(value):
                    return False
            else:
                if value != condition:
                    return False
        return True

    return predicate


def compile_file_filter(
    file_filter: "Callable[[File], bool] | dict | None",
) -> "Callable[[File], bool] | None":
    """Normalize ``file_filter`` into a ``File -> bool`` predicate (or ``None``).

    Accepts either a callable predicate over an open :class:`~zea.data.file.File`,
    or a declarative dotted-path dict (see :func:`_compile_dict_filter`). Returns
    ``None`` when ``file_filter`` is ``None`` (no filtering).
    """
    if file_filter is None:
        return None
    if isinstance(file_filter, dict):
        return _compile_dict_filter(file_filter)
    if callable(file_filter):
        return file_filter
    raise TypeError(
        "file_filter must be a callable 'File -> bool', a dotted-path dict, or None; "
        f"got {type(file_filter).__name__}."
    )


class H5FileHandleCache:
    """Cache for HDF5 file handles.

    This class manages a cache of HDF5 file handles to avoid reopening files
    multiple times. It uses an OrderedDict to maintain the order of file
    access and closes the least recently used file when the cache reaches
    its capacity."""

    def __init__(
        self,
        file_handle_cache_capacity: int = FILE_HANDLE_CACHE_CAPACITY,
    ):
        self._file_handle_cache = OrderedDict()
        self.file_handle_cache_capacity = file_handle_cache_capacity

    @staticmethod
    def _check_if_open(file):
        """Check if a file is open."""
        return bool(file.id.valid)

    def _open_file(self, file_path, revision: str | None = None) -> File:
        """Open one file for the cache, at the caller's ``revision``.

        A plain handle cache has no idea where its paths came from, so it opens
        read-only and never validates. Subclasses that do know override this to apply
        their own open policy -- see :meth:`Dataset._open_file`.
        """
        return File(file_path, "r", progress=False, revision=revision, validate=False)

    def get_file(self, file_path, revision: str | None = None) -> File:
        """Open an HDF5 file and cache it.

        ``revision`` is part of the cache key, not just the open: two revisions of one
        ``hf://`` path are different files, so one must never be served for the other.
        """
        cache_key = (file_path, revision)
        # If file is already in cache, return it and move it to the end
        if cache_key in self._file_handle_cache:
            self._file_handle_cache.move_to_end(cache_key)
            file = self._file_handle_cache[cache_key]
            # if file was closed, reopen:
            if not self._check_if_open(file):
                self._file_handle_cache[cache_key] = self._open_file(file_path, revision)
        # If file is not in cache, open it and add it to the cache
        else:
            # If cache is full, close the least recently used file
            if len(self._file_handle_cache) >= self.file_handle_cache_capacity:
                _, close_file = self._file_handle_cache.popitem(last=False)
                close_file.close()
            self._file_handle_cache[cache_key] = self._open_file(file_path, revision)

        return self._file_handle_cache[cache_key]

    def pop(self, file_path, revision: str | None = None):
        """Pop a file from the cache and close it."""
        file = self._file_handle_cache.pop((file_path, revision), None)
        if file is not None:
            try:
                file.close()
            except Exception:
                pass  # swallow exceptions during close

    def close(self):
        """Close all cached file handles."""
        cache: OrderedDict | None = getattr(self, "_file_handle_cache", None)
        if not cache:
            return

        # iterate over a static list to avoid mutation during iteration
        for fh in list(cache.values()):
            if fh is None:
                continue
            try:
                # attempt to close unconditionally and swallow exceptions
                fh.close()
            except Exception:
                # During interpreter shutdown or if the h5py internals are already
                # torn down, close() can raise weird errors (e.g. TypeError).
                # Swallow them here to avoid exceptions from __del__.
                pass

        cache.clear()  # clear the cache dict

    def __del__(self):
        self.close()


def _shape_and_metadata_gaps(file_path, key, metadata_paths, revision=None):
    """Describe one file: its shape for ``key``, plus what its metadata looks like.

    Both metadata passes ride along with the shape read so that a dataset is swept
    once, not three times: the file is open either way, and shapes and presence both
    come off the handles without reading data. Must stay importable at module level
    -- it runs in a :mod:`multiprocessing` worker.
    """
    # Imported here rather than at module scope: zea.data.metadata imports from us.
    from zea.data.metadata import metadata_signature, missing_metadata_paths

    # validate=False: the Dataset that owns these paths already validated them at
    # discovery (or was told not to), so re-checking every file here is wasted work.
    with File(file_path, mode="r", revision=revision, validate=False) as file:
        shape = file.shape(key)
        missing = missing_metadata_paths(file, metadata_paths)
        # Only meaningful once every file can answer, so skip the walk when one cannot.
        signature = {} if missing else metadata_signature(file, metadata_paths)
    return shape, missing, signature


def _map_over_files(func, file_paths, desc, verbose=True):
    """Apply *func* to every file path, in parallel when that is worth its startup cost.

    ``func`` must be importable at module level -- it runs in a
    :mod:`multiprocessing` worker. Set ``ZEA_FIND_H5_SHAPES_PARALLEL=0`` to force the
    main process (e.g. when calling from a worker yourself).
    """
    parallel = len(file_paths) >= _PARALLEL_MIN_FILES and os.environ.get(
        "ZEA_FIND_H5_SHAPES_PARALLEL", "1"
    ) in ("1", "true", "yes")

    if parallel:
        # make sure to call this from within a function or use if __name__ == "__main__"
        # to avoid freezing the main process
        with multiprocessing.Pool() as pool:
            return list(
                tqdm.tqdm(
                    pool.imap(func, file_paths),
                    total=len(file_paths),
                    desc=desc,
                    disable=not verbose,
                    delay=_PROGRESS_DELAY_S,
                )
            )

    return [
        func(file_path)
        for file_path in tqdm.tqdm(
            file_paths, desc=desc, disable=not verbose, delay=_PROGRESS_DELAY_S
        )
    ]


@cache_output("filepaths", "key", "metadata_paths", "_filepath_hash", "revision")
def _find_h5_file_shapes(
    filepaths, key, _filepath_hash, metadata_paths=(), verbose=True, revision=None
):
    # NOTE: we cache the output of this function such that file loading over the network is
    # faster for repeated calls with the same filepaths, key and _filepath_hash

    assert _filepath_hash is not None

    get_shape = functools.partial(
        _shape_and_metadata_gaps, key=key, metadata_paths=tuple(metadata_paths), revision=revision
    )
    results = _map_over_files(
        get_shape, filepaths, "Getting file shapes in each h5 file", verbose=verbose
    )

    file_shapes = [shape for shape, _, _ in results]
    metadata_gaps = {
        file_path: missing for file_path, (_, missing, _) in zip(filepaths, results) if missing
    }
    metadata_signatures = {
        file_path: signature for file_path, (_, _, signature) in zip(filepaths, results)
    }
    return file_shapes, metadata_gaps, metadata_signatures


def _file_hash(filepaths, revision: str | None = None):
    """Hash the identity of a list of files, to key caches that must follow their content.

    Local files are identified by size and modification time. ``hf://`` files are identified by
    content id (see :func:`~zea.internal.preset_utils._hf_content_id`), which moves when a file is
    re-uploaded or when ``revision`` points somewhere else.
    """
    # NOTE: this is really fast, even over network filesystems: the local branch stats
    # each file, and the remote one reads an already-memoized repo listing.
    total_size = 0
    modified_times = []
    content_ids = []
    hf_kwargs = {"revision": revision} if revision is not None else {}
    for fp in filepaths:
        fp = str(fp)
        if fp.startswith(HF_PREFIX):
            # Falling back to the path keeps the hash stable (not wrong) for a file the
            # listing cannot identify; it just stops distinguishing that file's versions.
            content_ids.append(_hf_content_id(fp, **hf_kwargs) or fp)
        elif os.path.isfile(fp):
            total_size += os.path.getsize(fp)
            modified_times.append(os.path.getmtime(fp))
    return hash_elements([total_size, modified_times, content_ids])


def _validate_h5_file(file_path):
    """Validate one file. Returns ``None`` when valid, else the error message.

    Must stay importable at module level -- it runs in a :mod:`multiprocessing` worker.
    """
    try:
        # validate=False on the open, then validate explicitly: the error belongs in the
        # returned report, not raised out of a worker process.
        with File(file_path, validate=False) as file:
            validate_file(file=file)
    except Exception as e:  # noqa: BLE001 -- reported per file, never raised from a worker
        return f"{type(e).__name__}: {e}"
    return None


@functools.lru_cache(maxsize=1)
def _validator_hash():
    """Hash of the source of :mod:`zea.data`, where the validation rules live."""
    sources = []
    try:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            sources.append(path.read_bytes().decode("utf-8", "replace"))
    except OSError:  # no source on disk (e.g. frozen install): the version has to do
        import zea

        sources = [zea.__version__]
    return hash_elements(sources)


@cache_output("file_paths", "_filepath_hash", "_validator_hash")
def _validate_h5_files(file_paths, _filepath_hash, _validator_hash, verbose=True) -> dict:
    """Validate every file against the zea format, caching the verdict.

    Cached on the file list, :func:`_file_hash` (size and modification time) and
    :func:`_validator_hash`, so a repeated call over unchanged files is a cache read
    rather than a sweep over every file on disk.

    Returns:
        dict: ``{file path: error message}`` for the files that failed validation.
        An empty dict means every file is a valid zea file.
    """
    errors = _map_over_files(
        _validate_h5_file,
        file_paths,
        "Checking dataset files on validity (zea format)",
        verbose=verbose,
    )
    return {path: error for path, error in zip(file_paths, errors) if error is not None}


def _raise_on_invalid_files(file_paths, verbose=True):
    """Validate *file_paths*, raising a single error that names the invalid files."""
    file_paths = [str(fp) for fp in file_paths]
    errors = _validate_h5_files(
        file_paths, _file_hash(file_paths), _validator_hash(), verbose=verbose
    )
    if not errors:
        return

    listed = list(errors.items())[:_MAX_LISTED_INVALID_FILES]
    details = "\n".join(f"  {path}: {error}" for path, error in listed)
    if len(errors) > len(listed):
        details += f"\n  ... and {len(errors) - len(listed)} more"
    raise ValueError(
        f"{len(errors)} of {len(file_paths)} file(s) are not valid zea files:\n{details}\n"
        "Pass validate=False to open them anyway."
    )


def _copy_h5_files(file_paths, base_path, to_path, key, mode=None, revision=None):
    """Copy ``key`` (or all keys) from each file to ``to_path``, mirroring structure.

    Each file's location relative to ``base_path`` is preserved under ``to_path``.
    Shared implementation for :meth:`Folder.copy` and :meth:`Dataset.copy`.

    Args:
        file_paths (list): Paths to the source HDF5 files.
        base_path (str or Path): Base directory the source files are made relative to
            when constructing their destination paths.
        to_path (str or Path): The destination directory where files will be copied.
        key (str): The key to copy from the source files. If ``"all"`` or ``"*"``, all
            keys are copied.
        mode (str, optional): The mode in which to open the destination files. Defaults
            to ``"a"`` (append), and ``"w"`` (write) if ``key`` is ``"all"`` or ``"*"``.
        revision (str, optional): HuggingFace revision used to download ``hf://`` sources.
    """
    all_keys = key == "all" or key == "*"

    if mode is None:
        mode = "a" if not all_keys else "w"

    if all_keys:
        key_msg = "Including all keys."
        if mode not in ["w", "x"]:
            raise ValueError(
                f"Invalid mode '{mode}' for copying all keys. Must be 'w' or 'x'."
                "If you want to copy all keys, the destination file must be opened "
                "in 'w' or 'x' mode, which means it will be overwritten or created."
            )
    else:
        key_msg = f"Only copying key '{key}'."
        if mode not in ["a", "w", "r+", "x"]:
            raise ValueError(
                f"Invalid mode '{mode}' for copying a specific key. "
                "Must be one of 'a', 'w', 'r+', or 'x'."
            )

    base_path = Path(base_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm.tqdm(
        file_paths,
        total=len(file_paths),
        desc=f"Copying {len(file_paths)} file(s) from {base_path} to {to_path}. {key_msg}",
    ):
        dst_path = to_path / Path(file_path).relative_to(base_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        # validate=False on the source: these paths come from a Folder/Dataset that has
        # already applied its own validate policy to them.
        with (
            File(file_path, stream=False, revision=revision, validate=False) as src,
            File(dst_path, mode) as dst,
        ):
            if all_keys:
                for obj in src.keys():
                    src.copy(obj, dst)
            else:
                src.copy_key(key, dst)


class Folder:
    """Group of HDF5 files in a folder that can be validated.
    Mostly used internally, you might want to use the Dataset class instead.
    """

    def __init__(
        self,
        folder_path: str | Path | HFPath,
        validate: bool = True,
        hf_cache_dir: str = HF_DATASETS_DIR,
        revision: str | None = None,
    ):
        single_file_error_msg = (
            f"Folder class requires a directory path, but got a single file: {str(folder_path)}. "
            "Use File class instead for single files."
        )

        if not isinstance(folder_path, (str, Path, HFPath)):
            raise ValueError(
                f"Invalid folder path: {folder_path}. Must be a string, Path, or HFPath."
            )

        # Hugging Face support
        folder_path_str = str(folder_path)
        if folder_path_str.startswith(HF_PREFIX):
            hf_kwargs = {}
            if revision is not None:
                hf_kwargs["revision"] = revision
            repo_id, subpath = _hf_parse_path(folder_path_str)
            files = _hf_list_files(repo_id, **hf_kwargs)

            # Check if it's a single file (not a directory)
            if subpath and any(f == subpath for f in files):
                raise ValueError(single_file_error_msg)

            # It's a directory, resolve to local cache
            folder_path = _hf_resolve_path(folder_path_str, cache_dir=hf_cache_dir, **hf_kwargs)

        # Check if the resolved path is a directory
        self.folder_path = Path(folder_path)
        if self.folder_path.is_file():
            raise ValueError(single_file_error_msg)

        # Find all hdf5 files in the folder
        self.file_paths = self.find_h5_files()
        if self.n_files == 0:
            raise ValueError(f"No HDF5 files found in folder: {folder_path}")

        if validate:
            self.validate_folder()

    def find_h5_files(self) -> List[str]:
        file_paths = list(search_file_tree(self.folder_path, filetypes=FILE_TYPES))
        return [str(fp) for fp in file_paths]  # to string

    def load_file_shapes(self, key: str, metadata_paths: Sequence[str] = ()):
        """Load the shapes of the datasets in each file.

        Args:
            key: HDF5 dataset key whose shape to read.
            metadata_paths: Dotted metadata paths to check for while each file is
                open. Default is ``()`` (no check).

        Returns:
            tuple: ``(file_shapes, metadata_gaps, metadata_signatures)``.
            ``metadata_gaps`` maps a file path to the requested paths that file
            lacks -- only files missing something appear, so an empty dict means
            every file can answer. ``metadata_signatures`` maps every file path to
            its ``{leaf path: shape}`` mapping, for checking that the files agree
            well enough to be batched together.
        """
        return _find_h5_file_shapes(
            self.file_paths, key, _file_hash(self.file_paths), tuple(metadata_paths)
        )

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.n_files

    @property
    def n_files(self):
        """Return number of files in dataset."""
        return len(self.file_paths)

    def validate_folder(self):
        """Validate that every file in the folder is a valid zea file.

        The verdict is cached (see :func:`_validate_h5_files`), so repeating this over
        an unchanged folder costs a ``stat`` per file rather than an open per file.

        Raises:
            ValueError: If any file in the folder is not a valid zea file. The message
                names the offending files.
        """
        _raise_on_invalid_files(self.file_paths)

    def __repr__(self):
        return f"Folder(n_files={self.n_files}, folder='{self.folder_path}')"

    def __str__(self):
        return f"Folder with {self.n_files} files in '{self.folder_path}'"

    def copy(self, to_path: str | Path, key: str, mode: str | None = None):
        """Copy the data for all or a specific key to a new location.

        Has the option to copy all keys or only a specific key. By default, it only copies if the
        destination file does not already contain the key. You can change the mode to 'w' to
        overwrite the destination file. Will always copy metadata such as dataset attributes and
        scan object.

        Args:
            to_path (str or Path): The destination path where files will be copied.
            key (str, optional): The key to copy from the source files.
                If 'all' or '*', all keys will be copied.
            mode (str): The mode in which to open the destination files.
                Defaults to 'a' (append mode), and 'w' (write mode) if key is 'all' or '*'.
                See: https://docs.h5py.org/en/stable/high/file.html#opening-creating-files
        """
        _copy_h5_files(self.file_paths, self.folder_path, to_path, key, mode=mode)


class Dataset(H5FileHandleCache):
    """Dataset class managing multiple :class:`~zea.data.file.File` objects."""

    def __init__(
        self,
        file_paths: Sequence[str | Path] | str | Path,
        validate: bool = True,
        directory_splits: list | None = None,
        revision: str | None = None,
        lazy: bool = True,
        file_filter: "Callable[[File], bool] | dict | None" = None,
        _suggest_lazy: bool = True,
        **kwargs,
    ):
        """Initializes the Dataset.

        Args:
            file_paths (str or list): (list of) path(s) to the folder(s) containing the HDF5 file(s)
                or list of HDF5 file paths. Can be a mixed list of folders and files.
            validate (bool, optional): Whether to validate the discovered files against
                the zea format, raising if any of them is not a valid zea file. Defaults
                to True. The verdict is cached, so only the first call over a given
                dataset opens every file. Lazy ``hf://`` files are validated when first
                opened instead.
            directory_splits (list, optional): List of directory split by. Is a list of floats
                between 0 and 1, with the same length as the number of file_paths given.
                If none, all files in file_paths are used.
            revision (str, optional): HuggingFace revision (branch, tag, or commit hash).
                Only used when file_paths contains ``hf://`` paths. Defaults to ``None``
                (uses HuggingFace Hub default, i.e. the ``main`` branch).
            lazy (bool, optional): If True, ``hf://`` files are not downloaded at init —
                each file is downloaded on first access. ``len(ds)`` returns the number of files
                (not total frames). Defaults to True.
            file_filter (callable or dict, optional): Keep only files whose content matches a
                predicate. Either a callable ``File -> bool`` (a file is kept when it returns
                ``True``), or a declarative dotted-path dict mapping a path on the
                :class:`~zea.data.file.File` (e.g. ``"metadata.subject.fat_percentage"``,
                ``"scan.center_frequency"``) to a condition: the :func:`EXISTS` helper
                (field must be present), a plain value (equality), or a callable on the
                resolved value. All dict entries are ANDed. Files whose predicate raises
                (e.g. no ``metadata`` group) are excluded, except that I/O errors
                (``OSError`` from HDF5/network reads) propagate rather than being treated
                as a mismatch. Evaluating the predicate reads
                each file; with ``lazy=True`` remote (``hf://``) files are streamed for just
                the metadata/scan bytes rather than downloaded in full, and the surviving
                files stay lazy. Defaults to ``None`` (no filtering).
        """
        super().__init__(**kwargs)
        self.validate = validate
        self.revision = revision
        self.lazy = lazy
        self.file_filter = compile_file_filter(file_filter)
        self._suggest_lazy = _suggest_lazy
        # Lazy hf:// paths that _open_file has already validated, so a handle that is
        # evicted and reopened does not stream the file again to re-check its structure.
        self._validated: set[str] = set()
        #: Bytes held by the files this dataset streams rather than downloads. Filled by
        #: `find_files` from the repo listing, so it costs nothing extra to know.
        self.remote_bytes = 0

        self.file_paths = self.find_files(file_paths)

        if directory_splits is not None:
            # Split the files according to their parent directories
            self.file_paths = split_files_by_directory(
                self.file_paths,
                directory_list=file_paths,
                directory_splits=directory_splits,
            )

        assert self.n_files > 0, f"No files in file_paths: {file_paths}"

        if self.validate:
            self._validate_files()

        if self.file_filter is not None:
            self.file_paths = self._apply_file_filter(self.file_paths)

    def _apply_file_filter(self, file_paths: List[str]) -> List[str]:
        """Return only the files for which ``self.file_filter`` returns ``True``.

        Each file is opened and passed to the predicate. If the predicate raises
        (e.g. the file has no ``metadata`` group), the file is excluded. I/O failures
        (``OSError`` from HDF5/network reads) are not mismatches and propagate instead.
        """
        assert self.file_filter is not None, "file_filter must be set before applying it"
        file_filter = self.file_filter
        kept: List[str] = []
        for path in file_paths:
            # Opening the file is file access, not filtering: let open/download/
            # permission failures propagate instead of silently dropping the file.
            # validate=False: these paths were validated just above, in __init__.
            with File(path, revision=self.revision, validate=False) as file:
                try:
                    keep = file_filter(file)
                except OSError:
                    # I/O failure (HDF5/network read while streaming a lazy remote
                    # file) is not a filter mismatch: propagate rather than silently
                    # dropping a file that may actually match.
                    raise
                except Exception as e:  # noqa: BLE001 — a predicate failure means "does not match"
                    log.debug(f"file_filter excluded '{path}': {type(e).__name__}: {e}")
                    continue
            if keep:
                kept.append(path)
            else:
                log.debug(f"file_filter excluded '{path}'.")

        n_removed = len(file_paths) - len(kept)
        if not kept:
            raise ValueError(
                f"file_filter removed all {len(file_paths)} files. "
                "Check that the filter matches the dataset's metadata/scan fields."
            )
        if n_removed:
            log.info(f"file_filter kept {len(kept)}/{len(file_paths)} files ({n_removed} removed).")
        return kept

    def load_file_shapes(self, key: str, metadata_paths: Sequence[str] = ()):
        """Load the shapes of the datasets in each file.

        Args:
            key: HDF5 dataset key whose shape to read.
            metadata_paths: Dotted metadata paths to check for while each file is
                open. Default is ``()`` (no check).

        Returns:
            tuple: ``(file_shapes, metadata_gaps, metadata_signatures)``.
            ``metadata_gaps`` maps a file path to the requested paths that file
            lacks -- only files missing something appear, so an empty dict means
            every file can answer. ``metadata_signatures`` maps every file path to
            its ``{leaf path: shape}`` mapping, for checking that the files agree
            well enough to be batched together.
        """
        return _find_h5_file_shapes(
            self.file_paths,
            key,
            _file_hash(self.file_paths, revision=self.revision),
            tuple(metadata_paths),
            revision=self.revision,
        )

    def _find_hf_files(self, hf_path: str) -> List[str]:
        """Resolve an HF path to a list of local paths (or HF strings if lazy)."""
        repo_id, _ = _hf_parse_path(hf_path)
        hf_kwargs = {}
        if self.revision is not None:
            hf_kwargs["revision"] = self.revision

        hf_files = _hf_list_h5_files(hf_path, **hf_kwargs)  # [(filename, size_bytes), ...]

        if not hf_files:
            raise FileNotFoundError(f"No HDF5 files found in {hf_path}")

        if len(hf_files) > 10 and not self.lazy:
            total_gb = sum(size for _, size in hf_files) / 1e9
            msg = (
                f"About to download {len(hf_files)} files ({total_gb:.1f} GB) "
                f"from '{hf_path}'. This may take a while."
            )
            if self._suggest_lazy:
                msg += (
                    f" Use {type(self).__name__}(..., lazy=True) to defer "
                    "downloads until each file is first accessed."
                )
            log.warning(msg)

        if self.lazy:
            self.remote_bytes += sum(size for _, size in hf_files)
            return [f"{HF_PREFIX}{repo_id}/{f}" for f, _ in hf_files]

        resolved = _hf_resolve_path(hf_path, **hf_kwargs)
        if resolved.is_dir():
            return sorted(str(fp) for fp in search_file_tree(resolved, filetypes=FILE_TYPES))
        return [str(resolved)]

    def find_files(self, paths) -> List[str]:
        """Expand folders, ``hf://`` paths and single files into a flat list of files."""
        file_paths = []

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        for file_path in paths:
            if isinstance(file_path, (list, tuple)):
                file_paths += self.find_files(file_path)
                continue

            file_path = str(file_path)

            if file_path.startswith(HF_PREFIX):
                file_paths += self._find_hf_files(file_path)
                continue

            file_path = Path(file_path)
            if file_path.is_dir():
                # validate=False: the whole discovered set is validated in one pass below.
                folder = Folder(file_path, validate=False, revision=self.revision)
                file_paths += folder.file_paths
                del folder
            elif file_path.is_file():
                file_paths.append(str(file_path))

        return file_paths

    def _open_file(self, file_path, revision: str | None = None) -> File:
        """Open one file at the dataset's revision."""
        # Only validate lazy hf:// paths that have not already been validated.
        # Local paths and non-lazy hf:// paths are validated at discovery.
        validate = (
            self.validate
            and str(file_path).startswith(HF_PREFIX)
            and str(file_path) not in self._validated
        )
        file = File(file_path, "r", progress=False, revision=revision, validate=validate)
        if validate:
            # Only on success: a raising open leaves the path unvalidated, to be retried.
            self._validated.add(str(file_path))
        return file

    def _validate_files(self):
        """Validate the discovered files in one cached pass.

        Lazy ``hf://`` pointers are skipped: they are validated when each file is first
        opened (see :meth:`_open_file`), because streaming every remote file here is
        exactly what ``lazy=True`` exists to avoid.
        """
        local_paths = [p for p in self.file_paths if not str(p).startswith(HF_PREFIX)]
        if local_paths:
            _raise_on_invalid_files(local_paths)

    @classmethod
    def from_config(cls, path, user=None, **kwargs) -> "Self":
        """Creates a Dataset from a config file."""
        resolved = format_data_path(path, user)
        reduced_params = reduce_to_signature(cls.__init__, kwargs)
        return cls(str(resolved), **reduced_params)

    def __len__(self):
        """Returns the number of files in the dataset."""
        return self.n_files

    @property
    def n_files(self):
        """Return number of files in dataset."""
        return len(self.file_paths)

    def copy(self, to_path: str | Path, key: str, mode: str | None = None):
        """Copy the data for all or a specific key to a new location.

        Works for a dataset built from a single file, a list of files, or a folder. Each
        file is written under ``to_path``, mirroring its location relative to the common
        parent directory of the dataset's files (for a single file, its own name is used).

        Has the option to copy all keys or only a specific key. By default, it only copies
        if the destination file does not already contain the key. You can change the mode to
        'w' to overwrite the destination file. Will always copy metadata such as dataset
        attributes and scan object.

        Args:
            to_path (str or Path): The destination folder where files will be copied.
            key (str): The key to copy from the source files.
                If 'all' or '*', all keys will be copied.
            mode (str, optional): The mode in which to open the destination files.
                Defaults to 'a' (append mode), and 'w' (write mode) if key is 'all' or '*'.
                See: https://docs.h5py.org/en/stable/high/file.html#opening-creating-files
        """
        if self.n_files == 1:
            base_path = Path(self.file_paths[0]).parent
        else:
            base_path = Path(os.path.commonpath([str(p) for p in self.file_paths]))
        _copy_h5_files(self.file_paths, base_path, to_path, key, mode=mode, revision=self.revision)

    def __getitem__(self, index) -> "File":
        """Retrieves an item from the dataset.

        Returns a :class:`~zea.data.file.File`. Lazy ``hf://`` files are opened with
        streaming enabled rather than downloaded in full: reads through the file
        (``dataset[0].data.raw_data[0]``) fetch only the chunks they touch — see
        :mod:`zea.data.chunk_reader`.
        """
        # Lazy datasets keep ``hf://`` URIs in ``file_paths``; pass them straight to
        # ``get_file`` so ``File`` streams them (stream=True is the default for hf://
        # paths) instead of resolving to a local path, which downloads the whole file.
        # Non-lazy HF paths were already resolved at init, and non-HF paths are local.
        return self.get_file(self.file_paths[index], revision=self.revision)

    def __iter__(self):
        """
        Generator that yields images from the hdf5 files.
        """
        for idx in range(self.n_files):
            yield self[idx]

    def __call__(self):
        return iter(self)

    @property
    def total_frames(self):
        """Return total number of frames in dataset."""
        return sum(
            self.get_file(file_path, revision=self.revision).n_frames
            for file_path in self.file_paths
        )

    def __repr__(self):
        return f"Dataset(n_files={self.n_files})"

    def __str__(self):
        return f"Dataset with {self.n_files} files"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def split_files_by_directory(file_names, directory_list, directory_splits):
    """Split files according to their parent directories and given split ratios.

    Args:
        file_names (list): List of file paths.
        directory_list (list): List of directory paths to split by.
        directory_splits (list): List of split ratios (0-1) for each directory.

    Returns:
        tuple: (split_file_names, split_file_shapes)
    """
    if isinstance(directory_list, str):
        directory_list = [directory_list]
    if isinstance(directory_splits, (float, int)):
        directory_splits = [directory_splits]

    assert len(directory_splits) == len(directory_list), (
        "Number of directory splits must be equal to the number of directories."
    )
    assert all(0 <= split <= 1 for split in directory_splits), (
        "Directory splits must be between 0 and 1."
    )

    # Get directory sizes using the new count function implementation
    directory_counts = count_samples_per_directory(file_names, directory_list)
    directory_sizes = [directory_counts[str(dir_path)] for dir_path in directory_list]

    # take percentage of the files from each directory
    split_indices = [int(split * size) for split, size in zip(directory_splits, directory_sizes)]

    # offset split indices by each total number of files
    start_datasets = [0] + list(np.cumsum(directory_sizes))
    split_indices = [
        (start_datasets[i], start_datasets[i] + split) for i, split in enumerate(split_indices)
    ]

    # split the files
    split_file_names = []

    for start, end in split_indices:
        split_file_names.extend(file_names[start:end])

    # verify the split size
    expected_size = sum((d * s) for d, s in zip(directory_sizes, directory_splits))
    expected_size = int(expected_size)
    assert len(split_file_names) == expected_size, (
        "Number of files in split directories does not match the expected number. "
        "Please check the directory splits."
    )

    return split_file_names


def count_samples_per_directory(file_names, directories):
    """Count number of samples per directory.

    Args:
        file_names (list): List of file paths
        directories (str or list): Directory or list of directories

    Returns:
        dict: Dictionary with directory paths as keys and sample counts as values
    """
    if not isinstance(directories, list):
        directories = [directories]

    # Convert all paths to strings with normalized separators
    dir_paths = [str(Path(d)) for d in directories]

    file_paths = [str(Path(f)) for f in file_names]

    # Count files per directory using string matching
    counts = {
        dir_path: sum(1 for f in file_paths if f.startswith(dir_path)) for dir_path in dir_paths
    }

    # Assert that the total counts match the number of files
    total_count = sum(counts.values())
    assert total_count == len(file_paths), (
        f"Total count of files ({total_count}) does not match the number of files provided "
        f"({len(file_paths)}). Some files may not belong to any of the specified directories."
    )

    return counts
