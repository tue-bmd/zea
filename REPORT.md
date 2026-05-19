# Tracks Feature Implementation Report

## Overview

This report documents the reimplementation of the **Tracks** multi-track HDF5 feature
onto the `feature/tracks_2` branch, ported from `feature/tracks` and simplified.

Tracks allow a single HDF5 file to store multiple independent acquisition sequences
(e.g. focused + diverging wave imaging in a single session). Each track has its own
`data/` and `scan/` sub-group, and an optional `track_schedule` dataset encodes the
interleaved transmit ordering for precise timestamp reconstruction.

---

## HDF5 Layout

**New format** (written by `FileSpec.save()` and `File.create()`):

```
acquisition.hdf5
├── attrs: probe_name, zea_version, …
├── track_schedule   # optional int32[n_total_tx]
├── metadata/
├── metrics/
└── tracks/
    ├── track_0/
    │   ├── data/    # raw_data, image, …
    │   └── scan/    # probe_geometry, t0_delays, …
    └── track_1/
        ├── data/
        └── scan/
```

**Legacy format** (read-only backwards compatibility): `data/` and `scan/` at root level.

Single-track new-format files are fully backwards-compatible — `file.data`, `file.scan()`,
`file["data"]`, `file["data/raw_data"]`, and `"data" in file` all continue to work
unchanged by transparently routing through `tracks/track_0/`.

---

## Key Design Decisions

### `spec.py`: `TrackSpec` + refactored `FileSpec`

- **`TrackSpec`** is a new `Spec` subclass with `SCHEMA = {"data": DataSpec, "scan": ScanSpec}`.
  It inherits `Spec.store_in_group` which automatically writes `data/` and `scan/` sub-groups.

- **`FileSpec`** is refactored:
  - New primary field: `tracks: list[TrackSpec]`
  - New optional field: `track_schedule: np.ndarray | None`
  - `data` and `scan` become read-only `@property` accessors (single-track only) for
    backwards compatibility with existing call sites
  - Custom `__init__` accepts either `tracks=[TrackSpec(...), ...]` **or** legacy
    `data=..., scan=...` kwargs (which are silently folded into `tracks[0]`)
  - `_SCHEMA_EXCLUDED_FIELDS = frozenset({"tracks"})` tells the schema consistency test
    to skip the `tracks` field (it is managed manually in `save`/`from_hdf5`)
  - `save()` writes `tracks/track_N/data` + `tracks/track_N/scan` layout
  - `from_hdf5()` detects and loads both new and legacy layouts

### `file.py`: `TrackProxy` + updated `File`

- **`TrackProxy`** is a lightweight proxy over a single `tracks/track_N/` HDF5 group.
  It exposes `.data` (a `GroupProxy`), `.scan(**kwargs)` (returns a `Scan`), `.timestamps`
  (computes absolute timestamps from `track_schedule` + `time_to_next_transmit`), and
  `__repr__`.

- **`File.__contains__`** and **`File.__getitem__`** are overridden to redirect
  `"data"`, `"scan"`, and sub-paths like `"data/segmentation"` to
  `tracks/track_0/data` / `tracks/track_0/scan` for single-track new-format files,
  preserving all existing code that accesses the file using legacy paths.

- **`File.data`** and **`File.scan()`** raise `AttributeError` with a helpful message
  when called on a multi-track file, directing the user to `file.tracks`.

- **`File.tracks`** returns a list of `TrackProxy` objects; raises `AttributeError` for
  legacy files.

- **`File._validate_file_impl`** updated to recognise `tracks/track_N/data` paths.

- **`File.copy_key`** updated to copy the scan group using `_scan_h5_group.name` so h5py
  can find it regardless of format.

### Simplifications vs `feature/tracks`

| `feature/tracks` | `feature/tracks_2` |
|---|---|
| `_data_h5_group` property (redundant with `data`) | Removed; internal callers use `self.data._group` |
| `_build_scan_from_dict` ignores its `scan_dict` argument and calls `self.get_scan_parameters()` | Fixed: uses the passed `scan_dict`, enabling correct per-track scan loading |
| `_scan_h5_group` used inconsistently | Kept as a clean internal property, used by `get_parameters`, `copy_key`, `scan`, `n_ax` |

The bug in `feature/tracks` (`_build_scan_from_dict` reading from `tracks/track_0/scan`
regardless of which track was requested) would have caused tracks 1..N to load the wrong
scan. This is fixed in our implementation.

---

## Files Changed

| File | Change |
|---|---|
| `zea/data/spec.py` | Added `TrackSpec`; refactored `FileSpec` with `tracks`/`track_schedule`, custom `__init__`, `data`/`scan` properties, `_SCHEMA_EXCLUDED_FIELDS`, updated `save`/`from_hdf5`/`to_dict` |
| `zea/data/file.py` | Added `TrackProxy`, `_compute_track_timestamps`; updated `File` with `__contains__`, `__getitem__`, `_n_tracks`, `tracks`, `track_schedule`, `_scan_h5_group`, updated `data`, `format_key`, `get_parameters`, `n_ax`, `scan`, `_build_scan_from_dict`, `get_probe_parameters`, `copy_key`, `_validate_file_impl` |
| `tests/data/test_spec.py` | Updated `test_spec_to_dict_is_recursive`, `test_saving_and_loading`, `test_data_custom_key_is_accepted` for new tracks layout; updated `test_schema_keys_match_dataclass_fields_for_all_specs` to respect `_SCHEMA_EXCLUDED_FIELDS` |
| `tests/data/test_file.py` | Added `TrackProxy`/`TrackSpec` imports; updated `TestFieldMetadataAttrs`; added `_make_two_track_spec` helper and `TestMultiTrackFile` (25 tests) |
| `tests/data/test_file_operations.py` | Updated 3 raw-h5py path checks to be format-aware |
| `tests/data/test_conversion_scripts.py` | Updated 1 raw-h5py path check for new tracks layout |

---

## Test Results

- All pre-existing tests continue to pass (1003 passed)
- 25 new `TestMultiTrackFile` tests added and passing
- Pre-existing unrelated failures (`test_metrics[ssim]`, `test_metrics_class_batch_size`,
  `test_multi_gpu_returns_list[jax]`) confirmed as pre-existing on base branch
