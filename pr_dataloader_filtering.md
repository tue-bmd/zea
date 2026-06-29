# Filter datasets by file content in `Dataset` / `Dataloader`

## Summary

Adds a `file_filter` parameter to `zea.Dataset` and `zea.Dataloader` so you can load a
folder but keep only the files whose **content** matches a predicate.

Previously, file discovery was purely filesystem-based (walk a directory, match `.hdf5`/`.h5`).
With the rich per-file metadata now exposed through the spec system, there was no way to
select files by what's *inside* them. That is addressed in this PR.

The filtering logic lives in **one place** (`zea.Dataset`); `zea.Dataloader` just
forwards the parameter, so there's no duplicated logic.

## API

`file_filter` accepts **either** form:

1. **Callable** `Callable[[File], bool]` — receives the open `zea.File` (so it can read
   `f.metadata`, `f.scan`, `f.probe`, `f.n_frames`, …) and returns `True` to keep the file.
   Maximum flexibility: cross-field logic, ranges, arithmetic — anything.

2. **Declarative dotted-dict** `dict[str, Any]` — maps a dotted attribute path on the `File`
   to a condition. The condition is the `EXISTS` helper (field must be present), a plain
   value (equality), or a callable on the resolved value. All entries are ANDed together.
   Config-friendly for the common presence/equality cases.

**Semantics:** a file is kept iff the predicate returns `True`. If the predicate *raises*
(e.g. the file has no `metadata` group, so `f.metadata` raises `KeyError`), the file is
**excluded** with a debug log — so "keep files that have X" reads naturally without
defensive guards. A filter that removes everything raises a clear `ValueError`. Because
filtering must read each file, it is incompatible with `lazy=True` (raises with guidance).

```python
from zea import Dataloader, EXISTS

# callable form — full flexibility
Dataloader(
    file_paths="/data/ds",
    file_filter=lambda f: f.metadata.subject is not None
    and f.metadata.subject.fat_percentage is not None,
)

# dict form — presence + equality + value-level predicate (all ANDed)
Dataloader(
    file_paths="/data/ds",
    file_filter={
        "metadata.subject.fat_percentage": EXISTS,
        "metadata.subject.sex": "f",
        "scan.center_frequency": lambda v: 4e6 <= v <= 6e6,
    },
)
```

## Try it (standalone)

This script writes two tiny zea files into a temp dir — one *with* a subject fat percentage,
one *without* — then shows both the dict and callable filter keeping only the right file.
It cleans up after itself.

```python
"""Standalone demo of Dataset/Dataloader content filtering. Run: python demo_filter.py"""

import tempfile
from pathlib import Path

import numpy as np

from zea import EXISTS, Dataset
from zea.data.file import File


def write_file(path, *, fat_percentage=None, center_frequency=5e6):
    """Write a minimal valid zea file with an `image` product and some metadata."""
    n_frames, n_tx, n_ax, n_el, n_ch, grid = 4, 2, 64, 8, 1, 16
    data = {
        "raw_data": np.ones((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
        "image": {"values": np.zeros((n_frames, grid, grid), dtype=np.uint8)},
    }
    scan = {
        "sampling_frequency": np.float32(40e6),
        "center_frequency": np.float32(center_frequency),
        "demodulation_frequency": np.float32(center_frequency),
        "initial_times": np.zeros((n_tx,), dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros((n_tx,), dtype=np.float32),
    }
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)

    subject = {"sex": "f"}
    if fat_percentage is not None:
        subject["fat_percentage"] = np.float32(fat_percentage)

    File.create(
        path,
        data=data,
        scan=scan,
        probe={"name": "demo", "probe_geometry": probe_geometry},
        metadata={"subject": subject},
        description="filtering demo",
        overwrite=True,
    )


with tempfile.TemporaryDirectory() as d:
    write_file(Path(d) / "with_fat.hdf5", fat_percentage=17.5, center_frequency=5e6)
    write_file(Path(d) / "no_fat.hdf5", fat_percentage=None, center_frequency=9e6)

    names = lambda ds: sorted(Path(p).name for p in ds.file_paths)

    # no filter: both files
    print("all files          :", names(Dataset(d, validate=False)))

    # dict filter: keep only files that record a subject fat percentage
    ds = Dataset(d, validate=False, file_filter={"metadata.subject.fat_percentage": EXISTS})
    print("has fat_percentage :", names(ds))

    # callable filter: keep only files in a center-frequency band
    ds = Dataset(d, validate=False, file_filter=lambda f: 4e6 <= float(f.scan.center_frequency) <= 6e6)
    print("center freq 4-6MHz :", names(ds))
```

Expected output:

```
all files          : ['no_fat.hdf5', 'with_fat.hdf5']
has fat_percentage : ['with_fat.hdf5']
center freq 4-6MHz : ['with_fat.hdf5']
```

(`Dataloader(d, key="data/image/values", file_filter=...)` filters identically — files are
dropped before any frames are indexed.)

## Changes

- **`zea/data/datasets.py`** — `EXISTS` helper, `_resolve_dotted_path`, `_compile_dict_filter`,
  `compile_file_filter`, and `Dataset.file_filter` (applied via `_apply_file_filter` after
  discovery; lazy guard; clear "removed all files" error).
- **`zea/data/dataloader.py`** — `file_filter` plumbed through `H5DataSource` → `Dataset` and
  exposed on `Dataloader`, with documented examples.
- **`zea/__init__.py`, `zea/data/__init__.py`** — export `EXISTS` (`from zea import EXISTS`).
- **`tests/data/test_dataloader.py`** — callable + parametrized dict filters, no-metadata
  exclusion, removed-all `ValueError`, lazy incompatibility, end-to-end `Dataloader`, and
  `compile_file_filter` unit tests.

## Notes / trade-offs

- Filtering opens each candidate file to read its (small) metadata, so there's an up-front
  pass over all files. For `hf://` datasets that would mean downloading everything, which is
  why `file_filter` + `lazy=True` raises rather than silently downloading.
- `EXISTS` is implemented as a plain callable (`value is not None`) rather than an identity
  sentinel, so it's evaluated rather than identity-compared — robust across module reloads
  and free of special-casing.

