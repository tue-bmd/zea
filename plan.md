# Plan: VirtualiZarr cloud-optimized read layer for zea (full build)

## Context

zea stores ultrasound datasets as many per-acquisition HDF5 files on HuggingFace
Hub. We added HTTP-range streaming (`zea.File("hf://...", stream=True)`) and
`hdf5plugin` compression support, but cold-open latency stays high — h5py walks the
HDF5 metadata (superblock → chunk B-trees) over HTTP before any read — and
cross-file `Dataset` access still **full-downloads every file**
(`datasets.py:769-779` → `_hf_resolve_path`). No concurrent or cross-file logical
array exists.

Goal: best cloud read performance for these files — **low-latency partial reads**,
**cross-file virtual datasets**, and **heavy parallel access** — via **VirtualiZarr**:
precompute a chunk manifest (offset/length per chunk) so reads go through Zarr +
obstore straight to byte ranges (no HDF5 metadata traversal), with concurrent
fetches and many files combined into one logical array.

## Phase 0 spike — VALIDATED (`scripts/spike_virtualizarr.py`)

Measured facts that drive this plan:

- **Feasibility ✅** VirtualiZarr's native `HDFParser(group="tracks/track_0/data")`
  virtualizes a **Blosc(zstd,shuffle)** zea file (and gzip) with zea's chunking and
  reads back **bit-identical** via Zarr. Parsing only the data group cleanly
  sidesteps vlen-string/scalar params.
- **kerchunk not needed ✅** JSON references (and Icechunk) are native to
  virtualizarr. Only the legacy **parquet** serializer imports `kerchunk`, and only
  at generation time. → use **JSON refs**, drop kerchunk.
- **Cross-file cold-open is the decisive win ✅** opening 8 files with h5py = **24
  HTTP requests / 0.64 s**; one combined virtual reference = **0 requests / 0.02 s**.
  Scales linearly with file count.
- **Concurrency works ✅** `obstore.get_ranges(8)` completes in one latency unit
  (fully parallel).
- **Chunk layout is the lever** the fine per-`(frame,tx)` layout yields many tiny
  chunks that zarr's sync read path fetches inconsistently (0.24–1.08 s); a coarser
  **per-frame** layout reads 3 frames in **3 requests / 0.09 s**, beating h5py.

## Locked decisions

1. **Codec**: default `lzf` → **Blosc(cname="zstd", shuffle=SHUFFLE)** (single HDF5
   filter 32001 → `numcodecs.Blosc`; Zarr-decodable, best ratio/speed). Make
   **`hdf5plugin` a core dependency**.
2. **Chunking default**: `DEFAULT_CHUNK_AXES` → **`("n_frames",)`** = one full frame
   per chunk (chunk `(1, n_tx, n_ax, n_el, n_ch)`). Best for the virtual/cloud path
   and fine for h5py+blockcache. **Caveat + planned refinement below.**
3. **References**: **kerchunk JSON** (native, no kerchunk package), published in the
   HF repo (`hf://<repo>/virtual/index.json`). Icechunk is a future option; parquet
   is avoided (would reintroduce kerchunk).
4. **Read API**: extend `Dataset` with **`lazy='virtual'`** reading the combined
   reference via Zarr + obstore.

### Chunking caveat (must document; drives a fast-follow)

Per-frame chunks get **large for high-`n_tx`** data (e.g. carotid `n_tx=149`,
`n_ax=2176`, `n_el=128` → ~148 MB/chunk uncompressed). That is bad for the HDF5
chunk cache and forces reading a whole frame to touch any transmit. **Fast-follow
(Phase 1b):** add a **max-transmits-per-chunk** option — chunk `(1, min(n_tx, CAP),
…)` — which needs an explicit chunk-size mechanism beyond the current
`chunk_axes` (which is 1-or-full only). Ship per-frame now; add the cap once
validated on real high-`n_tx` datasets.

### Not virtualized (side path, Phase 4)

vlen-string fields (probe `name`/`type`, `labels`, subject/annotations), scalar
floats, `complex64` maps, root HDF5 attributes. Only numeric bulk arrays
(`raw_data`, `image`, …) are virtualized.

## Phase 1 — Codec + chunking migration -- DONE

- `zea/data/spec.py:41`: `DEFAULT_COMPRESSION` `"lzf"` → the Blosc filter mapping
  (from `hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)`).
  `create_dataset` already accepts a `Mapping` (`spec.py:516-521`).
- `zea/data/spec.py:47`: `DEFAULT_CHUNK_AXES` `("n_frames", "n_tx")` → `("n_frames",)`.
- `pyproject.toml:36-55`: add `hdf5plugin` to core `dependencies`; add a central
  `import hdf5plugin` (top of `zea/data/file.py`) so the filter is registered for
  every read/write. Old `lzf` files stay readable (h5py native).
- Update `tests/data/test_data_format.py::test_chunk_axes` (default is now per-frame)
  and any compression/golden-size assertions.
- **Phase 1b (fast-follow):** max-tx-per-chunk option in `_resolve_chunks`.

## Phase 2 — Reference-generation tooling (`zea/data/virtual.py`) -- DONE

- `build_virtual_reference(paths, output_path, revision=...)` (+ `zea data virtualize`,
  via `file_operations.virtualize` and `_Virtualize` in `cli_args.py`).
- Files are enumerated with `Dataset(..., lazy=True)` and **sorted** (folder walks / HF
  listings are unordered) so the file index of a reference is reproducible.
- Per file the data group **and its subgroups** are parsed (`HDFParser` does not recurse),
  so `image/values`, `image/coordinates`, … are virtualized alongside `raw_data`.
- Chunks are parsed **at their final URL** (HF *resolve* URL for `hf://`, over HTTP range
  requests: metadata only, no download) → no `rename_paths` needed. Group structure is
  discovered by opening the file with `zea.File` (streams via `HfFileSystem`).
- Shape-homogeneous files are `xarray.concat`ed along a new `"file"` dim; files whose
  shapes differ (e.g. varying `n_frames`) become separate **shape groups** in the same
  reference. Each shape group / HDF5 subgroup is a nested Zarr group in one JSON
  (kerchunk refs are a flat key→[url, offset, len] map, so nesting = key prefixing);
  the root `.zattrs` carries a `zea_virtual` manifest (files + array→group map).

### Chunk filter masks — the one real gotcha (found in Phase 2)

When a filter does **not shrink** a chunk (incompressible data), HDF5 stores that chunk
**raw** and sets its per-chunk *filter mask*. Zarr has no equivalent — it applies the
codec to every chunk — and **VirtualiZarr ignores the mask entirely**, so such an array
would silently decode to garbage (or `RuntimeError: blosc decompression -1`).
Mitigation: generation checks every chunk's filter mask and **excludes** affected arrays
from the reference (warning; they stay readable via `zea.File`). Uniform-noise `uint8`
images hit this in practice. Old **lzf** files are rejected up front with resave guidance
(lzf has no Zarr codec at all).

## Phase 3 — `Dataset` integration (`lazy='virtual'`) -- DONE

- `Dataset(..., lazy="virtual")` + `virtual_index=` override; `Dataset.virtual` lazily
  opens `hf://<repo>/virtual/index.json` (or `<folder>/virtual/index.json` locally),
  erroring with `zea data virtualize` guidance when absent. `HF_TOKEN` is passed as a
  bearer header for private repos.
- **The read API is the file API** (revised after review — see Phase 3b): `dataset[i]`
  returns a `VirtualFile`, so `dataset[i].data.raw_data[0]` works exactly as on a real
  `File`, but reads chunk byte ranges (no download, no HDF5 open). `dataset[i]` is looked
  up **by path**, since the reference orders files by shape group.
- `ds.virtual` remains for the one thing the per-file API cannot express: reading across
  files in one concurrent expression (`ds.virtual["raw_data"][[0, 4], 0:2]`). A selection
  spanning two shape groups raises (their shapes cannot stack).
- `total_frames` uses the reference (no opens/downloads).
- DataLoader (grain) rejects `lazy` and needs local files (`dataloader.py:307-313`);
  feeding grain from the Zarr array is a **follow-up**, out of scope for the first cut.

## Phase 3b — Same API for a single file + paged HDF5 -- DONE

Prompted by review: *"why is this on Dataset and not File? can I not just stream one
file with the same API?"*

- **`VirtualFile` is the only read adapter.** `open_virtual_file(path)` builds a
  one-file manifest on the fly (one metadata pass, no download) and returns the same
  `VirtualFile` a dataset hands out — a lone file is just a reference with one file in
  it, so there is no second code path. Zarr cannot be fed back into h5py (the manifest
  exists precisely to bypass HDF5), so a thin proxy — mirroring `_GroupProxy` — is what
  makes the API identical.
- **Paged HDF5 for the plain (non-virtual) streaming path.** `File.create` now writes
  with `fs_strategy="page"` (64 KiB, `spec.PAGED_LAYOUT`). Measured over HTTP at 20 ms/req:
  cold open **3 → 2 requests, 0.16 s → 0.05 s**, for **~2%** file size. Requires HDF5
  ≥ 1.10.1 on readers.

### Measured: what the h5py route can and cannot do (drove the above)

- **Concurrency is impossible for h5py.** h5py serializes every call on a global lock
  (`h5py._objects.phil`) because the HDF5 C library is not concurrent-safe. 8 files, one
  frame each: **1.21 s serial vs 1.19 s with 8 threads**. `HDF5_USE_FILE_LOCKING=FALSE` is
  a *different* lock (POSIX `flock` on disk) and changes nothing — verified.
  Only multi-process helps (which grain already does).
- **A 0-request open is impossible for h5py**, because the metadata is exactly what it
  must fetch to know where anything is. Paging only cuts 3N → 2N requests for N files.
  Having that metadata locally *is* a manifest.
- **Zarr-native was considered and rejected** (see Risks): a zipped Zarr costs the same as
  HDF5 today (3 req open / 3 req read, no concurrency — fsspec's `ZipFileSystem` is sync),
  so it buys nothing; the un-zipped sharded form with consolidated metadata is ~parity with
  the reference (4 req vs 0) but gives up one-file-per-acquisition and the h5py ecosystem.

- Tests: `tests/data/test_virtual.py` (16, `importorskip`).

## Phase 4 — Parameters / metadata side path -- DONE

- `build_virtual_reference` also writes `params.json` next to the reference (published as
  `hf://<repo>/virtual/params.json`), from `to_scan_dict()`/`to_probe_dict()` + `n_ax`/
  `n_el`/`n_tx` — the same merge as `File.load_parameters`. Values are encoded with their
  dtype, so `VirtualReference.parameters(i)` reconstructs an object that compares **equal**
  to `File.load_parameters()`, without opening the HDF5 file.
- Identical parameter sets are **deduplicated** (one entry, files map to it): scan params
  carry `t0_delays` `(n_tx, n_el)` etc., which would otherwise dwarf the reference itself
  (a 149×128 array is ~150 kB of JSON *per file*).
- Params are read in the same file open as group discovery (`_inspect_file`), so
  generation still costs one open per file.

## Phase 5 — Publishing & existing-dataset migration -- DONE (`zea/data/publish.py`)

- `publish_dataset(input, repo_id, ...)` + `zea data publish <input> <repo_id>`:
  resave (Blosc + per-frame) → `create_repo` + `upload_folder` → virtualize the
  **uploaded** files pinned to that commit (`revision=data_commit.oid`) → upload
  `virtual/index.json` + `params.json`. Building the reference against the Hub rather
  than the local copies also **verifies** the published files are readable at their final
  URLs. Input may be an `hf://` path — that is the migration case for the existing lzf
  datasets. Never runs on its own; needs HF write auth.
- Returns `{repo_id, data_commit, virtual_commit, n_files}`.
- Fixed a Phase-1 leftover: the **`zea data resave` CLI still defaulted to
  `("n_frames", "n_tx")`** chunks (the function default had moved to `("n_frames",)`),
  which would have silently republished the old fine chunking.
- Tests: `tests/data/test_publish.py` (4, Hub calls stubbed — nothing is uploaded).

## Phase 6 — Benchmark, tests, docs

- Fold the spike's harness into a kept `scripts/benchmark_streaming.py`: h5py-streaming
  vs virtual on cold-open (single + cross-file), concurrent partial read, request count.
- Tests: mostly landed with Phases 2-5 (`tests/data/test_virtual.py`,
  `tests/data/test_publish.py`). Still missing: the cross-file **cold-open assertion**
  (virtual issues ~0 open requests) against a counting HTTP server, as in the spike.
- Virtual-access example in the pipeline notebook/docs.

## Dependencies (`pyproject.toml`) -- DONE

- **Core**: `hdf5plugin` (Phase 1).
- **Extra `zea[virtual]`**: `virtualizarr>=2.7`, `zarr>=3.2`, `numcodecs>=0.16`,
  `obstore>=0.11`, `xarray>=2025.1`. No `kerchunk` (parquet refs only), and no
  `fsspec`/`aiohttp` — the virtual path uses obstore, and `huggingface_hub` (core)
  already brings fsspec for the h5py streaming path.

## Verification

- **Round-trip**: `File.create(compression=Blosc, chunk_axes=("n_frames",))` → read
  via h5py and via Zarr → `np.array_equal`.
- **Cross-file**: `zea virtualize` a small local multi-file set → `Dataset(lazy='virtual')`
  reads match per-file h5py reads; `parameters(i)` matches `File.load_parameters()`;
  cold-open issues ~0 HTTP requests (assert via the spike's counting server).
- **Regression**: `tests/data/` (`test_data_format.py`, `test_spec.py`, `test_file.py`,
  `test_dataset.py`) after the codec+chunk change; `ruff` + `ty` clean.

## Risks

- **Large per-frame chunks for high-`n_tx`** → Phase 1b max-tx-per-chunk cap.
- **zarr sync fine-chunk reads underuse obstore concurrency** → mitigated by the
  coarse per-frame chunk default; revisit async/`get_ranges` batching if needed.
- **`hdf5plugin` core dep + new default codec**: old `lzf` files still readable, but
  new files need `hdf5plugin` on any reader — document.
- **Cross-file combine needs homogeneous shapes** → heterogeneous datasets grouped,
  not forced into one ragged array.
- **Reference staleness** → regenerate whenever underlying files change; pin
  `--revision <commit>` so a reference cannot silently drift from its data.
- **Incompressible chunks are stored raw by HDF5** (filter mask), which Zarr cannot
  express → detected at generation, affected arrays excluded (see Phase 2).
- **version churn** (virtualizarr/zarr/obstore) → pin to the spike's working set.
