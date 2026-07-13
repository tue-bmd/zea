# Plan: concurrent direct-chunk reads in `zea.File` (no VirtualiZarr)

**Supersedes** `plan.md` phases 2–6 (virtual references, `zea data virtualize`, params
sidecar, publishing) — see
*Relationship to the earlier plans* at the end. `plan.md` phase 1 (Blosc + per-frame
chunks) and the paged layout stay: they are load-bearing here too.

## Idea

`h5py` already exposes the chunk manifest: `dsid.get_chunk_info(i)` gives
`(byte_offset, size, filter_mask)` per chunk, and `read_direct_chunk()` returns a chunk's
**raw, still-compressed bytes**. So we can:

1. get the chunk offsets from h5py (metadata only),
2. fetch the chunk bytes ourselves — **concurrently** (`fs.cat_ranges`) when remote,
3. **decode them in parallel** in a `ThreadPoolExecutor` (Blosc/zlib release the GIL),
4. assemble and slice.

This bypasses the two things that make h5py slow — its global lock serializing decodes,
and its one-range-at-a-time I/O — while keeping h5py as the reader for everything else.
Extracting those offsets is *precisely* what VirtualiZarr does before handing them to
Zarr; doing it in-process removes the entire manifest/Zarr apparatus.

## Measured (this is why we pivot)

Same file (64 chunks × 6.3 MB, Blosc-zstd), bit-identical results in all cases.

**Local** (best of 3):

| chunks read | h5py | **direct + parallel decode** | zarr/virtualizarr |
|---|---|---|---|
| 1 | **5 ms** | 9 ms | 9 ms |
| 4 | 31 ms | **16 ms** | 24 ms |
| 16 | 129 ms | **91 ms** | 88 ms |
| 64 (403 MB) | 519 ms | **381 ms** | 301 ms |

**Cloud** (20 ms/request):

| chunks read | h5py stream | **direct + concurrent ranges** | zarr/virtualizarr |
|---|---|---|---|
| 1 | 0.17 s | **0.09 s** | 0.11 s |
| 4 | 0.17 s | **0.10 s** | 0.11 s |
| 16 | 0.57 s | **0.15 s** | 0.21 s |

Parity with the Zarr path, at a fraction of the machinery. (Supporting facts: h5py cannot
be made concurrent — its global lock serializes threads even across handles, and
`HDF5_USE_FILE_LOCKING=FALSE` is an unrelated lock; `BLOSC_NTHREADS` recovers only ~20%.)

## What this deletes

- **Dependencies**: `virtualizarr`, `zarr`, `obstore`, `xarray` (the `zea[virtual]` extra).
  Adds only `numcodecs` (small; already an indirect dep of the ecosystem) — fsspec is
  already present via `huggingface_hub`.
- **The manifest as an artifact**: no `virtual/index.json`, no `params.json`, no
  `zea data virtualize`, no publish step, no staleness, no revision pinning.
- **The backend refactor**: `zea.File` stays an `h5py.File` subclass. Scan/probe/metadata/
  custom/vlen strings keep working untouched, because h5py is still the reader.
- **Two gotchas found in the virtual path**:
  - *filter masks*: HDF5 stores an incompressible chunk **raw** and records it in the
    chunk's `filter_mask`. Zarr cannot express that (it silently decoded garbage; we had
    to detect and exclude such arrays). Here `filter_mask` is handed to us per chunk — we
    just skip that filter. **The bug cannot occur.**
  - *lzf*: has no Zarr codec, so old files were unvirtualizable. Here they simply fall back
    to the normal h5py read.

## What we give up

The **0-request cold open**. Opening N files still costs ~3 HTTP requests each (h5py's
metadata walk); a published manifest cost 0 — the one thing VirtualiZarr uniquely bought
(8 files: 24 → 0 requests). Mitigations, in order:

- paged HDF5 (already implemented) cuts it to ~2 requests/file;
- for large files (7 GB) the open is amortised into insignificance by the reads;
- if it ever matters, we can serialise **our own** offsets sidecar (Phase 4) — the manifest
  is just `(offset, size, filter_mask)` per chunk, which we now hold anyway. No Zarr needed.

## Design

### `zea/data/chunk_reader.py` (new)

```python
def read(dset: h5py.Dataset, selection) -> np.ndarray   # the whole public surface
```

1. **Eligibility.** Fast path requires: `dset.chunks is not None`, every filter in the
   pipeline has a decoder (registry below), and the selection is expressible as a set of
   whole chunks (any selection is — we read the intersecting chunks and slice after decode,
   exactly as HDF5 does internally). Otherwise → `dset[selection]` (plain h5py).
2. **Chunk enumeration.** Map the selection to the chunk grid; for zea's layout (one chunk
   per frame, full extent on the other axes) this is just the frame indices.
3. **Byte fetch.**
   - *local*: `dsid.read_direct_chunk(coord)` per chunk, serially (page cache; no decode).
   - *remote*: `get_chunk_info` for offsets, then one concurrent `fs.cat_ranges(...)`.
4. **Parallel decode.** `ThreadPoolExecutor`; per chunk, apply the filter pipeline in
   reverse, skipping filters whose bit is set in that chunk's `filter_mask`.
5. **Assemble + slice.**

**Codec registry** (filter id → decoder), everything else falls back:

| HDF5 filter | id | decoder |
|---|---|---|
| Blosc (zea default) | 32001 | `numcodecs.Blosc` (params read from the blosc header) |
| gzip/deflate | 1 | `zlib` |
| shuffle | 2 | `numcodecs.Shuffle` |
| lzf, everything else | — | *fallback to h5py* |

### Where it hooks in

`_GroupProxy` currently returns a raw `h5py.Dataset`. It will return a thin
`ChunkedDataset` wrapper that delegates everything (`shape`, `dtype`, `chunks`, `attrs`, …)
to the h5py dataset and overrides `__getitem__` to call `chunk_reader.read`. So
`f.data.raw_data[0:8]` gets the fast path with **no API change anywhere**, and
`f["data/raw_data"]` keeps returning the h5py dataset for anyone who wants the raw object.

### When the fast path is used

Fast path is a pure optimisation, so it must never be slower:

- **remote (`hf://`)**: always (0.09 s vs 0.17 s even for one chunk).
- **local**: only when ≥ N chunks are touched (measured crossover: ~4). Below that, h5py
  wins (5 ms vs 9 ms for one chunk) — the thread pool and buffer copies are not free.
- Threshold and worker count settle in Phase 5 on real files/hardware; expose as
  `zea.config` knobs, not call-site arguments.

## Phases

**1. Reader core.** `chunk_reader.py`: codec registry, chunk math, filter-mask handling,
local direct-chunk path, parallel decode, h5py fallback. Tests compare against h5py for a
matrix of selections × dtypes × codecs (Blosc, gzip, lzf→fallback, uncompressed,
incompressible/filter-masked chunks) — equality is the contract.

**2. Wire into `File`.** `ChunkedDataset` in `_GroupProxy`; local threshold policy. The
existing suite must pass **unchanged** — no API moves.

**3. Remote path.** Chunk offsets from the streamed h5py handle, then concurrent
`cat_ranges`. **Open question to settle first:** `HfFileSystem` is a *sync* fsspec
filesystem, so `cat_ranges` on it may serialize. If so, issue the ranges against the HF
*resolve* URL through the async `HTTPFileSystem` (with `Authorization: Bearer $HF_TOKEN`
for private repos) — the URL construction already exists from the virtual work. Verify
before building on it.

**4. (Optional) Offsets sidecar.** Only if cross-file cold open proves to matter: dump
`(offset, size, filter_mask)` per chunk to a compact sidecar so a remote open costs 0–1
requests instead of 2–3 per file. Our format, our reader, no Zarr.

**5. Benchmarks + defaults.** Extend the harness to sweep (chunks read) × (local | cloud) ×
(path) on **real 7 GB zea files on the target hardware**, and set the local threshold and
worker count from that. Also settles whether the chunk-size cap (`plan.md` phase 1b) is a
bigger lever than any of this: with per-frame chunks a 7 GB file has ~70 MB chunks, and
**one frame = one chunk decodes serially in every approach** — parallelism only helps when
a read touches many chunks.

**6. Remove the virtual path.** Delete `zea/data/virtual.py`, `zea/data/publish.py`, the
`zea data virtualize`/`publish` commands, the `zea[virtual]` extra, and their tests. (They
are committed at `e5b2c432`; the working tree already has them removed.) Keep `zea data
resave` and the Blosc/per-frame/paged defaults.

## Risks

- **Correctness is now ours.** Chunk math, partial chunks, edge chunks (a chunk may hang
  off the end of the array and must be cropped), endianness, scalar/string dtypes. Mitigated
  by: a narrow eligibility test, h5py fallback for anything unusual, and equality tests
  against h5py as the oracle. This is the main cost of the pivot — we trade Zarr's mature
  reader for ~300 lines we own.
- **Memory.** Fetching whole chunks is fine at 6 MB, dangerous at 70 MB: 16 concurrent
  chunks = 1.1 GB in flight. Cap concurrency **by bytes**, not by chunk count.
- **Filter pipelines beyond one filter.** HDF5 applies filters in order; we must reverse
  them and honour `filter_mask` bit-per-filter. Only Blosc (single filter) is the zea
  default, but gzip+shuffle exists in the wild → test it, fall back if unsure.
- **`cat_ranges` concurrency on HF** (see Phase 3) — the whole remote win depends on it.
- **Thread-pool overhead** could regress small local reads → threshold, benchmarked.

## Verification

- **Equality vs h5py** across the selection/codec matrix — the fast path must be a pure
  optimisation, never a semantic change.
- **Concurrency is real**: with a counting/latency-injecting HTTP server, N chunks complete
  in ~1 round trip, not N (as measured: 16 chunks in 1.2 round trips).
- **Fallbacks**: lzf, unknown filters, contiguous datasets, and filter-masked chunks all
  produce correct data (via fallback or mask handling).
- **No regression**: existing `tests/data/` suite passes untouched; local single-frame reads
  are no slower than today.

## Relationship to the earlier plans

- `plan.md` **phase 1** (Blosc default, per-frame chunks) and the **paged layout**: keep —
  Blosc is what makes chunks decodable in-process, and paging is what makes the open cheap.
- `plan.md` **phases 2–6** (virtual references, CLI, params sidecar, publishing): drop.
