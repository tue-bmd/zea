# Plan: concurrent direct-chunk reads in `zea.File` (no VirtualiZarr)

**Supersedes** `plan.md` phases 2–6 (virtual references, `zea data virtualize`, params
sidecar, publishing) — see
*Relationship to the earlier plans* at the end. `plan.md` phase 1 (Blosc + per-frame
chunks) and the paged layout stay: they are load-bearing here too.

## Idea

`h5py` already exposes the chunk manifest: `dsid.get_chunk_info(i)` gives
`(byte_offset, size, filter_mask)` per chunk. So we can:

1. get the chunk offsets from h5py (metadata only),
2. fetch the chunk bytes ourselves — **concurrently** when remote, straight from the file
   descriptor when local,
3. **decode them in parallel** in a `ThreadPoolExecutor` (Blosc/zlib release the GIL),
   writing each chunk *directly into the output array*,
4. slice.

This bypasses the two things that make h5py slow — its global lock serializing decodes,
and its one-range-at-a-time I/O — while keeping h5py as the reader for everything else.
Extracting those offsets is *precisely* what VirtualiZarr does before handing them to
Zarr; doing it in-process removes the entire manifest/Zarr apparatus.

## Measured (`scripts/benchmark_read_paths.py`)

Bit-identical results in every cell — the benchmark asserts equality against h5py.

**Local**, 16 frames × 12.6 MB chunks = 201 MB, warm page cache:

| chunks read | h5py | **direct** | zarr/virtualizarr |
|---|---|---|---|
| 1 | 10 ms | **7 ms** | 26 ms |
| 4 | 118 ms | **33 ms** | 74 ms |
| 8 | 199 ms | **42 ms** | 154 ms |
| 16 (201 MB) | 291 ms | **31 ms** | 256 ms |

**Cloud**, same file over HTTP at 20 ms/request (a local counting server: real HF is
bandwidth-bound and noisy by 3–4×, which drowns the round trips that actually differ):

| chunks read | h5py stream | **direct** | virtual (published ref) |
|---|---|---|---|
| 1 | 2 req open, 33 ms read | 2 req, **36 ms** | **0 req**, 55 ms |
| 4 | 4 req, 263 ms | 4 req, **110 ms** | 4 req, 62 ms |
| 16 | 18 req, 863 ms | 16 req, **126 ms** | 16 req, 170 ms |

h5py's reads are *serial*, so its read time scales with the chunk count. The direct path
issues the same ranges concurrently and reaches **read parity with Zarr** (126 vs 170 ms).
Virtual's entire remaining advantage is the 0-request open (~40 ms/file at 20 ms/req).
That is the trade this plan makes: parity on reads, minus the whole manifest apparatus.

### Two implementation details carry the local win

Both were wrong in the first draft of this plan, which is why it measured only 1.4× locally:

- **Do not fetch through `read_direct_chunk`.** It goes through h5py's global lock (so the
  fetch cannot parallelize) and copies into a fresh `bytes` per chunk. `os.pread` at the
  byte offset skips h5py entirely.
- **Do not "assemble" afterwards.** Decoding to a temporary and copying it into the output
  costs *more than the decode itself* (measured: 121 ms of copy vs 26 ms of decode for 16
  chunks), and the copy is serial, so it caps everything. `blosc.decompress(raw, dest=view)`
  writes each chunk straight into its slice of the output — no temporary, no copy, and the
  page faults spread across the worker threads.

Supporting fact: h5py cannot be made concurrent — its global lock serializes threads even
across handles, and `HDF5_USE_FILE_LOCKING=FALSE` is an unrelated lock.

## What this deletes

- **Dependencies**: `virtualizarr`, `zarr`, `obstore`, `xarray` (the `zea[virtual]` extra).
  Adds only `numcodecs` (small; already an indirect dep of the ecosystem) — fsspec is
  already present via `huggingface_hub`.
- **The manifest as an artifact**: no `virtual/index.json`, no `params.json`, no
  `zea data virtualize`, no `zea data publish`, no publish step, no staleness, no revision
  pinning.
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

The **0-request cold open**. Opening N files still costs ~2 HTTP requests each (h5py's
metadata walk, with the paged layout); a published manifest cost 0. Mitigations, in order:

- paged HDF5 (already implemented) already cut it from 3 to 2 requests/file;
- for large files the open is amortised into insignificance by the reads (2 requests ≈
  40 ms, against seconds of transfer);
- if it ever matters, we can serialise **our own** offsets sidecar (Phase 4) — the manifest
  is just `(offset, size, filter_mask)` per chunk, which we now hold anyway. No Zarr needed.

## Design

### `zea/data/chunk_reader.py` (new)

```python
def read(dset: h5py.Dataset, selection, fetcher) -> np.ndarray   # the whole public surface
```

1. **Eligibility.** Fast path requires: `dset.chunks is not None`, every filter in the
   pipeline has a decoder (registry below), a fetcher for this file's storage, and a
   selection of ints / unit-step slices / sorted-unique index lists. Anything else →
   `dset[selection]` (plain h5py). The read must also be big enough to be worth the thread
   hand-off (`MIN_BYTES`).
2. **Chunk enumeration.** Map the selection to the chunk grid; each touched chunk knows
   which region of the *output* it fills.
3. **Byte fetch** (`Fetcher`):
   - *local*: `os.pread(fd, size, offset)` per chunk, inside the worker threads.
   - *remote*: one concurrent `cat_ranges` per batch, against the HF **resolve** URL through
     the async `HTTPFileSystem` (see Phase 3 — `HfFileSystem` is serial).
4. **Parallel decode.** `ThreadPoolExecutor`; per chunk, apply the filter pipeline in
   reverse, skipping filters whose bit is set in that chunk's `filter_mask`. When the chunk
   maps onto a whole contiguous block of the output (zea's layouts always do), decompress
   **into that view**; otherwise decode to a temporary and copy the sub-selection out.
5. **Slice.**

**Codec registry** (filter id → decoder), everything else falls back:

| HDF5 filter | id | decoder |
|---|---|---|
| Blosc (zea default) | 32001 | `numcodecs.blosc` (params read from the blosc header) |
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

The fast path is a pure optimisation, so it must never be slower. Measured: it wins at
*every* chunk count, local and remote — including a single chunk (7 ms vs 10 ms locally),
because `pread` + a GIL-releasing decode beats h5py's locked read even with no parallelism.
So there is **no chunk-count threshold**. The only guards are `MIN_BYTES` (small reads,
where the thread hand-off is not worth it) and `MAX_BYTES_IN_FLIGHT`, which bounds how many
chunks are fetched at once — 166 MB chunks × 16 workers would otherwise be 2.6 GB in
flight. Both are module constants, not call-site arguments.

## Phases

**1. Reader core.** `chunk_reader.py`: codec registry, chunk math, filter-mask handling,
fetchers, parallel decode, h5py fallback. Tests compare against h5py for a matrix of
selections × dtypes × codecs (Blosc, gzip, lzf→fallback, uncompressed,
incompressible/filter-masked chunks) — equality is the contract.

**2. Wire into `File`.** `ChunkedDataset` in `_GroupProxy`. The existing suite must pass
**unchanged** — no API moves.

**3. Remote path.** — **question settled, and it matters.** `HfFileSystem.cat_ranges` is
*serial*: 16 ranges took **2745 ms**, against 149 ms for a single range. The async
`HTTPFileSystem` does the same 16 ranges in **177 ms** — one round trip. So the ranges must
be issued against the HF *resolve* URL through `HTTPFileSystem` (with
`Authorization: Bearer $HF_TOKEN` for private repos). The whole remote win depends on this.

**4. (Optional) Offsets sidecar.** Only if cross-file cold open proves to matter: dump
`(offset, size, filter_mask)` per chunk to a compact sidecar so a remote open costs 0–1
requests instead of 2 per file. Our format, our reader, no Zarr. **Not built.**

**5. Benchmarks + defaults.** — **done** (`scripts/benchmark_read_paths.py`; see Measured).
It also settles the question it was meant to: **the chunk-size cap (`plan.md` phase 1b) is a
bigger lever than any of this for high-`n_tx` data.** A real carotid file (`n_tx=149`) has
**166 MB chunks**, so one frame = one chunk, and a single-frame read has *nothing* to
parallelize in any approach. Splitting one chunk's byte range across sub-ranges does not
rescue it either (4/8/16/32 sub-ranges all land at ~157 MB/s — the link is saturated, not
latency-bound). The cap is independent of this plan and should be picked up next.

**6. Remove the virtual path.** Delete `zea/data/virtual.py`, `zea/data/publish.py`, the
`zea data virtualize` / `zea data publish` commands, `Dataset(lazy="virtual")` /
`Dataset.virtual`, the `zea[virtual]` extra, and their tests. (They stay in history at
`e5b2c432`.) Keep `zea data resave` and the Blosc/per-frame/paged defaults.

## Risks

- **Correctness is now ours.** Chunk math, partial chunks, edge chunks (a chunk may hang
  off the end of the array and must be cropped), endianness, scalar/string dtypes. Mitigated
  by: a narrow eligibility test, h5py fallback for anything unusual, and equality tests
  against h5py as the oracle. This is the main cost of the pivot — we trade Zarr's mature
  reader for ~300 lines we own.
- **Memory.** Fetching whole chunks is fine at 12 MB, dangerous at 166 MB. Concurrency is
  capped **by bytes** (`MAX_BYTES_IN_FLIGHT`), not by chunk count.
- **Filter pipelines beyond one filter.** HDF5 applies filters in order; we must reverse
  them and honour `filter_mask` bit-per-filter. Only Blosc (single filter) is the zea
  default, but gzip+shuffle exists in the wild → test it, fall back if unsure.
- **Thread-pool overhead** on small reads → `MIN_BYTES` guard.

## Verification

- **Equality vs h5py** across the selection/codec matrix — the fast path must be a pure
  optimisation, never a semantic change.
- **Concurrency is real**: with a counting/latency-injecting HTTP server, N chunks complete
  in ~1 round trip, not N (measured: 16 chunks in 126 ms at 20 ms/request).
- **Fallbacks**: lzf, unknown filters, contiguous datasets, and filter-masked chunks all
  produce correct data (via fallback or mask handling).
- **No regression**: existing `tests/data/` suite passes untouched.

## Relationship to the earlier plans

- `plan.md` **phase 1** (Blosc default, per-frame chunks) and the **paged layout**: keep —
  Blosc is what makes chunks decodable in-process, and paging is what makes the open cheap.
- `plan.md` **phases 2–6** (virtual references, CLI, params sidecar, publishing): drop.
- `plan.md` **phase 1b** (max-transmits-per-chunk cap): still open, and now the *biggest*
  remaining lever — see Phase 5.
