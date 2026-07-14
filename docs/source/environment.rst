Environment variables
================================

Here are the environment variables that ``zea`` uses at runtime. Arguably the most important environment variable is the Keras backend selection via ``KERAS_BACKEND``. See the :ref:`backend-installation` section for details on configuring the backend.

.. list-table::
   :header-rows: 1
   :widths: 20 80 20 20

   * - **Variable**
     - **Description**
     - **Default**
     - **Options**
   * - ``KERAS_BACKEND``
     - Select the Keras backend to use. This defines the ML framework that will be used for all tensor operations.
     - ``jax``
     - ``tensorflow``, ``torch``, ``jax``, ``numpy``
   * - ``ZEA_CACHE_DIR``
     - Directory to use for caching downloaded files, e.g. model weights or datasets from Hugging Face Hub.
     - ``~/.cache/zea``
     - ``str``
   * - ``ZEA_LOG_LEVEL``
     - Logging level for ``zea``.
     - ``DEBUG``
     - ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``
   * - ``ZEA_DISABLE_CACHE``
     - If set to ``1`` will write to a temporary cache directory that is deleted after the program exits.
     - ``0``
     - ``0``, ``1``
   * - ``ZEA_NVIDIA_SMI_TIMEOUT``
     - Timeout in seconds for calling ``nvidia-smi`` to get GPU information during :func:`zea.init_device`.
     - ``30``
     - Any positive integer, or ``<= 0`` to disable timeout.
   * - ``ZEA_DOWNLOAD_TIMEOUT``
     - Timeout in seconds for downloading files, e.g. during dataset conversion.
     - ``60``
     - Any positive integer, or ``<= 0`` to disable timeout.
   * - ``ZEA_FIND_H5_SHAPES_PARALLEL``
     - If set to ``1``, will use parallel processing when searching for HDF5 file shapes.
     - ``1``
     - ``0``, ``1``
   * - ``ZEA_TEST_DEVICE``
     - Can be used to only run the tests with a particular device.
     - ``auto:1``
     - Any valid device name as accepted by :func:`zea.init_device`. For example, ``cpu``,
       ``cuda:0``, ``auto:1``, etc.
   * - ``ZEA_CHUNK_CACHE``
     - Cache chunks fetched while streaming (``hf://``) on disk, under ``ZEA_CACHE_DIR``, so a
       repeated read is served locally instead of re-downloaded. Set to ``0`` to disable
       (equivalently, ``zea.File(..., cache=False)``). See :ref:`chunk-cache` below.
     - ``1``
     - ``0``, ``1``
   * - ``ZEA_CHUNK_CACHE_SIZE``
     - Byte budget for that cache. Once exceeded, the least-recently-*used* chunks are deleted
       until it fits.
     - ``10737418240`` (10 GiB)
     - Any positive integer (bytes).
   * - ``BLOSC_NTHREADS``
     - Threads Blosc uses to compress or decompress the blocks *within* a single HDF5 chunk.
       ``zea`` sets this to ``min(8, cpu_count)`` if you have not set it yourself, which is
       worth roughly **4x on writes** (measured 105 → 453 MB/s on real int16 channel data);
       see :ref:`blosc-nthreads` below.
     - ``min(8, cpu_count)``
     - Any positive integer.

.. _chunk-cache:

Streaming chunk cache (``ZEA_CHUNK_CACHE``)
-------------------------------------------

Streaming fetches only the chunks a read touches — but without a cache it re-fetches them
every time, so reading the same frames twice costs the network twice. ``zea`` therefore caches
the fetched (still compressed) chunks under ``ZEA_CACHE_DIR/chunks``. Measured on a 5-frame
read of a streamed 618 MB file: **4.4 s cold, 0.39 s once cached** — and the cache is on disk,
so it survives across processes, notebook restarts and training epochs.

This is *partial*-file caching, and that is the point: ``stream=False`` already caches whole
files via the HF hub, but downloading 6 GB to read 5 frames is what streaming exists to avoid.

Chunks are keyed by the file's **content hash**, not its URL — an ``hf://`` path resolves to a
mutable ref, so re-uploading a file changes the bytes behind the same URL. Keying on content
means a re-upload simply misses; it can never serve you the old file's data.

Disable per-file with ``zea.File(..., cache=False)`` or globally with ``ZEA_CHUNK_CACHE=0``.
``zea.data.chunk_cache.clear()`` empties it.

.. _blosc-nthreads:

Blosc threading (``BLOSC_NTHREADS``)
------------------------------------

HDF5 runs the Blosc filter **one chunk at a time, single-threaded**, but Blosc can process the
blocks *within* a chunk in parallel, and reads ``BLOSC_NTHREADS`` from the environment on every
call. So this variable is most of what determines write throughput. ``zea`` defaults it to
``min(8, cpu_count)`` via ``setdefault``, so an explicit value in your environment always wins.

Turn it **down** (to ``1`` or ``2``) when your writes are already parallel — several dataloader
workers each saving a file will multiply with this. Do not turn it far **up**: the gain reverses
once the blocks go memory-bound (32 threads measured slower than 8). It never changes what is
stored, only how fast it is produced.

.. note::

   ``numcodecs``, which decodes chunks on the read path (:mod:`zea.data.chunk_reader`), keeps its
   own Blosc thread setting and does not read this variable — so this affects writes, not reads.