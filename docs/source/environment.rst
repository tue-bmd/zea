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
     - Cache chunks fetched while streaming (``hf://``) under ``ZEA_CACHE_DIR/chunks``, so a
       repeated read is served from disk instead of re-downloaded. Keyed by content hash, so a
       re-uploaded file misses rather than serving stale bytes. Per-file: ``File(cache=False)``.
     - ``1``
     - ``0``, ``1``
   * - ``ZEA_CHUNK_CACHE_SIZE``
     - Byte budget for that cache; least-recently-used chunks are evicted once it is exceeded.
     - ``10737418240`` (10 GiB)
     - Any positive integer (bytes).
   * - ``BLOSC_NTHREADS``
     - Threads Blosc uses within a single HDF5 chunk, which is most of the write throughput
       (~4x). Turn it **down** when writes are already parallel (several dataloader workers each
       saving a file will multiply with it); turning it far up backfires, as the blocks go
       memory-bound. Only affects writes: the read path (:mod:`zea.data.chunk_reader`) uses
       ``numcodecs``, which has its own thread setting.
     - ``min(8, cpu_count)``
     - Any positive integer.