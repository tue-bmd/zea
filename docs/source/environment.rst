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
   * - ``BLOSC_NTHREADS``
     - Threads Blosc uses to compress or decompress the blocks *within* a single HDF5 chunk.
       ``zea`` sets this to ``min(8, cpu_count)`` if you have not set it yourself, which is
       worth roughly **4x on writes** (measured 105 → 453 MB/s on real int16 channel data);
       see :ref:`blosc-nthreads` below.
     - ``min(8, cpu_count)``
     - Any positive integer.

.. _blosc-nthreads:

Blosc threading (``BLOSC_NTHREADS``)
------------------------------------

``zea`` stores array data with the Blosc HDF5 filter, and HDF5 runs that filter **one chunk at a
time, single-threaded**. Blosc itself, however, splits each chunk into blocks and can process
those in parallel, and its HDF5 filter reads ``BLOSC_NTHREADS`` from the environment on every
call. So this variable is most of what determines write throughput.

``zea`` therefore sets a default of ``min(8, cpu_count)`` (in :mod:`zea.data.spec`), using
``setdefault`` — **an explicit ``BLOSC_NTHREADS`` in your environment always wins.**

Two reasons you might want to set it yourself:

* **Turn it down when your writes are already parallel.** Several dataloader workers each saving
  a file will multiply with this: 8 workers x 8 Blosc threads is 64 threads competing for memory
  bandwidth. Setting ``BLOSC_NTHREADS=1`` or ``2`` is usually better in that case.
* **Do not turn it far up.** The gain flattens and then reverses, because the blocks within one
  chunk are small: on real data, 32 threads measured *slower* than 8 (272 against 345 MB/s).

It does not change what is stored — files are byte-for-byte identical at any thread count, only
faster or slower to produce.

.. note::

   ``numcodecs`` (which ``zea`` uses to decode chunks concurrently on the read path, see
   :mod:`zea.data.chunk_reader`) keeps its **own** Blosc thread setting, which already defaults
   to 8 and is *not* read from this variable. Setting ``BLOSC_NTHREADS`` therefore affects
   writes, not reads.