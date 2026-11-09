Environment variables
================================

Here are the environment variables that ``zea`` uses at runtime. Arguably the most important environment variable is the Keras backend selection via ``KERAS_BACKEND``. See the :ref:`backend-installation` section for details on configuring the backend. For how the cache and streaming fit together, see :ref:`cache-directory` and :ref:`streaming` below.

.. list-table::
   :header-rows: 1
   :widths: 20 80 20 20

   * - **Variable**
     - **Description**
     - **Default**
     - **Options**
   * - ``KERAS_BACKEND``
     - Select the Keras backend to use. This defines the ML framework that will be used for all tensor operations. Takes precedence over the backend in ``keras.json``. If neither is set, ``zea`` picks an installed backend on import (preferring ``tensorflow``, then ``jax``, then ``torch``).
     - installed backend
     - ``tensorflow``, ``jax``, ``torch``, ``numpy``
   * - ``ZEA_CACHE_DIR``
     - Directory to use for caching downloaded files, e.g. model weights or datasets from Hugging Face Hub. See :ref:`cache-directory`.
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
   * - ``ZEA_TRUST_REMOTE_CODE``
     - If set to ``1``, allow a pipeline config's ``imports`` to fetch and execute a
       Python module from a remote location (``hf://``). Equivalent to passing
       ``--trust-remote-code`` to ``zea process``. Off by default, because it runs code
       from someone else's repository on your machine. See :ref:`pipeline-imports`.
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
     - If set to ``1``, will use parallel processing for the per-file sweeps over a dataset
       (reading HDF5 file shapes, and validating files against the zea format).
     - ``1``
     - ``0``, ``1``
   * - ``ZEA_FIND_H5_SHAPES_WORKERS``
     - Number of worker processes for those parallel sweeps. The sweeps mostly wait on file
       opens (over the network for ``hf://`` files), so more processes than this rarely help,
       and each one has to import ``zea``. Never more than the number of cores or files.
     - ``32``
     - Any positive integer.
   * - ``ZEA_TEST_DEVICE``
     - Can be used to only run the tests with a particular device.
     - ``auto:1``
     - Any valid device name as accepted by :func:`zea.init_device`. For example, ``cpu``,
       ``cuda:0``, ``auto:1``, etc.
   * - ``ZEA_TEST_BACKEND``
     - Backend to run the tests with. ``KERAS_BACKEND`` is ignored by the tests.
     - ``tensorflow``
     - ``tensorflow``, ``jax``, ``torch``
   * - ``ZEA_HF_CACHE_TTL``
     - Seconds a Hugging Face repository listing or resolved download path may be reused before
       ``zea`` asks the hub again. Keeps a dataset scan or preset load from repeating the same
       request; set to ``0`` if you need every call to hit the hub (e.g. a repo being written to
       from elsewhere while you read it).
     - ``300``
     - Any number of seconds, ``0`` to disable.
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
   * - ``ZEA_MAX_CONCURRENT_REQUESTS``
     - Maximum number of HTTP range requests in flight at once per process while streaming
       (``hf://``), shared by all open files and dataloaders. Streamed reads go to the Hugging
       Face CDN, which is not rate limited, so this only bounds connections and memory. Raise
       it when many dataloader threads stream at once; lower it on a slow or shared connection.
     - ``64``
     - Any positive integer.
   * - ``BLOSC_NTHREADS``
     - Threads Blosc uses within a single HDF5 chunk, which is most of the write throughput
       (~4x). Turn it **down** when writes are already parallel (several dataloader workers each
       saving a file will multiply with it); turning it far up backfires, as the blocks go
       memory-bound. Only affects writes: the read path (:mod:`zea.data.chunk_reader`) uses
       ``numcodecs``, which has its own thread setting.
     - ``min(8, cpu_count)``
     - Any positive integer.


.. _cache-directory:

Cache directory
---------------

Everything ``zea`` downloads or caches lives in one folder, ``ZEA_CACHE_DIR`` (``~/.cache/zea`` by default). Point it elsewhere to move all of it, for example to a scratch disk on a cluster, or set ``ZEA_DISABLE_CACHE=1`` to use a temporary folder that is removed when your program exits.

.. code-block:: text

   $ZEA_CACHE_DIR
   ├── huggingface/
   │   ├── datasets/   # downloaded files: configs, presets, .hdf5 files
   │   ├── models/     # model weights and presets
   │   └── listings/   # file lists of Hugging Face repos, reused for a few minutes
   ├── chunks/         # data read while streaming, up to 10 GiB
   ├── hf_locations/   # download links of streamed files, reused for 30 minutes
   └── cached_funcs/   # saved results of slow functions

Downloads stay until you delete them. The ``huggingface/`` folders use the same layout as the Hugging Face cache, so the ``hf`` command line tool can list and clean them:

.. code-block:: bash

   hf cache ls --cache-dir "${ZEA_CACHE_DIR:-$HOME/.cache/zea}/huggingface/datasets"
   hf cache ls --cache-dir "${ZEA_CACHE_DIR:-$HOME/.cache/zea}/huggingface/models"


.. _streaming:

Streaming
---------

Opening an ``hf://`` file, with :class:`zea.File` or a :class:`zea.Dataloader` over an ``hf://`` folder, streams it: ``zea`` only fetches the parts you actually read, instead of downloading the whole file first. That way you can work with datasets much larger than your disk. Pass ``stream=False`` to :class:`zea.File` to download the full file instead.

What you read is kept in ``chunks/``, so reading it again, such as in the next epoch, comes from disk. Once that folder reaches ``ZEA_CHUNK_CACHE_SIZE`` (10 GiB), the data used least recently makes room. If your dataset is bigger than that, each epoch downloads again what it reads: slower, but it works. Raise the limit if you have the disk space, or set ``ZEA_CHUNK_CACHE=0`` to keep nothing.

A few things you may want to tune:

- **Many readers at once.** ``ZEA_MAX_CONCURRENT_REQUESTS`` caps how many requests run in parallel. Raise it when many dataloader threads stream together, lower it on a slow or shared connection.
- **A repo that changes while you read it.** File lists are reused for ``ZEA_HF_CACHE_TTL`` seconds (5 minutes), so a new upload can take that long to show up. Set it to ``0`` to always ask Hugging Face.

Failed requests, such as a dropped connection or a busy server, are retried a few times before ``zea`` gives up.
