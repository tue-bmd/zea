Environment variables
================================

Here are the environment variables that ``zea`` uses at runtime. Below the table, the :ref:`backend <backend-selection>`, :ref:`cache-directory` and :ref:`streaming` sections explain how the most important ones fit together.

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


.. _backend-selection:

Backend
-------

``zea`` is built on Keras 3, so the same code runs on TensorFlow, JAX or PyTorch (or NumPy, for quick checks without any of them). Arguably the most important environment variable is therefore ``KERAS_BACKEND``, which picks the backend. Set it before you import ``zea``:

.. code-block:: bash

   export KERAS_BACKEND=jax

If you leave it unset, ``zea`` uses the backend from your ``keras.json`` if that one is installed, and otherwise picks one that is. See :ref:`backend-installation` for how to install each backend.


.. _cache-directory:

Cache directory
---------------

All files ``zea`` downloads or caches are saved to a single directory, namely the ``ZEA_CACHE_DIR`` (``~/.cache/zea`` by default). One can easily set a different location by setting the ``ZEA_CACHE_DIR`` environment variable, or use ``ZEA_DISABLE_CACHE=1`` to use a temporary folder that is removed when your program exits. The cache directory has the following structure:

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


As the tree shows, there are three main types of cached data that a user may interact with:

1. The Hugging Face cache, which is used for downloaded files. This is where you will find the downloaded datasets and models. It has the same layout as the cache of the Hugging Face CLI, but lives in its own folder, so pass ``--cache-dir`` to ``hf cache ls`` or ``hf cache rm`` to manage it (see above).
2. The chunk cache. ``zea`` does not always download the entire file when you open a file from Hugging Face. For instance when you use :class:`zea.File` or a :class:`zea.Dataloader`, which stream by default, it will only download the parts of the file that you actually read. The downloaded chunks are stored in ``chunks/``. More details on streaming can be found in the :ref:`streaming` section.
3. The cached functions. Some slow functions are cached to disk, so that they do not need to be recomputed on subsequent runs. The cached results are stored in ``cached_funcs/``.

.. _streaming:

Streaming
---------

Ultrasound datasets can be very large, and therefore it might not always be practical to download the entire dataset before starting to work with it. To address this, ``zea`` supports streaming datasets from Hugging Face Hub. This means that you can read data from a dataset without having to download the entire dataset first.

Streaming is the default whenever you open something on the Hub with an ``hf://`` path, whether that is a single file with :class:`zea.File` or a whole dataset with :class:`zea.Dataloader` (see :ref:`working-with-zea-data-files` for an example). ``zea`` then only fetches the chunks that hold the frames you read, so you can start right away and work with datasets that are much larger than your disk. If you do want the full file on disk, for instance to write to it, pass ``stream=False`` to :class:`zea.File` or ``lazy=False`` to :class:`zea.Dataloader`. For a dataset that easily fits on your disk, downloading it once is often the faster option.

All files that are downloaded through the streaming infrastructure are saved in the chunk cache. That way, when you need to grab part of a file a second time, it is read from disk instead of from the network. This cache is capped at ``ZEA_CHUNK_CACHE_SIZE`` (10 GiB by default) and makes room by dropping the data you used least recently. If you stream more than that, the next time you read it, it will be downloaded again. That still works, just slower, so raise the limit if you have the disk space, or set ``ZEA_CHUNK_CACHE=0`` if you would rather not keep anything at all.

There are two more settings you might want to change. When many dataloader threads stream at the same time, ``ZEA_MAX_CONCURRENT_REQUESTS`` (64 by default) limits how many requests run in parallel: raise it on a fast connection, or lower it on a slow or shared one. And because ``zea`` remembers the file list of a repository for ``ZEA_HF_CACHE_TTL`` seconds (5 minutes by default), a new upload can take a moment to show up. Set it to ``0`` when you read from a repository that is being written to at the same time.

Finally, reading over the network means that a connection will occasionally drop or a server will be busy. ``zea`` retries such requests a few times before giving up, so a short hiccup will not crash your program.
