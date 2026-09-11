Best Practices & FAQ
====================

This page collects practical tips and common pitfalls that come up when working
with ``zea``. Some depend on your specific setup (operating system, backend,
GPU), but the guidance below covers the most frequent issues.

.. tip::

   The example notebooks on the :doc:`examples` page are highly curated and
   demonstrate many of these patterns in context. They are a good place to copy
   patterns from.

Devices & GPUs
--------------

Most of ``zea`` runs comfortably on CPU. The parts that benefit most from GPU
acceleration are :class:`zea.ops.Beamform` and the :mod:`zea.models`. For small
acquisitions you can beamform on CPU, but with many transmits GPU beamforming is
strongly recommended, as it can become too slow or run out of memory on CPU.

**Call** ``zea.init_device()`` **at the top of your script.**
   On a machine with one or more GPUs, calling :func:`zea.init_device` early
   selects a device (based on available memory) and configures your backend.
   Call it *before* importing heavy libraries so it can hide the other GPUs from
   the framework.

   .. code-block:: python

      import zea

      zea.init_device()  # selects a GPU (or falls back to CPU)

**Check that your backend actually sees the GPU.**
   If processing is slower than expected, the backend may have fallen back to
   CPU because of a version or driver mismatch. When installing a backend (JAX /
   TensorFlow / PyTorch) on a GPU machine, check that its version is compatible
   with your platform, CUDA/cuDNN, and driver versions, and confirm the
   framework detects your GPUs:

   .. tab-set::

      .. tab-item:: JAX

         .. code-block:: python

            import jax
            print(jax.devices())

      .. tab-item:: PyTorch

         .. code-block:: python

            import torch
            print(torch.cuda.is_available())

      .. tab-item:: TensorFlow

         .. code-block:: python

            import tensorflow as tf
            print(tf.config.list_physical_devices())

**On Windows, use WSL for GPU support.**
   Recent versions of the ML backends have limited native Windows support,
   especially with GPU acceleration. If you are on Windows and want GPU support,
   run ``zea`` inside `WSL <https://learn.microsoft.com/windows/wsl/>`_ (Windows
   Subsystem for Linux).

.. seealso::

   See the :doc:`environment` and :doc:`installation` pages for more on backend
   selection and device management.

Data & Tensors
--------------

**Convert explicitly between NumPy (CPU) and tensors (GPU).**
   When moving a NumPy array onto the device as a tensor, or bringing a tensor
   back to the CPU as a NumPy array, use the Keras ops so the conversion works
   regardless of your backend:

   .. code-block:: python

      import keras

      tensor = keras.ops.convert_to_tensor(array)      # NumPy -> tensor (device)
      array = keras.ops.convert_to_numpy(tensor)       # tensor -> NumPy (CPU)

Loading data, configs & models
------------------------------

**Load directly from the Hugging Face Hub with** ``hf://`` **paths.**
   Anywhere ``zea`` accepts a path you can pass an ``hf://`` URL, and the file
   is downloaded and cached locally (under ``ZEA_CACHE_DIR``, default
   ``~/.cache/zea``). This works for data files, configs, and model presets:

   .. code-block:: python

      file = zea.File("hf://zeahub/picmus/.../carotid_cross.hdf5")
      config = zea.Config.from_path("hf://zeahub/configs/config_picmus_rf.yaml")

**Read only the frames or transmits you need.**
   :class:`zea.File` is lazy, like ``h5py``. Index the dataset to load a single
   frame or transmit instead of pulling the whole file into memory:

   .. code-block:: python

      with zea.File("my_acquisition.hdf5") as f:
          frame0 = f.data.raw_data[0]   # first frame only, not the whole file
          parameters = f.load_parameters()

**File vs. Dataset vs. Dataloader.**
   ``zea`` offers three levels for reading data, from a single file to a full
   training loader:

   - :class:`zea.File` wraps a single HDF5 acquisition file.
   - :class:`zea.Dataset` manages a collection of files and simply iterates over
     them, yielding one :class:`~zea.File` at a time. Because it does not stack
     anything, the files do not need to share the same shape.
   - :class:`zea.Dataloader` adds the usual dataloader features on top (batching,
     shuffling, resizing, augmentation, ...). Since it batches, it needs to
     produce a stackable array, so the samples it loads must be shape-compatible.

**Lazily stream a remote** ``hf://`` **dataset with** ``lazy=True``.
   When pointing :class:`zea.Dataset` at an ``hf://`` path, ``lazy=True`` avoids
   downloading the entire (potentially large) dataset up front: each file is
   downloaded on first access as you iterate. This is especially handy when you
   only need part of a big remote dataset:

   .. code-block:: python

      ds = zea.Dataset("hf://zeahub/picmus", lazy=True)  # nothing downloaded yet
      for file in ds:
          parameters = file.load_parameters()  # this file is fetched now
          ...

   See the :doc:`data-acquisition` page for more on datasets and dataloaders.

Plotting
--------

**Use** ``zea.visualize.set_mpl_style()`` **for consistent figures.**
   Call it once after importing to apply the ``zea`` matplotlib style to your
   plots:

   .. code-block:: python

      zea.visualize.set_mpl_style()

Pipeline
--------

**Define your pipeline once, outside the hot loop.**
   Building a :class:`zea.Pipeline` (and letting it JIT-compile) has a one-time
   cost. Construct the pipeline once and reuse it across frames/batches, rather
   than rebuilding it inside your processing loop. Likewise, call
   :meth:`~zea.Pipeline.prepare_parameters` once for a fixed acquisition and
   reuse the prepared inputs.

   .. code-block:: python

      # build once
      pipeline = zea.Pipeline.from_config(config, with_batch_dim=False)
      inputs = pipeline.prepare_parameters(parameters)

      # reuse inside the loop
      for frame in frames:
          image = pipeline(data=frame, **inputs)["data"]

**Debugging a pipeline error? Disable JIT compilation.**
   By default the pipeline compiles its operations (``jit_options="ops"``), and
   compiled code produces hard-to-read tracebacks. Set ``jit_options=None`` to
   run without compilation and get plain Python errors and stack traces:

   .. code-block:: python

      pipeline = zea.Pipeline.from_config(
          config,
          with_batch_dim=False,
          jit_options=None,  # easier debugging; re-enable ("ops"/"pipeline") when done
      )

   See the :doc:`pipeline` page for more on the ``jit_options`` modes
   (``"ops"``, ``"pipeline"``, or ``None``).

**Adding your own processing step?**
   ``zea`` operations are composable and you can register your own. See
   :ref:`custom operations <custom-ops>` for how to define an operation and use
   it in a pipeline, including from a ``config.yaml``.

Selecting a subset of transmits
-------------------------------

Use :meth:`~zea.Parameters.set_transmits` to work with fewer transmit events
(e.g. for faster reconstruction or to compare subsampling strategies). The
selection accepts an ``int`` (that many, evenly spaced), a list/array of
indices, a ``slice``, or a keyword such as ``"all"``, ``"center"``,
``"focused"``, ``"diverging"``, or ``"plane"``.

Changing the selection changes the parameters, so re-run
:meth:`~zea.Pipeline.prepare_parameters` afterwards, and apply the same
selection to the raw data along the transmit axis via
:attr:`~zea.Parameters.selected_transmits`:

.. code-block:: python

   parameters.set_transmits("focused")  # keep only the focused transmits

   # re-prepare parameters AFTER changing the selection
   inputs = pipeline.prepare_parameters(parameters)

   # apply the same selection to the raw data along the transmit axis
   # (data is typically (n_frames, n_tx, ...), so index the second axis)
   subsampled_data = data[:, parameters.selected_transmits]

   image = pipeline(data=subsampled_data, **inputs)["data"]

Beamforming
-----------

**Out of memory during beamforming? Increase the number of patches.**
   The beamforming grid is processed in patches by
   :class:`~zea.ops.PatchedGrid`. If you hit an out-of-memory (OOM) error, raise
   ``num_patches`` so each patch is smaller and uses less memory:

   .. code-block:: python

      pipeline = zea.Pipeline.from_default(num_patches=200)  # default is 100

**Enable pressure-field weighting for focused and grid-based beamforming.**
   For focused transmits and pixel- (grid-) based beamforming, pressure-field
   weighting computes the transmit pressure field and weights each pixel's
   contribution by it, suppressing contributions from regions a transmit did not
   insonify. Enable it on the ``beamform`` operation in your ``config.yaml``:

   .. code-block:: yaml

      pipeline:
        operations:
          - name: beamform
            params:
              enable_pfield: true

   or equivalently in code with ``Pipeline.from_default(enable_pfield=True)``.

**Sharp line around the focal point (focused acquisition)?**
   With focused transmits you can see a sharp artefact line near the focal plane
   where the delay model transitions. Set ``focal_region_length`` to a few
   millimetres to linearly blend the first- and last-arrival delays across the
   focal region, which smooths the transition. Set it in the ``parameters``
   section of your ``config.yaml``:

   .. code-block:: yaml

      parameters:
        focal_region_length: 0.005  # ~5 mm in meters (0.0 disables it)

   or in code with ``parameters.focal_region_length = 0.005``. Try a value and
   inspect the result.

FAQ
---

**Which backend should I use?**
   If you are unsure, use JAX: it is currently the fastest backend for ``zea``
   and the most thoroughly tested. PyTorch and TensorFlow work as well. See
   :ref:`the backend section <backend-installation>` for how to set the
   ``KERAS_BACKEND`` environment variable.

**Processing is slower than expected / runs on CPU.**
   Confirm your backend sees the GPU (see `Devices & GPUs`_), and build the
   pipeline once outside your loop (see `Pipeline`_).

**How do I control caching or logging verbosity?**
   ``zea`` reads a handful of environment variables at runtime, including
   ``ZEA_CACHE_DIR`` (where downloads are cached), ``ZEA_DISABLE_CACHE``, and
   ``ZEA_LOG_LEVEL`` (e.g. ``INFO`` for less verbose output). See the
   :doc:`environment` page for the full list.

**Where can I find more complete examples?**
   The :doc:`examples` page collects curated notebooks you can run directly in
   Google Colab.

Still stuck? Feel free to open an
`issue on GitHub <https://github.com/tue-bmd/zea/issues>`_.
