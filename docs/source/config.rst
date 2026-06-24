.. _config:

Parameters
==========

.. note::
   For the HDF5 data format and file I/O see :doc:`data-acquisition`.
   For pipeline operations see :doc:`pipeline`.

----------------------------
Parameters in the file
----------------------------

Every ``zea`` HDF5 file stores all parameters needed to process the acquisition
alongside the raw data.  They are split into two groups:

**Probe** (``probe/``)
   Fixed for the whole acquisition — element geometry, center frequency,
   bandwidth, lens properties.  Shared across all tracks.
   Defined by :class:`~zea.data.spec.ProbeSpec`.

**Scan** (``scan/``)
   Per-track transmit sequence — delays, apodizations, angles, waveforms,
   sound speed.  Each track has its own :class:`~zea.data.spec.ScanSpec`.

See the :ref:`group-reference` table for the complete field listing.

----------------------------
zea.Parameters
----------------------------

:meth:`~zea.File.load_parameters` merges the probe and scan groups into a
single :class:`~zea.Parameters` object and adds derived quantities
(``wavelength``, ``n_tx``, ``grid``, ``xlims``/``zlims``, ``selected_transmits``):

.. code-block:: python

   with zea.File("data.hdf5") as f:
       parameters = f.load_parameters()             # single-track
       parameters = f.tracks[0].load_parameters()   # multi-track

----------------------------
Config
----------------------------

A config is a YAML file (loaded as :class:`~zea.Config`) that specifies where
the data lives, the pipeline to run, the device to use, and any parameter
overrides.

.. doctest::

    >>> from zea import Config
    >>> from zea.config import check_config

    >>> config = Config.from_path("../configs/config_picmus_rf.yaml")
    >>> config = check_config(config)   # fills defaults, validates
    >>> config.pipeline.operations # doctest: +NORMALIZE_WHITESPACE
    ['demodulate',
     {'name': 'downsample', 'params': {'factor': 4}},
     {'name': 'beamform', 'params': {'beamformer': 'delay_and_sum', 'enable_pfield': False, 'num_patches': 200}},
     'envelope_detect',
     'normalize',
     'log_compress']

    >>> config.to_yaml("my_config.yaml")

.. testcleanup::

    import os
    os.remove("my_config.yaml")

Supported keys
~~~~~~~~~~~~~~

**data** — where to find the file

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Default
     - Description
   * - ``path``
     - ``null``
     - Full path to the HDF5 file. Supports local absolute paths, paths relative
       to the user data root (set in ``users.yaml``), and Hugging Face Hub paths
       (``hf://org/repo/path/to/file.hdf5``).
   * - ``local``
     - ``true``
     - Whether to use local data (``true``) or a network/NAS location (``false``).
   * - ``indices``
     - ``null``
     - Which frames to load: ``null`` (default), ``'all'``, a single ``int``, or
       a list of positive ints.

**parameters** — override any field from the :ref:`group-reference` or pass
custom keys straight through to the pipeline:

.. code-block:: yaml

   parameters:
     center_frequency: 5.0e6
     xlims: [-0.02, 0.02]
     grid_size_x: 512

**pipeline** — list of operations (see :doc:`pipeline`):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Default
     - Description
   * - ``operations``
     - ``[identity]``
     - Ordered list of operations. Each entry is either an operation name (string)
       or a mapping with ``name`` and optional ``params``.
   * - ``jit_options``
     - ``'ops'``
     - JIT scope: ``'ops'`` (compile each op), ``'pipeline'`` (compile the whole
       pipeline), or ``null`` (disable JIT).
   * - ``with_batch_dim``
     - ``true``
     - Whether operations expect a leading batch dimension.

**Top-level keys**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Default
     - Description
   * - ``device``
     - ``'auto:1'``
     - Target hardware: ``cpu``, ``gpu``, ``cuda``, ``gpu:0``, ``auto:1``
       (auto-select; ``-1`` for last device).
   * - ``hide_devices``
     - ``null``
     - Device indices to exclude from auto-selection (int or list of ints).
   * - ``git``
     - ``null``
     - Git commit or branch recorded for reproducibility.

The top-level config is **open**: arbitrary extra sections (``model:``, etc.)
are accepted and passed through unchanged.

----------------------------
API reference
----------------------------

.. autosummary::

   zea.Config

.. autofunction:: zea.config.check_config
   :no-index:
