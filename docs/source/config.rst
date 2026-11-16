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
     {'name': 'beamform', 'params': {'beamformer': 'delay_and_sum', 'enable_pfield': False}},
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
     - Whether to use local data (``true``) or data at a remote location
       (``false``). Use ``null`` when the ``users.yaml`` paths are single paths,
       shared by both.
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
Defining a beamforming grid
----------------------------

The extent of the final produced image is defined by :attr:`~zea.Parameters.xlims` and :attr:`~zea.Parameters.zlims`. For a cartesian beamforming grid (``grid_type="cartesian"``) this is also the extent of the beamforming grid. For a polar grid (``grid_type="polar"``) the rays fan out from an apex a distance :attr:`~zea.Parameters.distance_to_apex` *behind* the transducer surface. The ``rlims`` (radii from the apex) are then derived from :attr:`~zea.Parameters.zlims` and the ``distance_to_apex`` offset. The final image after scan conversion is then defined by :attr:`~zea.Parameters.xlims` and :attr:`~zea.Parameters.zlims` again.

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 3em 0;">
     <!-- Dark mode image -->
     <img
       src="_static/pixelgrid-Dark.svg"
       alt="Polar pixel grid geometry: xlims, zlims, rlims and distance_to_apex"
       style="display: none; width: 85%; padding-bottom: 1em;"
       class="only-dark"
     />
     <!-- Light mode image -->
     <img
       src="_static/pixelgrid-Light.svg"
       alt="Polar pixel grid geometry: xlims, zlims, rlims and distance_to_apex"
       style="display: none; width: 85%; padding-bottom: 1em;"
       class="only-light"
     />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
        The pixel grid of a polar <code>zea.Parameters</code> object on a curved probe.
     </div>
   </div>
   <style>
     @media (prefers-color-scheme: dark) {
       .only-dark { display: block !important; }
     }
     @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
       .only-light { display: block !important; }
     }
   </style>


.. _angle-conventions:

Angle and radius names
~~~~~~~~~~~~~~~~~~~~~~

zea uses two kinds of angles, in radians, both in the transducer frame (``x`` along
the array, ``y`` across it, ``+z`` into the medium):

- **Transmit directions** are `spherical coordinates <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_, as stored in the file format.
- **Grid and scan-conversion angles** are projections onto the x-z and y-z planes, so
  a 3D volume is a frustum that extends the 2D sector in elevation.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Symbol
     - Name
     - Meaning
   * - :math:`\theta`
     - ``polar_angles``
     - Spherical polar angle of each transmit: the angle between its direction and
       ``+z``, signed, positive toward ``+x``.
   * - :math:`\phi`
     - ``azimuth_angles``
     - Spherical azimuth angle of each transmit: the rotation of its direction around
       ``z``, from ``+x``. Zero for 2D imaging.
   * - :math:`r`, :math:`\rho`
     - ``rlims``
     - Radius from the apex of the polar grid, of the first and last row [m]. Not the
       depth: :attr:`~zea.Parameters.zlims` is measured from the array, so
       ``rlims = zlims + distance_to_apex``.
   * - :math:`\operatorname{atan}(x/z)`
     - ``polar_limits``
     - Polar angle of the first and last column of a polar grid, signed, positive toward
       ``+x``. In 3D, the angle of the projection onto the x-z plane.
   * - :math:`\operatorname{atan}(y/z)`
     - ``elevation_limits``
     - Elevation angle of the first and last sample of a 3D frustum, signed, positive
       toward ``+y``. Only in 3D.

To find where a transmit falls on the grid, convert its spherical angles:

.. math::

   \text{polar} = \operatorname{atan}(\tan\theta\cos\phi), \qquad
   \text{elevation} = \operatorname{atan}(\tan\theta\sin\phi)

A transmit steered only along the array has a grid polar angle equal to :math:`\theta`.
Once it is also steered in elevation, the two differ.

**Signed** :math:`\theta`\ **.** :math:`\theta` is stored with a sign, positive toward
``+x``. 2D data stores :math:`\phi = 0`. 3D data keeps :math:`\phi` in
:math:`(-\pi/2, \pi/2]`, so the sign of :math:`\theta` still gives the side in ``x``.

.. note::

   **Physics vs. mathematics convention.** zea uses the physics convention (ISO 80000-2):
   :math:`\theta` is the polar angle from ``+z`` and :math:`\phi` the azimuth around
   ``z``.

.. note::

   **Ultrasound jargon.** In ultrasound, "azimuth" usually names the *lateral* direction,
   along the array (``x``). We instead follow the
   `spherical coordinate system <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_ where
   azimuth :math:`\phi` is the rotation around the ``z`` axis.



----------------------------
API reference
----------------------------

.. autosummary::

   zea.Config

.. autofunction:: zea.config.check_config
   :no-index:
