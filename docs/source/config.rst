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

A :class:`zea.Parameters` object builds a default beamforming grid from what a file
records: the probe geometry, the transmits (``polar_angles``, ``transmit_origins`` and
``focus_distances``) and the record length. You choose the grid type with ``grid_type``
and ``enable_scanline``, and can tune the result with a few overrides. Positions are in
the transducer frame: ``z = 0`` at the middle of the array, ``+z`` into the medium, and
angles measured from ``+z``, positive toward ``+x`` (see :ref:`angle-conventions`).

:attr:`~zea.Parameters.zlims`
   Depths below the transducer. Defaults to the end of the record,
   ``sound_speed * n_ax / sampling_frequency / 2``.

:attr:`~zea.Parameters.xlims`
   The lateral extent of the grid. For a cartesian grid it covers what the transmits
   insonify down to ``zlims[1]``, and at least the probe width. For a polar grid it is the
   width of the sector; an explicit value sets the width of the scan-converted image
   instead (crop or pad).

:attr:`~zea.Parameters.grid_apex`
   The point a polar grid fans out from, ``(0, 0, -d)`` on the central axis at or behind
   the array. It is derived from where the transmits fan out from: the point where steered
   transmit axes meet (the middle of the array, a virtual apex behind a flat array whose
   transmits are spread over it, or a curved array's centre of curvature), the virtual
   source of unsteered diverging waves, or else a curved array's centre of curvature.

:attr:`~zea.Parameters.polar_limits`
   The angles of the first and last column of a polar grid: the range of the transmit
   angles, widened to cover what the transmits insonify at ``zlims[1]``. A focused
   transmit adds no margin; plane and diverging waves widen it. Set
   :attr:`~zea.Parameters.polar_margin` to widen by a fixed angle instead.

:attr:`~zea.Parameters.rlims`
   The radii of the first and last row of a polar grid, measured from the apex:
   ``zlims - grid_apex[2]``. Scan conversion reads the polar image through ``rlims`` and
   ``polar_limits``.

:attr:`~zea.Parameters.extent`
   For a polar grid, the window the scan-converted image fills: ``xlims`` wide, and on a
   flat array starting at ``zlims[0]``, which crops what a virtual apex places behind the
   array.

:attr:`~zea.Parameters.enable_scanline`
   One column per focused transmit: a polar grid at the transmit angles when their axes
   meet in one point, or a vertical column under each transmit when they are parallel.

:attr:`~zea.Parameters.f_number`
   The receive cone of each element. Defaults to the element's -6 dB acceptance angle, so
   0 (no cone) for most phased arrays. It masks pixels but does not change the grid.

Both figures below are drawn by the pixel grid explorer, a JavaScript port of the code
checked against reference values computed by zea (``docs/explorer``). **Open the
explorer** on either one to change the probe, transmits and overrides; the address then
holds every setting, so you can share a link to the exact grid.

.. raw:: html

   <script>
     // The explorer posts its content height, so each iframe fits the compact or
     // expanded view.
     window.addEventListener("message", (event) => {
       if (!event.data || event.data.type !== "zea-explorer-height") return;
       for (const frame of document.querySelectorAll("iframe.pixel-grid-explorer")) {
         if (frame.contentWindow === event.source) frame.style.height = `${event.data.height}px`;
       }
     });
   </script>
   <style>
     iframe.pixel-grid-explorer { display: block; width: 100%; height: 520px; border: 0; }
   </style>

   <figure style="margin: 2em 0;">
     <iframe class="pixel-grid-explorer" loading="lazy"
             src="_static/pixel-grid-explorer.html?preset=curved-docs-figure&amp;view=compact"
             title="Polar pixel grid of a curved array: probe_geometry, grid_apex, rlims, polar_limits, zlims, xlims and extent"></iframe>
     <figcaption><p>A polar grid on a curved array fans out from its centre of curvature.</p></figcaption>
   </figure>

   <figure style="margin: 2em 0;">
     <iframe class="pixel-grid-explorer" loading="lazy"
             src="_static/pixel-grid-explorer.html?preset=phased-docs-figure&amp;view=compact"
             title="Polar pixel grid of a flat phased array: probe_geometry, grid_apex, rlims, polar_limits, zlims, xlims and extent"></iframe>
     <figcaption><p>Focused lines that leave from across a phased array: their axes meet behind it in a virtual apex, and the grid fans out from there.</p></figcaption>
   </figure>


.. _angle-conventions:

Angle and radius names
~~~~~~~~~~~~~~~~~~~~~~

zea uses two kinds of angles, in radians, both in the transducer frame (``x`` along
the array, ``y`` across it, ``+z`` into the medium):

- **Transmit directions** are spherical coordinates, as stored in the file format.
- **Grid and scan-conversion angles** are projections onto the x-z and y-z planes, so
  a 3D volume is a frustum that extends the 2D sector in elevation. 3D adds one name,
  ``elevation_limits``; every 2D name keeps its meaning.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - Symbol
     - Name
     - Meaning
   * - :math:`\theta`
     - ``polar_angles``
     - Spherical polar angle of each transmit: the angle between its direction and
       ``+z``.
   * - :math:`\phi`
     - ``azimuth_angles``
     - Spherical azimuth angle of each transmit: the rotation of its direction around
       ``z``, from ``+x``. Zero for 2D imaging.
   * - :math:`r`, :math:`\rho`
     - ``rlims``
     - Radius from :attr:`~zea.Parameters.grid_apex`, of the first and last row [m].
       Not the depth: :attr:`~zea.Parameters.zlims` is measured from the array, so
       ``rlims = zlims - grid_apex[2]``.
   * - :math:`\operatorname{atan}(x/z)`
     - ``polar_limits``
     - Polar angle of the first and last column of a polar grid, signed, positive toward
       ``+x``. In 3D, the angle of the projection onto the x-z plane.
   * - :math:`\operatorname{atan}(y/z)`
     - ``elevation_limits``
     - Elevation angle of the first and last sample of a 3D frustum, signed, positive
       toward ``+y``. Only in 3D. Defaults to the transmits' elevation angles, widened
       by 15 % of their range, or by :attr:`~zea.Parameters.elevation_margin`.

A transmit points along
:math:`(\sin\theta\cos\phi,\ \sin\theta\sin\phi,\ \cos\theta)`. Its grid angles are
:math:`\operatorname{atan}(\tan\theta\cos\phi)` (polar) and
:math:`\operatorname{atan}(\tan\theta\sin\phi)` (elevation), so the two coincide with
the spherical angles only in the x-z plane (:math:`\phi = 0`) and the y-z plane
(:math:`\phi = \pm\pi/2`). That is why the grid angles are not called ``azimuth``.

**The zea standard.** 2D data stores :math:`\phi = 0` and a *signed* :math:`\theta`,
positive toward ``+x``, so :math:`\theta` equals the grid's polar angle. The spherical
convention would instead keep :math:`\theta \geq 0` and steer toward ``-x`` with
:math:`\phi = \pi`; both describe the same direction, but zea writes the signed form,
and 3D data keeps :math:`\phi` in :math:`(-\pi/2, \pi/2]` so that the sign of
:math:`\theta` still gives the side in ``x``.

.. note::

   **Physics vs. mathematics convention.** zea uses the physics convention (ISO 80000-2):
   :math:`\theta` is the polar angle from ``+z`` and :math:`\phi` the azimuth around
   ``z``. Many mathematics texts, and some libraries, swap the symbols: :math:`\theta`
   is the azimuth and :math:`\phi` the polar angle. When converting data, go by the
   names ``polar_angles`` and ``azimuth_angles``, not by the symbols of the source.

.. note::

   **Ultrasound jargon.** In ultrasound, "azimuth" usually names the *lateral* direction,
   along the array (``x``), and "elevation" the direction across it (``y``). zea's
   ``azimuth_angles`` is the spherical coordinate-system :math:`\phi` instead, not an
   angle toward ``x``. "Elevation" means the same in both: ``elevation_limits`` is the
   angle toward ``y``.


----------------------------
API reference
----------------------------

.. autosummary::

   zea.Config

.. autofunction:: zea.config.check_config
   :no-index:
