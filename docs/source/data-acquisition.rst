.. _data-acquisition:

Data
====

This page covers the ``zea`` data format, how files are structured, how to create and
read files, and where to get existing datasets. More detail data handling classes can
be found in :mod:`zea.data` module documentation.

.. note::
   For the configuration system (model, pipeline, and scan parameters in YAML), see
   :doc:`Config <config>`.  Example notebooks on data handling live in :doc:`Examples <examples>`.

The philosophy behind the zea data format is to store data alongside all necessary parameters to
process it (e.g. :class:`~zea.Parameters`), and additional metadata (e.g. acquisition conditions, patient info, etc.)
in a single file. This makes it easy to manage and share data, and ensures that all necessary information
is always available when loading a file.

Additionally, to support the :doc:`cognitive ultrasound framework <about>`, the zea data format is designed to
allow for flexible and efficient access to a part of the data (e.g. a single frame or transmit) without the need
to load the entire file into memory.

-------------------------------
Working with zea data files
-------------------------------

``zea`` stores each acquisition as a single HDF5 file following the :ref:`schema <data-spec>`.  The primary API is :class:`zea.File`. It operates similarly to `h5py.File <https://docs.h5py.org/en/latest/high/file.html>`_, but with an additional interface of parsing parameters into a :class:`~zea.Parameters` object (the merged probe + scan parameters, via :meth:`~zea.File.load_parameters`), and validating the file against the zea data spec.

**Open and read an existing file**

.. code-block:: python

    from zea import File

    with File("my_acquisition.hdf5") as f:
        raw   = f.data.raw_data[:]        # all frames
        raw0  = f.data.raw_data[0]        # first frame only
        parameters = f.load_parameters()  # returns zea.Parameters (merged probe + scan)
        scan  = f.scan                    # returns zea.data.spec.ScanSpec (bare scan group)
        probe = f.probe                   # returns zea.Probe

    # For remote files (Hugging Face Hub):
    with File("hf://zeahub/picmus/.../contrast_speckle.hdf5") as f:
        raw0 = f.data.raw_data[0]         # first frame

See :class:`zea.File` for the full API reference.

**Create a new file**

Use :meth:`zea.File.create` to build a validated file from NumPy arrays.
All inputs are checked against the full schema before anything is written to
disk.

.. doctest::

    >>> import numpy as np
    >>> from zea import File

    >>> n_frames, n_tx, n_el, n_ax = 2, 32, 128, 512
    >>> raw_data = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    >>> probe_geometry = np.zeros((n_el, 3), dtype=np.float32)

    >>> scan = {
    ...    "sampling_frequency": np.float32(40e6),
    ...    "center_frequency":   np.float32(7e6),
    ...    "demodulation_frequency": np.float32(7e6),
    ...    "initial_times":      np.zeros(n_tx, dtype=np.float32),
    ...    "t0_delays":          np.zeros((n_tx, n_el), dtype=np.float32),
    ...    "tx_apodizations":    np.ones((n_tx, n_el),  dtype=np.float32),
    ...    "focus_distances":    np.full(n_tx, np.inf,  dtype=np.float32),
    ...    "transmit_origins":   np.zeros((n_tx, 3),    dtype=np.float32),
    ...    "polar_angles":       np.zeros(n_tx, dtype=np.float32),
    ...    "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
    ... }

    >>> probe = {
    ...    "name": "verasonics_l11_4v",
    ...    "probe_geometry": probe_geometry,
    ... }

    >>> File.create(
    ...    "my_acquisition.hdf5",
    ...    data={"raw_data": raw_data},
    ...    scan=scan,
    ...    probe=probe,
    ... )

**Save from a Parameters object**

When you already hold a :class:`~zea.Parameters` object — e.g. loaded from an
existing file — you can round-trip it back to a new file using
:meth:`~zea.Parameters.to_scan_dict` and :meth:`~zea.Parameters.to_probe_dict`
to reconstruct the dicts that :meth:`~zea.File.create` expects.  No manual
field-by-field reconstruction is needed:

.. testsetup::

    import numpy as np
    from zea import File

    import numpy as np
    from zea import File

    n_frames, n_tx, n_el, n_ax = 2, 4, 8, 64
    raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    scan = {
            "sampling_frequency":     np.float32(40e6),
            "center_frequency":       np.float32(7e6),
            "demodulation_frequency": np.float32(7e6),
            "initial_times":          np.zeros(n_tx, dtype=np.float32),
            "t0_delays":              np.zeros((n_tx, n_el), dtype=np.float32),
            "tx_apodizations":        np.ones((n_tx, n_el), dtype=np.float32),
            "focus_distances":        np.full(n_tx, np.inf, dtype=np.float32),
            "transmit_origins":       np.zeros((n_tx, 3), dtype=np.float32),
            "polar_angles":           np.zeros(n_tx, dtype=np.float32),
            "time_to_next_transmit":  np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
    }
    File.create(
        "source.hdf5",
        data={"raw_data": raw},
        probe={"name": "test_probe", "probe_geometry": np.zeros((n_el, 3))},
        scan=scan, overwrite=True,
    )

.. doctest::

    >>> # load parameters from any file
    >>> with File("source.hdf5") as f:
    ...     parameters = f.load_parameters() # returns a `zea.Parameters` object
    ...     raw_data = f.data.raw_data[:]

    >>> # save those parameters to a new file, without manually reconstructing the scan and probe dicts
    >>> File.create(
    ...     "output.hdf5",
    ...     data={"raw_data": raw_data},
    ...     scan=parameters.to_scan_dict(),
    ...     probe=parameters.to_probe_dict() or None,
    ...     overwrite=True,
    ... )

.. testcleanup::

    import os
    os.remove("source.hdf5")
    os.remove("output.hdf5")

-------------------------------
Multi-track files
-------------------------------

Some acquisitions interleave multiple transmit sequences in a single recording. Sometimes
these sequences contain parameters that may not be expressed by a single ``ScanSpec`` + ``DataSpec`` pair.

.. admonition:: Example

   Imagine an acquisition that alternates between B-mode and Doppler pulses, where
   a different number of axial samples is recorded for each type. In this case, the ``n_ax``
   dimension of the raw data would differ between the two pulse types, and thus could not
   be represented by a single ``DataSpec``, which expects a non-ragged tensor for ``raw_data``.

Rather than splitting these into separate files, ``zea`` can store them as **Tracks**: 
self-contained bundles of raw data and scan parameters in a single HDF5 file, 
with a shared :class:`~zea.Probe` and metadata. Each track exposes its own :class:`~zea.Parameters` object (via
``track.load_parameters()``), containing the parameters
necessary to beamform the raw data in that track. This allows us to specify a :class:`~zea.Pipeline`
*per-track*, which can be applied independently to each frame in that track. Taking the example above,
we could specify a B-mode pipeline to apply to the B-mode track, and a Doppler pipeline to apply to the Doppler track.
Global timing information can be stored in the optional ``track_schedule`` parameter, which
indicates which track each transmit event belongs to. Provided the
``time_to_next_transmit`` for each transmit event, this allows us to reconstruct
the full timing of the acquisition.

.. raw:: html

   <div style="display: flex; flex-direction: column; align-items: center; margin: 3em 0;">
     <!-- Dark mode image -->
     <img
       src="_static/tracks-Dark.svg"
       alt="zea data acquisition with multiple tracks"
       style="display: none; width: 60%; padding-bottom: 1em;"
       class="only-dark"
     />
     <!-- Light mode image -->
     <img
       src="_static/tracks-Light.svg"
       alt="zea data acquisition with multiple tracks"
       style="display: none; width: 60%; padding-bottom: 1em;"
       class="only-light"
     />
     <div style="text-align: center; font-style: italic; color: var(--color-foreground-secondary, #666);">
        Illustrative example of a zea file with two tracks.
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

**HDF5 layout**

.. code-block:: text

    acquisition.hdf5
    ├── attrs:  us_machine, description, zea_version
    ├── probe/                  # probe_geometry, probe_center_frequency, …
    ├── metadata/               # credit, annotations, subject, …
    ├── metrics/                # optional evaluation metrics
    ├── track_schedule          # optional int32[n_total_tx]
    └── tracks/
        ├── track_0/
        │   ├── attrs:  label="focused_bmode"
        │   ├── data/           # raw_data, image, …
        │   └── scan/           # focus_distances, t0_delays, …
        └── track_1/
            ├── attrs:  label="planewave_doppler"
            ├── data/
            └── scan/

**Write — create a file with multiple tracks**

.. doctest::

    >>> import numpy as np
    >>> from zea import File
    >>> from zea.probes import create_probe_geometry

    >>> n_frames, n_ax, n_el = 2, 512, 128
    >>> n_tx_focused, n_tx_pw = 3, 2
    >>> pitch = 0.0003

    >>> probe_geometry = create_probe_geometry(n_el, pitch)

    >>> # One track index per global transmit event across all frames
    >>> track_schedule = np.tile(
    ...     [0] * n_tx_focused + [1] * n_tx_pw, n_frames
    ... ).astype(np.int32)

    >>> File.create(
    ...     "acquisition.hdf5",
    ...     tracks=[
    ...         # Track 0: focused B-mode
    ...         {
    ...             "label": "focused_bmode",
    ...             "data": {"raw_data": np.zeros((n_frames, n_tx_focused, n_ax, n_el, 1))},
    ...             "scan": {
    ...                 "sampling_frequency":     40e6,
    ...                 "center_frequency":       7e6,
    ...                 "demodulation_frequency": 7e6,
    ...                 "initial_times":          np.zeros(n_tx_focused),
    ...                 "t0_delays":              np.zeros((n_tx_focused, n_el)),
    ...                 "tx_apodizations":        np.ones((n_tx_focused, n_el)),
    ...                 "focus_distances":        np.full(n_tx_focused, np.inf),
    ...                 "transmit_origins":       np.zeros((n_tx_focused, 3)),
    ...                 "polar_angles":           np.zeros(n_tx_focused),
    ...                 "time_to_next_transmit": np.ones((n_frames, n_tx_focused)) * 1e-4,
    ...             },
    ...         },
    ...         # Track 1: plane-wave Doppler
    ...         {
    ...             "label": "planewave_doppler",
    ...             "data": {"raw_data": np.zeros((n_frames, n_tx_pw, n_ax, n_el, 1))},
    ...             "scan": {
    ...                 "sampling_frequency":     40e6,
    ...                 "center_frequency":       7e6,
    ...                 "demodulation_frequency": 7e6,
    ...                 "initial_times":          np.zeros(n_tx_pw),
    ...                 "t0_delays":              np.zeros((n_tx_pw, n_el)),
    ...                 "tx_apodizations":        np.ones((n_tx_pw, n_el)),
    ...                 "focus_distances":        np.full(n_tx_pw, np.inf),
    ...                 "transmit_origins":       np.zeros((n_tx_pw, 3)),
    ...                 "polar_angles":           np.zeros(n_tx_pw),
    ...                 "time_to_next_transmit": np.ones((n_frames, n_tx_pw)) * 2e-4,
    ...             },
    ...         },
    ...     ],
    ...     probe={"name": "L11-4v", "probe_geometry": probe_geometry},
    ...     track_schedule=track_schedule,
    ...     overwrite=True,
    ... )

**Read — unpack multiple tracks from a file**

.. doctest::

    >>> import zea

    >>> with zea.File("acquisition.hdf5") as f:
    ...     probe = f.probe             # probe is shared across all tracks
    ...     # See track labels:
    ...     print(f.track_labels)          # ['focused_bmode', 'planewave_doppler']
    ...     # Unpack in the same order as track_labels — always safe:
    ...     focused_track, planewave_track = f.tracks
    ...     # Or fetch a specific track by name:
    ...     focused_track = f.get_track("focused_bmode")
    ...     focused_parameters = focused_track.load_parameters()
    ...     focused_raw  = focused_track.data.raw_data[:]
    ...     # access the global timing information for the focused track:
    ...     focused_track.timestamps
    ...     # ... process with e.g. a focused B-mode pipeline
    ...     planewave_parameters = planewave_track.load_parameters()
    ...     planewave_raw  = planewave_track.data.raw_data[:]
    ...     # access the global timing information for the planewave track:
    ...     planewave_track.timestamps
    ...     # ... process with e.g. a plane-wave Doppler pipeline
    ['focused_bmode', 'planewave_doppler']
    array([[0.    , 0.0001, 0.0002],
           [0.0007, 0.0008, 0.0009]], dtype=float32)
    array([[0.0003, 0.0005],
           [0.001 , 0.0012]], dtype=float32)

.. testcleanup::

    import os
    os.remove("acquisition.hdf5")

-------------------------------
``zea`` data format reference
-------------------------------

Files created with zea 0.1.0 and later are fully described by the
:class:`~zea.data.spec.FileSpec` class.

.. note::

   The spec is the single source of truth.  The documentation below is
   **automatically generated** from :mod:`zea.data.spec`.
   Run ``python docs/source/spec_doc.py`` to refresh it after spec changes.

.. _data-spec:

.. include:: _spec_ref.rst

-------------------------------
Custom fields
-------------------------------

Beyond the standard data types (``raw_data``, ``beamformed_data``, …), you can attach arbitrary
**custom spatial maps** and **custom metadata** to any zea file.

**Custom spatial maps** (``data`` group)

A custom map is a named entry in the ``data`` group that associates a pixel array with a
per-pixel Cartesian coordinate grid.  Each map is then a function from Cartesian space to
some real values.  Pass it as a sub-dict under the key you want:

.. code-block:: python

    import numpy as np
    from zea import File
    from zea.beamform.pixelgrid import cartesian_pixel_grid

    n_frames = 2
    values = np.zeros((n_frames, 64, 64, 1), dtype=np.uint8)   # (frames, z, x[, channels])

    # Build a coordinate grid matching the values spatial shape.
    # cartesian_pixel_grid returns shape (nz, nx, 3); broadcast to add the frame dimension.
    coords_2d = cartesian_pixel_grid(
        xlims=(-0.02, 0.02), zlims=(-0.03, 0.0), grid_size_x=64, grid_size_z=64
    )  # shape (64, 64, 3), last axis = [x, y, z] in metres
    coordinates = np.broadcast_to(coords_2d, (n_frames, 64, 64, 3)).copy()
    # For a simple placeholder without a real grid:
    # coordinates = np.zeros((n_frames, 64, 64, 3), dtype=np.float32)

    File.create(
        "my_acquisition.hdf5",
        data={
            "raw_data": raw,
            "my_overlay": {          # <-- Example of a custom field not in the zea spec
                "values":      values,
                "coordinates": coordinates,  # shape (*spatial_dims, 3)
                # optional: "labels", "description", "unit"
            },
        },
        scan=scan,
    )

    # Reading back
    with File("my_acquisition.hdf5") as f:
        overlay_values      = f.data.my_overlay.values[:]
        overlay_coordinates = f.data.my_overlay.coordinates[:]

.. note::

   :func:`~zea.beamform.pixelgrid.cartesian_pixel_grid` and
   :func:`~zea.beamform.pixelgrid.polar_pixel_grid` are convenient helpers for
   constructing coordinate grids that match typical beamformed images.  See their
   docstrings for full details.


**Custom metadata** (``metadata`` group)

Standard metadata fields (``credit``, ``annotations``, ``text_report``, ``subject``, ``ecg``, …)
are validated by :class:`~zea.data.spec.MetadataSpec`.  Pass a plain dict to ``File.create``
metadata argument.

.. code-block:: python

    File.create(
        "my_acquisition.hdf5",
        data={"raw_data": raw},
        scan=scan,
        metadata={
            "credit": "My Lab, 2024",
            "text_report": "Normal acquisition, no pathology.",
            "annotations": {
                "label": np.array(["healthy", "healthy"]),
            },
        },
    )

Custom signal keys (anything beyond the standard names) are accepted and stored as
:class:`~zea.data.spec.SignalND` entries: a dict with ``samples``, ``start_time_offset``, and
``sampling_frequency``:

.. code-block:: python

    import numpy as np
    from zea import File

    n_samples = 500
    respiratory_signal = {
        "samples":            np.sin(np.linspace(0, 2 * np.pi, n_samples)).astype(np.float32),
        "start_time_offset":  np.float32(-0.5),   # seconds before first transmit
        "sampling_frequency": np.float32(10.0),   # Hz
    }

    File.create(
        "my_acquisition.hdf5",
        data={"raw_data": raw},
        scan=scan,
        metadata={
            "credit": "My Lab, 2024",
            "respiratory_signal": respiratory_signal,   # <-- custom SignalND field
        },
    )

    # Reading back
    with File("my_acquisition.hdf5") as f:
        meta = f.metadata()
        samples = meta.respiratory_signal.samples        # numpy array
        fs      = meta.respiratory_signal.sampling_frequency

See :class:`~zea.data.spec.MetadataSpec` for the full list of supported standard fields.

-------------------------------
Supported datasets & conversion
-------------------------------

The ``zea`` toolbox supports several public and research ultrasound datasets.
Conversion scripts live in
`zea/data/convert/ <https://github.com/tue-bmd/zea/tree/main/zea/data/convert/>`__
and can be invoked as:

.. code-block:: shell

    python -m zea.data.convert --dataset "echonet"  --src <src> --dst <dst>
    python -m zea.data.convert --dataset "camus"    --src <src> --dst <dst>
    python -m zea.data.convert --dataset "picmus"   --src <src> --dst <dst>

**Supported datasets:**

- **EchoNet-Dynamic** — large-scale cardiac ultrasound.
- **EchoNet-LVH** — cardiac dataset for left ventricular hypertrophy.
- **CAMUS** — Cardiac Acquisitions for Multi-structure Ultrasound Segmentation.
- **PICMUS** — Plane-wave Imaging Challenge in Medical Ultrasound.
- **Custom** — any dataset can be converted by following the layout described above.

-------------------------------
Data acquisition platforms
-------------------------------

**Verasonics**

Record data with your Verasonics script, save the workspace to ``.mat``, then convert:

.. code-block:: shell

    python -m zea.data.convert --dataset "verasonics" --src <src> --dst <dst>

See :mod:`zea.data.convert.verasonics` for details.

**us4us** — to be added in a future release.
