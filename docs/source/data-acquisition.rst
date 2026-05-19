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
process it (e.g. :class:`~zea.Scan` parameters), and additional metadata (e.g. acquisition conditions, patient info, etc.)
in a single file. This makes it easy to manage and share data, and ensures that all necessary information
is always available when loading a file.

Additionally, to support the :doc:`cognitive ultrasound framework <about>`, the zea data format is designed to
allow for flexible and efficient access to a part of the data (e.g. a single frame or transmit) without the need
to load the entire file into memory.

-------------------------------
Working with zea data files
-------------------------------

``zea`` stores each acquisition as a single HDF5 file following the :ref:`schema <data-spec>`.  The primary API is :class:`zea.File`. It operates similarly to `h5py.File <https://docs.h5py.org/en/latest/high/file.html>`_, but with an additional interface of parsing parameters into :class:`~zea.Scan` and :class:`~zea.Probe` objects, and validating the file against the zea data spec.

**Open and read an existing file**

.. code-block:: python

    from zea import File

    with File("my_acquisition.hdf5") as f:
        raw   = f.data.raw_data[:]        # all frames
        raw0  = f.data.raw_data[0]        # first frame only
        scan  = f.scan()                  # returns zea.Scan
        probe = f.probe()                 # returns zea.Probe

    # For remote files (Hugging Face Hub):
    with File("hf://zeahub/picmus/.../contrast_speckle.hdf5") as f:
        raw0 = f.data.raw_data[0]         # first frame

See :class:`zea.File` for the full API reference.

**Create a new file**

Use :meth:`zea.File.create` to build a validated file from NumPy arrays.
All inputs are checked against the full schema before anything is written to
disk::

    import numpy as np
    from zea import File

    n_frames, n_tx, n_el, n_ax = 2, 32, 128, 512
    raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    geom = np.zeros((n_el, 3), dtype=np.float32)

    scan = {
        "probe_geometry": geom,
        "sampling_frequency": np.float32(40e6),
        "center_frequency":   np.float32(7e6),
        "demodulation_frequency": np.float32(7e6),
        "initial_times":      np.zeros(n_tx, dtype=np.float32),
        "t0_delays":          np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations":    np.ones((n_tx, n_el),  dtype=np.float32),
        "focus_distances":    np.full(n_tx, np.inf,  dtype=np.float32),
        "transmit_origins":   np.zeros((n_tx, 3),    dtype=np.float32),
        "polar_angles":       np.zeros(n_tx, dtype=np.float32),
        "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
    }

    f = File.create(
        "my_acquisition.hdf5",
        data={"raw_data": raw},
        scan=scan,
        probe_name="L11-4v",
    )
    f.close()


-------------------------------
``zea`` data format reference
-------------------------------

Files created with zea 0.0.12 and later are fully described by the
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

    f = File.create(
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
    f.close()

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

    f = File.create(
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

    f = File.create(
        "my_acquisition.hdf5",
        data={"raw_data": raw},
        scan=scan,
        metadata={
            "credit": "My Lab, 2024",
            "respiratory_signal": respiratory_signal,   # <-- custom SignalND field
        },
    )
    f.close()

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
