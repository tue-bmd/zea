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
Multi-track files
-------------------------------

Some acquisitions interleave multiple transmit sequences in a single recording — for example,
alternating focused and diverging-wave pulses.  Rather than splitting these into separate files
or losing the interleaving information, ``zea`` can store them as **tracks**: self-contained
bundles of raw data and scan parameters, all inside one HDF5 file.

**Why tracks?**

- A single file stays self-describing: each track carries its own :class:`~zea.data.spec.ScanSpec`
  so it can be beamformed independently.
- The original firing order is preserved via the optional ``track_schedule`` field.

**HDF5 layout**

.. code-block:: text

    acquisition.hdf5
    ├── attrs:  probe_name, us_machine, zea_version, …
    ├── track_schedule          # optional int32[n_total_tx]
    └── tracks/
        ├── track_0/
        │   ├── data/           # raw_data, image, …
        │   └── scan/           # probe_geometry, t0_delays, …
        └── track_1/
            ├── data/
            └── scan/

**Write — create a file with two tracks**

.. code-block:: python

    import numpy as np
    from zea.data.spec import FileSpec

    n_frames, n_ax, n_el = 4, 512, 128
    n_tx_focused, n_tx_pw = 16, 8

    probe_geom = np.zeros((n_el, 3), dtype=np.float32)

    spec = FileSpec(
        tracks=[
            # Track 1: e.g. focused B-mode
            {
                "data": {"raw_data": np.zeros((n_frames, n_tx_focused, n_ax, n_el, 1), dtype=np.float32)},
                "scan": {
                    "probe_geometry":         probe_geom,
                    "sampling_frequency":     np.float32(40e6),
                    "center_frequency":       np.float32(7e6),
                    "demodulation_frequency": np.float32(7e6),
                    "initial_times":          np.zeros(n_tx_focused, dtype=np.float32),
                    "t0_delays":              np.zeros((n_tx_focused, n_el), dtype=np.float32),
                    "tx_apodizations":        np.ones((n_tx_focused, n_el), dtype=np.float32),
                    "focus_distances":        np.full(n_tx_focused, np.inf, dtype=np.float32),
                    "transmit_origins":       np.zeros((n_tx_focused, 3), dtype=np.float32),
                    "polar_angles":           np.zeros(n_tx_focused, dtype=np.float32),
                },
            },
            # Track 2: e.g. plane-wave Doppler
            {
                "data": {"raw_data": np.zeros((n_frames, n_tx_pw, n_ax, n_el, 1), dtype=np.float32)},
                "scan": {
                    "probe_geometry":         probe_geom,
                    "sampling_frequency":     np.float32(40e6),
                    "center_frequency":       np.float32(7e6),
                    "demodulation_frequency": np.float32(7e6),
                    "initial_times":          np.zeros(n_tx_pw, dtype=np.float32),
                    "t0_delays":              np.zeros((n_tx_pw, n_el), dtype=np.float32),
                    "tx_apodizations":        np.ones((n_tx_pw, n_el), dtype=np.float32),
                    "focus_distances":        np.full(n_tx_pw, np.inf, dtype=np.float32),
                    "transmit_origins":       np.zeros((n_tx_pw, 3), dtype=np.float32),
                    "polar_angles":           np.zeros(n_tx_pw, dtype=np.float32),
                },
            },
        ],
        probe_name="L11-4v",
    )
    spec.save("acquisition.hdf5")

**Read — iterate over tracks**

.. code-block:: python

    import zea

    with zea.File("acquisition.hdf5") as f:
        probe = f.probe()              # probe hardware is shared across all tracks
        for track in f.tracks:
            scan = track.scan()        # returns a zea.Scan for this track
            raw  = track.data.raw_data[:]
            print(scan.n_tx, raw.shape)

    # Accessing f.data or f.scan() on a multi-track file raises AttributeError
    # with a hint to use f.tracks instead.

**Beamform each track with its own pipeline**

.. code-block:: python

    import zea

    # Load per-track configs (focused B-mode and diverging Doppler)
    bmode_config   = zea.Config.from_path("hf://zeahub/zea-cardiac-2026/config.yaml")
    doppler_config = zea.Config.from_path("configs/config_cardiac_diverging_doppler.yaml")

    bmode_pipeline   = zea.Pipeline.from_config(bmode_config)
    doppler_pipeline = zea.Pipeline.from_config(doppler_config)

    with zea.File("acquisition.hdf5") as f:
        probe = f.probe()
        focused_track, diverging_track = f.tracks

        focused_scan   = focused_track.scan(**bmode_config.scan)
        diverging_scan = diverging_track.scan(**doppler_config.scan)

        focused_raw   = focused_track.data.raw_data[:]
        diverging_raw = diverging_track.data.raw_data[:]

    bmode_params   = bmode_pipeline.prepare_parameters(probe, focused_scan, bmode_config.scan)
    doppler_params = doppler_pipeline.prepare_parameters(probe, diverging_scan, doppler_config.scan)

    bmode_image = bmode_pipeline(data=focused_raw, **bmode_params)["data"]
    iq_data     = doppler_pipeline(data=diverging_raw, **doppler_params)["data"]

See ``example-multiple-tracks.py`` in the repository root for a complete end-to-end
example that saves a two-track cardiac file and exports a side-by-side B-mode / Color
Doppler animated GIF.

-------------------------------
``zea`` data format reference
-------------------------------



.. note::

   The spec is the single source of truth.  The documentation below is
   **automatically generated** from :mod:`zea.data.spec`.
   Run ``python docs/source/spec_doc.py`` to refresh it after spec changes.

.. _data-spec:

.. include:: _spec_ref.rst

-------------------------------
Custom fields
-------------------------------

Beyond the standard data types (``raw_data``, ``image_sc``, …), you can attach arbitrary
**custom spatial maps** and **custom metadata** to any zea file.

**Custom spatial maps** (``data`` group)

A custom map is a named entry in the ``data`` group that associates a pixel array with a physical
extent.  Pass it as a sub-dict under the key you want:

.. code-block:: python

    import numpy as np
    from zea import File

    n_frames = 2
    values = np.zeros((n_frames, 64, 64, 1), dtype=np.uint8)   # (frames, z, x[, channels])
    extent = np.array([x_min, x_max, y_min, y_max, z_min, z_max], dtype=np.float32)  # metres

    f = File.create(
        "my_acquisition.hdf5",
        data={
            "raw_data": raw,
            "my_overlay": {          # any name not already in the spec
                "values":  values,
                "extent":  extent,
                # optional: "labels", "description", "unit"
            },
        },
        scan=scan,
    )
    f.close()

    # Reading back
    with File("my_acquisition.hdf5") as f:
        overlay_values = f.data.my_overlay.values[:]
        overlay_extent = f.data.my_overlay.extent[:]


**Custom metadata** (``metadata`` group)

Standard metadata fields (``credit``, ``annotations``, ``text_report``, ``subject``, ``ecg``, …)
are validated by :class:`~zea.data.spec.MetadataSpec`.  Pass a plain dict to ``File.create`` or to
:func:`~zea.data.file_operations.save_file`:

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
:class:`~zea.data.spec.SignalND` entries.  See :class:`~zea.data.spec.MetadataSpec` for the full
list of supported fields.

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
