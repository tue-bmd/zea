.. _data-acquisition:

Data
====

This page covers the ``zea`` data format, how files are structured, how to create and
read files, and where to get existing datasets. More detail data handling classes can
be found in :mod:`zea.data` module documentation.

For the configuration system (model, pipeline, and scan parameters in YAML), see
:doc:`Config <config>`.  Example notebooks on data handling live in :doc:`Examples <examples>`.

-------------------------------
Working with zea data files
-------------------------------

``zea`` stores each acquisition as a single HDF5 file following the
schema :ref:`data-spec`.  The primary API is :class:`zea.File`.

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
