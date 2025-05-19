.. _data-acquisition:

Data
================================

This page provides a comprehensive overview of how ultrasound data is acquired, structured, and managed within the **usbmd** toolbox.
For a quick start, see :doc:`Getting Started <getting-started>`.
For a full reference of all config parameters, see :doc:`Parameters <parameters>`.

-------------------------------
usbmd data format
-------------------------------

The **usbmd** toolbox uses a custom data format based on the HDF5 standard to ensure consistency, reproducibility, and ease of use across different projects and platforms.

**Key Features:**
- All data and metadata are stored in a single `.hdf5` file per sequence.
- The format is designed to be extensible and self-describing.
- Data is organized into logical groups: `data`, `scan`, and `settings`.

**File Structure Overview:**

.. code-block:: text

    data_file.hdf5
    ├── data
    │    ├── raw_data
    │    ├── aligned_data
    │    ├── envelope_data
    │    ├── beamformed_data
    │    ├── image
    │    └── image_sc
    ├── scan
    │    ├── probe_geometry
    │    ├── t0_delays
    │    ├── n_frames
    │    ├── n_tx
    │    ├── n_el
    │    ├── n_ax
    │    ├── n_ch
    │    ├── sampling_frequency
    │    ├── center_frequency
    │    ├── initial_times
    │    ├── bandwidth_percent
    │    └── time_to_next_transmit
    └── settings
         └── ... (all acquisition and processing parameters)

**Key Groups and Entries:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Entry**
     - **Description**
   * - ``data/raw_data``
     - Raw channel data as acquired from the ultrasound system. Contains either RF (radio-frequency) or IQ (in-phase/quadrature) samples for each element, transmit, and frame. Shape: [n_frames, n_tx, n_el, n_ax, n_ch]
   * - ``data/aligned_data``
     - Time-of-flight corrected data. The raw data is shifted in time so that echoes from the same spatial location are aligned across channels. Shape: [n_frames, n_tx, n_el, n_ax, n_ch]
   * - ``data/envelope_data``
     - Envelope-detected data. The analytic signal is computed (e.g., via Hilbert transform) and the magnitude is taken, removing the carrier frequency. Shape: [n_frames, n_z, n_x]
   * - ``data/beamformed_data``
     - Data after beamforming. The aligned channel data is summed (with apodization) to form spatially resolved signals. Shape: [n_frames, n_z, n_x]
   * - ``data/image``
     - Log-compressed image (in dB). The envelope data is compressed for visualization. Shape: [n_frames, n_z, n_x]
   * - ``data/image_sc``
     - Scan-converted image (in dB). The image is mapped to a Cartesian grid, correcting for probe geometry (e.g., curved arrays). Shape: [n_frames, output_size_z, output_size_x]
   * - ``scan/probe_geometry``
     - 3D coordinates (in meters) of each transducer element in the probe, shape [n_el, 3].
   * - ``scan/t0_delays``
     - Time delays (in seconds) applied to each element for each transmit, shape [n_tx, n_el]. Determines beam steering and focusing.
   * - ``scan/n_frames``
     - Number of frames in the dataset (temporal dimension).
   * - ``scan/n_tx``
     - Number of transmit events per frame.
   * - ``scan/n_el``
     - Number of elements in the transducer array.
   * - ``scan/n_ax``
     - Number of axial (depth) samples per transmit.
   * - ``scan/n_ch``
     - Number of channels in the data (typically 1 for RF, 2 for IQ).
   * - ``scan/sampling_frequency``
     - Sampling frequency of the data acquisition (in Hz).
   * - ``scan/center_frequency``
     - Center frequency of the transducer (in Hz).
   * - ``scan/initial_times``
     - Time (in seconds) when the A/D converter starts sampling for each transmit, shape [n_tx]. Represents the delay between transmit and first recorded sample.
   * - ``scan/bandwidth_percent``
     - Receive bandwidth of the RF signal, as a percentage of the center frequency.
   * - ``scan/time_to_next_transmit``
     - Time interval (in seconds) between subsequent transmit events, shape [n_tx * n_frames].
   * - ``settings``
     - All acquisition and processing parameters required to interpret the data (e.g., reconstruction settings, system configuration).

**Tips:**

- Use :py:meth:`usbmd.File.summary` to inspect datasets.
- Use :py:func:`usbmd.data.data_format.generate_usbmd_dataset` to create new datasets in the correct format.
- `HDFView <https://www.hdfgroup.org/downloads/hdfview/>`__ can be used for manual inspection.


-------------------------------
Supported Datasets & Conversion
-------------------------------

The **usbmd** toolbox supports several public and research ultrasound datasets. For each, we provide scripts to download and convert the data into the usbmd format for integration with the toolbox.

**Supported Datasets:**

- **EchoNet-Dynamic**: Large-scale cardiac ultrasound dataset.
- **CAMUS**: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation.
- **PICMUS**: Plane-wave Imaging Challenge in Medical Ultrasound.
- **Custom Datasets**: You can add your own datasets by following the usbmd format.

**Conversion Scripts:**
- Scripts are provided in the ``usbmd/data/convert/`` directory to automate downloading and conversion.
- Example usage:

  .. code-block:: bash

      python usbmd/data/convert/echonet.py --output-dir <your_data_dir>
      python usbmd/data/convert/camus.py --output-dir <your_data_dir>
      python usbmd/data/convert/picmus.py --output-dir <your_data_dir>

- These scripts will fetch the raw data, process it, and store it in the standardized usbmd format.

-------------------------------
Data Acquisition Platforms
-------------------------------

The **usbmd** toolbox is designed to work with data from multiple ultrasound acquisition systems. We provide tools and documentation for integrating data from the following platforms:

**Verasonics**
- Record data using your preferred Verasonics script.
- Save entire workspace to a `.mat` file.
- Use ``usbmd/data/convert/matlab.py`` to convert the MATLAB workspace files to usbmd format.
- Example:

  .. code-block:: bash

      python usbmd/data/convert/matlab.py --input <verasonics_mat_file> --output <usbmd_hdf5_file>

**us4us**
- TBA
- See ``usbmd/data/convert/us4us.py`` for details.
