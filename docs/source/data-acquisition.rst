.. _data-acquisition:

Data
================================

This page provides a comprehensive overview of how ultrasound data is acquired, structured, and managed within the ``zea`` toolbox.

For a quick start, see :doc:`Getting Started <getting-started>`.
For a full reference of all config parameters, see :doc:`Parameters <parameters>`. Lastly, some example notebooks on data handling can be found in :doc:`Examples <examples>`.

-------------------------------
``zea`` data handling
-------------------------------

For information on how to handle data in the ``zea`` toolbox, see the :mod:`zea.data` module documentation.

-------------------------------
``zea`` data format
-------------------------------

The ``zea`` toolbox uses a custom data format based on the HDF5 standard to store ultrasound data. It is convenient as it allows for efficient storage and retrieval of large datasets, with easy indexing that does not require loading the entire dataset into memory.

**Key Features:**
- All data and metadata are stored in a single `.hdf5` file per sequence.
- The format is designed to be extensible and self-describing.
- Data is organized into logical groups: `data` and `scan` (custom parameters allowed in `scan`).

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
    └── scan
         ├── n_ax
         ├── n_el
         ├── n_tx
         ├── n_ch
         ├── n_frames
         ├── sound_speed
         ├── probe_geometry
         ├── sampling_frequency
         ├── center_frequency
         ├── demodulation_frequency
         ├── initial_times
         ├── t0_delays
         ├── tx_apodizations
         ├── focus_distances
         ├── transmit_origins
         ├── polar_angles
         ├── azimuth_angles
         ├── bandwidth_percent
         ├── time_to_next_transmit
         ├── tgc_gain_curve
         ├── element_width
         ├── tx_waveform_indices
         ├── waveforms_one_way
         ├── waveforms_two_way
         ├── lens_correction
         └── ... (custom parameters allowed)

**Parameter Descriptions:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Entry**
     - **Description**
   * - ``data/raw_data``
     - Raw channel data as acquired from the ultrasound system. Shape: [n_frames, n_tx, n_ax, n_el, n_ch]
   * - ``data/aligned_data``
     - Time-of-flight corrected data. Shape: [n_frames, n_tx, n_ax, n_el, n_ch]
   * - ``data/envelope_data``
     - Envelope-detected data. Shape: [n_frames, grid_size_z, grid_size_x]
   * - ``data/beamformed_data``
     - Data after beamforming. Shape: [n_frames, grid_size_z, grid_size_x]
   * - ``data/image``
     - Log-compressed image (in dB). Shape: [n_frames, grid_size_z, grid_size_x]
   * - ``data/image_sc``
     - Scan-converted image (in dB). Shape: [n_frames, output_size_z, output_size_x]
   * - ``scan/n_ax``
     - Number of axial (depth) samples per transmit.
   * - ``scan/n_el``
     - Number of elements in the transducer array.
   * - ``scan/n_tx``
     - Number of transmit events per frame.
   * - ``scan/n_ch``
     - Number of channels in the data (typically 1 for RF, 2 for IQ).
   * - ``scan/n_frames``
     - Number of frames in the dataset (temporal dimension).
   * - ``scan/sound_speed``
     - Speed of sound in m/s.
   * - ``scan/probe_geometry``
     - 3D coordinates (in meters) of each transducer element, shape [n_el, 3].
   * - ``scan/sampling_frequency``
     - Sampling frequency of the data acquisition (in Hz).
   * - ``scan/center_frequency``
     - Center frequency of the transmit waveform in Hz.
   * - ``scan/demodulation_frequency``
     - The demodulation frequency of the data in Hz. This is the assumed center frequency of the transmit waveform used to demodulate the rf data to iq data.
   * - ``scan/initial_times``
     - Time (in seconds) when the A/D converter starts sampling for each transmit, shape [n_tx].
   * - ``scan/t0_delays``
     - Time delays (in seconds) applied to each element for each transmit, shape [n_tx, n_el].
   * - ``scan/tx_apodizations``
     - Transmit apodization values, shape [n_tx, n_el].
   * - ``scan/focus_distances``
     - Distance from the origin point on the transducer to where the beam comes to focus for each transmit, in meters, shape [n_tx].
   * - ``scan/transmit_origins``
     - 3D coordinates (in meters) of the origin point for each transmit, shape [n_tx, 3].
   * - ``scan/polar_angles``
     - Polar angles of transmit beams in radians, shape [n_tx].
   * - ``scan/azimuth_angles``
     - Azimuthal angles of transmit beams in radians, shape [n_tx].
   * - ``scan/bandwidth_percent``
     - Receive bandwidth as a percentage of center frequency.
   * - ``scan/time_to_next_transmit``
     - Time interval (in seconds) between subsequent transmit events, shape [n_frames, n_tx].
   * - ``scan/tgc_gain_curve``
     - Time-gain-compensation curve, shape [n_ax].
   * - ``scan/element_width``
     - Width of the elements in the probe (meters).
   * - ``scan/tx_waveform_indices``
     - Indices for transmit waveforms, shape [n_tx].
   * - ``scan/waveforms_one_way``
     - List of one-way waveforms (simulated, 250MHz).
   * - ``scan/waveforms_two_way``
     - List of two-way waveforms (simulated, 250MHz).
   * - ``scan/lens_correction``
     - Lens correction parameter (optional).
   * - ``scan/...``
     - Any additional custom parameters.

.. note::

  All datasets in the `scan` group should have `unit` and `description` attributes.
  Custom parameters can be added directly to the `scan` group as needed.

-------------------------------
How to Generate a zea Dataset
-------------------------------

Here is a minimal example of how to generate and save a zea dataset:

.. doctest::

  >>> import numpy as np
  >>> from zea.data.data_format import DatasetElement, generate_zea_dataset

  >>> # Example data (replace with your actual data)
  >>> raw_data = np.random.randn(2, 11, 2048, 128, 1)
  >>> image = np.random.randn(2, 512, 512)
  >>> probe_geometry = np.zeros((128, 3))
  >>> t0_delays = np.zeros((11, 128))
  >>> initial_times = np.zeros((11,))
  >>> sampling_frequency = 40e6
  >>> center_frequency = 7e6

  >>> # Optionally define a custom dataset element
  >>> custom_dataset_element = DatasetElement(
  ...    group_name="scan",
  ...    dataset_name="custom_element",
  ...    data=np.random.rand(10, 10),
  ...    description="custom description",
  ...    unit="m",
  ... )

  >>> # Save the dataset to disk
  >>> generate_zea_dataset(
  ...    "output_file.hdf5",
  ...    raw_data=raw_data,
  ...    image=image,
  ...    probe_geometry=probe_geometry,
  ...    t0_delays=t0_delays,
  ...    initial_times=initial_times,
  ...    sampling_frequency=sampling_frequency,
  ...    center_frequency=center_frequency,
  ...    sound_speed=1540,
  ...    probe_name="generic",
  ...    description="Example dataset",
  ...    additional_elements=[custom_dataset_element],
  ...    overwrite=True,
  ... )

.. testcleanup::

    import os

    os.remove("output_file.hdf5")

For more advanced usage, see :py:func:`zea.data.data_format.generate_zea_dataset`.

-------------------------------
Supported Datasets & Conversion
-------------------------------

The ``zea`` toolbox supports several public and research ultrasound datasets. For each, we provide scripts to download and convert the data into the ``zea`` format for integration with the toolbox. In general any dataset can be converted to the ``zea`` format by following the structure outlined above.

**Supported Datasets:**

- **EchoNet-Dynamic**: Large-scale cardiac ultrasound dataset.
- **EchoNet-LVH**: Large-scale cardiac dataset for left ventricular hypertrophy detection.
- **CAMUS**: Cardiac Acquisitions for Multi-structure Ultrasound Segmentation.
- **PICMUS**: Plane-wave Imaging Challenge in Medical Ultrasound.
- **Custom Datasets**: You can add your own datasets by following the ``zea`` format.

**Conversion Scripts:**

- Scripts are provided in the `zea/data/convert/ <https://github.com/tue-bmd/zea/tree/main/zea/data/convert/>`__ directory to automate downloading and conversion.
- Example usage:

  .. code-block:: shell

      python -m zea.data.convert --dataset "echonet" --src <source_folder> --dst <destination_folder>
      python -m zea.data.convert --dataset "camus" --src <source_folder> --dst <destination_folder>
      python -m zea.data.convert --dataset "picmus" --src <source_folder> --dst <destination_folder>

- These scripts will fetch the raw data, process it, and store it in the standardized ``zea`` format.

-------------------------------
Data Acquisition Platforms
-------------------------------

One can also acquire data using various ultrasound platforms and convert it to the ``zea`` format. Of course this can be done manually, using a similar snippet as above, but we try to provide scripts for popular ultrasound systems to automate this process. Note that this is still a work in progress, and we will add more information in the future.

**Verasonics**

- Record data using your preferred Verasonics script.
- Save entire workspace to a `.mat` file.
- Use the ``--dataset "verasonics"`` flag to convert the MATLAB workspace files to ``zea`` format. More info can be found in the :mod:`zea.data.convert.verasonics` module documentation.
- Example:

  .. code-block:: shell

    python -m zea.data.convert --dataset "verasonics" --src <source_folder> --dst <destination_folder>

**us4us**

- To be added in a future release.
