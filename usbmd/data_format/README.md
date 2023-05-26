The USBMD data format has the following structure:

- The dataset always contains a `data` and a `settings` group.
- The `data` group contains the actual data.
- The `settings` group contains all the settings that are necessary to interpret the data.
- The `data` group contains one or more of the following keys:
    - `raw_data`: The raw data of the ultrasound measurement. This can be
    either rf-data or iq-data.
    - `aligned_data`: The time-of-flight corrected data. This can be either
    rf-data or iq-data.
    - `envelope_data`: The envelope of the aligned data.
    - `beamformed_data`: The data after beamforming.
    - `image`: The processed image.
    - `image_sc`: The scan converted image.
- The `scan` group must contain the following keys:
    - `probe_geometry`: The geometry of the probe.
    - `t0_delays`: The time delays of the transducer elements.
    - `n_frames`: The number of frames in the dataset.
    - `n_tx`: The number of transmits per frame.
    - `n_el`: The number of elements in the transducer.
    - `n_ax`: The number of axial samples per transmit.
    - `n_ch`: The number of rf/iq channels in the data (either 1 or 2).
    - `sampling_frequency`: The sampling frequency of the data.
    - `center_frequency`: The center frequency of the transducer.
    - `initial_times`: The times when the A/D converter starts sampling in seconds of shape (n_tx,). This is the time between the first element firing and the first recorded sample.

## Viewing a dataset
To view what is inside an existing dataset you can use the `usbmd.utils.utils.print_hdf5_attrs` function, which prints all the keys, shapes, and attributes of a dataset. Alternatively you can use the tool [HDFView](https://www.hdfgroup.org/downloads/hdfview/) from the HDF group.

## Saving data
When store data or converting a dataset to USBMD format it is recommended to always use the `usbmd.data_format.usbmd_data_format.generate_usbmd_dataset` function. This ensures all data is stored in exactly the same way and makes it easy to propagate changes to the USBMD format to all datasets.