.. _parameters:

Parameters
=============================

This page provides a comprehensive overview of all configuration parameters available in usbmd.
These parameters are used in the YAML config files to control data loading, preprocessing, model settings, scan parameters, and more.

You can use these configs to run the pipeline, UI, or training scripts.
Configs are written in YAML format and can be loaded, edited, and saved using the usbmd API.

-------------------------------
How to Load and Save a Config
-------------------------------

Here is a minimal example of how to load and save a config file using usbmd:

.. code-block:: python

   from usbmd import Config

   # Load a config from file
   config = Config.from_yaml("configs/config_picmus_rf.yaml")

   # Access parameters
   print(config.model.batch_size)
   config.model.batch_size = 8

   # Save the config back to file
   config.to_yaml("configs/config_picmus_rf_modified.yaml")

-------------------------------
Parameter List
-------------------------------

Below is a hierarchical list of all configuration parameters, grouped by section.
Descriptions are shown for each parameter.

.. contents::
   :local:
   :depth: 2

-------------------------------
Parameters Reference
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Parameter**
     - **Description**
   * - ``data``
     - The data section contains the parameters for the data.
   * - ``data.apodization``
     - The receive apodization to use.
   * - ``data.dataset_folder``
     - The path of the folder to load data files from (relative to the user data root as set in users.yaml)
   * - ``data.dtype``
     - The form of data to load (raw_data, rf_data, iq_data, beamformed_data, envelope_data, image, image_sc)
   * - ``data.dynamic_range``
     - The dynamic range for showing data in db [min, max]
   * - ``data.file_path``
     - The path of the file to load when running the UI (either an absolute path or one relative to the dataset folder)
   * - ``data.frame_no``
     - The frame number to load when running the UI (null, int, 'all')
   * - ``data.input_range``
     - The range of the input data in db (null, [min, max])
   * - ``data.local``
     - true: use local data on this device, false: use data from NAS
   * - ``data.modtype``
     - The modulation type of the data (rf, iq, null)
   * - ``data.output_size``
     - The size of the output data (e.g. the number of pixels in the image)
   * - ``data.subset``
     - ?
   * - ``data.to_dtype``
     - The type of data to convert to (raw_data, aligned_data, beamformed_data, envelope_data, image, image_sc)
   * - ``data.user``
     - The user to use when loading data (null, dict)
   * - ``device``
     - The device to run on ('cpu', 'gpu:0', 'gpu:1', ...)
   * - ``model``
     - The model section contains the parameters for the model.
   * - ``model.batch_size``
     - The number of frames to process in a batch
   * - ``model.beamformer``
     - Settings used to configure the beamformer.
   * - ``model.beamformer.auto_pressure_weighting``
     - True: enables automatic field-based weighting of Tx events in compounding.False: disables automatic field-based weighting of Tx events in compounding.
   * - ``model.beamformer.proxtype``
     - The type of proximal operator to use (null, wavelet, softthres, fourier, neural)
   * - ``model.beamformer.type``
     - The beamforming method to use (das,)
   * - ``model.patch_shape``
     - The shape of the patches to use for training the model. e.g. [8, 8] for 8x8 patches.
   * - ``plot``
     - Settings pertaining to plotting when running the UI (`usbmd --config <path-to-config.yaml>`)
   * - ``plot.fliplr``
     - Set to true to flip the image left to right
   * - ``plot.headless``
     - Set to true to run the UI in headless mode
   * - ``plot.image_extension``
     - The file extension to use when saving the image (png, jpg)
   * - ``plot.plot_lib``
     - The plotting library to use (opencv, matplotlib)
   * - ``plot.save``
     - Set to true to save the plots to disk, false to only display them in the UI
   * - ``plot.tag``
     - The name for the plot
   * - ``plot.video_extension``
     - The file extension to use when saving the video (mp4, gif)
   * - ``postprocess``
     - The postprocess section contains the parameters for the postprocessing.
   * - ``postprocess.bm3d``
     - Settings for the bm3d algorithm.
   * - ``postprocess.bm3d.sigma``
     - The sigma value for the bm3d algorithm
   * - ``postprocess.bm3d.stage``
     - The stage of the bm3d algorithm to use (all_stages, hard_thresholding)
   * - ``postprocess.contrast_boost``
     - Settings for the contrast boost.
   * - ``postprocess.contrast_boost.k_n``
     - The negative contrast boost factor
   * - ``postprocess.contrast_boost.k_p``
     - The positive contrast boost factor
   * - ``postprocess.contrast_boost.threshold``
     - The threshold for the contrast boost
   * - ``postprocess.lista``
     - Set to true to use the lista algorithm
   * - ``postprocess.thresholding``
     - Settings for the thresholding.
   * - ``postprocess.thresholding.below_threshold``
     - Set to true to threshold below the threshold
   * - ``postprocess.thresholding.fill_value``
     - The value to fill the data with when thresholding (min, max, threshold, any_number)
   * - ``postprocess.thresholding.percentile``
     - The percentile to use for thresholding
   * - ``postprocess.thresholding.threshold``
     - The threshold to use for thresholding
   * - ``postprocess.thresholding.threshold_type``
     - The type of thresholding to use (soft, hard)
   * - ``preprocess``
     - The preprocess section contains the parameters for the preprocessing.
   * - ``preprocess.demodulation``
     - The demodulation method to use (manual, hilbert, gabor)
   * - ``preprocess.elevation_compounding``
     - The method to use for elevation compounding (null, int, max, mean)
   * - ``preprocess.multi_bpf``
     - Settings for the multi bandpass filter.
   * - ``preprocess.multi_bpf.bandwidths``
     - The bandwidths of the filter bands
   * - ``preprocess.multi_bpf.freqs``
     - The center frequencies of the filter bands
   * - ``preprocess.multi_bpf.num_taps``
     - The number of taps in the filter
   * - ``preprocess.multi_bpf.units``
     - The units of the frequencies and bandwidths (Hz, kHz, MHz, GHz)
   * - ``scan``
     - The scan section contains the parameters pertaining to the reconstruction.
   * - ``scan.Nx``
     - The number of pixels in the beamforming grid in the x-direction
   * - ``scan.Nz``
     - The number of pixels in the beamforming grid in the z-direction
   * - ``scan.apply_lens_correction``
     - Set to true to apply lens correction in the time-of-flight calculation
   * - ``scan.center_frequency``
     - The center frequency of the transducer in Hz
   * - ``scan.demodulation_frequency``
     - The demodulation frequency of the data in Hz. This is the assumed center frequency of the transmit waveform used to demodulate the rf data to iq data.
   * - ``scan.downsample``
     - The decimation factor to use for downsampling the data from rf to iq. If 1, no downsampling is performed.
   * - ``scan.lens_sound_speed``
     - The speed of sound in the lens in m/s. Usually around 1000 m/s
   * - ``scan.lens_thickness``
     - The thickness of the lens in meters
   * - ``scan.n_ch``
     - The number of channels in the raw data (1 for rf data, 2 for iq data)
   * - ``scan.sampling_frequency``
     - The sampling frequency of the data in Hz
   * - ``scan.selected_transmits``
     - The number of transmits in a frame. Can be 'all' for all transmits, an integer for a specific number of transmits selected evenly from the transmits in the frame, or a list of integers for specific transmits to select from the frame.
   * - ``scan.xlims``
     - The limits of the x-axis in the scan in meters (null, [min, max])
   * - ``scan.ylims``
     - The limits of the y-axis in the scan in meters (null, [min, max])
   * - ``scan.zlims``
     - The limits of the z-axis in the scan in meters (null, [min, max])
