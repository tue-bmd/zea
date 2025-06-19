"""Parameter descriptions for the config file."""

from zea.internal.config.validation import _ALLOWED_PLOT_LIBS, _DATA_TYPES


def allows_type_to_str(allowed_types):
    """Transforms a list of allowed types into a string for use in a comment."""
    ouput_str = ", ".join([str(a) if a is not None else "null" for a in allowed_types])
    return ouput_str


PARAMETER_DESCRIPTIONS = {
    "data": {
        "description": "The data section contains the parameters for the data.",
        "dataset_folder": (
            "The path of the folder to load data files from (relative to the user data "
            "root as set in users.yaml)"
        ),
        "to_dtype": (f"The type of data to convert to ({allows_type_to_str(_DATA_TYPES)})"),
        "file_path": (
            "The path of the file to load when running the UI (either an absolute path "
            "or one relative to the dataset folder)"
        ),
        "frame_no": "The frame number to load when running the UI (null, int, 'all')",
        "input_range": "The range of the input data in db (null, [min, max])",
        "apodization": "The receive apodization to use.",
        "output_range": ("The output range to which the data should be mapped (e.g. [0, 1])."),
        "resolution": ("The spatial resolution of the data in meters per pixel (float, optional)."),
        "local": "true: use local data on this device, false: use data from NAS",
        "dtype": (
            "The form of data to load (raw_data, rf_data, iq_data, beamformed_data, "
            "envelope_data, image, image_sc)"
        ),
        "dynamic_range": "The dynamic range for showing data in db [min, max]",
        "user": "The user to use when loading data (null, dict)",
    },
    "scan": {
        "description": (
            "The scan section contains the parameters pertaining to the reconstruction."
        ),
        "selected_transmits": (
            "The number of transmits in a frame. Can be 'all' for all transmits, an "
            "integer for a specific number of transmits selected evenly from the "
            "transmits in the frame, or a list of integers for specific transmits to "
            "select from the frame."
        ),
        "Nx": "The number of pixels in the beamforming grid in the x-direction",
        "Nz": "The number of pixels in the beamforming grid in the z-direction",
        "n_ch": "The number of channels in the raw data (1 for rf data, 2 for iq data)",
        "n_ax": "The number of samples in a receive recording per channel.",
        "xlims": "The limits of the x-axis in the scan in meters (null, [min, max])",
        "ylims": "The limits of the y-axis in the scan in meters (null, [min, max])",
        "zlims": "The limits of the z-axis in the scan in meters (null, [min, max])",
        "center_frequency": "The center frequency of the transducer in Hz",
        "sampling_frequency": "The sampling frequency of the data in Hz",
        "demodulation_frequency": (
            "The demodulation frequency of the data in Hz. This is the assumed center "
            "frequency of the transmit waveform used to demodulate the rf data to iq "
            "data."
        ),
        "apply_lens_correction": (
            "Set to true to apply lens correction in the time-of-flight calculation"
        ),
        "lens_thickness": "The thickness of the lens in meters",
        "lens_sound_speed": ("The speed of sound in the lens in m/s. Usually around 1000 m/s"),
        "f_number": (
            "The receive f-number for apodization. Set to zero to disable masking. "
            "The f-number is the ratio between the distance from the transducer and the "
            "size of the aperture."
        ),
        "fill_value": (
            "Value to fill the image with outside the defined region (float, default 0.0)."
        ),
        "phi_range": (
            "The range of phi values in radians for 3D scan conversion (null, [min, max])."
        ),
        "theta_range": (
            "The range of theta values in radians for scan conversion (null, [min, max])."
        ),
        "rho_range": ("The range of rho values in meters for scan conversion (null, [min, max])."),
        "resolution": ("The resolution for scan conversion in meters per pixel (float, optional)."),
    },
    "pipeline": {
        "description": "This section contains the necessary parameters for building the pipeline.",
        "operations": (
            "The operations to perform on the data. This is a list of dictionaries, "
            "where each dictionary contains the parameters for a single operation."
        ),
        "with_batch_dim": (
            "Whether operations should expect a batch dimension in the input. Defaults to True."
        ),
        "jit_options": (
            "The JIT options to use. Must be 'pipeline', 'ops', or None. "
            "'pipeline' compiles the entire pipeline as a single function. "
            "'ops' compiles each operation separately. None disables JIT compilation. "
            "Defaults to 'ops'."
        ),
        "jit_kwargs": ("Additional keyword arguments for the JIT compiler. Defaults to None."),
        "name": ("The name of the pipeline. Defaults to 'pipeline'."),
        "validate": ("Whether to validate the pipeline. Defaults to True."),
    },
    "device": "The device to run on ('cpu', 'gpu:0', 'gpu:1', ...)",
    "plot": {
        "description": (
            "Settings pertaining to plotting when running the UI "
            "(`zea --config <path-to-config.yaml>`)"
        ),
        "save": ("Set to true to save the plots to disk, false to only display them in the UI"),
        "plot_lib": (f"The plotting library to use ({allows_type_to_str(_ALLOWED_PLOT_LIBS)})"),
        "fps": "Frames per second for video output.",
        "tag": "The name for the plot",
        "fliplr": "Set to true to flip the image left to right",
        "image_extension": "The file extension to use when saving the image (png, jpg)",
        "video_extension": "The file extension to use when saving the video (mp4, gif)",
        "headless": "Set to true to run the UI in headless mode",
        "selector": (
            "Type of selector to use for ROI selection in the UI ('rectangle', 'lasso', or None)."
        ),
        "selector_metric": ("Metric to use for evaluating selected regions (e.g., 'gcnr')."),
    },
    "git": "The git commit hash or branch for reproducibility (string, optional).",
    "hide_devices": ("List of device indices to hide from selection (list of int, optional)."),
}
