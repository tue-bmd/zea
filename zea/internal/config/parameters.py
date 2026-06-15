"""Parameter descriptions for the config file."""

PARAMETER_DESCRIPTIONS = {
    "data": {
        "description": "Data path and loading settings.",
        "path": (
            "Full path to the data file. Supports absolute paths, paths relative to "
            "the user data root (set in users.yaml), and Hugging Face Hub paths "
            "(hf://org/repo/path/to/file.hdf5)."
        ),
        "local": "true: use local data on this device, false: use data from NAS",
        "indices": (
            "Indices into the data to load. null loads the default, 'all' loads every frame, "
            "int loads a single frame, list loads specific frames."
        ),
        "user": "User path overrides set automatically by setup_zea (null, dict).",
    },
    "parameters": {
        "description": (
            "Open mapping of scan/probe/custom parameters that overwrite values loaded "
            "from the data file. ProbeSpec and ScanSpec are the authoritative sources "
            "for valid parameter names — see the spec reference in data-acquisition. "
            "Arbitrary custom parameters are forwarded to the pipeline unchanged."
        ),
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
        "jit_kwargs": "Additional keyword arguments for the JIT compiler. Defaults to None.",
        "name": "The name of the pipeline. Defaults to 'pipeline'.",
        "validate": "Whether to validate the pipeline. Defaults to True.",
    },
    "device": "The device to run on ('cpu', 'gpu:0', 'gpu:1', 'auto:1', ...)",
    "git": "The git commit hash or branch for reproducibility (string, optional).",
    "hide_devices": "List of device indices to hide from selection (list of int, optional).",
}
