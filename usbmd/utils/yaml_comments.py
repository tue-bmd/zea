import os
import re

import yaml

from usbmd.utils.config_validation import (
    _ALLOWED_KEYS_PROXTYPE,
    _BEAMFORMER_TYPES,
    _DATA_TYPES,
    _ML_LIBRARIES,
    _MOD_TYPES,
)


def allows_type_to_str(allowed_types):
    ouput_str = ", ".join([str(a) if a is not None else "null" for a in allowed_types])
    if len(allowed_types) > 1:
        ouput_str = f"{ouput_str[:-2]}"
    return ouput_str


descriptions = {
    "model": {
        "description": "The model section contains the parameters for the model.",
        "batch_size": "The number of frames to process in a batch",
        "patch_shape": "The shape of the patches to use for training the model. e.g. [8, 8] for 8x8 patches.",
        "beamformer": {
            "description": "Settings used to configure the beamformer.",
            "type": f"The beamforming method to use ({allows_type_to_str(_BEAMFORMER_TYPES)})",
            "proxtype": f"The type of proximal operator to use ({allows_type_to_str(_ALLOWED_KEYS_PROXTYPE)})",
        },
    },
    #     "dataset_name": str,
    # "dtype": Or(*_DATA_TYPES),
    # Optional("output_size", default=500): positive_integer,
    # Optional("to_dtype", default="image"): Or(*_DATA_TYPES),
    # Optional("file_path", default=None): Or(None, str, Path),
    # Optional("local", default=True): bool,
    # Optional("subset", default=None): Or(None, str),
    # Optional("frame_no", default=None): Or(None, "all", int),
    # Optional("dynamic_range", default=[-60, 0]): list_of_size_two,
    # Optional("input_range", default=None): Or(None, list_of_size_two),
    # Optional("apodization", default=None): Or(None, str),
    # Optional("modtype", default=None): Or(
    #     *_MOD_TYPES
    # ),  # ONLY FOR LEGACY DATASET
    # Optional("from_modtype", default=None): Or(
    #     *_MOD_TYPES
    # ),  # ONLY FOR LEGACY DATASET
    # Optional("user", default=None): Or(None, dict),
    # Optional("dataset_folder", default=None): Or(None, str),
    "data": {
        "description": "The data section contains the parameters for the data.",
        "dataset_folder": "The path of the folder to load data files from (relative to the user data root as set in users.yaml)",
        "output_size": "The size of the output data (e.g. the number of pixels in the image)",
        "to_dtype": f"The type of data to convert to ({allows_type_to_str(_DATA_TYPES)})",
        "file_path": "The path of the file to load when running the UI (either an absolute path or one relative to the dataset folder)",
        "subset": "?",
        "frame_no": "The frame number to load when running the UI (null, int, 'all')",
        "input_range": "The range of the input data in db (null, [min, max])",
        "apodization": "The receive apodization to use.",
        "modtype": f"The modulation type of the data ({allows_type_to_str(_MOD_TYPES)})",
        "local": "true: use local data on this device, false: use data from NAS",
        "dataset_name": "Determines the dataset class that is initialized (usbmd, picmus, verasonics, abledata, vsms2020, vsms2022, dummy)",
        "dtype": "The form of data to load (raw_data, rf_data, iq_data, beamformed_data, envelope_data)",
        "dynamic_range": "The dynamic range for showing data in db [min, max]",
        "user": "The user to use when loading data (null, dict)",
    },
    "scan": {
        # The number of transmits in a frame
        "selected_transmits": "The number of transmits in a frame. Can be 'all' for all transmits, an integer for a specific number of transmits selected evenly from the transmits in the frame, or a list of integers for specific transmits to select from the frame.",
        "downsample": "The decimation factor to use for downsampling the data from rf to iq. If 1, no downsampling is performed.",
        "Nx": "The number of pixels in the beamforming grid in the x-direction",
        "Nz": "The number of pixels in the beamforming grid in the z-direction",
        "n_ch": "The number of channels in the raw data (1 for rf data, 2 for iq data)",
        "xlims": "The limits of the x-axis in the scan in meters (null, [min, max])",
        "ylims": "The limits of the y-axis in the scan in meters (null, [min, max])",
        "zlims": "The limits of the z-axis in the scan in meters (null, [min, max])",
        "center_frequency": "The center frequency of the transducer in Hz",
        "sampling_frequency": "The sampling frequency of the data in Hz",
        "demodulation_frequency": "The demodulation frequency of the data in Hz",
    },
    "preprocess": {
        "description": "The preprocess section contains the parameters for the preprocessing.",
        "demodulation": "",
    },
    "device": "The device to run on ('cpu', 'gpu:0', 'gpu:1', ...)",
    "ml_library": f"The library to use ({allows_type_to_str(_ML_LIBRARIES)})",
    "plot": {
        "description": "Settings pertaining to plotting when running the UI (`usbmd/ui.py`)",
        "save": "Set to true to save the plots to disk, false to only display them in the UI",
        "plot_lib": "The plotting library to use ('matplotlib', 'plotly')",
        "tag": "test",
        "fliplr": "Set to true to flip the image left to right",
    },
}


def wrap_string_as_comment(
    input_string, indent_level=0, max_line_length=100, indent_size=2
):
    # Calculate the prefix (indent + comment symbol)
    indent = " " * (indent_level * indent_size)  # Assuming 4 spaces per indent level
    prefix = indent + "# "
    prefix_length = len(prefix)

    # Ensure max line length is at least larger than the prefix
    if max_line_length <= prefix_length:
        raise ValueError(
            "max_line_length must be greater than the length of the prefix"
        )

    # Break the string into words
    words = input_string.split()

    # Prepare the result list to hold lines
    result_lines = []
    current_line = prefix

    for word in words:
        # Check if adding the next word exceeds the max line length
        if len(current_line) + len(word) + 1 > max_line_length:
            # Add the current line to the result and start a new one
            result_lines.append(current_line)
            current_line = prefix + word
        else:
            # If this is not the first word in the line, add a space
            if current_line != prefix:
                current_line += " "
            current_line += word

    # Add the last line if it's not empty
    if current_line:
        result_lines.append(current_line)

    return "\n".join(result_lines) + "\n"


def process_yaml_content(lines, descriptions, indent_size=2):
    """
    Recursive function to process YAML content line by line and add comments.
    """
    modified_lines = []
    current_keys = []

    # Go through all the lines. If the line contains a key, try to look for its description
    # and add it. If the description is not found, add no comment.
    for line in lines:
        # Continue if it's a comment
        if re.match(r"^\s*#", line):
            continue
        # If it matches a key with colon
        if re.match(r"^\s*\w+\s*:", line):
            indent_level = len(re.match(r"^\s*", line).group(0)) // indent_size
            current_keys = current_keys[:indent_level]
            key = line.split(":")[0].strip()
            current_keys.append(key)

            # Recursively index into the descriptions dictionary to find the description
            description = descriptions

            try:
                for key in current_keys:
                    description = description[key]

                if isinstance(description, dict):
                    description = description["description"]
            except KeyError:
                description = "-"

            comment_lines = wrap_string_as_comment(
                description, indent_level, max_line_length=80, indent_size=indent_size
            )

            # Add the description as a comment
            modified_lines.append(comment_lines)
            modified_lines.append(line)
        else:
            modified_lines.append(line)
    return modified_lines


def add_comments_to_yaml(file_path, descriptions):
    # Read the original YAML content
    with open(file_path, "r") as file:
        content = file.readlines()

    # Process and add comments to the YAML content
    modified_content = process_yaml_content(content, descriptions)

    with open("modified.yaml", "w") as file:
        file.writelines(modified_content)


# Assuming your YAML files are in the current directory
yaml_files = [f for f in os.listdir(".") if f.endswith(".yaml") or f.endswith(".yml")]
yaml_files = [r"configs/config_usbmd_cirs.yaml"]

# Add comments to each YAML file
for file_name in yaml_files:
    add_comments_to_yaml(file_name, descriptions)
