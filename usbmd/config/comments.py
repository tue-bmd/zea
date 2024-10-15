"""Functionality to automatically add comments to a YAML file based on a descriptions
dictionary. Running this script will add comments to all YAML files in the `configs`
directory. The comments are based on the descriptions dictionary defined in this file.

The updated YAML files will be saved in place and should be manually commited.
"""

import os
import re
from pathlib import Path

from usbmd.config.validation import (
    _ALLOWED_DEMODULATION,
    _ALLOWED_KEYS_PROXTYPE,
    _ALLOWED_PLOT_LIBS,
    _BACKENDS,
    _DATA_TYPES,
    _MOD_TYPES,
)


def allows_type_to_str(allowed_types):
    """Transforms a list of allowed types into a string for use in a comment."""
    ouput_str = ", ".join([str(a) if a is not None else "null" for a in allowed_types])
    return ouput_str


DESCRIPTIONS = {
    "model": {
        "description": "The model section contains the parameters for the model.",
        "batch_size": "The number of frames to process in a batch",
        "patch_shape": (
            "The shape of the patches to use for training the model. e.g. [8, 8] for "
            "8x8 patches."
        ),
        "beamformer": {
            "description": "Settings used to configure the beamformer.",
            "type": "The beamforming method to use (das,)",
            "auto_pressure_weighting": (
                "True: enables automatic field-based weighting of Tx events in compounding."
                "False: disables automatic field-based weighting of Tx events in compounding."
            ),
            "proxtype": (
                "The type of proximal operator to use "
                f"({allows_type_to_str(_ALLOWED_KEYS_PROXTYPE)})"
            ),
        },
    },
    "data": {
        "description": "The data section contains the parameters for the data.",
        "dataset_folder": (
            "The path of the folder to load data files from (relative to the user data "
            "root as set in users.yaml)"
        ),
        "output_size": (
            "The size of the output data (e.g. the number of pixels in the image)"
        ),
        "to_dtype": (
            f"The type of data to convert to ({allows_type_to_str(_DATA_TYPES)})"
        ),
        "file_path": (
            "The path of the file to load when running the UI (either an absolute path "
            "or one relative to the dataset folder)"
        ),
        "subset": "?",
        "frame_no": "The frame number to load when running the UI (null, int, 'all')",
        "input_range": "The range of the input data in db (null, [min, max])",
        "apodization": "The receive apodization to use.",
        "modtype": (
            f"The modulation type of the data ({allows_type_to_str(_MOD_TYPES)})"
        ),
        "local": "true: use local data on this device, false: use data from NAS",
        "dataset_name": (
            "Determines the dataset class that is initialized, most likely will be `usbmd`, else "
            "one of the legacy datasets (picmus, verasonics, abledata, vsms2020, vsms2022, dummy)"
        ),
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
        "downsample": (
            "The decimation factor to use for downsampling the data from rf "
            "to iq. If 1, no downsampling is performed."
        ),
        "Nx": "The number of pixels in the beamforming grid in the x-direction",
        "Nz": "The number of pixels in the beamforming grid in the z-direction",
        "n_ch": "The number of channels in the raw data (1 for rf data, 2 for iq data)",
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
        "lens_sound_speed": (
            "The speed of sound in the lens in m/s. Usually around 1000 m/s"
        ),
    },
    "preprocess": {
        "description": (
            "The preprocess section contains the parameters for the preprocessing."
        ),
        "elevation_compounding": (
            "The method to use for elevation compounding (null, int, max, mean)"
        ),
        "multi_bpf": {
            "description": "Settings for the multi bandpass filter.",
            "num_taps": "The number of taps in the filter",
            "freqs": "The center frequencies of the filter bands",
            "bandwidths": "The bandwidths of the filter bands",
            "units": (
                "The units of the frequencies and bandwidths (Hz, kHz, MHz, GHz)"
            ),
        },
        "demodulation": (
            "The demodulation method to use "
            f"({allows_type_to_str(_ALLOWED_DEMODULATION)})"
        ),
    },
    "postprocess": {
        "description": (
            "The postprocess section contains the parameters for the postprocessing."
        ),
        "contrast_boost": {
            "description": "Settings for the contrast boost.",
            "k_p": "The positive contrast boost factor",
            "k_n": "The negative contrast boost factor",
            "threshold": "The threshold for the contrast boost",
        },
        "thresholding": {
            "description": "Settings for the thresholding.",
            "percentile": "The percentile to use for thresholding",
            "threshold": "The threshold to use for thresholding",
            "fill_value": (
                "The value to fill the data with when thresholding (min, max, threshold, "
                "any_number)"
            ),
            "below_threshold": "Set to true to threshold below the threshold",
            "threshold_type": "The type of thresholding to use (soft, hard)",
        },
        "lista": "Set to true to use the lista algorithm",
        "bm3d": {
            "description": "Settings for the bm3d algorithm.",
            "sigma": "The sigma value for the bm3d algorithm",
            "stage": "The stage of the bm3d algorithm to use (all_stages, hard_thresholding)",
        },
    },
    "device": "The device to run on ('cpu', 'gpu:0', 'gpu:1', ...)",
    "ml_library": f"The library to use ({allows_type_to_str(_BACKENDS)})",
    "plot": {
        "description": (
            "Settings pertaining to plotting when running the UI "
            "(`usbmd --config <path-to-config.yaml>`)"
        ),
        "save": (
            "Set to true to save the plots to disk, false to only display them in the UI"
        ),
        "plot_lib": (
            f"The plotting library to use ({allows_type_to_str(_ALLOWED_PLOT_LIBS)})"
        ),
        "tag": "The name for the plot",
        "fliplr": "Set to true to flip the image left to right",
        "image_extension": "The file extension to use when saving the image (png, jpg)",
        "video_extension": "The file extension to use when saving the video (mp4, gif)",
        "headless": "Set to true to run the UI in headless mode",
    },
}


def wrap_string_as_comment(
    input_string, indent_level=0, max_line_length=100, indent_size=2
):
    """Limit the length of lines in a string and adds a comment prefix."""
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
    If no `data` and `plot` keys are found in the top level, the function assumes that
    the YAML file is not in the correct format and does not add comments.

    Args:
        lines (list): List of lines in the YAML file.
        descriptions (dict): Dictionary with descriptions for the keys in the YAML file.
        indent_size (int): Number of spaces per indent level.

    Returns:
        list: List of lines with comments added.
    """
    modified_lines = []
    current_keys = []

    data_found = False
    plot_found = False

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

            if key == "data" and indent_level == 0:
                data_found = True
            if key == "plot" and indent_level == 0:
                plot_found = True

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
    if data_found and plot_found:
        return modified_lines
    else:
        print("data and/or plot key not found. Not adding comments.")
        return lines


def add_comments_to_yaml(file_path, descriptions):
    """Adds comments to a YAML file based on a descriptions dictionary."""
    # Read the original YAML content
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.readlines()

    # Process and add comments to the YAML content
    modified_content = process_yaml_content(content, descriptions)

    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines(modified_content)


def compile_comments_to_markdown_doc(descriptions, indent_level=0):
    """Compiles the descriptions dictionary into a nicely formatted markdown document.
    descriptions are in the form of nested dictionaries, where the keys are the sections
    and the values are either strings or nested dictionaries. The function recursively
    processes the dictionary and compiles it into a markdown document.
    """
    markdown_doc = ""
    indent = " " * 4 * indent_level

    for key, value in descriptions.items():
        if isinstance(value, str):
            markdown_doc += f"{indent}- **{key}**: {value}\n"
        elif isinstance(value, dict):
            markdown_doc += f"{indent}- **{key}**:\n"
            markdown_doc += compile_comments_to_markdown_doc(value, indent_level + 1)

    return markdown_doc


if __name__ == "__main__":
    # Assuming your YAML files are in the configs directory
    config_dir = Path("configs")

    # List all YAML files in the directory
    yaml_files = [
        f for f in os.listdir(config_dir) if f.endswith(".yaml") or f.endswith(".yml")
    ]
    # exclude the probe.yaml file
    yaml_files = [f for f in yaml_files if f != "probes.yaml"]

    # Add comments to each YAML file
    for file_name in yaml_files:
        print(f"Adding comments to {config_dir/file_name}")
        add_comments_to_yaml(config_dir / file_name, DESCRIPTIONS)

    # Compile the descriptions into a markdown document
    markdown_doc = compile_comments_to_markdown_doc(DESCRIPTIONS)
    markdown_doc_path = config_dir / "config_doc.md"
    with open(markdown_doc_path, "w", encoding="utf-8") as file:
        file.write(markdown_doc)
    print(f"Markdown document updated and saved to {markdown_doc_path}")
