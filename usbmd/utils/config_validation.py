"""Validate configuration yaml files
Author(s): Tristan Stevens
Date: 31/01/2023
https://www.andrewvillazon.com/validate-yaml-python-schema/
"""
from schema import And, Optional, Or, Schema

from usbmd.processing import _BEAMFORMER_TYPES, _DATA_TYPES, _MOD_TYPES, _ML_LIBRARIES

list_of_size_two = And(list, lambda l: len(l) == 2)
positive_integer = And(int, lambda i: i > 0)

config_schema = Schema({
    "data": {
        "dataset_name": str,
        "dtype": Or(*_DATA_TYPES),
        Optional("output_size"): positive_integer,
        Optional("to_dtype"): Or(*_DATA_TYPES),
        Optional("file_path"): Or(None, str),
        Optional("local"): bool,
        Optional("subset"): Or(None, str),
        Optional("frame_no"): Or(None, "all", int),
        Optional("dynamic_range"): list_of_size_two,
        Optional("input_range"): list_of_size_two,
        Optional("apodization"): Or(None, str),
        Optional("modtype"): Or(*_MOD_TYPES),
        Optional("from_modtype"): Or(*_MOD_TYPES),
        Optional("downsample"): positive_integer,
        Optional("n_angles"): Or(None, int, list),
    },
    "plot": {
        "save": bool,
        "axis": bool,
        Optional("fps"): int,
    },
    Optional("scan"): {
        "xlims": list_of_size_two,
        "zlims": list_of_size_two,
    },
    Optional("model"): {
        Optional("batch_size"): positive_integer,
        Optional("beamformer"): {
            "type": Or(None, *_BEAMFORMER_TYPES),
        },
    },
    Optional("preprocess"): {
        Optional("elevation_compounding"): Or(int, "max", "mean"),
        Optional("multi_bpf"): bool,
    },
    Optional("device"): Or("cpu", "gpu"),
    Optional("ml_library"): Or(None, *_ML_LIBRARIES),
})

def check_config(config: dict):
    """Check a config given dictionary"""
    config_schema.validate(dict(config))
    print('Config is correct')
