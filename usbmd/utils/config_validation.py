"""Validate configuration yaml files
Author(s): Tristan Stevens
Date: 31/01/2023
https://github.com/keleshev/schema
https://www.andrewvillazon.com/validate-yaml-python-schema/

This file specifies bare bone structure of the config files.
Furthermore it check the config file you create for validity and sets
missing (if optional) parameters to default values. When adding functionality
that needs parameters from the config file, make sure to add those paremeters here.
Also if that parameter is optional, add a default value.
"""
from schema import And, Optional, Or, Regex, Schema

from usbmd.processing import (_BEAMFORMER_TYPES, _DATA_TYPES, _ML_LIBRARIES,
                              _MOD_TYPES)

# predefined checks, later used in schema to check validity of parameter
list_of_size_two = And(list, lambda l: len(l) == 2)
positive_integer = And(int, lambda i: i > 0)

# optional sub schemas go here to allow for nested defaults
model_schema = Schema({
    Optional("batch_size", default=8): positive_integer,
    Optional("beamformer", default=None): {
        "type": Or(None, *_BEAMFORMER_TYPES),
    }
})

preprocess_schema = Schema({
    Optional("elevation_compounding", default="max"): Or(int, "max", "mean"),
    Optional("multi_bpf", default=False): bool,
})

postprocess_schema = Schema({
    Optional("contrast_boost", default=None): {
        "k_p": float,
        "k_n": float,
        "threshold": float,
        },
    Optional("lista", default=None): bool,
})

scan_schema = Schema({
    Optional("xlims", default=None): list_of_size_two,
    Optional("zlims", default=None): list_of_size_two,
    Optional("n_angles", default=None): Or(None, int, list),
    Optional("Nx", default=None): Or(None, int),
    Optional("Nz", default=None): Or(None, int),
})

# top level schema
config_schema = Schema({
    "data": {
        "dataset_name": str,
        "dtype": Or(*_DATA_TYPES),
        Optional("output_size", default=500): positive_integer,
        Optional("to_dtype", default="image"): Or(*_DATA_TYPES),
        Optional("file_path", default=None): Or(None, str),
        Optional("local", default=True): bool,
        Optional("subset", default=None): Or(None, str),
        Optional("frame_no", default=None): Or(None, "all", int),
        Optional("dynamic_range", default=[-60, 0]): list_of_size_two,
        Optional("input_range", default=None): Or(None, list_of_size_two),
        Optional("apodization", default=None): Or(None, str),
        Optional("modtype", default=None): Or(*_MOD_TYPES),
        Optional("from_modtype", default=None): Or(*_MOD_TYPES),
        Optional("downsample", default=None): positive_integer,
    },
    "plot": {
        "save": bool,
        "axis": bool,
        Optional("fps", default=20): int,
        Optional("tag", default=None): str,
        Optional("headless", default=None): bool,
    },
    Optional("model", default=model_schema.validate({})): model_schema,
    Optional("preprocess", default=preprocess_schema.validate({})): preprocess_schema,
    Optional("postprocess", default=postprocess_schema.validate({})): postprocess_schema,
    Optional("scan", default=scan_schema.validate({})): scan_schema,

    Optional("device", default="cpu"): \
        Or("cpu", "gpu", "cuda", Regex(r"cuda:\d+"), Regex(r"gpu:\d+")),
    Optional("ml_library", default=None): Or(None, *_ML_LIBRARIES, 'disable'),
})

def check_config(config: dict):
    """Check a config given dictionary"""
    config = config_schema.validate(dict(config))
    print('Config is correct')
    return config
