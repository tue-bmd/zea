"""Validate configuration yaml files.

https://github.com/keleshev/schema
https://www.andrewvillazon.com/validate-yaml-python-schema/

This file specifies bare bone structure of the config files.
Furthermore it check the config file you create for validity and sets
missing (if optional) parameters to default values. When adding functionality
that needs parameters from the config file, make sure to add those paremeters here.
Also if that parameter is optional, add a default value.

- **Author(s)**     : Tristan Stevens
- **Date**          : 31/01/2023
"""
import importlib
from pathlib import Path
from typing import Union

from schema import And, Optional, Or, Regex, Schema

_ML_LIBRARIES = [None, "torch", "tensorflow"]

# need to import ML libraries first for registry
for lib in _ML_LIBRARIES:
    if importlib.util.find_spec(str(lib)):
        if lib == "torch":
            # pylint: disable=unused-import
            import usbmd.pytorch_ultrasound
        if lib == "tensorflow":
            # pylint: disable=unused-import
            import usbmd.tensorflow_ultrasound

# Register beamforing types in registry
from usbmd.registry import tf_beamformer_registry, torch_beamformer_registry

_BEAMFORMER_TYPES = set(
    tf_beamformer_registry.registered_names()
    + torch_beamformer_registry.registered_names()
)

from usbmd.processing import _DATA_TYPES, _ML_LIBRARIES, _MOD_TYPES
from usbmd.utils.config import Config
from usbmd.utils.metrics import _METRICS

# predefined checks, later used in schema to check validity of parameter
any_number = Or(
    int,
    float,
    error="Must be a number, scientific notation should be of form x.xe+xx, "
    "otherwise interpreted as string",
)
list_of_size_two = And(list, lambda l: len(l) == 2)
positive_integer = And(int, lambda i: i > 0)
list_of_floats = And(list, lambda l: all(isinstance(_l, float) for _l in l))
percentage = And(any_number, lambda f: 0 <= f <= 100)

# optional sub schemas go here, to allow for nested defaults

# model
model_schema = Schema(
    {
        Optional("batch_size", default=8): positive_integer,
        Optional("patch_shape", default=[8, 8]): list_of_size_two,
        Optional("beamformer", default=None): {
            "type": Or(None, *_BEAMFORMER_TYPES),
            Optional("folds", default=1): positive_integer,
            Optional("end_with_prox", default=False): bool,
            Optional("proxtype", default="softthres"): Or(
                None, "wavelet", "softthres", "fourier", "neural"
            ),
            Optional("kernel_size", default=3): positive_integer,
            Optional("aux_inputs", default=None): Or(None, list),
        },
    }
)

# preprocess
preprocess_schema = Schema(
    {
        Optional("elevation_compounding", default=None): Or(int, "max", "mean", None),
        Optional("multi_bpf", default=None): Or(
            None,
            {
                "num_taps": positive_integer,
                "freqs": list_of_floats,
                "bandwidths": list_of_floats,
                # Optional("units", default="Hz"): Or("Hz", "kHz", "MHz", "GHz"),
            },
        ),
        Optional("demodulation", default="manual"): Or("manual", "hilbert", "gabor"),
    }
)

# postprocess
postprocess_schema = Schema(
    {
        Optional("contrast_boost", default=None): Or(
            None,
            {
                "k_p": float,
                "k_n": float,
                "threshold": float,
            },
        ),
        Optional("thresholding", default=None): Or(
            None,
            {
                Optional("percentile", default=None): Or(None, percentage),
                Optional("threshold", default=None): Or(None, any_number),
                Optional("fill_value", default="min"): Or(
                    "min", "max", "threshold", any_number
                ),
                Optional("below_threshold", default=True): bool,
                Optional("threshold_type", default="hard"): "hard",
            },
        ),
        Optional("lista", default=False): bool,
    }
)

# scan
scan_schema = Schema(
    {
        Optional("xlims", default=None): Or(None, list_of_size_two),
        Optional("zlims", default=None): Or(None, list_of_size_two),
        Optional("ylims", default=None): Or(None, list_of_size_two),
        # TODO: n_angles and N_tx are overlapping parameters
        Optional("n_angles", default=None): Or(None, int, list),
        Optional("N_tx", default=None): Or(None, int),
        Optional("Nx", default=None): Or(None, positive_integer),
        Optional("Nz", default=None): Or(None, positive_integer),
        Optional("N_ax", default=None): Or(None, int),
        Optional("fc", default=None): Or(None, any_number),
        Optional("fs", default=None): Or(None, any_number),
        Optional("downsample", default=None): Or(None, positive_integer),
    }
)

# top level schema
config_schema = Schema(
    {
        "data": {
            "dataset_name": str,
            "dtype": Or(*_DATA_TYPES),
            Optional("output_size", default=500): positive_integer,
            Optional("to_dtype", default="image"): Or(*_DATA_TYPES),
            Optional("file_path", default=None): Or(None, str, Path),
            Optional("local", default=True): bool,
            Optional("subset", default=None): Or(None, str),
            Optional("frame_no", default=None): Or(None, "all", int),
            Optional("dynamic_range", default=[-60, 0]): list_of_size_two,
            Optional("input_range", default=None): Or(None, list_of_size_two),
            Optional("apodization", default=None): Or(None, str),
            Optional("modtype", default=None): Or(*_MOD_TYPES),
            Optional("from_modtype", default=None): Or(*_MOD_TYPES),
            Optional("user", default=None): Or(None, dict),
            Optional("dataset_folder", default=None): Or(None, str),
        },
        "plot": {
            "save": bool,
            "axis": bool,
            Optional("fps", default=20): int,
            Optional("tag", default=None): Or(None, str),
            Optional("headless", default=False): bool,
            Optional("selector", default=None): Or(None, "rectangle", "lasso"),
            Optional("selector_metric", default="gcnr"): Or(*_METRICS),
        },
        Optional("model", default=model_schema.validate({})): model_schema,
        Optional(
            "preprocess", default=preprocess_schema.validate({})
        ): preprocess_schema,
        Optional(
            "postprocess", default=postprocess_schema.validate({})
        ): postprocess_schema,
        Optional("scan", default=scan_schema.validate({})): scan_schema,
        Optional("device", default="auto:1"): Or(
            "cpu",
            "gpu",
            "cuda",
            Regex(r"cuda:\d+"),
            Regex(r"gpu:\d+"),
            Regex(r"auto:\d+"),
            None,
        ),
        Optional("ml_library", default=None): Or(None, *_ML_LIBRARIES, "disable"),
        Optional("git", default=None): Or(None, str),
    }
)


def check_config(config: Union[dict, Config], verbose: bool = False):
    """Check a config given dictionary"""
    assert type(config) in [
        dict,
        Config,
    ], f"Config must be a dictionary or Config object, not {type(config)}"
    if isinstance(config, Config):
        config = config.serialize()
        config = config_schema.validate(config)
        config = Config(config)
    else:
        config = config_schema.validate(config)
    if verbose:
        print("Config is correct")
    return config
