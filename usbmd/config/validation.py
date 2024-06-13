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

from usbmd import Config
from usbmd.registry import metrics_registry
from usbmd.utils import log
from usbmd.utils.checks import _DATA_TYPES, _ML_LIBRARIES, _MOD_TYPES

_ML_LIBRARIES = [None, "torch", "tensorflow", "numpy"]

# need to import ML libraries first for registry
_ML_LIB_SET = False
for lib in _ML_LIBRARIES:
    if importlib.util.find_spec(str(lib)):
        if lib == "torch":
            # pylint: disable=unused-import
            import usbmd.backend.pytorch

            _ML_LIB_SET = True
        if lib == "tensorflow":
            # pylint: disable=unused-import
            import usbmd.backend.tensorflow

            _ML_LIB_SET = True

# pylint: disable=unused-import
import usbmd.utils.metrics

# Register beamforing types in registry
from usbmd.registry import tf_beamformer_registry, torch_beamformer_registry

_BEAMFORMER_TYPES = set(
    tf_beamformer_registry.registered_names()
    + torch_beamformer_registry.registered_names()
)


# predefined checks, later used in schema to check validity of parameter
any_number = Or(
    int,
    float,
    error="Must be a number, scientific notation should be of form x.xe+xx, "
    "otherwise interpreted as string",
)
list_of_size_two = And(list, lambda l: len(l) == 2)
positive_integer = And(int, lambda i: i > 0)
positive_integer_and_zero = And(int, lambda i: i >= 0)
positive_float = And(float, lambda f: f > 0)
list_of_floats = And(list, lambda l: all(isinstance(_l, float) for _l in l))
list_of_positive_integers = And(list, lambda l: all(_l >= 0 for _l in l))
percentage = And(any_number, lambda f: 0 <= f <= 100)

_ALLOWED_KEYS_PROXTYPE = (None, "wavelet", "softthres", "fourier", "neural")
_ALLOWED_DEMODULATION = ("manual", "hilbert", "gabor")
_ALLOWED_PLOT_LIBS = ("opencv", "matplotlib")

# optional sub schemas go here, to allow for nested defaults

# model
model_schema = Schema(
    {
        Optional("batch_size", default=1): positive_integer,
        Optional("patch_shape", default=None): Or(None, list_of_size_two),
        Optional("beamformer", default={}): {
            Optional("type", default=None): Or(None, *_BEAMFORMER_TYPES),
            Optional("folds", default=1): positive_integer,
            Optional("end_with_prox", default=False): bool,
            Optional("proxtype", default="softthres"): Or(*_ALLOWED_KEYS_PROXTYPE),
            Optional("kernel_size", default=3): positive_integer,
            Optional("aux_inputs", default=None): Or(None, list),
            Optional("patches", default=1): positive_integer,
            Optional("jit", default=False): bool,
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
                Optional("units", default="Hz"): Or("Hz", "kHz", "MHz", "GHz"),
            },
        ),
        Optional("demodulation", default="manual"): Or(*_ALLOWED_DEMODULATION),
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
                Optional("threshold_type", default="hard"): Or("hard", "soft"),
            },
        ),
        Optional("lista", default=None): Or(bool, None),
        Optional("bm3d", default=None): Or(
            None,
            {
                Optional("sigma", default=0.1): positive_float,
                Optional("stage", default="all_stages"): Or(
                    "all_stages", "hard_thresholding"
                ),
            },
        ),
    }
)

# scan
scan_schema = Schema(
    {
        Optional("xlims", default=None): Or(None, list_of_size_two),
        Optional("zlims", default=None): Or(None, list_of_size_two),
        Optional("ylims", default=None): Or(None, list_of_size_two),
        Optional("selected_transmits", default=None): Or(
            None,
            positive_integer,
            list_of_positive_integers,
            "all",
            "center",
        ),
        Optional("Nx", default=None): Or(None, positive_integer),
        Optional("Nz", default=None): Or(None, positive_integer),
        Optional("n_ch", default=None): Or(None, int),
        Optional("n_ax", default=None): Or(None, int),
        Optional("center_frequency", default=None): Or(None, any_number),
        Optional("sampling_frequency", default=None): Or(None, any_number),
        Optional("demodulation_frequency", default=None): Or(None, any_number),
        Optional("downsample", default=None): Or(None, positive_integer),
        Optional("f_number", default=None): Or(None, positive_float),
    }
)

# plot
plot_schema = Schema(
    {
        Optional("save", default=False): bool,
        Optional("plot_lib", default="opencv"): Or(*_ALLOWED_PLOT_LIBS),
        Optional("fps", default=20): int,
        Optional("tag", default=None): Or(None, str),
        Optional("headless", default=False): bool,
        Optional("selector", default=None): Or(None, "rectangle", "lasso"),
        Optional("selector_metric", default="gcnr"): Or(
            *metrics_registry.registered_names()
        ),
        Optional("fliplr", default=False): bool,
        Optional("image_extension", default="png"): Or("png", "jpg"),
        Optional("video_extension", default="gif"): Or("mp4", "gif"),
    }
)

data_schema = Schema(
    {
        "dtype": Or(*_DATA_TYPES),
        "dataset_folder": str,
        Optional("dataset_name", default="usbmd"): str,
        Optional("output_size", default=500): positive_integer,
        Optional("to_dtype", default="image"): Or(*_DATA_TYPES),
        Optional("file_path", default=None): Or(None, str, Path),
        Optional("local", default=True): bool,
        Optional("subset", default=None): Or(None, str),
        Optional("frame_no", default=None): Or(None, "all", int),
        Optional("dynamic_range", default=[-60, 0]): list_of_size_two,
        Optional("input_range", default=None): Or(None, list_of_size_two),
        Optional("apodization", default=None): Or(None, str),
        Optional("modtype", default=None): Or(*_MOD_TYPES),  # ONLY FOR LEGACY DATASET
        Optional("from_modtype", default=None): Or(
            *_MOD_TYPES
        ),  # ONLY FOR LEGACY DATASET
        Optional("user", default=None): Or(None, dict),
    }
)

# top level schema
config_schema = Schema(
    {
        "data": data_schema,
        Optional("plot", default=plot_schema.validate({})): plot_schema,
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
            Regex(r"auto:-\d+"),
            None,
        ),
        Optional("hide_devices", default=None): Or(
            None, list_of_positive_integers, positive_integer_and_zero
        ),
        Optional("ml_library", default="numpy"): Or(None, *_ML_LIBRARIES),
        Optional("git", default=None): Or(None, str),
    }
)


def check_config(config: Union[dict, Config], verbose: bool = False):
    """Check a config given dictionary"""

    def _try_validate_config(config):
        if not _ML_LIB_SET:
            log.warning(
                "No ML library (i.e. `torch` or `tensorflow` was found or set, "
                "note that some functionality may not be available. "
            )
            if config.get("ml_library") != "numpy":
                log.warning(
                    "Setting `ml_library` to `numpy`. "
                    "Make sure to not use any ml_library specific functionality or parameters."
                )
                config["ml_library"] = "numpy"
        try:
            config = config_schema.validate(config)
            return config
        except Exception as e:
            log.error(f"Config is not valid: {e}")
            raise e

    assert type(config) in [
        dict,
        Config,
    ], f"Config must be a dictionary or Config object, not {type(config)}"
    if isinstance(config, Config):
        config = config.serialize()
        config = _try_validate_config(config)
        config = Config(config)
        config.freeze()  # freeze because schema will add all defaults
    else:
        config = _try_validate_config(config)
    if verbose:
        log.success("Config is correct")
    return config
