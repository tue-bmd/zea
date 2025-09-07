"""Validate configuration yaml files.

https://github.com/keleshev/schema
https://www.andrewvillazon.com/validate-yaml-python-schema/

This file specifies bare bone structure of the config files.
Furthermore it check the config file you create for validity and sets
missing (if optional) parameters to default values. When adding functionality
that needs parameters from the config file, make sure to add those paremeters here.
Also if that parameter is optional, add a default value.

"""

from pathlib import Path

from schema import And, Optional, Or, Regex, Schema

import zea.metrics  # noqa: F401
from zea.internal.checks import _DATA_TYPES
from zea.internal.registry import metrics_registry

# predefined checks, later used in schema to check validity of parameter
any_number = Or(
    int,
    float,
    error="Must be a number, scientific notation should be of form x.xe+xx, "
    "otherwise interpreted as string",
)
list_of_size_two = And(list, lambda _list: len(_list) == 2)
positive_integer = And(int, lambda i: i > 0)
positive_integer_and_zero = And(int, lambda i: i >= 0)
positive_float = And(float, lambda f: f > 0)
list_of_floats = And(list, lambda _list: all(isinstance(_l, float) for _l in _list))
list_of_positive_integers = And(list, lambda _list: all(_l >= 0 for _l in _list))
percentage = And(any_number, lambda f: 0 <= f <= 100)

_ALLOWED_PLOT_LIBS = ("opencv", "matplotlib")

# pipeline / operations
pipeline_schema = Schema(
    {
        Optional("operations", default=["identity"]): Or(
            None, [Or(str, {"name": str, "params": dict}, {"name": str})]
        ),
        Optional("with_batch_dim", default=True): bool,
        Optional("jit_options", default="ops"): Or(None, "ops", "pipeline"),
        Optional("jit_kwargs", default=None): Or(None, dict),
        Optional("name", default="pipeline"): str,
        Optional("validate", default=True): bool,
    }
)

# postprocess DEPRECATED
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
                Optional("fill_value", default="min"): Or("min", "max", "threshold", any_number),
                Optional("below_threshold", default=True): bool,
                Optional("threshold_type", default="hard"): Or("hard", "soft"),
            },
        ),
        Optional("lista", default=None): Or(bool, None),
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
        Optional("grid_size_x", default=None): Or(None, positive_integer),
        Optional("grid_size_z", default=None): Or(None, positive_integer),
        Optional("n_ch", default=None): Or(None, int),
        Optional("n_ax", default=None): Or(None, int),
        Optional("center_frequency", default=None): Or(None, any_number),
        Optional("sampling_frequency", default=None): Or(None, any_number),
        Optional("demodulation_frequency", default=None): Or(None, any_number),
        Optional("f_number", default=None): Or(None, positive_float),
        Optional("apply_lens_correction", default=False): bool,
        Optional("lens_thickness", default=1e-3): positive_float,
        Optional("lens_sound_speed", default=1000): Or(positive_float, positive_integer),
        Optional("theta_range", default=None): Or(None, list_of_size_two),
        Optional("phi_range", default=None): Or(None, list_of_size_two),
        Optional("rho_range", default=None): Or(None, list_of_size_two),
        Optional("fill_value", default=0.0): any_number,
        Optional("resolution", default=None): Or(None, positive_float),
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
        Optional("selector_metric", default="gcnr"): Or(*metrics_registry.registered_names()),
        Optional("fliplr", default=False): bool,
        Optional("image_extension", default="png"): Or("png", "jpg"),
        Optional("video_extension", default="gif"): Or("mp4", "gif"),
    }
)

data_schema = Schema(
    {
        "dtype": Or(*_DATA_TYPES),
        "dataset_folder": str,
        Optional("resolution", default=None): Or(None, positive_float),
        Optional("to_dtype", default="image"): Or(*_DATA_TYPES),
        Optional("file_path", default=None): Or(None, str, Path),
        Optional("local", default=True): bool,
        Optional("frame_no", default=None): Or(None, "all", int),
        Optional("dynamic_range", default=[-60, 0]): list_of_size_two,
        Optional("input_range", default=None): Or(None, list_of_size_two),
        Optional("output_range", default=None): Or(None, list_of_size_two),
        Optional("apodization", default=None): Or(None, str),
        Optional("user", default=None): Or(None, dict),
    }
)

# top level schema
config_schema = Schema(
    {
        "data": data_schema,
        Optional("plot", default=plot_schema.validate({})): plot_schema,
        Optional("pipeline", default=pipeline_schema.validate({})): pipeline_schema,
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
        Optional("git", default=None): Or(None, str),
    }
)
