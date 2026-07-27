"""Loading and saving of zea model presets.

Mostly from ``keras_hub.src.utils.preset_utils``, trimmed to what zea needs and
routed through the zea model registry. The Hugging Face plumbing (handle parsing,
authentication, downloading) is shared with the data stack and lives in
:mod:`zea.internal.preset_utils`.
"""

import collections
import datetime
import inspect
import json
import os
from pathlib import Path

import keras
from huggingface_hub.utils import EntryNotFoundError, HFValidationError

import zea
import zea.models.base
from zea.internal.preset_utils import (
    HF_MODELS_DIR,
    HF_PREFIX,  # noqa: F401 — re-exported, this module used to define it
    HF_SCHEME,
    _hf_download,
    _hf_parse_path,
)
from zea.internal.registry import model_registry

ASSET_DIR = "assets"

# Config file names.
CONFIG_FILE = "config.json"
IMAGE_CONVERTER_CONFIG_FILE = "image_converter.json"
PREPROCESSOR_CONFIG_FILE = "preprocessor.json"
METADATA_FILE = "metadata.json"

# Weight file names.
MODEL_WEIGHTS_FILE = "model.weights.h5"

# HuggingFace filenames.
README_FILE = "README.md"
HF_CONFIG_FILE = "config.json"

# Global state for preset registry.
BUILTIN_PRESETS = {}
BUILTIN_PRESETS_FOR_MODEL = collections.defaultdict(dict)


def register_presets(presets, model_cls):
    """Register built-in presets for a set of classes.

    Note that this is intended only for models and presets shipped in the
    library itself.
    """
    for preset in presets:
        BUILTIN_PRESETS[preset] = presets[preset]
        BUILTIN_PRESETS_FOR_MODEL[model_cls][preset] = presets[preset]


def builtin_presets(cls):
    """Find all registered built-in presets for a class."""
    presets = {}
    if cls in BUILTIN_PRESETS_FOR_MODEL:
        presets.update(BUILTIN_PRESETS_FOR_MODEL[cls])
    return presets


def _resolve_builtin_preset(preset):
    """Resolve a built-in preset identifier to its handle or local path."""
    entry = BUILTIN_PRESETS.get(preset)
    if entry is None:
        return preset
    return entry["hf_handle"] if "hf_handle" in entry else entry["path"]


def _get_hf_file(preset, path):
    """Download ``path`` from a ``hf://`` preset and return the local path."""
    repo_id, subpath = _hf_parse_path(preset)
    filename = f"{subpath}/{path}" if subpath else path
    try:
        return _hf_download(repo_id, filename, cache_dir=HF_MODELS_DIR, repo_type="model")
    except HFValidationError as e:
        raise ValueError(
            "Unexpected Hugging Face preset. Hugging Face model handles "
            "should have the form 'hf://{org}/{model}'. For example, "
            f"'hf://username/bert_base_en'. Received: preset={preset}."
        ) from e
    except EntryNotFoundError as e:
        message = str(e)
        if message.find("403 Client Error"):
            raise FileNotFoundError(
                f"`{path}` doesn't exist in preset directory `{preset}`."
            ) from e
        raise ValueError(message) from e


def _get_local_file(preset, path):
    """Return the local path of ``path`` inside a local preset directory."""
    preset_dir = Path(preset).resolve()
    local_path = (preset_dir / path).resolve()
    # Guard against a `path` that escapes the preset directory (keras-hub does the
    # same for its cache directory).
    if preset_dir != local_path and preset_dir not in local_path.parents:
        raise ValueError(f"Invalid path: '{path}'. It escapes the preset directory.")
    if not local_path.exists():
        raise FileNotFoundError(f"`{path}` doesn't exist in preset directory `{preset}`.")
    return str(Path(preset) / path)


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    if not isinstance(preset, str):
        raise ValueError(f"A preset identifier must be a string. Received: preset={preset}")

    preset = _resolve_builtin_preset(preset)

    scheme = None
    if "://" in preset:
        scheme = preset.split("://")[0].lower()

    if scheme == HF_SCHEME:
        return _get_hf_file(preset, path)
    elif Path(preset).exists():
        # Assume a local filepath
        return _get_local_file(preset, path)
    else:
        raise ValueError(
            "Unknown preset identifier. A preset must be a one of:\n"
            "1) a built-in preset identifier like `'taesdxl'`\n"
            "2) a Hugging Face handle like `'hf://zea/taesdxl'`\n"
            "3) a path to a local preset directory like `'./taesdxl`\n"
            "Use `print(cls.presets.keys())` to view all built-in presets for "
            "API symbol `cls`.\n"
            f"Received: preset='{preset}'"
        )


def load_json(preset, config_file=CONFIG_FILE):
    """Load a JSON file from a preset."""
    config_path = get_file(preset, config_file)
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def load_serialized_object(config, cls, **kwargs):
    """Load a serialized Keras object from a config."""
    # `dtype` in config might be a serialized `DTypePolicy` or `DTypePolicyMap`.
    # Ensure that `dtype` is properly configured.
    dtype = kwargs.pop("dtype", None)
    config = set_dtype_in_config(config, dtype)

    config["config"] = {**config["config"], **kwargs}
    # return keras.saving.deserialize_keras_object(config)
    return zea.models.base.deserialize_zea_object(config, cls)


def check_config_class(config):
    """Validate a preset is being loaded on the correct class."""
    registered_name = config["registered_name"]
    if registered_name in ("Functional", "Sequential"):
        return keras.Model
    # cls = keras.saving.get_registered_object(registered_name)
    name = keras_to_zea_registry(registered_name, model_registry)

    cls = model_registry[name]

    if cls is None:
        raise ValueError(
            f"Attempting to load class {registered_name} with "
            "`from_preset()`, but there is no class registered with zea "
            f"for {registered_name}. Make sure to register any custom "
            "classes with `zea.registry.model_registry()`."
        )
    return cls


def jax_memory_cleanup(layer):
    """Cleanup memory for JAX models."""
    # For jax, delete all previous allocated memory to avoid temporarily
    # duplicating variable allocations. torch and tensorflow have stateful
    # variable types and do not need this fix.
    if keras.config.backend() == "jax":
        for weight in layer.weights:
            value = getattr(weight, "_value", None)
            # Do not delete sharded arrays, as they may be referenced in JAX's
            # distributed computation graph and deletion can cause errors.
            if value is not None and getattr(value, "sharding", None) is None:
                value.delete()


def set_dtype_in_config(config, dtype=None):
    """Set the `dtype` in a serialized Keras config."""
    if dtype is None:
        return config

    config = config.copy()
    if "dtype" not in config["config"]:
        # Forward `dtype` to the config.
        config["config"]["dtype"] = dtype
    elif (
        "dtype" in config["config"]
        and isinstance(config["config"]["dtype"], dict)
        and "DTypePolicyMap" in config["config"]["dtype"]["class_name"]
    ):
        # If it is `DTypePolicyMap` in `config`, forward `dtype` as its default
        # policy.
        policy_map_config = config["config"]["dtype"]["config"]
        policy_map_config["default_policy"] = dtype
        for k in policy_map_config["policy_map"].keys():
            policy_map_config["policy_map"][k]["config"]["source_name"] = dtype
    return config


def check_file_exists(preset, path):
    """Check if a file exists in a preset."""
    try:
        get_file(preset, path)
    except FileNotFoundError:
        return False
    return True


def _assert_file_exists(preset, path):
    try:
        get_file(preset, path)
    except FileNotFoundError as e:
        raise ValueError(
            f"Preset {preset} has no {path}. Make sure the URL or "
            "directory you are trying to load is a valid KerasHub preset and "
            "and that you have permissions to read/download from this location."
        ) from e


def keras_to_zea_registry(keras_name, zea_registry):
    """Convert a Keras class name to a zea registry name."""
    for registry_name, entry in zea_registry.registry.items():
        if entry.__name__ == keras_name:
            return registry_name
    raise ValueError(
        f"Class {keras_name} not found in `zea` registry. "
        "Make sure to register any custom classes with `zea.registry.model_registry()`. "
        "Currently, the `zea` registry contains: "
        f"{zea_registry.registry.items()}"
    )


class PresetLoader:
    """Base class for loading a model from a preset."""

    def __init__(self, preset, config):
        """Initialize a preset loader."""
        self.config = config
        self.preset = preset

    def get_model_kwargs(self, **kwargs):
        """Extract model kwargs from the preset."""
        model_kwargs = {}

        # Forward `dtype` to model
        model_kwargs["dtype"] = kwargs.pop("dtype", None)

        # Forward `height` and `width` to model
        if "image_shape" in kwargs:
            model_kwargs["image_shape"] = kwargs.pop("image_shape", None)

        return model_kwargs, kwargs

    def load_model(self, cls, load_weights, **kwargs):
        """Load the backbone model from the preset."""
        raise NotImplementedError

    def load_preprocessor(self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs):
        """Load a prepocessor layer from the preset."""
        kwargs = cls._add_missing_kwargs(self, kwargs)
        return cls(**kwargs)


class KerasPresetLoader(PresetLoader):
    """Loader for Keras serialized presets."""

    def check_model_class(self):
        """Check the model class is correct for the preset."""
        return check_config_class(self.config)

    def load_model(self, cls, load_weights, **kwargs):
        """Load a model from a serialized Keras config."""
        model = load_serialized_object(self.config, cls=cls, **kwargs)

        if hasattr(model, "custom_load_weights"):
            jax_memory_cleanup(model)
            # Only pass load_weights if the method explicitly declares it as a parameter.
            # Other models may have **kwargs that forward to Keras APIs which reject it.
            sig = inspect.signature(model.custom_load_weights)
            if "load_weights" in sig.parameters:
                model.custom_load_weights(self.preset, load_weights=load_weights)
            else:
                if load_weights:
                    model.custom_load_weights(self.preset)
            return model

        if not load_weights:
            return model

        jax_memory_cleanup(model)

        # try to build with image_shape or input_shape if not built yet ->
        # but preferred way to build is to have a build_config in the json!
        if not model.built:
            if hasattr(model, "image_shape"):
                model.build(input_shape=model.image_shape)
            elif hasattr(model, "input_shape"):
                model.build(input_shape=model.input_shape)
            else:
                raise ValueError(
                    "Model could not be built. Make sure to add a build_config to the json "
                    "or set the input_shape or image_shape attribute before loading weights."
                )
        model.load_weights(get_file(self.preset, MODEL_WEIGHTS_FILE))

        return model

    def load_image_converter(self, cls, **kwargs):
        """Load an image converter from the preset."""
        converter_config = load_json(self.preset, IMAGE_CONVERTER_CONFIG_FILE)
        return load_serialized_object(converter_config, cls, **kwargs)

    def get_file(self, path):
        """Get a file from the preset."""
        return get_file(self.preset, path)

    def load_preprocessor(self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs):
        """Load a preprocessor from the preset."""
        # If there is no `preprocessing.json` or it's for the wrong class,
        # delegate to the super class loader.
        if not check_file_exists(self.preset, config_file):
            return super().load_preprocessor(cls, **kwargs)
        preprocessor_json = load_json(self.preset, config_file)
        if not issubclass(check_config_class(preprocessor_json), cls):
            return super().load_preprocessor(cls, **kwargs)
        # We found a `preprocessing.json` with a complete config for our class.
        preprocessor = load_serialized_object(preprocessor_json, cls, **kwargs)
        if hasattr(preprocessor, "load_preset_assets"):
            preprocessor.load_preset_assets(self.preset)
        return preprocessor


class KerasPresetSaver:
    """Saver for Keras serialized presets."""

    def __init__(self, preset_dir):
        """Initialize a preset saver."""
        os.makedirs(preset_dir, exist_ok=True)
        self.preset_dir = preset_dir

    def save_model(self, model):
        """Save a model to a preset."""
        self._save_serialized_object(model, config_file=CONFIG_FILE)
        model_weight_path = os.path.join(self.preset_dir, MODEL_WEIGHTS_FILE)
        model.save_weights(model_weight_path)
        self._save_metadata(model)

    def save_image_converter(self, converter):
        """Save an image converter to a preset."""
        self._save_serialized_object(converter, IMAGE_CONVERTER_CONFIG_FILE)

    def save_preprocessor(self, preprocessor):
        """Save a preprocessor to a preset."""
        config_file = PREPROCESSOR_CONFIG_FILE
        if hasattr(preprocessor, "config_file"):
            config_file = preprocessor.config_file
        self._save_serialized_object(preprocessor, config_file)
        for layer in preprocessor._flatten_layers(include_self=False):
            if hasattr(layer, "save_to_preset"):
                layer.save_to_preset(self.preset_dir)

    def _recursive_pop(self, config, key):
        """Remove a key from a nested config object"""
        config.pop(key, None)
        for value in config.values():
            if isinstance(value, dict):
                self._recursive_pop(value, key)

    def _save_serialized_object(self, layer, config_file):
        config_path = os.path.join(self.preset_dir, config_file)
        config = keras.saving.serialize_keras_object(layer)
        config_to_skip = ["compile_config", "build_config"]
        for key in config_to_skip:
            self._recursive_pop(config, key)
        with open(config_path, "w", encoding="utf-8") as config_file:
            config_file.write(json.dumps(config, indent=4))

    def _save_metadata(self, layer):
        zea_version = zea.__version__
        keras_version = keras.version() if hasattr(keras, "version") else None

        metadata = {
            "keras_version": keras_version,
            "parameter_count": layer.count_params(),
            "zea_version": zea_version,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
        metadata_path = os.path.join(self.preset_dir, METADATA_FILE)
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))


def get_preset_saver(preset):
    """Get a preset saver."""
    # We only support one form of saving; Keras serialized
    # configs and saved weights.
    return KerasPresetSaver(preset)


def get_preset_loader(preset):
    """Get a preset loader."""
    _assert_file_exists(preset, CONFIG_FILE)
    # We currently assume all formats we support have a `config.json`, this is
    # true, for Keras, Transformers, and timm. We infer the on disk format by
    # inspecting the `config.json` file.
    config = load_json(preset, CONFIG_FILE)
    if "registered_name" in config:
        # If we see registered_name, we assume a serialized Keras object.
        return KerasPresetLoader(preset, config)
    else:
        contents = json.dumps(config, indent=4)
        raise ValueError(
            f"Unrecognized format for {CONFIG_FILE} in {preset}. "
            "Create a preset with the `save_to_preset` utility on KerasHub "
            f"models. Contents of {CONFIG_FILE}:\n{contents}"
        )
