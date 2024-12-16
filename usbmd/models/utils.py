import collections
import datetime
import json
import os
from pathlib import Path

import huggingface_hub
import keras
from huggingface_hub.utils import EntryNotFoundError, HFValidationError

import usbmd

HF_PREFIX = "hf://"

HF_SCHEME = "hf"

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
BUILTIN_PRESETS_FOR_BACKBONE = collections.defaultdict(dict)


def register_presets(presets, backbone_cls):
    """Register built-in presets for a set of classes.

    Note that this is intended only for models and presets shipped in the
    library itself.
    """
    for preset in presets:
        BUILTIN_PRESETS[preset] = presets[preset]
        BUILTIN_PRESETS_FOR_BACKBONE[backbone_cls][preset] = presets[preset]


def builtin_presets(cls):
    """Find all registered built-in presets for a class."""
    presets = {}
    if cls in BUILTIN_PRESETS_FOR_BACKBONE:
        presets.update(BUILTIN_PRESETS_FOR_BACKBONE[cls])
    name = getattr(cls, "name", None)
    presets.update(builtin_presets(name))
    return presets


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    if not isinstance(preset, str):
        raise ValueError(
            f"A preset identifier must be a string. Received: preset={preset}"
        )

    scheme = None
    if "://" in preset:
        scheme = preset.split("://")[0].lower()

    if scheme == HF_SCHEME:
        if huggingface_hub is None:
            raise ImportError(
                f"`from_preset()` requires the `huggingface_hub` package to load from '{preset}'. "
                "Please install with `pip install huggingface_hub`."
            )
        hf_handle = preset.removeprefix(HF_SCHEME + "://")
        try:
            return huggingface_hub.hf_hub_download(repo_id=hf_handle, filename=path)
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
                )
            else:
                raise ValueError(message)
    elif Path(preset).exists():
        # Assume a local filepath
        local_path = Path(preset) / path
        if not local_path.exists():
            raise FileNotFoundError(
                f"`{path}` doesn't exist in preset directory `{preset}`."
            )
        return str(local_path)
    else:
        raise ValueError(
            "Unknown preset identifier. A preset must be a one of:\n"
            "1) a built-in preset identifier like `'taesdxl'`\n"
            "2) a Hugging Face handle like `'hf://usbmd/taesdxl'`\n"
            "3) a path to a local preset directory like `'./taesdxl`\n"
            "Use `print(cls.presets.keys())` to view all built-in presets for "
            "API symbol `cls`.\n"
            f"Received: preset='{preset}'"
        )


def load_json(preset, config_file=CONFIG_FILE):
    config_path = get_file(preset, config_file)
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


class PresetLoader:
    def __init__(self, preset, config):
        self.config = config
        self.preset = preset

    def get_model_kwargs(self, **kwargs):
        model_kwargs = {}

        # Forward `dtype` to backbone.
        model_kwargs["dtype"] = kwargs.pop("dtype", None)

        # Forward `height` and `width` to backbone when using `TextToImage`.
        if "image_shape" in kwargs:
            model_kwargs["image_shape"] = kwargs.pop("image_shape", None)

        return model_kwargs, kwargs

    def load_model(self, cls, load_weights, **kwargs):
        """Load the backbone model from the preset."""
        raise NotImplementedError

    def load_image_converter(self, cls, **kwargs):
        """Load an image converter layer from the preset."""
        raise NotImplementedError

    def load_preprocessor(self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs):
        """Load a prepocessor layer from the preset.

        By default, we create a preprocessor from a tokenizer with default
        arguments. This allow us to support transformers checkpoints by
        only converting the backbone and tokenizer.
        """
        kwargs = cls._add_missing_kwargs(self, kwargs)
        return cls(**kwargs)


def load_serialized_object(config, **kwargs):
    # `dtype` in config might be a serialized `DTypePolicy` or `DTypePolicyMap`.
    # Ensure that `dtype` is properly configured.
    dtype = kwargs.pop("dtype", None)
    config = set_dtype_in_config(config, dtype)

    config["config"] = {**config["config"], **kwargs}
    return keras.saving.deserialize_keras_object(config)


def check_config_class(config):
    """Validate a preset is being loaded on the correct class."""
    registered_name = config["registered_name"]
    if registered_name in ("Functional", "Sequential"):
        return keras.Model
    cls = keras.saving.get_registered_object(registered_name)
    if cls is None:
        raise ValueError(
            f"Attempting to load class {registered_name} with "
            "`from_preset()`, but there is no class registered with Keras "
            f"for {registered_name}. Make sure to register any custom "
            "classes with `register_keras_serializable()`."
        )
    return cls


def jax_memory_cleanup(layer):
    # For jax, delete all previous allocated memory to avoid temporarily
    # duplicating variable allocations. torch and tensorflow have stateful
    # variable types and do not need this fix.
    if keras.config.backend() == "jax":
        for weight in layer.weights:
            if getattr(weight, "_value", None) is not None:
                weight._value.delete()


def check_file_exists(preset, path):
    try:
        get_file(preset, path)
    except FileNotFoundError:
        return False
    return True


class KerasPresetLoader(PresetLoader):
    def check_backbone_class(self):
        return check_config_class(self.config)

    def load_backbone(self, cls, load_weights, **kwargs):
        backbone = load_serialized_object(self.config, **kwargs)
        if load_weights:
            jax_memory_cleanup(backbone)
            backbone.load_weights(get_file(self.preset, MODEL_WEIGHTS_FILE))
        return backbone

    def load_image_converter(self, cls, **kwargs):
        converter_config = load_json(self.preset, IMAGE_CONVERTER_CONFIG_FILE)
        return load_serialized_object(converter_config, **kwargs)

    def load_preprocessor(self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs):
        # If there is no `preprocessing.json` or it's for the wrong class,
        # delegate to the super class loader.
        if not check_file_exists(self.preset, config_file):
            return super().load_preprocessor(cls, **kwargs)
        preprocessor_json = load_json(self.preset, config_file)
        if not issubclass(check_config_class(preprocessor_json), cls):
            return super().load_preprocessor(cls, **kwargs)
        # We found a `preprocessing.json` with a complete config for our class.
        preprocessor = load_serialized_object(preprocessor_json, **kwargs)
        if hasattr(preprocessor, "load_preset_assets"):
            preprocessor.load_preset_assets(self.preset)
        return preprocessor


class KerasPresetSaver:
    def __init__(self, preset_dir):
        os.makedirs(preset_dir, exist_ok=True)
        self.preset_dir = preset_dir

    def save_backbone(self, backbone):
        self._save_serialized_object(backbone, config_file=CONFIG_FILE)
        backbone_weight_path = os.path.join(self.preset_dir, MODEL_WEIGHTS_FILE)
        backbone.save_weights(backbone_weight_path)
        self._save_metadata(backbone)

    def save_image_converter(self, converter):
        self._save_serialized_object(converter, IMAGE_CONVERTER_CONFIG_FILE)

    def save_preprocessor(self, preprocessor):
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
        with open(config_path, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

    def _save_metadata(self, layer):
        usbmd_version = usbmd.__version__
        keras_version = keras.version() if hasattr(keras, "version") else None

        metadata = {
            "keras_version": keras_version,
            "parameter_count": layer.count_params(),
            "usbmd_version": usbmd_version,
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
        metadata_path = os.path.join(self.preset_dir, METADATA_FILE)
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))
