"""Base model class for all zea Keras models.

This module provides the `BaseModel` class for all zea Keras models.
"""

import importlib

import keras
from keras.src.saving.serialization_lib import record_object_after_deserialization

from zea.internal.core import classproperty
from zea.models.preset_utils import builtin_presets, get_preset_loader, get_preset_saver


class BaseModel(keras.models.Model):
    """Base class for all zea Keras models.

    A ``BaseModel`` is the basic model for zea.
    """

    @classmethod
    def from_config(cls, config):
        """Create a model instance from a configuration dictionary.

        The default ``from_config()`` for functional models will return a
        vanilla ``keras.Model``. This override ensures a subclass instance is returned.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            BaseModel: An instance of the model subclass.

        """
        return cls(**config)

    @classproperty
    def presets(cls):
        """List built-in presets for a ``BaseModel`` subclass.

        Returns:
            dict: Dictionary of available built-in presets.
        """
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a model from a preset.

        A preset is a directory of configs, weights, and other file assets used
        to save and load a pre-trained model. The ``preset`` can be passed as one of:

            1. a built-in preset identifier like ``'bert_base_en'``
            2. a Kaggle Models handle like ``'kaggle://user/bert/keras/bert_base_en'``
            3. a Hugging Face handle like ``'hf://user/bert_base_en'``
            4. a path to a local preset directory like ``'./bert_base_en'``

        This constructor can be called in one of two ways: either from the base
        class like ``keras_hub.models.Backbone.from_preset()``, or from
        a model class like ``keras_hub.models.GemmaBackbone.from_preset()``.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        For any ``Backbone`` subclass, you can run ``cls.presets.keys()`` to list
        all built-in presets available on the class.

        Args:
            preset (str): A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights (bool): If ``True``, the weights will be loaded into the
                model architecture. If ``False``, the weights will be randomly
                initialized.
            **kwargs: Additional keyword arguments.

        Examples:
            .. code-block:: python

                # Load a Gemma backbone with pre-trained weights.
                model = keras_hub.models.Backbone.from_preset(
                    "gemma_2b_en",
                )

                # Load a Bert backbone with a pre-trained config and random weights.
                model = keras_hub.models.Backbone.from_preset(
                    "bert_base_en",
                    load_weights=False,
                )

        Returns:
            BaseModel: The loaded model instance.

        """
        loader = get_preset_loader(preset)
        model_cls = loader.check_model_class()
        if not issubclass(model_cls, cls):
            raise ValueError(
                f"Saved preset has type `{model_cls.__name__}` which is not "
                f"a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{model_cls.__name__}` instead."
            )
        return loader.load_model(model_cls, load_weights, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save backbone to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_model(self)


def deserialize_zea_object(config):
    """Retrieve the object by deserializing the config dict.

    Need to borrow this function from keras and customize a bit to allow
    deserialization of custom (zea) objects. See the original function here:
    `keras.utils.deserialize_keras_object()`. As from  the following keras
    PR did not work on none Keras objects anymore:
    - https://github.com/keras-team/keras/pull/20751

    Args:
        config (dict): The configuration dictionary
    Returns:
        obj (Object): The deserialized object
    """
    class_name = config["class_name"]
    inner_config = config["config"] or {}

    module = config.get("module", None)
    registered_name = config.get("registered_name", class_name)

    cls = _retrieve_class(module, registered_name, config)

    if not hasattr(cls, "from_config"):
        raise TypeError(
            f"Unable to reconstruct an instance of '{class_name}' because "
            f"the class is missing a `from_config()` method. "
            f"Full object config: {config}"
        )

    try:
        instance = cls.from_config(inner_config)
    except TypeError as e:
        raise TypeError(
            f"{cls} could not be deserialized properly. Please"
            " ensure that components that are Python object"
            " instances (layers, models, etc.) returned by"
            " `get_config()` are explicitly deserialized in the"
            " model's `from_config()` method."
            f"\n\nconfig={config}.\n\nException encountered: {e}"
        ) from e
    build_config = config.get("build_config", None)
    if build_config and not instance.built:
        instance.build_from_config(build_config)
        instance.built = True
    compile_config = config.get("compile_config", None)
    if compile_config:
        instance.compile_from_config(compile_config)
        instance.compiled = True

    if "shared_object_id" in config:
        record_object_after_deserialization(instance, config["shared_object_id"])
    return instance


def _retrieve_class(module, class_name, config):
    # Attempt to retrieve the class object given the `module`
    # and `class_name`. Import the module, find the class.

    package = module.split(".", maxsplit=1)[0]
    if package in {"zea", "zea-addons"}:
        try:
            mod = importlib.import_module(module)
            obj = vars(mod).get(class_name, None)
            if obj is not None:
                return obj
        except ModuleNotFoundError as exc:
            raise TypeError(
                f"Could not deserialize class '{class_name}' because "
                f"its parent module {module} cannot be imported. "
                f"Full object config: {config}"
            ) from exc

    raise TypeError(
        f"Could not locate class '{class_name}'. "
        "Make sure custom classes are decorated with "
        "`@zea.registry.model_registry().`"
        f"Full object config: {config}"
    )
