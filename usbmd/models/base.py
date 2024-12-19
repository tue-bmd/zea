import keras

from usbmd.models.preset_utils import (
    builtin_presets,
    get_preset_loader,
    get_preset_saver,
)


class classproperty(property):
    """Define a class level property."""

    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)


class BaseModel(keras.models.Model):
    """Class class for all USBMD Keras models.

    A `BaseModel` is the basic model.
    """

    def __init__(self, *args, dtype=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        """List built-in presets for a `BaseModel` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a `keras_hub.models.Backbone` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built-in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.Backbone.from_preset()`, or from
        a model class like `keras_hub.models.GemmaBackbone.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        For any `Backbone` subclass, you can run `cls.presets.keys()` to list
        all built-in presets available on the class.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a Gemma backbone with pre-trained weights.
        model = keras_hub.models.Backbone.from_preset(
            "gemma_2b_en",
        )

        # Load a Bert backbone with a pre-trained config and random weights.
        model = keras_hub.models.Backbone.from_preset(
            "bert_base_en",
            load_weights=False,
        )
        ```
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
