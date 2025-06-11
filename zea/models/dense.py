"""Dense models and architectures"""

import keras
from keras import layers

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.layers import sinusoidal_embedding
from zea.models.preset_utils import register_presets
from zea.models.presets import dense_presets


@model_registry(name="dense")
class DenseNet(BaseModel):
    """Simple dense model"""

    def __init__(
        self,
        input_dim,
        widths,
        output_dim,
        name="dense",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.widths = widths
        self.output_dim = output_dim
        self.network = get_dense_network(self.input_dim, self.widths, self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "widths": self.widths,
                "output_dim": self.output_dim,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_dense_network(input_dim, widths, output_dim):
    """Simple feedforward network"""
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for width in widths:
        x = layers.Dense(width, activation="relu")(x)
    outputs = layers.Dense(output_dim, kernel_initializer="zeros")(x)
    return keras.Model(inputs, outputs, name="dense_net")


@model_registry(name="dense_time_conditional")
class DenseTimeConditionalNet(BaseModel):
    """Dense model with time-conditional sinusoidal embedding"""

    def __init__(
        self,
        input_dim,
        widths,
        output_dim,
        embedding_min_frequency=1.0,
        embedding_max_frequency=1000.0,
        embedding_dims=32,
        name="dense_time_conditional",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.widths = widths
        self.output_dim = output_dim
        self.embedding_min_frequency = embedding_min_frequency
        self.embedding_max_frequency = embedding_max_frequency
        self.embedding_dims = embedding_dims
        self.network = get_time_conditional_dense_network(
            self.input_dim,
            self.widths,
            self.output_dim,
            self.embedding_min_frequency,
            self.embedding_max_frequency,
            self.embedding_dims,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "widths": self.widths,
                "output_dim": self.output_dim,
                "embedding_min_frequency": self.embedding_min_frequency,
                "embedding_max_frequency": self.embedding_max_frequency,
                "embedding_dims": self.embedding_dims,
            }
        )
        return config

    def call(self, *args, **kwargs):
        return self.network(*args, **kwargs)


def get_time_conditional_dense_network(
    input_dim,
    widths,
    output_dim,
    embedding_min_frequency=1.0,
    embedding_max_frequency=1000.0,
    embedding_dims=32,
):
    """Dense network with time-conditional sinusoidal embedding"""
    inputs = keras.Input(shape=(input_dim,))
    time_input = keras.Input(shape=(1,))

    @keras.saving.register_keras_serializable()
    def _sinusoidal_embedding(x):
        return sinusoidal_embedding(
            x, embedding_min_frequency, embedding_max_frequency, embedding_dims
        )

    e = layers.Lambda(_sinusoidal_embedding, output_shape=(embedding_dims,))(time_input)
    x = layers.Concatenate()([inputs, e])
    for width in widths:
        x = layers.Dense(width, activation="relu")(x)
    outputs = layers.Dense(output_dim, kernel_initializer="zeros")(x)
    return keras.Model([inputs, time_input], outputs, name="dense_time_conditional_net")


register_presets(dense_presets, DenseNet)
register_presets(dense_presets, DenseTimeConditionalNet)
