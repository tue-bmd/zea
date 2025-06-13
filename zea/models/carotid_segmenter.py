"""Carotid segmentation model.

Original implementation of paper:
    - "Unsupervised domain adaptation method for segmenting cross-sectional CCA images"
    - https://doi.org/10.1016/j.cmpb.2022.107037
    - Author: Luuk van Knippenberg
"""

import keras
from keras import ops
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
)

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.preset_utils import register_presets
from zea.models.presets import carotid_segmenter_presets

INFERENCE_SIZE = 256


@model_registry(name="carotid-segmenter")
class CarotidSegmenter(BaseModel):
    """Carotid segmentation model."""

    def __init__(
        self,
        input_shape=(INFERENCE_SIZE, INFERENCE_SIZE, 1),
        input_range=(0, 1),
        name="carotid_segmenter",
        **kwargs,
    ):
        """Initializes the carotid segmenter model.

        Based on U-Net architecture.

        Original implementation of paper:
            - "Unsupervised domain adaptation method for segmenting cross-sectional CCA images"
            - https://doi.org/10.1016/j.cmpb.2022.107037
        """

        super().__init__(
            name=name,
            **kwargs,
        )
        self.input_shape = input_shape
        self.input_range = input_range

        self.network = _get_network(self.input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "input_shape": self.input_shape,
                "input_range": self.input_range,
            }
        )
        return config

    def call(self, inputs):
        """Segment the input image."""
        if self.network is None:
            raise ValueError(
                "Please load model using `CarotidSegmenter.from_preset()` before calling."
            )

        assert inputs.ndim == 4, (
            f"Input should have 4 dimensions (B, H, W, C), but has {inputs.ndim}."
        )

        assert inputs.shape[-1] == 1, f"Input should have 1 channel, but has {inputs.shape[-1]}."

        original_size = ops.shape(inputs)[1:3]
        inputs = ops.image.resize(inputs, [INFERENCE_SIZE, INFERENCE_SIZE])

        mask = self.network(inputs)

        # resize output to original size
        output = ops.image.resize(mask, original_size)

        return output


def _conv_block(x, filters, block, convs=2, dropout=0.5, pool=True):
    for i in range(convs):
        x = Conv2D(filters, 3, activation="relu", padding="same", name=f"Conv{block}_{i + 1}")(x)
        x = BatchNormalization(name=f"BN{block}_{i + 1}")(x)
    if pool:
        x_pooled = MaxPooling2D(pool_size=(2, 2), name=f"P{block}")(x)
        x_pooled = Dropout(dropout, name=f"DO{block}")(x_pooled)
        return x, x_pooled
    else:
        x = Dropout(dropout, name=f"DO{block}")(x)
        return x


def _up_block(x, skip, filters, block, convs=2, final_conv_filters=None):
    x = Conv2D(filters, 2, activation="relu", padding="same", name=f"Conv{block}_1")(
        UpSampling2D(size=(2, 2))(x)
    )
    x = BatchNormalization(name=f"BN{block}_1")(x)
    x = concatenate([skip, x], axis=3, name=f"Merge{block}")
    for i in range(2, convs + 2):
        x = Conv2D(filters, 3, activation="relu", padding="same", name=f"Conv{block}_{i}")(x)
        x = BatchNormalization(name=f"BN{block}_{i}")(x)
    # For block 9, add Conv9_4 with 2 filters and BN9_4
    if final_conv_filters is not None:
        x = Conv2D(
            final_conv_filters,
            3,
            activation="relu",
            padding="same",
            name=f"Conv{block}_4",
        )(x)
        x = BatchNormalization(name=f"BN{block}_4")(x)
    x = Dropout(0, name=f"DO{block}")(x)
    return x


def _get_network(input_size=(256, 256, 1)):
    inputs = Input(input_size, name="Input")
    x = Dropout(0.2, name="DO0")(inputs)
    NrFeaturesPerLayer = 64

    # Encoder
    skips = []
    for block in range(1, 5):
        x, x_pooled = _conv_block(x, NrFeaturesPerLayer, block, convs=2, dropout=0.5, pool=True)
        skips.append(x)
        x = x_pooled

    # Bottleneck
    x = _conv_block(x, NrFeaturesPerLayer, 5, convs=2, dropout=0.5, pool=False)

    # Decoder
    for block, skip in zip(range(6, 9), reversed(skips[1:])):
        x = _up_block(x, skip, NrFeaturesPerLayer, block, convs=2)

    up9 = Conv2D(NrFeaturesPerLayer, 2, activation="relu", padding="same", name="Conv9_1")(
        UpSampling2D(size=(2, 2))(x)
    )
    bn20 = BatchNormalization(name="BN9_1")(up9)
    merge9 = concatenate([skips[0], bn20], axis=3, name="Merge9")
    conv9_2 = Conv2D(NrFeaturesPerLayer, 3, activation="relu", padding="same", name="Conv9_2")(
        merge9
    )
    bn21 = BatchNormalization(name="BN9_2")(conv9_2)
    conv9_3 = Conv2D(NrFeaturesPerLayer, 3, activation="relu", padding="same", name="Conv9_3")(bn21)
    bn22 = BatchNormalization(name="BN9_3")(conv9_3)
    conv9_4 = Conv2D(2, 3, activation="relu", padding="same", name="Conv9_4")(bn22)
    bn23 = BatchNormalization(name="BN9_4")(conv9_4)
    x = Dropout(0, name="DO9")(bn23)

    # Final layer
    x = Conv2D(1, 1, activation="sigmoid", name="Segmentation")(x)

    return keras.Model(inputs=inputs, outputs=x, name="CustomUnet")


register_presets(carotid_segmenter_presets, CarotidSegmenter)
