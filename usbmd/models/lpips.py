"""LPIPS model for perceptual similarity.

- **Author(s)**     : Tristan Stevens
- **Date**          : 20/01/2025
"""

import keras
from keras import ops
from keras.api.layers import Conv2D, Dropout, Input, Lambda

from usbmd.models.base import BaseModel
from usbmd.models.preset_utils import get_preset_loader, register_presets
from usbmd.models.presets import lpips_presets
from usbmd.registry import model_registry


@model_registry(name="lpips")
class LPIPS(BaseModel):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric.
    Images should be in the range [-1, 1].
    """

    def __init__(
        self,
        net_type="vgg",
        **kwargs,
    ):
        """Initialize the LPIPS model.
        Expects weights to be in the [-1, 1] range.

        Exported weights using:
            https://github.com/moono/lpips-tf2.x/blob/master/example_export_script/convert_to_tensorflow.py

        Args:
            net_type (str, optional): Type of network to use. Defaults to "vgg".
        """
        super().__init__(**kwargs)

        assert net_type == "vgg", "Only VGG model is supported"

        self.net = perceptual_model()
        self.lin = linear_model()

    def custom_load_weights(self, preset, **kwargs):  # pylint: disable=unused-argument
        """TFSM layer does not support loading weights."""
        loader = get_preset_loader(preset)

        vgg_file = "vgg/vgg.weights.h5"
        lin_file = "lin/lin.weights.h5"
        vgg_file = loader.get_file(vgg_file)
        lin_file = loader.get_file(lin_file)

        self.net.load_weights(vgg_file, **kwargs)
        self.lin.load_weights(lin_file, **kwargs)

    def call(self, inputs):  # pylint: disable=arguments-differ
        """Compute the LPIPS metric.

        Args:
            inputs (list): List of two input images of shape [B, H, W, C].
                Images should be in the range [-1, 1].

        """
        input1, input2 = inputs

        # preprocess input images
        net_out1 = Lambda(lambda x: self.preprocess_input(x))(input1)
        net_out2 = Lambda(lambda x: self.preprocess_input(x))(input2)

        # run vgg model first
        net_out1 = self.net(net_out1)
        net_out2 = self.net(net_out2)

        # normalize
        net_out1 = [
            Lambda(
                lambda x: x * ops.rsqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))
            )(t)
            for t in net_out1
        ]
        net_out2 = [
            Lambda(
                lambda x: x * ops.rsqrt(ops.sum(ops.square(x), axis=-1, keepdims=True))
            )(t)
            for t in net_out2
        ]

        # subtract
        diffs = [
            Lambda(lambda x: ops.square(x[0] - x[1]))([t1, t2])
            for t1, t2 in zip(net_out1, net_out2)
        ]

        # run on learned linear model
        lin_out = self.lin(diffs)

        # take spatial average: list([N, 1], [N, 1], [N, 1], [N, 1], [N, 1])
        lin_out = ops.convert_to_tensor(
            [
                Lambda(lambda x: ops.mean(x, axis=[1, 2], keepdims=False))(t)
                for t in lin_out
            ]
        )

        # take sum of all layers: [N, 1]
        lin_out = Lambda(lambda x: ops.sum(x, axis=0))(lin_out)

        # squeeze: [N, ]
        lin_out = Lambda(lambda x: ops.squeeze(x, axis=-1))(lin_out)

        return lin_out

    @staticmethod
    def preprocess_input(image):
        """Preprocess the input images

        Args:
            image (Tensor): Input image tensor of shape [B, H, W, C]
                and values in the range [0, 1].
        Returns:
            Tensor: Preprocessed image tensor of shape [B, H, W, C]
                and standardized values for VGG model.
        """

        scale = ops.convert_to_tensor([0.458, 0.448, 0.450])[None, None, None, :]
        shift = ops.convert_to_tensor([-0.030, -0.088, -0.188])[None, None, None, :]
        image = (image - shift) / scale
        return image


def perceptual_model():
    """Get the VGG16 model for perceptual loss."""
    layers = [
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
        "block4_conv3",
        "block5_conv3",
    ]
    vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet")

    vgg16_output_layers = [l.output for l in vgg16.layers if l.name in layers]
    model = keras.Model(vgg16.input, vgg16_output_layers, name="perceptual_model")
    return model


def linear_model():
    """Get the linear head model for LPIPS."""
    vgg_channels = [64, 128, 256, 512, 512]
    inputs, outputs = [], []
    for ii, channel in enumerate(vgg_channels):
        name = f"lin{ii}"

        model_input = Input(shape=(None, None, channel), dtype="float32")
        model_output = Dropout(rate=0.5, dtype="float32")(model_input)
        model_output = Conv2D(
            filters=1,
            kernel_size=1,
            strides=1,
            use_bias=False,
            dtype="float32",
            data_format="channels_last",
            name=name,
        )(model_output)
        inputs.append(model_input)
        outputs.append(model_output)

    model = keras.Model(inputs=inputs, outputs=outputs, name="linear_model")
    return model


register_presets(lpips_presets, LPIPS)
