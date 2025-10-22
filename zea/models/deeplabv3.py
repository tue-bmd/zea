"""DeepLabV3+ architecture for multi-class segmentation. For more details see https://arxiv.org/abs/1802.02611."""

import keras
from keras import layers, ops


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    """
    Create a convolution block with batch normalization and ReLU activation.

    This is a standard building block used throughout the DeepLabV3+ architecture,
    consisting of Conv2D -> BatchNormalization -> ReLU.

    Args:
        block_input (Tensor): Input tensor to the convolution block
        num_filters (int): Number of output filters/channels. Defaults to 256.
        kernel_size (int): Size of the convolution kernel. Defaults to 3.
        dilation_rate (int): Dilation rate for dilated convolution. Defaults to 1.
        use_bias (bool): Whether to use bias in the convolution layer. Defaults to False.

    Returns:
        Tensor: Output tensor after convolution, batch normalization, and ReLU
    """
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    """
    Implement Atrous Spatial Pyramid Pooling (ASPP) module.

    ASPP captures multi-scale context by applying parallel atrous convolutions
    with different dilation rates. This helps the model understand objects
    at multiple scales.

    The module consists of:
    - Global average pooling branch
    - 1x1 convolution branch
    - 3x3 convolutions with dilation rates 6, 12, and 18

    Reference: https://arxiv.org/abs/1706.05587

    Args:
        dspp_input (Tensor): Input feature tensor from encoder

    Returns:
        Tensor: Multi-scale feature representation
    """
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_shape, num_classes, pretrained_weights=None):
    """
    Build DeepLabV3+ model for semantic segmentation.

    DeepLabV3+ combines the benefits of spatial pyramid pooling and encoder-decoder
    architecture. It uses a ResNet50 backbone as encoder, ASPP for multi-scale
    feature extraction, and a simple decoder for recovering spatial details.

    Architecture:
    1. Encoder: ResNet50 backbone with atrous convolutions
    2. ASPP: Multi-scale feature extraction
    3. Decoder: Simple decoder with skip connections
    4. Output: Final segmentation prediction

    Reference: https://arxiv.org/abs/1802.02611

    Args:
        image_shape (tuple): Input image shape as (height, width, channels)
        num_classes (int): Number of output classes for segmentation
        pretrained_weights (str, optional): Pretrained weights for ResNet50 backbone.
                                          Defaults to None.

    Returns:
        keras.Model: Complete DeepLabV3+ model
    """
    model_input = keras.Input(shape=image_shape)
    # 3-channel grayscale as repeated single channel for ResNet50
    model_input_3_channel = ops.concatenate([model_input, model_input, model_input], axis=-1)
    preprocessed = keras.applications.resnet50.preprocess_input(model_input_3_channel)
    resnet50 = keras.applications.ResNet50(
        weights=pretrained_weights, include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_shape[0] // 4 // x.shape[1], image_shape[1] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_shape[0] // x.shape[1], image_shape[1] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)
