import keras
from keras import layers


def conv_bn_relu(x, filters, k=3, d=1):
    """
    Simple convolution -> batch normalization -> ReLU block.

    This helper implements a common pattern used throughout the DeepLabV3+
    decoder and ASPP: a 2D convolution followed by BatchNormalization and
    a ReLU activation.

    Args:
        x (Tensor): Input tensor.
        filters (int): Number of output channels for the convolution.
        k (int): Kernel size for the convolution. Defaults to 3.
        d (int): Dilation rate for the convolution (atrous conv). Defaults to 1.

    Returns:
        Tensor: Activated output tensor after Conv2D -> BatchNormalization -> ReLU.
    """
    x = layers.Conv2D(
        filters, k, padding="same", dilation_rate=d, use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def ASPP(x, filters):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.

    ASPP captures multi-scale context by computing several parallel branches:
    - Image-level pooling branch (global context)
    - 1x1 convolution branch
    - 3x3 dilated convolutions with dilation rates 6, 12, 18

    The outputs of all branches are concatenated and followed by a 1x1
    convolution to fuse the multi-scale features. This is the ASPP used in
    DeepLabV3/V3+ for robust multi-scale context aggregation.

    Args:
        x (Tensor): Input feature map (typically the backbone features at OS=16).
        filters (int): Number of filters to use inside each ASPP branch and for
                       the final projection.

    Returns:
        Tensor: Output tensor of the ASPP module with fused multi-scale features.
    """
    # Image pooling branch (correct way)
    pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pool = conv_bn_relu(pool, filters, k=1)
    pool = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation="bilinear")(pool)

    b1 = conv_bn_relu(x, filters, k=1, d=1)
    b6 = conv_bn_relu(x, filters, k=3, d=6)
    b12 = conv_bn_relu(x, filters, k=3, d=12)
    b18 = conv_bn_relu(x, filters, k=3, d=18)

    x = layers.Concatenate()([pool, b1, b6, b12, b18])
    return conv_bn_relu(x, filters, k=1)


def DeeplabV3PlusSmall(image_shape, num_classes, alpha=0.5, pretrained_weights=None):
    """
    DeepLabV3+ model using MobileNetV3Small backbone.

    Constructs a DeepLabV3+ segmentation model that uses MobileNetV3Small as the
    encoder (backbone) and a lightweight ASPP+decoder head. The implementation
    follows the standard DeepLabV3+ geometry:
      - Backbone provides two feature maps:
          * low-level features at output stride 4 (for decoder skip connection)
          * high-level features at output stride 16 (for ASPP)
      - ASPP applied to the OS=16 features
      - ASPP output is upsampled 4x to match OS=4 features
      - Low-level features are projected and concatenated with upsampled ASPP
      - Decoder refines concatenated features and upsamples 4x to original resolution

    Channel widths are scaled by `alpha`:
      - ASPP internal width: int(256 * alpha)
      - Low-level projection width: int(48 * alpha)
      - Decoder internal width: int(256 * alpha)

    Notes:
        - image_shape height/width should be divisible by 16 to guarantee OS=16 taps.
        - MobileNetV3Small supports alpha scaling natively; use alpha values supported
          by the Keras implementation when loading imagenet weights (e.g. 0.75, 1.0).
        - `weights=None` disables pretrained weights.

    Args:
        image_shape (tuple): Input image shape as (height, width, channels).
        num_classes (int): Number of segmentation classes (output channels).
        alpha (float): Width multiplier passed to MobileNetV3Small and used to scale
                       ASPP/decoder channel widths. Defaults to 0.5.
        weights (str or None): Pretrained weights for MobileNetV3Small (e.g. "imagenet")
                               or None for random init.

    Returns:
        keras.Model: Keras Model instance for DeepLabV3+ with MobileNetV3Small backbone.
    """
    aspp_ch = max(1, int(256 * alpha))
    low_ch = max(1, int(48 * alpha))
    dec_ch = max(1, int(256 * alpha))

    inp = keras.Input(shape=image_shape)

    if image_shape[-1] == 1:
        x = layers.Concatenate()([inp, inp, inp])
    else:
        x = inp

    backbone = keras.applications.MobileNetV3Small(
        input_shape=(image_shape[0], image_shape[1], 3),
        alpha=alpha,
        include_top=False,
        weights=pretrained_weights,
        input_tensor=x,
        include_preprocessing=True,
    )

    # OS=4 feature
    low = backbone.get_layer("expanded_conv_project_bn").output

    # OS=16 feature
    # NOTE: this may change for different input sizes
    high = backbone.get_layer("expanded_conv_7_project_bn").output

    # ASPP
    x = ASPP(high, aspp_ch)

    # OS16 -> OS4
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    # Decoder
    low = conv_bn_relu(low, low_ch, k=1)
    x = layers.Concatenate()([x, low])
    x = conv_bn_relu(x, dec_ch)
    x = conv_bn_relu(x, dec_ch)

    # OS4 -> full
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    out = layers.Conv2D(num_classes, 1, padding="same")(x)

    return keras.Model(inp, out)
