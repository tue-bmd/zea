"""Metrics for ultrasound images."""

from functools import partial
from typing import List

import keras
import numpy as np
from keras import ops

from zea import log
from zea.backend import func_on_device
from zea.func import tensor
from zea.func.tensor import translate
from zea.internal.registry import metrics_registry
from zea.internal.utils import reduce_to_signature
from zea.models.lpips import LPIPS


def get_metric(name, **kwargs):
    """Get metric function given name."""
    metric_fn = metrics_registry[name]
    if not metric_fn.__name__.startswith("get_"):
        return partial(metric_fn, **kwargs)

    log.info(f"Initializing metric: {log.green(name)}")
    return metric_fn(**kwargs)


def _reduce_mean(array, keep_batch_dim=True):
    """Reduce array by taking the mean.
    Preserves batch dimension if keep_batch_dim=True.
    """
    if keep_batch_dim:
        ndim = ops.ndim(array)
        axis = tuple(range(max(0, ndim - 3), ndim))
    else:
        axis = None
    return ops.mean(array, axis=axis)


@metrics_registry(name="cnr", paired=True)
def cnr(x, y):
    """Calculate contrast to noise ratio"""
    mu_x = ops.mean(x)
    mu_y = ops.mean(y)

    var_x = ops.var(x)
    var_y = ops.var(y)

    return 20 * ops.log10(ops.abs(mu_x - mu_y) / ops.sqrt((var_x + var_y) / 2))


@metrics_registry(name="contrast", paired=True)
def contrast(x, y):
    """Contrast ratio"""
    return 20 * ops.log10(ops.mean(x) / ops.mean(y))


@metrics_registry(name="gcnr", paired=True)
def gcnr(x, y, bins=256):
    """Generalized contrast-to-noise-ratio"""
    x = ops.convert_to_numpy(x)
    y = ops.convert_to_numpy(y)
    x = np.ravel(x)
    y = np.ravel(y)
    _, bins = np.histogram(np.concatenate((x, y)), bins=bins)
    f, _ = np.histogram(x, bins=bins, density=True)
    g, _ = np.histogram(y, bins=bins, density=True)
    f /= np.sum(f)
    g /= np.sum(g)
    return 1 - np.sum(np.minimum(f, g))


@metrics_registry(name="fwhm", paired=False)
def fwhm(img):
    """Resolution full width half maxima"""
    mask = ops.nonzero(img >= 0.5 * ops.amax(img))[0]
    return mask[-1] - mask[0]


@metrics_registry(name="snr", paired=False)
def snr(img):
    """Signal to noise ratio"""
    return ops.mean(img) / ops.std(img)


@metrics_registry(name="wopt_mae", paired=True)
def wopt_mae(ref, img):
    """Find the optimal weight that minimizes the mean absolute error"""
    wopt = ops.median(ref / img)
    return wopt


@metrics_registry(name="wopt_mse", paired=True)
def wopt_mse(ref, img):
    """Find the optimal weight that minimizes the mean squared error"""
    wopt = ops.sum(ref * img) / ops.sum(img * img)
    return wopt


@metrics_registry(name="psnr", paired=True)
def psnr(y_true, y_pred, *, max_val=255):
    """Peak Signal to Noise Ratio (PSNR) for two input tensors.

    PSNR = 20 * log10(max_val) - 10 * log10(mean(square(y_true - y_pred)))

    Args:
        y_true (tensor): [None, height, width, channels]
        y_pred (tensor): [None, height, width, channels]
        max_val: The dynamic range of the images

    Returns:
        Tensor (float): PSNR score for each image in the batch.
    """
    mse = _reduce_mean(ops.square(y_true - y_pred))
    psnr = 20 * ops.log10(max_val) - 10 * ops.log10(mse)
    return psnr


@metrics_registry(name="mse", paired=True)
def mse(y_true, y_pred):
    """Gives the MSE for two input tensors.
    Args:
        y_true (tensor)
        y_pred (tensor)
    Returns:
        (float): mean squared error between y_true and y_pred. L2 loss.

    """
    return _reduce_mean(ops.square(y_true - y_pred))


@metrics_registry(name="mae", paired=True)
def mae(y_true, y_pred):
    """Gives the MAE for two input tensors.
    Args:
        y_true (tensor)
        y_pred (tensor)
    Returns:
        (float): mean absolute error between y_true and y_pred. L1 loss.

    """
    return _reduce_mean(ops.abs(y_true - y_pred))


@metrics_registry(name="ssim", paired=True)
def ssim(
    a,
    b,
    *,
    max_val: float = 255.0,
    filter_size: int = 11,
    filter_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    filter_fn=None,
):
    """Computes the structural similarity index (SSIM) between image pairs.

    This function is based on the standard SSIM implementation from:
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

    This function copied from [`dm_pix.ssim`](https://dm-pix.readthedocs.io/en/latest/api.html#dm_pix.ssim),
    which is part of the DeepMind's `dm_pix` library. They modeled their implementation
    after the `tf.image.ssim` function.

    Note: the true SSIM is only defined on grayscale. This function does not
    perform any colorspace transform. If the input is in a color space, then it
    will compute the average SSIM.

    Args:
        a: First image (or set of images).
        b: Second image (or set of images).
        max_val: The maximum magnitude that `a` or `b` can have.
        filter_size: Window size (>= 1). Image dims must be at least this small.
        filter_sigma: The bandwidth of the Gaussian used for filtering (> 0.).
        k1: One of the SSIM dampening parameters (> 0.).
        k2: One of the SSIM dampening parameters (> 0.).
        return_map: If True, will cause the per-pixel SSIM "map" to be returned.
        filter_fn: An optional argument for overriding the filter function used by
            SSIM, which would otherwise be a 2D Gaussian blur specified by filter_size
            and filter_sigma.

    Returns:
        Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """

    if filter_fn is None:
        # Construct a 1D Gaussian blur filter.
        hw = filter_size // 2
        shift = (2 * hw - filter_size + 1) / 2
        f_i = ((ops.cast(ops.arange(filter_size), "float32") - hw + shift) / filter_sigma) ** 2
        filt = ops.exp(-0.5 * f_i)
        filt /= ops.sum(filt)

        # Construct a 1D convolution.
        def filter_fn_1(z):
            return tensor.correlate(z, ops.flip(filt), mode="valid")

        # Apply the vectorized filter along the y axis.
        def filter_fn_y(z):
            z_flat = ops.reshape(ops.moveaxis(z, -3, -1), (-1, z.shape[-3]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-2],
                z.shape[-1],
                -1,
            )
            _z_filtered = ops.vectorized_map(filter_fn_1, z_flat)
            z_filtered = ops.moveaxis(ops.reshape(_z_filtered, z_filtered_shape), -1, -3)
            return z_filtered

        # Apply the vectorized filter along the x axis.
        def filter_fn_x(z):
            z_flat = ops.reshape(ops.moveaxis(z, -2, -1), (-1, z.shape[-2]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-3],
                z.shape[-1],
                -1,
            )
            _z_filtered = ops.vectorized_map(filter_fn_1, z_flat)
            z_filtered = ops.moveaxis(ops.reshape(_z_filtered, z_filtered_shape), -1, -2)
            return z_filtered

        # Apply the blur in both x and y.
        filter_fn = lambda z: filter_fn_y(filter_fn_x(z))

    mu0 = filter_fn(a)
    mu1 = filter_fn(b)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filter_fn(a**2) - mu00
    sigma11 = filter_fn(b**2) - mu11
    sigma01 = filter_fn(a * b) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = keras.config.epsilon()
    sigma00 = ops.maximum(epsilon, sigma00)
    sigma11 = ops.maximum(epsilon, sigma11)
    sigma01 = ops.sign(sigma01) * ops.minimum(ops.sqrt(sigma00 * sigma11), ops.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim_value = ops.mean(ssim_map, axis=tuple(range(-3, 0)))
    return ssim_map if return_map else ssim_value


@metrics_registry(name="ncc", paired=True)
def ncc(x, y):
    """Normalized cross correlation"""
    num = ops.sum(x * y)
    denom = ops.sqrt(ops.sum(x**2) * ops.sum(y**2))
    return num / ops.maximum(denom, keras.config.epsilon())


@metrics_registry(name="lpips", paired=True)
def get_lpips(image_range, batch_size=None, clip=False):
    """
    Get the Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        image_range (list): The range of the images. Will be translated to [-1, 1] for LPIPS.
        batch_size (int): The batch size for the LPIPS model.
        clip (bool): Whether to clip the images to `image_range`.

    Returns:
        The LPIPS metric function which can be used with [..., h, w, c] tensors in
        the range `image_range`.
    """
    # Get the LPIPS model
    _lpips = LPIPS.from_preset("lpips")
    _lpips.trainable = False
    _lpips.disable_checks = True

    def unstack_lpips(imgs):
        """Unstack the images and calculate the LPIPS metric."""
        img1, img2 = ops.unstack(imgs, num=2, axis=-1)
        return _lpips([img1, img2])

    def lpips(img1, img2, **kwargs):
        """
        The LPIPS metric function.
        Args:
            img1 (tensor) with shape (..., h, w, c)
            img2 (tensor) with shape (..., h, w, c)
        Returns (float): The LPIPS metric between img1 and img2 with shape [...]
        """
        # clip and translate images to [-1, 1]
        if clip:
            img1 = ops.clip(img1, *image_range)
            img2 = ops.clip(img2, *image_range)
        img1 = translate(img1, image_range, [-1, 1])
        img2 = translate(img2, image_range, [-1, 1])

        imgs = ops.stack([img1, img2], axis=-1)
        n_batch_dims = ops.ndim(img1) - 3
        return tensor.func_with_one_batch_dim(
            unstack_lpips, imgs, n_batch_dims, batch_size=batch_size
        )

    return lpips


class Metrics:
    """Class for calculating multiple paired metrics. Also useful for batch processing.

    Will preprocess images by translating to [0, 255], clipping, and quantizing to uint8
    if specified.

    Example:
        .. doctest::

            >>> from zea import metrics
            >>> import numpy as np

            >>> metrics = metrics.Metrics(["psnr", "lpips"], image_range=[0, 255])
            >>> y_true = np.random.rand(4, 128, 128, 1)
            >>> y_pred = np.random.rand(4, 128, 128, 1)
            >>> result = metrics(y_true, y_pred)
            >>> result = {k: float(v) for k, v in result.items()}
            >>> print(result)  # doctest: +ELLIPSIS
            {'psnr': ..., 'lpips': ...}
    """

    def __init__(
        self,
        metrics: List[str],
        image_range: tuple,
        quantize: bool = False,
        clip: bool = False,
        **kwargs,
    ):
        """Initialize the Metrics class.

        Args:
            metrics (list): List of metric names to calculate.
            image_range (tuple): The range of the images. Used for metrics like PSNR and LPIPS.
            quantize (bool): Whether to quantize the images to uint8 before calculating metrics.
            clip (bool): Whether to clip the images to `image_range` before calculating metrics.
            kwargs: Additional keyword arguments to pass to the metric functions.
        """
        # Assert all metrics are paired
        for m in metrics:
            assert metrics_registry.get_parameter(m, "paired"), (
                f"Metric {m} is not a paired metric."
            )

        # Add image_range to kwargs for metrics that require it
        kwargs["image_range"] = image_range
        self.image_range = image_range

        # Initialize all metrics
        self.metrics = {
            m: get_metric(m, **reduce_to_signature(metrics_registry[m], kwargs)) for m in metrics
        }

        # Other settings
        self.quantize = quantize
        self.clip = clip

    @staticmethod
    def _call_metric_fn(
        fun, y_true, y_pred, average_batches, return_numpy, device, mapped_batch_size=None
    ):
        num_batch_axes = max(0, ops.ndim(y_true) - 3)

        # Because most metric functions do not support batching, we vmap over the batch axes.
        metric_fn = fun
        for _ in range(num_batch_axes):
            # recursively vmap the leading axis
            metric_fn = tensor.vmap(
                metric_fn, in_axes=0, _use_torch_vmap=True, batch_size=mapped_batch_size
            )

        out = func_on_device(metric_fn, device, y_true, y_pred)

        if average_batches:
            out = ops.mean(out)

        if return_numpy:
            out = ops.convert_to_numpy(out)
        return out

    def _prepocess(self, tensor):
        tensor = translate(tensor, self.image_range, [0, 255])
        if self.clip:
            tensor = ops.clip(tensor, 0, 255)
        if self.quantize:
            tensor = ops.cast(tensor, "uint8")
        tensor = ops.cast(tensor, "float32")  # Some metrics require float32
        return tensor

    def __call__(
        self,
        y_true,
        y_pred,
        average_batches=True,
        mapped_batch_size=None,
        return_numpy=True,
        device=None,
    ):
        """Calculate all metrics and return as a dictionary.
        Assumes input shape [..., h, w, c], i.e. images of shape [h, w, c] with
        any number of leading batch dimensions. The metrics will be calculated
        on these 2d images and mapped across all leading batch dimensions.

        Args:
            y_true (tensor): Ground truth images with shape [..., h, w, c]
            y_pred (tensor): Predicted images with shape [..., h, w, c]
            average_batches (bool): Whether to average the metrics over the batch dimensions.
            mapped_batch_size (optional int): The batch size to use for computing
                metric values in parallel.
                You may want to decrease this if you run into memory issues, e.g. with LPIPS.
            return_numpy (bool): Whether to return the metrics as numpy arrays. If False, will
                return as tensors.
            device (str): The device to run the metric calculations on. If None, will use the
                default device.
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = self._call_metric_fn(
                metric,
                self._prepocess(y_true),
                self._prepocess(y_pred),
                average_batches,
                return_numpy,
                device,
                mapped_batch_size=mapped_batch_size,
            )
        return results


def _sector_reweight_image(image, sector_angle, axis):
    """
    Reweights image according to the amount of area each
    row of pixels will occupy if that image is scan converted
    with angle sector_angle.
    This 'image' could be e.g. a pixelwise loss or metric.

    We can compute this by viewing the scan converted image as the sector
    of a circle with a known central angle, and radius given by depth.
    See: https://en.wikipedia.org/wiki/Circular_sector

    Params:
        image (ndarray or Tensor): image to be re-weighted, any shape
        sector_angle (float | int): angle in degrees
        axis (int): axis corresponding to the height/depth dimension.

    Returns:
        reweighted_image (ndarray): image with pixels reweighted to area occupied by each
            pixel post-scan-conversion.
    """
    height = image.shape[axis]
    depths = ops.arange(height, dtype="float32") + 0.5  # center of the pixel as its depth
    reweighting_factors = (sector_angle / 360) * 2 * np.pi * depths
    # Reshape reweighting_factors to broadcast along the specified axis
    shape = [1] * ops.ndim(image)
    shape[axis] = height
    reweighting_factors = ops.reshape(reweighting_factors, shape)
    return reweighting_factors * image
