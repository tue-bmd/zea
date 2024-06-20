"""This module defines quality metrics for ultrasound images.

- **Author(s)**     : Tristan Stevens, Ben Luijten
- **Date**          : 18 Nov 2021
"""

import numpy as np

from usbmd.registry import metrics_registry


def get_metric(name):
    """Get metric function given name."""
    return metrics_registry[name]


@metrics_registry(name="cnr", framework="numpy", supervised=True)
def cnr(x, y):
    """Calculate contrast to noise ratio"""
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    return 20 * np.log10(np.abs(mu_x - mu_y) / np.sqrt((var_x + var_y) / 2))


@metrics_registry(name="contrast", framework="numpy", supervised=True)
def contrast(x, y):
    """Contrast ratio"""
    return 20 * np.log10(x.mean() / y.mean())


@metrics_registry(name="gcnr", framework="numpy", supervised=True)
def gcnr(x, y, bins=256):
    """Generalized contrast-to-noise-ratio"""
    x = x.flatten()
    y = y.flatten()
    _, bins = np.histogram(np.concatenate((x, y)), bins=bins)
    f, _ = np.histogram(x, bins=bins, density=True)
    g, _ = np.histogram(y, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))


@metrics_registry(name="fwhm", framework="numpy", supervised=False)
def fwhm(img):
    """Resolution full width half maxima"""
    mask = np.nonzero(img >= 0.5 * np.amax(img))[0]
    return mask[-1] - mask[0]


@metrics_registry(name="speckle_res", framework="numpy", supervised=False)
def speckle_res(img):
    """TODO: Write speckle edge-spread function resolution code"""
    raise NotImplementedError


@metrics_registry(name="snr", framework="numpy", supervised=False)
def snr(img):
    """Signal to noise ratio"""
    return img.mean() / img.std()


@metrics_registry(name="wopt_mae", framework="numpy", supervised=True)
def wopt_mae(ref, img):
    """Find the optimal weight that minimizes the mean absolute error"""
    wopt = np.median(ref / img)
    return wopt


@metrics_registry(name="wopt_mse", framework="numpy", supervised=True)
def wopt_mse(ref, img):
    """Find the optimal weight that minimizes the mean squared error"""
    wopt = np.sum(ref * img) / np.sum(img * img)
    return wopt


@metrics_registry(name="l1loss", framework="numpy", supervised=True)
def l1loss(x, y):
    """L1 loss"""
    return np.abs(x - y).mean()


@metrics_registry(name="l2loss", framework="numpy", supervised=True)
def l2loss(x, y):
    """L2 loss"""
    return np.sqrt(((x - y) ** 2).mean())


@metrics_registry(name="psnr", framework="numpy", supervised=True)
def psnr(x, y):
    """Peak signal to noise ratio"""
    dynamic_range = max(x.max(), y.max()) - min(x.min(), y.min())
    return 20 * np.log10(dynamic_range / l2loss(x, y))


@metrics_registry(name="ncc", framework="numpy", supervised=True)
def ncc(x, y):
    """Normalized cross correlation"""
    return (x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum())


@metrics_registry(name="image_entropy", framework="numpy", supervised=False)
def image_entropy(image):
    """Calculate the entropy of the image

    Args:
        image (ndarray): The image for which the entropy is calculated

    Returns:
        float: The entropy of the image
    """
    marg = np.histogramdd(np.ravel(image), bins=256)[0] / image.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy


@metrics_registry(name="image_sharpness", framework="numpy", supervised=False)
def image_sharpness(image):
    """Calculate the sharpness of the image

    Args:
        image (ndarray): The image for which the sharpness is calculated

    Returns:
        float: The sharpness of the image
    """
    return np.mean(np.abs(np.gradient(image)))

def sector_reweight_image(image, sector_angle):
    """
    Reweights image according to the amount of area each
    row of pixels will occupy if that image is scan converted
    with angle sector_angle.
    This 'image' could be e.g. a pixelwise loss or metric.

    We can compute this by viewing the scan converted image as the sector
    of a circle with a known central angle, and radius given by depth.
    See: https://en.wikipedia.org/wiki/Circular_sector

    Params:
        image (tensor of shape (img_height, img_width, channels)): image to be re-weighted
        sector_angle (float | int): angle in degrees
    Returns:
        reweighted_image (tensor of shape (img_height, img_width, channels)): image according with pixels reweighted
            to area occupied by each pixel post-scan-conversion.
    """
    assert len(image.shape) == 3, "image should have shape (height, width, channels)"
    height = image.shape[0]
    depths = (
        np.arange(height) + 0.5
    )  # add 0.5 to measure the center of the pixel as its depth
    # Reweighting factor is set as the arc length of the sector: https://en.wikipedia.org/wiki/Circular_sector#Arc_length
    reweighting_factors = (sector_angle / 360) * 2 * np.pi * depths
    return reweighting_factors[:, None] * image