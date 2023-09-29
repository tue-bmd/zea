"""This module defines quality metrics for ultrasound images.

- **Author(s)**     : Tristan Stevens, Ben Luijten
- **Date**          : 18 Nov 2021
"""

import numpy as np

_METRICS = {}


def register_metric(func=None, *, name=None):
    """A decorator for registering metric functions."""

    def _register(func):
        if name is None:
            local_name = func.__name__
        else:
            local_name = name
        if local_name in _METRICS:
            raise ValueError(f"Already registered metric with name: {local_name}")
        _METRICS[local_name] = func
        return func

    if func is None:
        return _register
    else:
        return _register(func)


def get_metric(name):
    """Get metric function given name."""
    return _METRICS[name]


@register_metric(name="cnr")
def cnr(x, y):
    """Calculate contrast to noise ratio"""
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    return 20 * np.log10(np.abs(mu_x - mu_y) / np.sqrt((var_x + var_y) / 2))


@register_metric(name="contrast")
def contrast(x, y):
    """Contrast ratio"""
    return 20 * np.log10(x.mean() / y.mean())


@register_metric(name="gcnr")
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


@register_metric(name="fwhm")
def fwhm(img):
    """Resolution full width half maxima"""
    mask = np.nonzero(img >= 0.5 * np.amax(img))[0]
    return mask[-1] - mask[0]


@register_metric(name="speckle_res")
def speckle_res(img):
    """TODO: Write speckle edge-spread function resolution code"""
    raise NotImplementedError


@register_metric(name="snr")
def snr(img):
    """Signal to noise ratio"""
    return img.mean() / img.std()


@register_metric(name="wopt_mae")
def wopt_mae(ref, img):
    """Find the optimal weight that minimizes the mean absolute error"""
    wopt = np.median(ref / img)
    return wopt


@register_metric(name="wopt_mse")
def wopt_mse(ref, img):
    """Find the optimal weight that minimizes the mean squared error"""
    wopt = np.sum(ref * img) / np.sum(img * img)
    return wopt


@register_metric(name="l1")
def l1loss(x, y):
    """L1 loss"""
    return np.abs(x - y).mean()


@register_metric(name="l2")
def l2loss(x, y):
    """L2 loss"""
    return np.sqrt(((x - y) ** 2).mean())


@register_metric(name="psnr")
def psnr(x, y):
    """Peak signal to noise ratio"""
    dynamic_range = max(x.max(), y.max()) - min(x.min(), y.min())
    return 20 * np.log10(dynamic_range / l2loss(x, y))


@register_metric(name="ncc")
def ncc(x, y):
    """Normalized cross correlation"""
    return (x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum())


@register_metric(name="image_entropy")
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


@register_metric(name="image_sharpness")
def image_sharpness(image):
    """Calculate the sharpness of the image

    Args:
        image (ndarray): The image for which the sharpness is calculated

    Returns:
        float: The sharpness of the image
    """
    return np.mean(np.abs(np.gradient(image)))


if __name__ == "__main__":
    x = np.random.rayleigh(2, (80, 50))
    y = np.random.rayleigh(1, (80, 50))

    print(f"Contrast [dB]:  {(20 * np.log10(contrast(x, y)))}")
    print(f"CNR:            {cnr(x, y)}")
    print(f"SNR:            {snr(x)}")
    print(f"GCNR:           {gcnr(x, y)}")
    print(f"L1 Loss:        {l1loss(x, y)}")
    print(f"L2 Loss:        {l2loss(x, y)}")
    print(f"PSNR [dB]:      {psnr(x, y)}")
    print(f"NCC:            {ncc(x, y)}")
