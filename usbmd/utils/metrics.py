"""
==============================================================================
    Eindhoven University of Technology
==============================================================================

    Source Name   : metrics.py

    Author(s)     : Tristan Stevens, Ben Luijten
    Date          : 18 Nov 2021

==============================================================================
"""

import numpy as np

def cnr(x, y):
    """Calculate contrast to noise ratio"""
    mu_x = np.mean(x)
    mu_y = np.mean(y)

    var_x = np.var(x)
    var_y = np.var(y)

    return 20 * np.log10(np.abs(mu_x - mu_y) / np.sqrt((var_x + var_y) / 2))

def contrast(x, y):
    """Contrast ratio"""
    return 20 * np.log10(x.mean() / y.mean())

def gcnr(x, y):
    """Generalized contrast-to-noise-ratio"""
    _, bins = np.histogram(np.concatenate((x, y)), bins=256)
    f, _ = np.histogram(x, bins=bins, density=True)
    g, _ = np.histogram(y, bins=bins, density=True)
    f /= f.sum()
    g /= g.sum()
    return 1 - np.sum(np.minimum(f, g))

def fwhm(img):
    """Resolution full width half maxima"""
    mask = np.nonzero(img >= 0.5 * np.amax(img))[0]
    return mask[-1] - mask[0]

def speckle_res(img):
    """TODO: Write speckle edge-spread function resolution code"""
    raise NotImplementedError

def snr(img):
    """Signal to noise ratio"""
    return img.mean() / img.std()

def wopt_mae(ref, img):
    """Find the optimal weight that minimizes the mean absolute error"""
    wopt = np.median(ref / img)
    return wopt

def wopt_mse(ref, img):
    """Find the optimal weight that minimizes the mean squared error"""
    wopt = np.sum(ref * img) / np.sum(img * img)
    return wopt

def l1loss(x, y):
    """L1 loss"""
    return np.abs(x - y).mean()

def l2loss(x, y):
    """L2 loss"""
    return np.sqrt(((x - y) ** 2).mean())

def psnr(x, y):
    """Peak signal to noise ratio"""
    dynamic_range = max(x.max(), y.max()) - min(x.min(), y.min())
    return 20 * np.log10(dynamic_range / l2loss(x, y))

def ncc(x, y):
    """Normalized cross correlation"""
    return (x * y).sum() / np.sqrt((x ** 2).sum() * (y ** 2).sum())


if __name__ == "__main__":
    x = np.random.rayleigh(2, (80, 50))
    y = np.random.rayleigh(1, (80, 50))

    print(f'Contrast [dB]:  {(20 * np.log10(contrast(x, y)))}')
    print(f'CNR:            {cnr(x, y)}')
    print(f'SNR:            {snr(x)}')
    print(f'GCNR:           {gcnr(x, y)}')
    print(f'L1 Loss:        {l1loss(x, y)}')
    print(f'L2 Loss:        {l2loss(x, y)}')
    print(f'PSNR [dB]:      {psnr(x, y)}')
    print(f'NCC:            {ncc(x, y)}')
