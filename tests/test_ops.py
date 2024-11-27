"""
Tests for some of the usbmd.ops functions.
"""

import numpy as np
import pytest
from keras import ops
from scipy.ndimage import gaussian_filter

from usbmd.ops import GaussianBlur, LeeFilter


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_gaussian_blur(sigma):
    # output for default args should be equalivalent to scipy.ndimage.gaussian_filter
    blur = GaussianBlur(sigma=sigma, with_batch_dim=False)

    rng = np.random.default_rng(seed=42)
    image = rng.normal(size=(32, 32)).astype(np.float32)
    image_tensor = ops.convert_to_tensor(image[..., None])

    blurred_scipy = gaussian_filter(image, sigma=sigma)
    blurred_usbmd = blur(image_tensor)[..., 0]

    np.testing.assert_allclose(blurred_scipy, blurred_usbmd, rtol=1e-4)


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_lee_filter(sigma):
    rng = np.random.default_rng(seed=42)
    image = rng.normal(size=(32, 32)).astype(np.float32)

    lee = LeeFilter(sigma=sigma, with_batch_dim=False)

    image_tensor = ops.convert_to_tensor(image[..., None])
    filtered = lee(image_tensor)[..., 0]

    assert ops.var(filtered) < ops.var(image_tensor), "Variance should be reduced"
