import numpy as np
import pytest

from zea.data.spec import Segmentation


def test_segmentation_spec():
    # Correct usage
    pixels = np.zeros((10, 256, 256, 1), dtype=np.uint8)
    labels = np.array(["background", "label1", "label2", "label3"], dtype=np.str_)
    extent = np.array([0.0, 1.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    segmentation = Segmentation(pixels=pixels, labels=labels, extent=extent)
    assert segmentation.pixels.shape == (10, 256, 256, 1)
    assert segmentation.labels.shape == (4,)
    assert segmentation.extent.shape == (6,)

    # Incorrect usage: pixel values do not correspond to labels
    pixels_invalid = np.array([[[[0], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]]]], dtype=np.uint8)
    with pytest.raises(
        ValueError, match="Segmentation pixels contain values that do not correspond to any label"
    ):
        Segmentation(pixels=pixels_invalid, labels=labels, extent=extent)
