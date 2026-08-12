"""Tests for the LPIPS perceptual similarity metric.

The trained linear head only exists in the preset, so with random weights the
metric is not meaningful in absolute terms. What *is* testable offline is its
structure: identical inputs are at distance zero, the batch dimension is handled,
and the input checks fire.
"""

from pathlib import Path

import numpy as np
import pytest
from keras import ops

from zea.models.lpips import LPIPS, linear_model, perceptual_model

IMAGE_SHAPE = (32, 32, 3)


@pytest.fixture(scope="module")
def model():
    """LPIPS with random weights (module-scoped: building VGG16 is the slow part)."""
    return LPIPS()


def test_only_vgg_is_supported():
    """The other LPIPS backbones were never ported."""
    with pytest.raises(AssertionError, match="Only VGG"):
        LPIPS(net_type="alex")


def test_weights_are_frozen(model):
    """LPIPS is a metric, not something to train."""
    assert model.trainable is False


def test_perceptual_model_returns_five_feature_maps():
    """One feature map per VGG16 block, at halving resolutions."""
    net = perceptual_model()
    outputs = net(np.zeros((1, 32, 32, 3), dtype="float32"))

    assert [tuple(out.shape) for out in outputs] == [
        (1, 32, 32, 64),
        (1, 16, 16, 128),
        (1, 8, 8, 256),
        (1, 4, 4, 512),
        (1, 2, 2, 512),
    ]


def test_linear_model_projects_each_feature_map_to_one_channel():
    """The learned head is a 1x1 convolution per VGG16 block."""
    lin = linear_model()
    features = [
        np.zeros((1, 8, 8, channels), dtype="float32") for channels in (64, 128, 256, 512, 512)
    ]

    outputs = lin(features)

    assert [tuple(out.shape) for out in outputs] == [(1, 8, 8, 1)] * 5


def test_normalize_tensor_gives_unit_norm_along_channels():
    """Features are unit-normalized per pixel before being compared."""
    features = np.arange(2 * 3, dtype="float32").reshape(1, 1, 2, 3) + 1.0

    normalized = ops.convert_to_numpy(LPIPS._normalize_tensor(features))

    np.testing.assert_allclose(np.linalg.norm(normalized, axis=-1), 1.0, rtol=1e-5)


def test_preprocess_input_standardizes_with_the_vgg_statistics():
    """The input is shifted and scaled with the constants the VGG weights expect."""
    image = np.zeros((1, 2, 2, 3), dtype="float32")

    out = ops.convert_to_numpy(LPIPS.preprocess_input(image))

    expected = -np.array([-0.030, -0.088, -0.188]) / np.array([0.458, 0.448, 0.450])
    np.testing.assert_allclose(out[0, 0, 0], expected, rtol=1e-5)


class TestCall:
    """The metric itself."""

    def test_identical_images_have_zero_distance(self, model, rng):
        x = rng.uniform(-1, 1, (2, *IMAGE_SHAPE)).astype("float32")

        distance = ops.convert_to_numpy(model([x, x]))

        assert distance.shape == (2,)
        np.testing.assert_allclose(distance, 0.0, atol=1e-6)

    def test_different_images_have_a_nonzero_distance(self, model, rng):
        """The sign only becomes meaningful with the trained (non-negative) head."""
        x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")

        distance = ops.convert_to_numpy(model([x, -x]))

        assert distance[0] != 0.0

    def test_unbatched_input_gives_a_scalar(self, model, rng):
        """Without a batch dimension in, there is no batch dimension out."""
        x = rng.uniform(-1, 1, IMAGE_SHAPE).astype("float32")

        distance = ops.convert_to_numpy(model([x, x]))

        assert distance.shape == ()

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(np.full((1, *IMAGE_SHAPE), 2.0, dtype="float32"), id="out_of_range"),
            pytest.param(np.zeros((1, 32, 32, 2), dtype="float32"), id="bad_channels"),
            pytest.param(np.zeros((1, 1, 32, 32, 3), dtype="float32"), id="too_many_dims"),
        ],
    )
    def test_rejects_input_that_is_not_a_normalized_image(self, model, rng, bad):
        """Inputs are expected to be [-1, 1] images with one or three channels."""
        good = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")

        with pytest.raises(ValueError, match=r"\[-1, 1\] range"):
            model([good, bad])

    def test_checks_can_be_disabled(self, rng):
        """``disable_checks`` exists so the metric can run inside a TensorFlow graph."""
        model = LPIPS(disable_checks=True)
        x = rng.uniform(0, 255, (1, *IMAGE_SHAPE)).astype("float32")

        assert ops.convert_to_numpy(model([x, x])).shape == (1,)


def test_custom_load_weights_reads_both_halves_from_the_preset(model, local_preset):
    """The VGG backbone and the learned linear head are separate files in the preset."""
    preset = local_preset(LPIPS)
    for name in ("vgg/vgg.weights.h5", "lin/lin.weights.h5"):
        path = Path(preset) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    loaded = {}
    model.net.load_weights = lambda filepath, **kwargs: loaded.setdefault("net", filepath)
    model.lin.load_weights = lambda filepath, **kwargs: loaded.setdefault("lin", filepath)
    try:
        model.custom_load_weights(preset)
    finally:
        del model.net.load_weights, model.lin.load_weights

    assert loaded["net"].endswith("vgg/vgg.weights.h5")
    assert loaded["lin"].endswith("lin/lin.weights.h5")
