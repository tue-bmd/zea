"""Tests for the UltraPIPS perceptual similarity metric.

Most tests run on random weights and check the structure of the metric and of
the TUSA Swin encoder. The regression test against the upstream PyTorch
implementation needs the trained weights from the preset.
"""

import os

import numpy as np
import pytest
from keras import ops

from zea.models.ultrapips import (
    SwinViT,
    UltraPIPS,
    load_torch_state_dict,
    relative_position_index,
    shifted_window_mask,
    window_partition,
    window_reverse,
)

IMAGE_SHAPE = (64, 48, 1)


@pytest.fixture(scope="module")
def model():
    """UltraPIPS with random weights (module-scoped: building the encoder is the slow part)."""
    model = UltraPIPS()
    model.build()
    return model


def test_only_tusa_is_supported():
    """The other backbones of the reference implementation were not ported."""
    with pytest.raises(ValueError, match="Unknown UltraPIPS backbone"):
        UltraPIPS(backbone="usfm")


def test_weights_are_frozen(model):
    """UltraPIPS is a metric, not something to train."""
    assert model.trainable is False


def test_parameter_count_matches_the_tusa_encoder(model):
    """Same number of parameters as the ``swinViT`` of the TUSA SwinUNETR."""
    assert model.count_params() == 9_120_756


def test_encoder_returns_five_feature_maps():
    """Patch embedding plus one map per stage, at halving resolutions and doubling width."""
    net = SwinViT()
    outputs = net(np.zeros((1, 128, 128, 1), dtype="float32"))

    assert [tuple(out.shape) for out in outputs] == [
        (1, 64, 64, 36),
        (1, 32, 32, 72),
        (1, 16, 16, 144),
        (1, 8, 8, 288),
        (1, 4, 4, 576),
    ]


def test_encoder_handles_maps_not_divisible_by_the_window(rng):
    """Feature maps are padded to whole windows, and small maps shrink the window."""
    net = SwinViT(feature_size=12, depths=(2, 2, 2, 2), num_heads=(1, 2, 4, 8))
    x = rng.standard_normal((1, 30, 22, 1)).astype("float32")

    outputs = net(x)

    assert [tuple(out.shape[1:3]) for out in outputs] == [(15, 11), (8, 6), (4, 3), (2, 2), (1, 1)]
    assert all(np.isfinite(ops.convert_to_numpy(out)).all() for out in outputs)


def test_window_reverse_inverts_window_partition(rng):
    x = rng.standard_normal((2, 14, 21, 3)).astype("float32")

    windows = window_partition(x, (7, 7))
    restored = window_reverse(windows, (7, 7), 14, 21)

    assert tuple(windows.shape) == (2 * 2 * 3, 49, 3)
    np.testing.assert_array_equal(ops.convert_to_numpy(restored), x)


def test_relative_position_index_covers_the_bias_table():
    """Every relative offset in a 7x7 window maps to one of the 13 * 13 table rows."""
    index = relative_position_index((7, 7))

    assert index.shape == (49, 49)
    assert set(np.unique(index)) == set(range(13 * 13))
    # A token's offset to itself is the center of the table
    np.testing.assert_array_equal(np.diag(index), 6 * 13 + 6)


def test_shifted_window_mask_only_blocks_wrapped_regions():
    """Windows in the interior are unmasked; the wrapped-around border windows are split."""
    mask = shifted_window_mask(14, 14, (7, 7), (3, 3))

    assert mask.shape == (4, 49, 49)
    assert set(np.unique(mask)) == {0.0, -100.0}
    np.testing.assert_array_equal(mask[0], 0.0)
    assert (mask[3] != 0).any()
    np.testing.assert_array_equal(mask, np.swapaxes(mask, 1, 2))


class TestCall:
    """The metric itself."""

    def test_identical_images_have_zero_distance(self, model, rng):
        x = rng.uniform(-1, 1, (2, *IMAGE_SHAPE)).astype("float32")

        distance = ops.convert_to_numpy(model([x, x]))

        assert distance.shape == (2,)
        np.testing.assert_allclose(distance, 0.0, atol=1e-6)

    def test_different_images_have_a_positive_distance(self, model, rng):
        """Without a learned head the distance is a sum of squares, so never negative."""
        x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")

        distance = ops.convert_to_numpy(model([x, -x]))

        assert distance[0] > 0.0

    def test_is_symmetric(self, model, rng):
        x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
        y = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")

        np.testing.assert_allclose(
            ops.convert_to_numpy(model([x, y])), ops.convert_to_numpy(model([y, x])), rtol=1e-5
        )

    def test_samples_in_a_batch_are_independent(self, model, rng):
        """Batching only stacks samples: every normalization in the encoder is per sample."""
        x = rng.uniform(-1, 1, (3, *IMAGE_SHAPE)).astype("float32")
        y = rng.uniform(-1, 1, (3, *IMAGE_SHAPE)).astype("float32")

        batched = ops.convert_to_numpy(model([x, y]))
        single = [float(ops.convert_to_numpy(model([x[i], y[i]]))) for i in range(3)]

        # Batched GPU kernels may reduce in a different order (~1e-5 relative on TF).
        np.testing.assert_allclose(batched, single, rtol=1e-4)

    def test_unbatched_input_gives_a_scalar(self, model, rng):
        x = rng.uniform(-1, 1, IMAGE_SHAPE).astype("float32")

        assert ops.convert_to_numpy(model([x, x])).shape == ()

    def test_gray_rgb_is_the_same_as_one_channel(self, model, rng):
        """RGB input is converted to grayscale, and a gray RGB image stays the same image."""
        x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
        y = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
        rgb = lambda a: np.repeat(a, 3, axis=-1)  # noqa: E731

        np.testing.assert_allclose(
            ops.convert_to_numpy(model([rgb(x), rgb(y)])),
            ops.convert_to_numpy(model([x, y])),
            rtol=1e-3,
        )

    def test_preprocess_resizes_to_the_backbone_resolution(self, model, rng):
        x = rng.uniform(-1, 1, (2, *IMAGE_SHAPE)).astype("float32")

        assert tuple(model.preprocess_input(x).shape) == (2, 128, 128, 1)

    @pytest.mark.parametrize(
        "bad",
        [
            pytest.param(np.full((1, *IMAGE_SHAPE), 2.0, dtype="float32"), id="out_of_range"),
            pytest.param(np.zeros((1, 64, 48, 2), dtype="float32"), id="bad_channels"),
            pytest.param(np.zeros((1, 1, 64, 48, 1), dtype="float32"), id="too_many_dims"),
        ],
    )
    def test_rejects_input_that_is_not_a_normalized_image(self, model, rng, bad):
        good = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")

        with pytest.raises(ValueError, match=r"\[-1, 1\] range"):
            model([good, bad])


def _torch_state_dict_from(net):
    """Write a Keras encoder's weights out in the layout of the TUSA PyTorch checkpoint."""
    state = {}

    def conv(kernel):
        return np.transpose(kernel, (3, 2, 0, 1))

    state["swinViT.patch_embed.proj.weight"] = conv(net.patch_embed.kernel.numpy())
    state["swinViT.patch_embed.proj.bias"] = net.patch_embed.bias.numpy()
    for i, (res_block, stage) in enumerate(zip(net.res_blocks, net.stages)):
        res = f"swinViT.layers{i + 1}c.0.layer"
        state[f"{res}.conv1.conv.weight"] = conv(res_block.conv1.kernel.numpy())
        state[f"{res}.conv2.conv.weight"] = conv(res_block.conv2.kernel.numpy())
        layer = f"swinViT.layers{i + 1}.0"
        for j, block in enumerate(stage.blocks):
            key = f"{layer}.blocks.{j}"
            for name, norm in (("norm1", block.norm1), ("norm2", block.norm2)):
                state[f"{key}.{name}.weight"] = norm.gamma.numpy()
                state[f"{key}.{name}.bias"] = norm.beta.numpy()
            attn = block.attn
            state[f"{key}.attn.relative_position_bias_table"] = (
                attn.relative_position_bias_table.numpy()
            )
            state[f"{key}.attn.relative_position_index"] = relative_position_index((7, 7))
            for name, dense in (
                ("attn.qkv", attn.qkv),
                ("attn.proj", attn.proj),
                ("mlp.linear1", block.fc1),
                ("mlp.linear2", block.fc2),
            ):
                state[f"{key}.{name}.weight"] = dense.kernel.numpy().T
                state[f"{key}.{name}.bias"] = dense.bias.numpy()
        state[f"{layer}.downsample.norm.weight"] = stage.downsample.norm.gamma.numpy()
        state[f"{layer}.downsample.norm.bias"] = stage.downsample.norm.beta.numpy()
        state[f"{layer}.downsample.reduction.weight"] = stage.downsample.reduction.kernel.numpy().T
    return state


class TestLoadTorchStateDict:
    """Conversion of the TUSA PyTorch checkpoint."""

    def test_round_trips_the_weights(self, rng):
        source = SwinViT()
        source.build((None, 128, 128, 1))
        state = _torch_state_dict_from(source)
        # Decoder weights of the full SwinUNETR are ignored
        state["encoder1.layer.conv1.conv.weight"] = np.zeros((36, 1, 3, 3))

        target = SwinViT()
        target.build((None, 128, 128, 1))
        load_torch_state_dict(target, state)

        x = rng.standard_normal((1, 128, 128, 1)).astype("float32")
        for out, expected in zip(target(x), source(x)):
            np.testing.assert_allclose(
                ops.convert_to_numpy(out), ops.convert_to_numpy(expected), atol=1e-6
            )

    def test_rejects_a_state_dict_with_unknown_encoder_keys(self):
        net = SwinViT()
        net.build((None, 128, 128, 1))
        state = _torch_state_dict_from(net)
        state["swinViT.layers5.0.blocks.0.norm1.weight"] = np.zeros(36)

        with pytest.raises(KeyError, match="layers5"):
            load_torch_state_dict(net, state)


def test_round_trips_through_a_local_preset(model, tmp_path, rng):
    """``save_to_preset`` and ``from_preset`` give back the same metric."""
    model.save_to_preset(str(tmp_path))
    reloaded = UltraPIPS.from_preset(str(tmp_path))

    x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
    y = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
    np.testing.assert_allclose(
        ops.convert_to_numpy(reloaded([x, y])), ops.convert_to_numpy(model([x, y])), rtol=1e-6
    )


def test_loads_the_torch_checkpoint_from_a_preset(model, tmp_path, rng):
    """``custom_load_weights(..., backend="torch")`` converts the ``unet.pt`` in the preset."""
    torch = pytest.importorskip("torch")
    model.save_to_preset(str(tmp_path))
    state = _torch_state_dict_from(model.net)
    torch.save(
        {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in state.items()},
        tmp_path / "unet.pt",
    )

    reloaded = UltraPIPS()
    reloaded.custom_load_weights(str(tmp_path), backend="torch")

    x = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
    y = rng.uniform(-1, 1, (1, *IMAGE_SHAPE)).astype("float32")
    np.testing.assert_allclose(
        ops.convert_to_numpy(reloaded([x, y])), ops.convert_to_numpy(model([x, y])), rtol=1e-6
    )


def _regression_inputs():
    """Smooth anatomy-like pattern against noisy, speckled and inverted versions of it."""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:96, 0:112] / 96.0
    clean = 0.5 + 0.25 * np.sin(9 * xx + 2 * yy) * np.cos(6 * yy)
    speckle = clean * rng.rayleigh(0.8, size=(3,) + clean.shape)
    x = np.clip(np.stack([clean, clean, clean]), 0, 1)
    y = np.clip(
        np.stack([clean + 0.05 * rng.standard_normal(clean.shape), speckle[1], 1 - clean]), 0, 1
    )
    return x[..., None].astype("float32"), y[..., None].astype("float32")


@pytest.mark.heavy
def test_matches_the_reference_implementation():
    """Distances of the upstream PyTorch UltraPIPS (``backbone="tusa_vit"``).

    Computed with https://github.com/talg2324/UltraPIPS on the same inputs,
    which it takes in ``[0, 1]`` and channels-first.
    """
    model = UltraPIPS.from_preset(os.environ.get("ZEA_ULTRAPIPS_PRESET", "ultrapips-tusa"))
    x, y = _regression_inputs()

    distance = ops.convert_to_numpy(model([x * 2 - 1, y * 2 - 1]))

    np.testing.assert_allclose(distance, [0.25759995, 1.7587278, 5.3710604], rtol=1e-4)
