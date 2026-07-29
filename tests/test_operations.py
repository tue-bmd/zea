"""Tests for different operations.

Note that in each test, we have to reimport all modules involving keras, such
that when the backend is switched, the functions inside are reimported with the
correct backend.
"""

import math
from unittest import mock

import keras
import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert as hilbert_scipy

from zea import ops
from zea.beamform.delays import compute_t0_delays_focused
from zea.func.ultrasound import (
    channels_to_complex,
    complex_to_channels,
    compute_time_to_peak_stack,
    construct_acquisition_from_synthetic_aperture,
    decode_hadamard,
    make_tgc_curve,
)
from zea.ops import Pipeline, Simulate, beamformer_registry
from zea.parameters import Parameters
from zea.simulator import simulate_rf

from . import DEFAULT_TEST_SEED, backend_equality_check


@pytest.mark.parametrize(
    "comp_type, size, parameter_value_range",
    [
        ("a", (2, 1, 128, 32), (50, 200)),
        ("a", (512, 512), (50, 200)),
        ("mu", (2, 1, 128, 32), (50, 300)),
        ("mu", (512, 512), (50, 300)),
    ],
)
@backend_equality_check(decimal=4)
def test_companding(comp_type, size, parameter_value_range):
    """Test companding function"""

    import keras

    from zea import ops

    rng = np.random.default_rng(42)

    for parameter_value in np.linspace(*parameter_value_range, 10):
        A = parameter_value if comp_type == "a" else 0
        mu = parameter_value if comp_type == "mu" else 0

        companding = ops.Companding(comp_type=comp_type, expand=False)
        rng = np.random.default_rng(DEFAULT_TEST_SEED)
        signal = np.clip((rng.standard_normal(size) - 0.5) * 2, -1, 1)
        signal = signal.astype("float32")
        signal = keras.ops.convert_to_tensor(signal)

        signal_out = companding(data=signal, A=A, mu=mu)["data"]

        signal = keras.ops.convert_to_numpy(signal)
        signal_out = keras.ops.convert_to_numpy(signal_out)

        assert np.any(np.not_equal(signal, signal_out)), (
            "Companding failed, arrays should not be equal"
        )

        companding = ops.Companding(comp_type=comp_type, expand=True)

        signal_out = keras.ops.convert_to_tensor(signal_out)
        signal_out = companding(data=signal_out, A=A, mu=mu)["data"]

        signal = keras.ops.convert_to_numpy(signal)
        signal_out = keras.ops.convert_to_numpy(signal_out)

        np.testing.assert_almost_equal(signal, signal_out, decimal=6)
    return signal_out


@pytest.mark.parametrize(
    "size, dynamic_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), None),
        ((512, 512), None, (0, 1)),
        ((512, 600), None, (-10, 300)),
        ((1, 128, 32), None, None),
    ],
)
@backend_equality_check(decimal=4)
def test_converting_to_image(size, dynamic_range, input_range):
    """Test converting to image functions"""

    import keras

    from zea import ops

    if dynamic_range is None:
        _dynamic_range = (-60, 0)
    else:
        _dynamic_range = dynamic_range
    if input_range is None:
        _input_range = (0, 1)
    else:
        _input_range = input_range

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal(size) * (_input_range[1] - _input_range[0]) + _input_range[0]
    output_range = (0, 1)
    normalize = ops.Normalize(output_range, input_range)
    log_compress = ops.LogCompress()

    data = keras.ops.convert_to_tensor(data)

    _data = normalize(data=data)["data"]
    _data = log_compress(data=_data, dynamic_range=dynamic_range)["data"]

    _data = keras.ops.convert_to_numpy(_data)

    # data should be in dynamic range
    assert np.all(
        np.logical_and(_data >= _dynamic_range[0], _data <= _dynamic_range[1]),
    ), f"Data is not in dynamic range after converting to image {_dynamic_range}"
    return _data


@pytest.mark.parametrize(
    "size, output_range, input_range",
    [
        ((2, 1, 128, 32), (-30, -5), (0, 1)),
        ((512, 512), (-2, -1), (-3, 50)),
        ((1, 128, 32), (50, 51), (-2.2, 3.0)),
    ],
)
@backend_equality_check(decimal=4)
def test_normalize(size, output_range, input_range):
    """Test normalize function"""

    import keras

    from zea import ops

    normalize = ops.Normalize(output_range, input_range)

    _input_range = output_range
    _output_range = input_range
    normalize_back = ops.Normalize(_output_range, _input_range)

    # create random data between input range
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.random(size) * (input_range[1] - input_range[0]) + input_range[0]

    data = keras.ops.convert_to_tensor(data)

    _data = normalize(data=data)["data"]

    input_range, output_range = output_range, input_range
    _data = normalize_back(data=_data)["data"]

    _data = keras.ops.convert_to_numpy(_data)
    data = keras.ops.convert_to_numpy(data)

    np.testing.assert_almost_equal(data, _data, decimal=4)
    return _data


@pytest.mark.parametrize(
    "size, axis",
    [
        ((2, 1, 128, 32), (-1)),
        ((512, 512), (-1)),
        ((1, 128, 32), (-1)),
    ],
)
def test_complex_to_channels(size, axis):
    """Test complex to channels and back"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.random(size) + 1j * rng.random(size)
    _data = complex_to_channels(data, axis=axis)
    __data = channels_to_complex(_data)
    np.testing.assert_almost_equal(data, __data)


@pytest.mark.parametrize(
    "size, axis",
    [
        ((222, 1, 2, 2), (-1)),
        ((512, 512, 2), (-1)),
        ((2, 20, 128, 2), (-1)),
    ],
)
def test_channels_to_complex(size, axis):
    """Test channels to complex and back"""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.random(size)
    _data = channels_to_complex(data)
    __data = complex_to_channels(_data, axis=axis)
    np.testing.assert_almost_equal(data, __data)


@pytest.mark.parametrize(
    "size, axis",
    [
        ((2, 1, 128, 32), -1),
        ((2, 20, 8), 1),
        ((512, 512), -1),
    ],
)
@backend_equality_check(decimal=5)
def test_complex_channels_operations(size, axis):
    """Round-trip the ComplexToChannels and ChannelsToComplex operations.

    ``ComplexToChannels`` splits complex data into two real channels placed at
    ``axis``, while ``ChannelsToComplex`` reads the two channels from the last
    axis, so we move them back before converting to complex again.
    """
    import keras

    from zea import ops

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    complex_data = (rng.random(size) + 1j * rng.random(size)).astype("complex64")

    channels = ops.ComplexToChannels(axis=axis)(data=keras.ops.convert_to_tensor(complex_data))[
        "data"
    ]
    channels = keras.ops.convert_to_numpy(channels)
    assert channels.shape[axis] == 2, "Real/imaginary channels should live on `axis`."

    restored = ops.ChannelsToComplex()(
        data=keras.ops.convert_to_tensor(np.moveaxis(channels, axis, -1))
    )["data"]
    restored = keras.ops.convert_to_numpy(restored)

    np.testing.assert_almost_equal(restored, complex_data, decimal=5)
    return restored


# NOTE: torch is excluded because keras' correlate drops the complex dtype on the
# torch backend, which the complex_channels path relies on.
@backend_equality_check(decimal=4, backends=["tensorflow", "jax"])
def test_fir_filter_complex_channels():
    """FirFilter with complex_channels=True filters IQ data given as two real channels.

    Because the filter taps are real, filtering the complex signal is equivalent to
    filtering the real and imaginary channels independently, so the complex-channel
    path must match a plain real-valued FirFilter applied along the same axis.
    """
    import keras

    from zea import ops

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # (batch, n_ax, complex_channels) — filter along the n_ax axis.
    signal = rng.standard_normal((2, 64, 2)).astype("float32")
    taps = rng.standard_normal((7,)).astype("float32")
    signal_tensor = keras.ops.convert_to_tensor(signal)
    taps_tensor = keras.ops.convert_to_tensor(taps)

    result = ops.FirFilter(axis=1, complex_channels=True, with_batch_dim=True)(
        data=signal_tensor, fir_filter_taps=taps_tensor
    )["data"]
    result = keras.ops.convert_to_numpy(result)

    assert result.shape == signal.shape, "Real/imaginary channels should be restored."
    assert not np.allclose(result, signal), "Filtering should change the signal."

    reference = ops.FirFilter(axis=1, complex_channels=False, with_batch_dim=True)(
        data=signal_tensor, fir_filter_taps=taps_tensor
    )["data"]
    reference = keras.ops.convert_to_numpy(reference)

    np.testing.assert_almost_equal(result, reference, decimal=4)

    # Last axis holds the complex channels, so it may not be the filter axis.
    with pytest.raises(AssertionError):
        ops.FirFilter(axis=-1, complex_channels=True, with_batch_dim=True)(
            data=signal_tensor, fir_filter_taps=taps_tensor
        )

    return result


@pytest.mark.parametrize(
    "factor, batch_size",
    [
        (1, 2),
        (4, 1),
        (2, 3),
    ],
)
def test_up_and_down_conversion(factor, batch_size):
    """Test rf2iq and iq2rf in sequence"""
    n_el = 128
    n_scat = 3
    n_tx = 2
    n_ax = 512

    aperture = 30e-3

    tx_apodizations = np.ones((n_tx, n_el))
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    t0_delays = np.stack(
        [
            np.linspace(0, 1e-6, n_el),
            np.linspace(1e-6, 0, n_el),
        ]
    )

    parameters = Parameters(
        n_tx=n_tx,
        n_ax=n_ax,
        n_el=n_el,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        apply_lens_correction=True,
        lens_sound_speed=1440.0,
        lens_thickness=1e-3,
        initial_times=np.zeros((n_tx,)),
        attenuation_coef=0.7,
        n_ch=1,
        selected_transmits="all",
    )

    # use pipeline here so it is easy to propagate the scan parameters
    simulator_pipeline = Pipeline(
        [
            Simulate(output_key="simulated_data"),
            ops.Demodulate(key="simulated_data", output_key="data"),
            ops.Downsample(factor=factor),
            ops.UpMix(upsampling_rate=factor),
        ]
    )
    inputs = simulator_pipeline.prepare_parameters(parameters)

    data = []
    _data = []
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    for _ in range(batch_size):
        # Define scatterers with random variation
        scat_x_base, scat_z_base = np.meshgrid(
            np.linspace(-10e-3, 10e-3, 5),
            np.linspace(5e-3, 30e-3, 5),
            indexing="ij",
        )
        # Add random perturbations
        scat_x = np.ravel(scat_x_base) + rng.uniform(-1e-3, 1e-3, 25)
        scat_z = np.ravel(scat_z_base) + rng.uniform(-1e-3, 1e-3, 25)
        n_scat = len(scat_x)
        # Select random subset of scatterers
        idx = rng.choice(n_scat, n_scat, replace=False)[:n_scat]
        scat_positions = np.stack(
            [
                scat_x[idx],
                np.zeros_like(scat_x[idx]),
                scat_z[idx],
            ],
            axis=1,
        )
        scat_positions = np.expand_dims(scat_positions, axis=0)  # add batch dimension

        output = simulator_pipeline(
            **inputs,
            scatterer_positions=scat_positions.astype(np.float32),
            scatterer_magnitudes=np.ones((1, n_scat), dtype=np.float32),
        )

        data.append(keras.ops.convert_to_numpy(output["data"]))
        _data.append(keras.ops.convert_to_numpy(output["simulated_data"]))

    data = np.concatenate(data)
    _data = np.concatenate(_data)

    np.testing.assert_almost_equal(
        data,
        _data,
        decimal=2,
        err_msg="Data is not equal after up and down conversion.",
    )


@pytest.mark.parametrize(
    "shape, axis, N",
    [
        # 1D tests
        ((500,), -1, None),
        ((500,), -1, 500),
        ((500,), -1, 1024),
        # 2D tests
        ((128, 500), -1, None),
        ((128, 500), -1, 512),
        # 3D tests
        ((2, 500, 128), 1, None),
        ((2, 500, 128), 1, 600),
        # 4D tests (original test case)
        ((2, 500, 128, 1), -3, None),
        ((2, 500, 128, 1), -3, 512),
    ],
)
@backend_equality_check(decimal=4)
def test_hilbert_transform(shape, axis, N):
    """Test hilbert transform with various shapes and N values"""

    import keras

    from zea import func

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Create sinusoidal data along the specified axis
    n_samples = shape[axis]
    base_signal = np.sin(np.linspace(0, 2 * math.e * np.pi, n_samples))

    # Reshape to match target shape
    reshape_spec = [1] * len(shape)
    reshape_spec[axis] = n_samples
    data = base_signal.reshape(reshape_spec)

    # Tile to full shape
    tile_spec = list(shape)
    tile_spec[axis] = 1
    data = np.tile(data, tile_spec)

    # Add small noise
    data = data + rng.random(data.shape) * 0.1

    data = keras.ops.convert_to_tensor(data)

    data_iq = func.hilbert(data, N=N, axis=axis)
    assert keras.ops.dtype(data_iq) in [
        "complex64",
        "complex128",
    ], f"Data type should be complex, got {keras.ops.dtype(data_iq)} instead."

    data_iq = keras.ops.convert_to_numpy(data_iq)

    reference_data_iq = hilbert_scipy(data, N=N, axis=axis)
    np.testing.assert_almost_equal(reference_data_iq, data_iq, decimal=4)

    return data_iq


def test_hilbert_transform_invalid_N():
    """Test that hilbert raises ValueError when N < n_ax"""
    import keras

    from zea import func

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.random((100,))
    data = keras.ops.convert_to_tensor(data)

    # N=50 is less than n_ax=100, should raise ValueError
    with pytest.raises(ValueError, match="N must be greater or equal to n_ax"):
        func.hilbert(data, N=50, axis=-1)


@pytest.fixture(scope="module")
def spiral_image():
    """
    Fixture for generating a synthetic spiral image and noisy variants.
    Returns:
        dict: {
            "spiral": clean spiral image of size (64, 64),
            "noisy": additive Gaussian noise,
            "speckle": multiplicative speckle noise
        }
    """
    x = np.linspace(-1, 1, 64)
    y = np.linspace(-1, 1, 64)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    theta = np.arctan2(yv, xv)
    spiral = np.sin(8 * theta + 8 * r)
    spiral = (spiral - spiral.min()) / (spiral.max() - spiral.min())

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    noisy = spiral + 0.2 * rng.normal(size=spiral.shape)
    noisy = np.clip(noisy, 0, 1)
    speckle = spiral * (1 + 0.5 * rng.normal(size=spiral.shape))
    speckle = np.clip(speckle, 0, 1)

    return {
        "spiral": spiral.astype(np.float32),
        "noisy": noisy.astype(np.float32),
        "speckle": speckle.astype(np.float32),
    }


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
@backend_equality_check(decimal=4, backends=["tensorflow", "jax"])
def test_gaussian_blur(sigma, spiral_image):
    """
    Test `ops.GaussianBlur against scipy.ndimage.gaussian_filter.`
    `GaussianBlur` with default args should be equivalent to scipy.

    NOTE: We don't test torch backend here because of differences in padding behavior.
    """
    import keras

    from zea import ops

    blur = ops.GaussianBlur(sigma=sigma, with_batch_dim=False)

    # Use spiral image for testing
    image = spiral_image["spiral"]
    image_tensor = keras.ops.convert_to_tensor(image[..., None])

    blurred_scipy = gaussian_filter(image, sigma=sigma)
    blurred_zea = blur(data=image_tensor)["data"][..., 0]

    blurred_zea = keras.ops.convert_to_numpy(blurred_zea)

    np.testing.assert_allclose(blurred_scipy, blurred_zea, rtol=1e-5, atol=1e-5)

    return blurred_zea


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@backend_equality_check(decimal=5, backends=["tensorflow", "jax"])
def test_lee_filter(sigma, spiral_image):
    """
    Test `ops.LeeFilter`, checks if variance is reduced and if with and without
    batch dimension give the same result.

    # NOTE: We don't test torch backend here because of differences in padding behavior.
    """
    import keras

    from zea import ops

    # Use spiral image for testing
    image = spiral_image["spiral"]
    image = image[..., None]  # add channel dimension

    lee = ops.LeeFilter(sigma=sigma, with_batch_dim=False)
    lee_batched = ops.LeeFilter(sigma=sigma, with_batch_dim=True)

    image_tensor = keras.ops.convert_to_tensor(image)
    filtered = lee(data=image_tensor)["data"][..., 0]
    filtered_batched = lee_batched(data=image_tensor[None, ...])["data"][0, ..., 0]

    filtered = keras.ops.convert_to_numpy(filtered)
    filtered_batched = keras.ops.convert_to_numpy(filtered_batched)

    assert np.allclose(filtered, filtered_batched), (
        "LeeFilter with and without batch dim should give the same result."
    )

    assert np.var(filtered) < np.var(image), (
        "LeeFilter should reduce variance of the processed image"
    )

    return filtered


@pytest.mark.parametrize(
    "threshold_type,below_threshold,fill_value,threshold_param",
    [
        ("hard", True, "min", {"percentile": 50}),
        ("hard", False, "max", {"percentile": 75}),
        ("soft", True, 0.0, {"threshold": 0.5}),
        ("soft", False, 1.0, {"threshold": 0.3}),
    ],
)
@backend_equality_check()
def test_threshold_op(spiral_image, threshold_type, below_threshold, fill_value, threshold_param):
    """Test `ops.Threshold` operation on a synthetic spiral image."""
    import keras

    from zea import ops

    spiral = spiral_image["spiral"]
    spiral_tensor = keras.ops.convert_to_tensor(spiral)

    threshold = ops.Threshold(
        threshold_type=threshold_type,
        below_threshold=below_threshold,
        fill_value=fill_value,
    )

    # Set the correct parameter (either percentile or threshold) and the other to None
    percentile = threshold_param.get("percentile", None)
    threshold_value = threshold_param.get("threshold", None)

    out = threshold(data=spiral_tensor, percentile=percentile, threshold=threshold_value)
    out_np = keras.ops.convert_to_numpy(out["data"])

    # Quantitative: check that thresholding changes the image
    assert not np.allclose(spiral, out_np)

    return out_np


@pytest.mark.parametrize(
    "niter,lmbda",
    [
        (5, 0.5),
        (10, 0.25),
    ],
)
@backend_equality_check()
def test_anisotropic_diffusion_op(spiral_image, niter, lmbda):
    """Test `ops.AnisotropicDiffusion` operation on a noisy synthetic image."""

    import keras

    from zea import ops

    speckle = spiral_image["speckle"]
    speckle_tensor = keras.ops.convert_to_tensor(speckle)

    srad = ops.AnisotropicDiffusion(with_batch_dim=False)
    filtered = srad(data=speckle_tensor, niter=niter, lmbda=lmbda)
    filtered_np = keras.ops.convert_to_numpy(filtered["data"])

    # Quantitative: variance should be reduced, but mean should be similar
    assert np.var(filtered_np) < np.var(speckle)
    assert np.abs(np.mean(filtered_np) - np.mean(speckle)) < 0.1

    return filtered_np


def test_compute_time_to_peak():
    """Test compute_time_to_peak function."""
    t = np.arange(512) / 250e6

    t_peak = 1e-6

    center_frequency0 = 5e6
    waveform0 = np.exp(-((t - t_peak) ** 2) / (2 * (0.2e-6) ** 2)) * np.cos(
        2 * np.pi * center_frequency0 * (t - t_peak)
    )

    center_frequency1 = 10e6
    waveform1 = np.exp(-((t - t_peak) ** 2) / (2 * (0.2e-6) ** 2)) * np.cos(
        2 * np.pi * center_frequency1 * (t - t_peak)
    )

    center_frequencies = np.array([center_frequency0, center_frequency1])
    waveforms = np.array([waveform0, waveform1])

    t_peak = compute_time_to_peak_stack(waveforms, center_frequencies, 250e6)

    assert np.allclose(t_peak, 1e-6, atol=1e-8), f"t_peak should be close to 1e-6, got {t_peak}"


@pytest.mark.parametrize("beamformer_name", beamformer_registry.registered_names())
@backend_equality_check()
def test_beamformers(beamformer_name):
    """All beamformer operations produce the correct output shape and agree across backends."""

    import keras

    from zea.ops import beamformer_registry

    n_tx, n_pix, n_el, n_ch = 3, 7, 4, 2
    operation = beamformer_registry[beamformer_name](with_batch_dim=True)
    data = keras.ops.zeros((1, n_tx, n_pix, n_el, n_ch))
    output = operation(data=data)["data"]
    assert output.shape == (1, n_pix, n_ch), (
        f"Expected shape (1, {n_pix}, {n_ch}) for '{beamformer_name}', got {output.shape}"
    )
    return output


def test_dmas_requires_iq_data():
    """DMAS should raise ValueError when given single-channel (RF) data."""

    import keras

    from zea import ops

    operation = ops.DelayMultiplyAndSum(with_batch_dim=True)
    with pytest.raises(ValueError):
        data = keras.ops.zeros((1, 3, 7, 4, 1))
        operation(data=data)


@backend_equality_check()
def test_coherence_factor_coherent_signal():
    """CF weight is 1 for a perfectly coherent signal, so output matches DAS across backends."""

    import keras

    from zea import ops

    n_tx, n_pix, n_el, n_ch = 3, 7, 4, 2
    data = keras.ops.ones((1, n_tx, n_pix, n_el, n_ch))

    cf_out = ops.CoherenceFactor(with_batch_dim=True)(data=data)["data"]
    das_out = ops.DelayAndSum(with_batch_dim=True)(data=data)["data"]

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(cf_out),
        keras.ops.convert_to_numpy(das_out),
        rtol=1e-5,
    )
    return cf_out


@backend_equality_check()
def test_coherence_factor_effective_aperture_and_exponent():
    """Masked-out channels are excluded from N, and the exponent shapes the weight."""

    import keras

    from zea import ops

    n_tx, n_pix, n_el, n_ch = 3, 7, 4, 2
    # Half the aperture is masked out, e.g. by the f-number mask in TOF correction
    data = keras.ops.concatenate(
        [
            keras.ops.ones((1, n_tx, n_pix, n_el // 2, n_ch)),
            keras.ops.zeros((1, n_tx, n_pix, n_el // 2, n_ch)),
        ],
        axis=-2,
    )
    das_out = keras.ops.convert_to_numpy(ops.DelayAndSum(with_batch_dim=True)(data=data)["data"])

    # The contributing channels are perfectly coherent, so CF is 1 over the
    # effective aperture, but only n_el_eff / n_el = 0.5 over the full one.
    effective_out = ops.CoherenceFactor(effective_aperture=True, with_batch_dim=True)(data=data)[
        "data"
    ]
    full_out = ops.CoherenceFactor(with_batch_dim=True)(data=data)["data"]
    squared_out = ops.CoherenceFactor(exponent=2.0, with_batch_dim=True)(data=data)["data"]

    np.testing.assert_allclose(keras.ops.convert_to_numpy(effective_out), das_out, rtol=1e-5)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(full_out), 0.5 * das_out, rtol=1e-5)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(squared_out), 0.25 * das_out, rtol=1e-5)
    return effective_out


@backend_equality_check()
def test_generalized_coherence_factor_coherent_signal():
    """GCF weight is 1 for a perfectly coherent signal, so output matches DAS across backends."""

    import keras

    from zea import ops

    n_tx, n_pix, n_el, n_ch = 3, 7, 4, 2
    data = keras.ops.ones((1, n_tx, n_pix, n_el, n_ch))

    gcf_out = ops.GeneralizedCoherenceFactor(with_batch_dim=True)(data=data)["data"]
    das_out = ops.DelayAndSum(with_batch_dim=True)(data=data)["data"]

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(gcf_out),
        keras.ops.convert_to_numpy(das_out),
        rtol=1e-5,
    )
    return gcf_out


@backend_equality_check()
def test_generalized_coherence_factor_m_zero_passthrough():
    """m_zero passed via call kwargs overrides the instance default."""

    import keras

    from zea import ops

    n_tx, n_pix, n_el, n_ch = 3, 7, 8, 2
    rng = np.random.default_rng(42)
    data = keras.ops.convert_to_tensor(
        rng.standard_normal((1, n_tx, n_pix, n_el, n_ch)).astype(np.float32)
    )

    # m_zero=2 at init vs. m_zero=2 at call time should give identical results
    out_init = ops.GeneralizedCoherenceFactor(m_zero=2, with_batch_dim=True)(data=data)["data"]
    out_call = ops.GeneralizedCoherenceFactor(with_batch_dim=True)(data=data, m_zero=2)["data"]
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out_init),
        keras.ops.convert_to_numpy(out_call),
        rtol=1e-5,
    )

    # Different m_zero values should produce different results
    out_default = ops.GeneralizedCoherenceFactor(with_batch_dim=True)(data=data)["data"]
    out_m0 = ops.GeneralizedCoherenceFactor(with_batch_dim=True)(data=data, m_zero=0)["data"]
    assert not np.allclose(
        keras.ops.convert_to_numpy(out_default),
        keras.ops.convert_to_numpy(out_m0),
    ), "Expected different results for different m_zero values"

    return out_init


def _reference_capon(x, subarray_size, diagonal_loading, axial_k=0, axial_stride=1):
    """Textbook Capon/MVDR with forward spatial smoothing, in plain numpy.

    Args:
        x: complex array of shape ``(n_tx, n_pix, n_el)``.
        axial_k: half-width of the axial covariance averaging window.
        axial_stride: flat-index distance between axially adjacent pixels.

    Returns:
        Complex array of shape ``(n_pix,)``: sub-aperture-averaged, transmit-compounded.
    """
    n_tx, n_pix, n_el = x.shape
    num_subarrays = n_el - subarray_size + 1
    steering = np.ones(subarray_size, dtype=np.complex128)
    out = np.zeros(n_pix, dtype=np.complex128)

    # each transmit is beamformed with its own weights, then compounded
    for tx in range(n_tx):
        subs = [
            np.array([x[tx, pix, start : start + subarray_size] for start in range(num_subarrays)])
            for pix in range(n_pix)
        ]
        covariances = [np.einsum("li,lj->ij", s, s.conj()) / num_subarrays for s in subs]
        for pix in range(n_pix):
            neighbours = [
                pix + k * axial_stride
                for k in range(-axial_k, axial_k + 1)
                if 0 <= pix + k * axial_stride < n_pix
            ]
            cov = sum(covariances[n] for n in neighbours) / len(neighbours)
            # loading relative to the mean eigenvalue, i.e. trace / subarray_size
            loading = diagonal_loading * np.trace(cov).real / subarray_size
            cov_inv_a = np.linalg.solve(cov + loading * np.eye(subarray_size), steering)
            weights = cov_inv_a / (steering.conj() @ cov_inv_a)
            out[pix] += (weights.conj() @ subs[pix].sum(axis=0)) / num_subarrays
    return out


@pytest.mark.parametrize("subarray_size", [None, 3, 8])
@backend_equality_check()
def test_minimum_variance_matches_reference(subarray_size):
    """MV output matches a plain-numpy Capon implementation, and has the right shape."""
    import keras

    from zea import ops

    rng = np.random.default_rng(42)
    n_tx, n_pix, n_el = 3, 9, 8
    complex_data = rng.standard_normal((n_tx, n_pix, n_el)) + 1j * rng.standard_normal(
        (n_tx, n_pix, n_el)
    )
    data = np.stack([complex_data.real, complex_data.imag], axis=-1).astype(np.float32)

    op = ops.MinimumVariance(subarray_size=subarray_size, axial_averaging=0, with_batch_dim=True)
    out = op(data=keras.ops.convert_to_tensor(data[None]))["data"]
    assert out.shape == (1, n_pix, 2)

    expected = _reference_capon(
        complex_data,
        subarray_size if subarray_size is not None else n_el // 2,
        op.diagonal_loading,
    )
    out_np = keras.ops.convert_to_numpy(out)[0]
    assert np.allclose(out_np[:, 0], expected.real, atol=1e-4), "MV real part differs from Capon"
    assert np.allclose(out_np[:, 1], expected.imag, atol=1e-4), "MV imag part differs from Capon"
    return out


@pytest.mark.parametrize("axial_averaging", [1, 2])
@backend_equality_check()
def test_minimum_variance_axial_averaging_matches_reference(axial_averaging):
    """Axial covariance averaging matches the reference, including at the grid edges."""
    import keras

    from zea import ops

    rng = np.random.default_rng(42)
    n_tx, n_el = 2, 8
    grid_size_z, grid_size_x = 7, 3
    n_pix = grid_size_z * grid_size_x
    complex_data = rng.standard_normal((n_tx, n_pix, n_el)) + 1j * rng.standard_normal(
        (n_tx, n_pix, n_el)
    )
    data = np.stack([complex_data.real, complex_data.imag], axis=-1).astype(np.float32)
    # (n_z, n_x, 3); only its shape is used, to locate axially adjacent pixels
    grid = np.zeros((grid_size_z, grid_size_x, 3), dtype=np.float32)

    op = ops.MinimumVariance(subarray_size=4, axial_averaging=axial_averaging, with_batch_dim=True)
    out = op(data=keras.ops.convert_to_tensor(data[None]), grid=grid)["data"]

    expected = _reference_capon(
        complex_data,
        4,
        op.diagonal_loading,
        axial_k=axial_averaging,
        axial_stride=grid_size_x,
    )
    out_np = keras.ops.convert_to_numpy(out)[0]
    assert np.allclose(out_np[:, 0], expected.real, atol=1e-4), "MV real part differs from Capon"
    assert np.allclose(out_np[:, 1], expected.imag, atol=1e-4), "MV imag part differs from Capon"
    return out


@backend_equality_check()
def test_minimum_variance_axial_averaging_changes_output():
    """Axial averaging is actually applied when a grid is available."""
    import keras

    from zea import ops

    rng = np.random.default_rng(0)
    n_tx, n_el = 2, 8
    grid_size_z, grid_size_x = 6, 4
    data = rng.standard_normal((1, n_tx, grid_size_z * grid_size_x, n_el, 2)).astype(np.float32)
    tensor = keras.ops.convert_to_tensor(data)
    grid = np.zeros((grid_size_z, grid_size_x, 3), dtype=np.float32)

    smoothed = ops.MinimumVariance(axial_averaging=2, with_batch_dim=True)(data=tensor, grid=grid)[
        "data"
    ]
    plain = ops.MinimumVariance(axial_averaging=0, with_batch_dim=True)(data=tensor, grid=grid)[
        "data"
    ]
    assert not np.allclose(
        keras.ops.convert_to_numpy(smoothed), keras.ops.convert_to_numpy(plain)
    ), "axial_averaging had no effect"
    return smoothed


@backend_equality_check()
def test_minimum_variance_large_loading_tends_to_das():
    """With heavy diagonal loading the Capon weights collapse to uniform, i.e. DAS."""
    import keras

    from zea import ops

    rng = np.random.default_rng(42)
    n_tx, n_pix, n_el = 3, 7, 8
    data = rng.standard_normal((1, n_tx, n_pix, n_el, 2)).astype(np.float32)
    tensor = keras.ops.convert_to_tensor(data)

    # A single sub-aperture spanning the full aperture makes the DAS limit exact:
    # w -> 1/n_el, so the output is the DAS sum divided by n_el.
    out = ops.MinimumVariance(subarray_size=n_el, diagonal_loading=1e8, with_batch_dim=True)(
        data=tensor
    )["data"]
    das = ops.DelayAndSum(with_batch_dim=True)(data=tensor)["data"]

    assert np.allclose(
        keras.ops.convert_to_numpy(out),
        keras.ops.convert_to_numpy(das) / n_el,
        atol=1e-5,
    ), "Heavily loaded MV should reduce to DAS"
    return out


@backend_equality_check()
def test_minimum_variance_is_scale_equivariant():
    """Scaling the input scales the output by the same factor.

    The Capon weights are scale invariant (both the covariance and the trace-relative
    diagonal loading scale with the signal power), so the beamformer must be linear in
    the input amplitude. Any absolute epsilon in the weight normalisation breaks this.
    """
    import keras

    from zea import ops

    rng = np.random.default_rng(42)
    data = rng.standard_normal((1, 3, 7, 8, 2)).astype(np.float32)
    op = ops.MinimumVariance(with_batch_dim=True)

    factor = 1e3
    out = op(data=keras.ops.convert_to_tensor(data))["data"]
    out_scaled = op(data=keras.ops.convert_to_tensor(data * factor))["data"]

    assert np.allclose(
        keras.ops.convert_to_numpy(out) * factor,
        keras.ops.convert_to_numpy(out_scaled),
        rtol=1e-3,
    ), "MV output should scale linearly with the input amplitude"
    return out


@backend_equality_check()
def test_minimum_variance_masked_aperture_is_finite():
    """A partially masked receive aperture (f-number mask) must not produce NaN/Inf."""
    import keras

    from zea import ops

    rng = np.random.default_rng(42)
    n_tx, n_pix, n_el = 4, 6, 16
    data = rng.standard_normal((1, n_tx, n_pix, n_el, 2)).astype(np.float32)
    data[..., n_el // 2 :, :] = 0.0  # half the aperture masked out
    out = ops.MinimumVariance(subarray_size=8, with_batch_dim=True)(
        data=keras.ops.convert_to_tensor(data)
    )["data"]
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(out))), (
        "MV produced non-finite values for a partially masked aperture"
    )
    return out


def test_minimum_variance_requires_iq_data():
    """MV should raise ValueError when given single-channel (RF) data."""
    import keras

    from zea import ops

    data = keras.ops.zeros((1, 2, 5, 8, 1))
    with pytest.raises(ValueError, match="requires IQ data"):
        ops.MinimumVariance(with_batch_dim=True)(data=data)


@pytest.mark.parametrize("kwargs", [{"subarray_size": 0}, {"subarray_size": 2.5}])
def test_minimum_variance_invalid_subarray_size(kwargs):
    """Invalid subarray_size is rejected at construction time."""
    from zea import ops

    with pytest.raises(ValueError, match="subarray_size"):
        ops.MinimumVariance(**kwargs)


@backend_equality_check()
def test_minimum_variance_zero_data_is_finite():
    """All-zero input (boundary pixels) must produce finite output, not NaN.

    TOFCorrection returns zeros outside the image boundary.  The covariance for
    those pixels is the zero matrix, whose trace is zero, so diagonal loading
    collapses unless the trace is clamped away from zero.  This test exercises
    that guard directly.
    """
    import keras

    from zea import ops

    n_tx, n_pix, n_el = 2, 5, 8
    data = keras.ops.zeros((1, n_tx, n_pix, n_el, 2))
    out = ops.MinimumVariance(with_batch_dim=True)(data=data)["data"]
    out_np = keras.ops.convert_to_numpy(out)
    assert np.all(np.isfinite(out_np)), (
        f"MV produced non-finite values for all-zero input: "
        f"nan={np.isnan(out_np).sum()}, inf={np.isinf(out_np).sum()}"
    )
    return out


@pytest.mark.parametrize(
    "axis, size, start, end, window_type",
    [
        (0, 64, 64, 32, "hanning"),
        (1, 32, 32, 16, "linear"),
    ],
)
@backend_equality_check(decimal=5)
def test_apply_window(axis, size, start, end, window_type):
    """Test ApplyWindow operation."""

    import keras

    from zea.ops.ultrasound import ApplyWindow

    operation = ApplyWindow(axis=axis, size=size, start=start, end=end, window_type=window_type)

    data = keras.ops.ones((256, 128, 64))
    data_out = operation(data=data)["data"]
    if axis == 0:
        assert keras.ops.convert_to_numpy(data_out[:start, :, :]).sum() == 0.0, (
            "Start region not zeroed correctly."
        )
        assert keras.ops.convert_to_numpy(data_out[-end:, :, :]).sum() == 0.0, (
            "End region not zeroed correctly."
        )
    elif axis == 1:
        assert keras.ops.convert_to_numpy(data_out[:, :start, :]).sum() == 0.0, (
            "Start region not zeroed correctly."
        )
        assert keras.ops.convert_to_numpy(data_out[:, -end:, :]).sum() == 0.0, (
            "End region not zeroed correctly."
        )
    return data_out


@backend_equality_check(decimal=4)
def test_band_pass_filter():
    """Test BandPassFilter operation."""

    import keras

    from zea import ops

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal((2, 1, 128, 16, 1)).astype("float32")
    data = keras.ops.convert_to_tensor(data)

    operation = ops.BandPassFilter(
        axis=-3,
        with_batch_dim=True,
        passband=(3.5e6, 6.5e6),
    )

    # Uses init-time passband.
    result_init_passband = operation(
        data=data,
        sampling_frequency=40e6,
    )["data"]

    # Call-time passband should override init-time passband.
    result_call_passband = operation(
        data=data,
        sampling_frequency=40e6,
        passband=(4e6, 6e6),
    )["data"]

    # With init-time passband set, demod/bandwidth should not override it.
    result_demod_frequency = operation(
        data=data,
        sampling_frequency=40e6,
        demodulation_frequency=5e6,
        bandwidth=2e6,
    )["data"]

    # Test the frequency/bandwidth input mode when no fixed passband is set.
    operation = ops.BandPassFilter(axis=-3, with_batch_dim=True)
    result_default = operation(
        data=data,
        sampling_frequency=40e6,
        demodulation_frequency=5e6,
        bandwidth=3e6,
    )["data"]

    rtol, atol = 1e-5, 1e-6
    result_init_passband = keras.ops.convert_to_numpy(result_init_passband)
    result_call_passband = keras.ops.convert_to_numpy(result_call_passband)
    result_demod_frequency = keras.ops.convert_to_numpy(result_demod_frequency)

    # Compare the three passband-related modes.
    assert np.allclose(result_init_passband, result_demod_frequency, rtol=rtol, atol=atol)
    assert not np.allclose(result_init_passband, result_call_passband, rtol=rtol, atol=atol)
    assert not np.allclose(result_demod_frequency, result_call_passband, rtol=rtol, atol=atol)

    with pytest.raises(ValueError, match="passband must be an iterable of two numeric values"):
        operation(
            data=data,
            sampling_frequency=40e6,
            passband=(4e6,),
        )

    return result_default


def test_band_pass_filter_validates_in_eager_mode():
    """Frequency validation in BandPassFilter.call must run in eager (non-JIT) mode."""
    import keras

    from zea import ops

    data = keras.ops.ones((1, 2, 128, 4, 1), dtype="float32")

    # Eager (jit_compile=False): invalid frequencies must raise.
    op = ops.BandPassFilter(
        axis=-3,
        with_batch_dim=True,
        passband=(100e6, 200e6),  # far above Nyquist — always invalid
        jit_compile=False,
    )
    with pytest.raises(ValueError, match="Invalid cutoff frequency"):
        op(data=data, sampling_frequency=40e6)

    # jit_compile=True: values are abstract inside the JIT trace, so float() fails
    # and the try/except in get_band_pass_filter silently skips validation. This is a
    # known limitation — test that it at least does not crash with valid frequencies.
    op_jit = ops.BandPassFilter(
        axis=-3,
        with_batch_dim=True,
        passband=(3.5e6, 6.5e6),
        jit_compile=True,
    )
    op_jit(data=data, sampling_frequency=40e6)  # must not raise


def test_make_tgc_curve():
    """Test that TGC curve is monotonically increasing with depth."""
    n_ax = 1000
    attenuation_coef = 0.5  # dB/cm/MHz (typical for soft tissue)
    sampling_frequency = 40e6  # 40 MHz
    center_frequency = 5e6  # 5 MHz
    sound_speed = 1540  # m/s

    tgc_curve = make_tgc_curve(
        n_ax=n_ax,
        attenuation_coef=attenuation_coef,
        sampling_frequency=sampling_frequency,
        center_frequency=center_frequency,
        sound_speed=sound_speed,
    )

    # Check output shape
    assert tgc_curve.shape == (n_ax,), f"Expected shape ({n_ax},), got {tgc_curve.shape}"

    # Check output dtype
    assert tgc_curve.dtype == np.float32, f"Expected dtype float32, got {tgc_curve.dtype}"

    # Check that the curve is monotonically increasing (TGC should increase with depth)
    differences = np.diff(tgc_curve)
    assert np.all(differences >= 0), "TGC curve should be monotonically increasing with depth"

    # Check that the first value is 1 (no gain at zero depth)
    assert np.isclose(tgc_curve[0], 1.0, rtol=1e-5), (
        f"TGC curve should start at 1.0 (no gain at zero depth), got {tgc_curve[0]}"
    )

    # Check that values are positive
    assert np.all(tgc_curve > 0), "TGC curve values should be positive"


@pytest.mark.parametrize(
    "n_tx, n_pix, n_rx, with_batch",
    [
        (32, 50, 32, False),  # Square aperture, no batch
        (32, 50, 32, True),  # Square aperture, with batch
        (48, 100, 48, False),  # Larger aperture, no batch
        (20, 25, 20, False),  # Smaller aperture, no batch
    ],
)
@backend_equality_check(decimal=5)
def test_common_midpoint_phase_error(n_tx, n_pix, n_rx, with_batch):
    """Test CommonMidpointPhaseError operation.

    This operation computes the Common Midpoint Phase Error (CMPE) between
    translated transmit and receive subapertures, used for autofocusing.
    """

    import keras

    from zea import ops

    # Create synthetic TOF-corrected IQ data
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Create data with some coherent structure (simulating actual TOF-corrected data)
    # Add random phase variations to simulate realistic ultrasound data
    # IQ data has 2 channels (I and Q)
    phase = rng.random((n_tx, n_pix, n_rx)) * 2 * np.pi
    amplitude = rng.random((n_tx, n_pix, n_rx)) * 0.5 + 0.5  # Amplitude between 0.5 and 1

    # Convert to IQ format
    data = np.stack(
        [
            amplitude * np.cos(phase),  # I channel
            amplitude * np.sin(phase),  # Q channel
        ],
        axis=-1,
    ).astype(np.float32)

    # Test with batch dimension if requested
    if with_batch:
        batch_size = 2
        # create two batches with different phase patterns
        data = np.stack([data, data * np.exp(1j * 0.1)], axis=0)

        cmpe = ops.CommonMidpointPhaseError(with_batch_dim=True)
        expected_output_shape = (batch_size, n_pix)
    else:
        cmpe = ops.CommonMidpointPhaseError(with_batch_dim=False)
        expected_output_shape = (n_pix,)

    data_tensor = keras.ops.convert_to_tensor(data)

    # Compute CMPE
    output = cmpe(data=data_tensor)
    phase_error = output["data"]

    # Convert to numpy for assertions
    phase_error_np = keras.ops.convert_to_numpy(phase_error)

    # Check output shape
    assert phase_error_np.shape == expected_output_shape, (
        f"Expected shape {expected_output_shape}, got {phase_error_np.shape}"
    )

    # Check that output values are in valid range [0, π]
    # Phase differences should be absolute values bounded by π
    assert np.all(phase_error_np >= 0), "Phase errors should be non-negative"
    assert np.all(phase_error_np <= np.pi + 1e-5), (
        f"Phase errors should be <= π, got max: {np.max(phase_error_np)}"
    )

    # Check that not all values are zero (would indicate wrong computation)
    assert np.any(phase_error_np > 0), (
        "Phase errors should have some non-zero values for random data"
    )

    # Check for NaN values
    assert not np.any(np.isnan(phase_error_np)), "Phase errors should not contain NaN values"

    if with_batch:
        # Check that the two batches are not identical (due to different phase patterns)
        assert not np.allclose(phase_error_np[0], phase_error_np[1], rtol=1e-7), (
            "Phase errors for different batches should not be identical"
        )

    return phase_error_np


@backend_equality_check()
def test_common_midpoint_phase_error_with_zeros():
    """Test CommonMidpointPhaseError operation handles zeros correctly.

    The operation should handle cases where some data points are zero
    (e.g., points outside the field of view).
    """

    import keras

    from zea import ops

    # Create small synthetic data
    n_tx, n_pix, n_rx = 24, 30, 24

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    # Create data with some zeros
    # IQ data has 2 channels (I and Q)
    phase = rng.random((n_tx, n_pix, n_rx)) * 2 * np.pi
    amplitude = rng.random((n_tx, n_pix, n_rx)) * 0.5 + 0.5

    data = np.stack(
        [
            amplitude * np.cos(phase),
            amplitude * np.sin(phase),
        ],
        axis=-1,
    ).astype(np.float32)

    # Set some regions to zero (simulating out-of-FOV points)
    data[:5, :, :5, :] = 0  # Zero out corner region
    data[-5:, :, -5:, :] = 0  # Zero out another corner

    cmpe = ops.CommonMidpointPhaseError(with_batch_dim=False)
    data_tensor = keras.ops.convert_to_tensor(data)

    output = cmpe(data=data_tensor)
    phase_error = keras.ops.convert_to_numpy(output["data"])

    # Check that computation doesn't crash and produces valid output
    assert phase_error.shape == (n_pix,), f"Expected shape ({n_pix},), got {phase_error.shape}"

    # Check for inf values (should never occur)
    assert not np.any(np.isinf(phase_error)), "Phase errors should not contain infinities"

    # NaNs are acceptable where all subapertures are invalid (division by zero)
    # but not all values should be NaN
    n_nan = np.sum(np.isnan(phase_error))
    assert n_nan < n_pix, "At least some phase errors should be finite (not all NaN)"

    # Check that finite values are in valid range [0, π]
    finite_mask = np.isfinite(phase_error)
    if np.any(finite_mask):
        finite_values = phase_error[finite_mask]
        assert np.all(finite_values >= 0) and np.all(finite_values <= np.pi + 1e-5), (
            "Finite phase errors should be in [0, π] range"
        )
    return phase_error


@backend_equality_check()
def test_common_midpoint_phase_error_coherent_data():
    """Test CommonMidpointPhaseError with perfectly coherent data.

    With perfectly coherent data (same phase everywhere), the phase error
    should be close to zero.
    """

    import keras

    from zea import ops

    n_tx, n_pix, n_rx, n_ch = 32, 40, 32, 2

    # Create perfectly coherent data (same phase for all elements)
    amplitude = 1.0
    phase = np.pi / 4  # Single phase value

    data = np.full((n_tx, n_pix, n_rx, n_ch), 0.0, dtype=np.float32)
    data[..., 0] = amplitude * np.cos(phase)  # I channel
    data[..., 1] = amplitude * np.sin(phase)  # Q channel

    cmpe = ops.CommonMidpointPhaseError(with_batch_dim=False)
    data_tensor = keras.ops.convert_to_tensor(data)

    output = cmpe(data=data_tensor)
    phase_error = keras.ops.convert_to_numpy(output["data"])

    # For perfectly coherent data, phase errors should be very small
    assert np.all(phase_error < 0.01), (
        f"Phase errors for coherent data should be near zero, got max: {np.max(phase_error)}"
    )

    # Check shape
    assert phase_error.shape == (n_pix,), f"Expected shape ({n_pix},), got {phase_error.shape}"

    return phase_error


@pytest.mark.parametrize("with_batch_dim", [False, True])
@backend_equality_check(decimal=5)
def test_apply_aligned_apodization(with_batch_dim):
    """A per-pixel, per-transmit weight scales each transmit's contribution to a
    pixel, e.g. a one-hot mask isolates a single owning transmit per pixel
    (the scanline / line-by-line special case of pixel-based DAS)."""
    from zea.func.ultrasound import apply_aligned_apodization

    n_tx, n_pix, n_el, n_ch = 3, 2, 4, 2
    data = keras.ops.ones((n_tx, n_pix, n_el, n_ch))
    if with_batch_dim:
        data = keras.ops.expand_dims(data, axis=0)

    # Pixel 0 belongs to transmit 0; pixel 1 belongs to transmit 2.
    apodization = keras.ops.convert_to_tensor(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )

    out = keras.ops.convert_to_numpy(apply_aligned_apodization(data, apodization, with_batch_dim))

    expected = np.zeros((n_tx, n_pix, n_el, n_ch), dtype=np.float32)
    expected[0, 0] = 1.0
    expected[2, 1] = 1.0
    if with_batch_dim:
        expected = expected[None]

    np.testing.assert_allclose(out, expected, rtol=1e-5)
    return out


@pytest.mark.parametrize("with_batch_dim", [False, True])
def test_aligned_apodization_op(with_batch_dim):
    """AlignedApodization applies flat_aligned_apodization to aligned data; with no
    mask supplied (flat_aligned_apodization=None) it passes the data through
    unchanged."""
    from zea.ops import AlignedApodization

    n_tx, n_pix, n_el, n_ch = 3, 2, 4, 2
    data = keras.ops.ones((n_tx, n_pix, n_el, n_ch))
    if with_batch_dim:
        data = keras.ops.expand_dims(data, axis=0)

    op = AlignedApodization(with_batch_dim=with_batch_dim)

    out_noop = op(data=data)["data"]
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out_noop), keras.ops.convert_to_numpy(data)
    )

    apodization = keras.ops.convert_to_tensor(
        np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )
    out = keras.ops.convert_to_numpy(op(data=data, flat_aligned_apodization=apodization)["data"])

    expected = np.zeros((n_tx, n_pix, n_el, n_ch), dtype=np.float32)
    expected[0, 0] = 1.0
    expected[2, 1] = 1.0
    if with_batch_dim:
        expected = expected[None]

    np.testing.assert_allclose(out, expected, rtol=1e-5)


@pytest.mark.parametrize("with_batch_dim", [False, True])
@backend_equality_check(decimal=5)
def test_apply_receive_apodization(with_batch_dim):
    """A per-pixel, per-element weight scales each receive element's contribution
    to a pixel (custom receive-aperture apodization), broadcasting uniformly over
    the transmit and channel axes."""
    from zea.func.ultrasound import apply_receive_apodization

    n_tx, n_pix, n_el, n_ch = 3, 2, 4, 2
    data = keras.ops.ones((n_tx, n_pix, n_el, n_ch))
    if with_batch_dim:
        data = keras.ops.expand_dims(data, axis=0)

    # Per-pixel taper over the receive elements (independent of transmit/channel).
    apod_np = np.array([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    apodization = keras.ops.convert_to_tensor(apod_np)

    out = keras.ops.convert_to_numpy(apply_receive_apodization(data, apodization, with_batch_dim))

    # Broadcast the (n_pix, n_el) weight over n_tx (leading) and n_ch (trailing).
    expected = np.broadcast_to(apod_np[None, :, :, None], (n_tx, n_pix, n_el, n_ch)).copy()
    if with_batch_dim:
        expected = expected[None]

    np.testing.assert_allclose(out, expected, rtol=1e-5)
    return out


@pytest.mark.parametrize("with_batch_dim", [False, True])
def test_receive_apodization_op(with_batch_dim):
    """ReceiveApodization applies a custom per-pixel, per-element mask to aligned
    data; with no mask supplied (flat_receive_apodization=None) it passes the data
    through unchanged, and a uniform ones-weight is the identity."""
    from zea.ops import ReceiveApodization

    n_tx, n_pix, n_el, n_ch = 3, 2, 4, 2
    rng = np.random.default_rng(0)
    data_np = rng.standard_normal((n_tx, n_pix, n_el, n_ch)).astype(np.float32)
    data = keras.ops.convert_to_tensor(data_np)
    if with_batch_dim:
        data = keras.ops.expand_dims(data, axis=0)

    op = ReceiveApodization(with_batch_dim=with_batch_dim)

    # None -> pass-through
    out_noop = op(data=data)["data"]
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out_noop), keras.ops.convert_to_numpy(data)
    )

    # Ones -> identity
    ones = keras.ops.ones((n_pix, n_el))
    out_ones = op(data=data, flat_receive_apodization=ones)["data"]
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(out_ones),
        keras.ops.convert_to_numpy(data),
        rtol=1e-5,
    )

    # Per-element taper -> scales the expected elements
    apod_np = np.array([[1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 1.0]], dtype=np.float32)
    apodization = keras.ops.convert_to_tensor(apod_np)
    out = keras.ops.convert_to_numpy(op(data=data, flat_receive_apodization=apodization)["data"])

    expected = data_np * apod_np[None, :, :, None]
    if with_batch_dim:
        expected = expected[None]

    np.testing.assert_allclose(out, expected, rtol=1e-5)


@backend_equality_check(decimal=2)
def test_prepare_parameters_pfield_all_backends():
    """pipeline.prepare_parameters must work on all backends when pfield is enabled.

    Regression: Parameters stores n_el with dtype=np.int32. The torch backend's
    torch.zeros rejects a numpy scalar as the size argument, causing a TypeError
    inside compute_pfield when reached via prepare_parameters → to_tensor → flat_pfield.
    """

    import keras

    from zea.beamform.delays import compute_t0_delays_planewave
    from zea.internal.cache import cache_disabled
    from zea.ops import Pipeline
    from zea.parameters import Parameters

    n_el = 8
    n_tx = 2
    pitch = 3e-4
    probe_geometry = np.stack(
        [
            np.arange(n_el) * pitch - (n_el - 1) * pitch / 2,
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=-1,
    ).astype(np.float32)
    angles = np.linspace(-5, 5, n_tx) * np.pi / 180
    t0_delays = compute_t0_delays_planewave(probe_geometry, angles)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)

    parameters = Parameters(
        probe_geometry=probe_geometry,
        n_tx=n_tx,
        n_el=n_el,
        xlims=(-5e-3, 5e-3),
        zlims=(5e-3, 20e-3),
        n_ax=64,
        sampling_frequency=5e6 * 4,
        center_frequency=5e6,
        polar_angles=angles,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
    )
    parameters.grid_size_x = 8
    parameters.grid_size_z = 8
    # default downsample=10 collapses this 8x8 grid to 1 point, making the field
    # constant up to backend rounding noise, which flakes the quantile threshold.
    parameters.pfield_kwargs = {"downsample": 2}

    # Disable the on-disk result cache so ops are actually executed in each backend
    # subprocess (the cache would serve a stale pickle and hide crashes).
    with cache_disabled():
        pipeline = Pipeline.from_default(enable_pfield=True)
        inputs = pipeline.prepare_parameters(parameters)
        return keras.ops.convert_to_numpy(inputs["flat_pfield"])


@backend_equality_check(decimal=3)
def test_tissue_suppression():
    """Test that TissueSuppression reduces stationary tissue component."""
    import keras

    from zea import ops

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    n_frames, n_tx, n_ax, n_ch = 10, 4, 32, 8
    shape = (n_frames, n_tx, n_ax, n_ch)

    # Stationary tissue: same signal repeated across all frames

    gradient = np.linspace(0, 1, n_ax).reshape(1, 1, n_ax, 1)
    tissue = np.ones(shape) * gradient

    # Blood component: random per frame
    blood = rng.standard_normal(shape) * 0.1

    data = (tissue + blood).astype(np.float32)

    cutoff = 3
    op = ops.TissueSuppression(cutoff=cutoff)
    data_tensor = keras.ops.convert_to_tensor(data)
    output = keras.ops.convert_to_numpy(op(data=data_tensor)["data"])

    assert output.shape == shape, f"Expected shape {shape}, got {output.shape}"
    assert output.dtype == data.dtype, f"Expected dtype {data.dtype}, got {output.dtype}"

    # After suppression, energy should be lower than the original tissue-dominated signal
    assert np.mean(output**2) < np.mean(data**2), "Tissue suppression should reduce signal energy"

    correlation_across_frames = np.corrcoef(output.reshape(n_frames, -1))
    # The correlation across frames should be reduced after tissue suppression
    assert np.mean(correlation_across_frames) < 0.5, (
        "Tissue suppression should reduce correlation across frames, got mean correlation: "
        f"{np.mean(correlation_across_frames)}"
    )

    return output


def hadamard_encode(data, hadamard_matrix):
    """Encodes transmits as Hadamard combinations: encoded[j] = sum_t H[j, t] * data[t]."""
    return np.einsum("itklm,jt->ijklm", data, hadamard_matrix)


def test_decode_hadamard():
    """Hadamard encoding followed by decoding recovers the original data up to a factor n_tx."""
    n_el = 32
    hadamard_size = 32
    n_tx = hadamard_size
    participating_channels = np.random.choice(n_el, size=hadamard_size, replace=False)
    hadamard_matrix = hadamard(hadamard_size).astype(np.float32)
    tx_apodizations_hadamard = np.zeros((hadamard_size, n_el), dtype=np.float32)
    tx_apodizations_hadamard[:, participating_channels] = hadamard_matrix
    tx_apodizations_synthetic_aperture = np.zeros((hadamard_size, n_el), dtype=np.float32)
    tx_apodizations_synthetic_aperture[:, participating_channels] = np.eye(hadamard_size)
    center_frequency = 5e6
    sampling_frequency = 4 * center_frequency
    sound_speed = 1540.0
    wavelength = sound_speed / center_frequency
    element_width = wavelength / 2 * 0.9
    shared_kwargs = dict(
        scatterer_positions=np.array([[0.0, 0.0, 3e-3]], dtype=np.float32),
        scatterer_magnitudes=np.array([1.0], dtype=np.float32),
        probe_geometry=linear_probe_geometry(n_elements=n_el, pitch=wavelength / 2).astype(
            np.float32
        ),
        apply_lens_correction=False,
        lens_thickness=0.0,
        lens_sound_speed=1000.0,
        sound_speed=sound_speed,
        n_ax=256,
        center_frequency=center_frequency,
        sampling_frequency=sampling_frequency,
        t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
        initial_times=np.zeros((n_tx,), dtype=np.float32),
        element_width=element_width,
        attenuation_coef=0.0,
    )
    raw_data_encoded = simulate_rf(tx_apodizations=tx_apodizations_hadamard, **shared_kwargs)[None]
    raw_data_synthetic_aperture = simulate_rf(
        tx_apodizations=tx_apodizations_synthetic_aperture, **shared_kwargs
    )[None]
    raw_data_decoded, tx_apodizations_decoded = decode_hadamard(raw_data_encoded, hadamard_matrix)
    np.testing.assert_allclose(
        raw_data_decoded, raw_data_synthetic_aperture * hadamard_size, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(tx_apodizations_decoded, np.eye(hadamard_size))


def test_decode_hadamard_warns_on_non_orthogonal_matrix():
    """A non-orthogonal tx_apodizations matrix triggers a warning."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal((1, 2, 4, 2, 1))
    non_orthogonal = np.array([[1.0, 1.0], [1.0, 1.0]])
    with mock.patch("zea.log.warning") as warning:
        decode_hadamard(data, non_orthogonal)
    warning.assert_called_once()


def test_sa_to_virtual_focus(tmp_path):
    """sa_to_virtual_focus converts a synthetic aperture file into a single-transmit
    virtual-focus file, wiring the scan metadata and raw data through
    construct_acquisition_from_synthetic_aperture."""
    from zea.data.file import File, validate_file
    from zea.data.file_operations import sa_to_virtual_focus

    from .data import generate_example_dataset

    # Synthetic aperture data has one transmit per element (n_tx == n_el).
    n_el = 8
    input_path = tmp_path / "sa.hdf5"
    output_path = tmp_path / "virtual_focus.hdf5"
    generate_example_dataset(input_path, n_el=n_el, n_tx=n_el, n_ax=128, n_frames=1)

    focus_distance = 20e-3
    sa_to_virtual_focus(
        input_path,
        output_path,
        polar_angle=0.0,
        azimuth_angle=0.0,
        focus_distance=focus_distance,
        tx_apodization="hanning",
    )

    validate_file(output_path)

    with File(input_path) as f:
        raw_data = f.data.raw_data[:]
        parameters = f.load_parameters()

    tx_apodization = keras.ops.expand_dims(keras.ops.cast(keras.ops.hanning(n_el), "float32"), 0)
    expected_raw_data, expected_t0_delays = construct_acquisition_from_synthetic_aperture(
        raw_data=raw_data,
        probe_geometry=parameters["probe_geometry"],
        sampling_frequency=parameters["sampling_frequency"],
        polar_angle=0.0,
        azimuth_angle=0.0,
        focus_distance=focus_distance,
        transmit_origin=(0.0, 0.0, 0.0),
        sound_speed=parameters["sound_speed"],
        tx_apodization=tx_apodization,
    )
    expected_raw_data = keras.ops.convert_to_numpy(expected_raw_data)

    with File(output_path) as f:
        out_raw_data = f.data.raw_data[:]
        out_parameters = f.load_parameters()

    # A single transmit remains, and its data matches the directly-constructed acquisition.
    assert out_raw_data.shape == (1, 1, 128, n_el, 1)
    np.testing.assert_allclose(out_raw_data, expected_raw_data, atol=1e-4)

    # The transmit scheme metadata reflects the requested virtual focus.
    np.testing.assert_allclose(out_parameters["t0_delays"], np.asarray(expected_t0_delays))
    np.testing.assert_allclose(out_parameters["polar_angles"], [0.0])
    np.testing.assert_allclose(out_parameters["focus_distances"], [focus_distance])
    np.testing.assert_allclose(out_parameters["transmit_origins"], [[0.0, 0.0, 0.0]])
    assert out_parameters["tx_apodizations"].shape == (1, n_el)


def test_sa_to_virtual_focus_requires_probe_geometry(tmp_path):
    """sa_to_virtual_focus raises when the input file has no probe_geometry."""
    from zea.data.file import File
    from zea.data.file_operations import sa_to_virtual_focus

    # A file with only an image has no probe, so probe_geometry is undefined.
    input_path = tmp_path / "no_probe.hdf5"
    output_path = tmp_path / "out.hdf5"
    File.create(
        input_path,
        data={"image": {"values": np.zeros((1, 16, 16), dtype=np.float32)}},
        overwrite=True,
    )

    with pytest.raises(ValueError, match="requires a probe with a defined probe_geometry"):
        sa_to_virtual_focus(
            input_path,
            output_path,
            polar_angle=0.0,
            azimuth_angle=0.0,
            focus_distance=np.inf,
        )


def test_sa_to_virtual_focus_refuses_existing_output(tmp_path):
    """An existing output path is only overwritten when overwrite=True."""
    from zea.data.file import File
    from zea.data.file_operations import sa_to_virtual_focus

    from .data import generate_example_dataset

    n_el = 8
    input_path = tmp_path / "sa.hdf5"
    output_path = tmp_path / "existing.hdf5"
    generate_example_dataset(input_path, n_el=n_el, n_tx=n_el, n_ax=64, n_frames=1)
    output_path.touch()  # pre-existing output file

    convert = lambda **kwargs: sa_to_virtual_focus(
        input_path,
        output_path,
        polar_angle=0.0,
        azimuth_angle=0.0,
        focus_distance=np.inf,
        **kwargs,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        convert()

    # With overwrite=True the existing file is replaced by a valid virtual-focus file.
    convert(overwrite=True)
    with File(output_path) as f:
        assert f.data.raw_data[:].shape[1] == 1


def linear_probe_geometry(n_elements, pitch):
    """Returns a linear array geometry of shape (n_elements, 3) along the x-axis."""
    x = np.arange(n_elements) * pitch
    return np.stack([x, np.zeros(n_elements), np.zeros(n_elements)], axis=1)


def synthetic_aperture_acquisition(raw_data, probe_geometry, polar_angle=0.0, **kwargs):
    """Runs construct_acquisition_from_synthetic_aperture and returns numpy outputs."""
    raw_data = keras.ops.convert_to_tensor(raw_data)
    constructed, t0_delays = construct_acquisition_from_synthetic_aperture(
        raw_data,
        probe_geometry,
        polar_angle=polar_angle,
        azimuth_angle=0.0,
        focus_distance=kwargs.pop("focus_distance", np.inf),
        sampling_frequency=kwargs.pop("sampling_frequency", 1.0),
        sound_speed=kwargs.pop("sound_speed", 1.0),
        **kwargs,
    )
    return keras.ops.convert_to_numpy(constructed), np.asarray(t0_delays)


def test_construct_acquisition_zero_angle_sums_transmits():
    """A zero-angle plane wave has zero delays, so the output is the sum over transmits."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n_frames, n_el, n_ax, n_ch = 2, 4, 32, 1
    raw_data = rng.standard_normal((n_frames, n_el, n_ax, n_el, n_ch)).astype(np.float32)
    probe_geometry = linear_probe_geometry(n_el, pitch=1.0)

    constructed, t0_delays = synthetic_aperture_acquisition(raw_data, probe_geometry)

    assert constructed.shape == (n_frames, 1, n_ax, n_el, n_ch)
    assert t0_delays.shape == (1, n_el)
    np.testing.assert_allclose(t0_delays, 0.0, atol=1e-12)
    np.testing.assert_allclose(constructed[:, 0], raw_data.sum(axis=1), atol=1e-4)


def test_construct_acquisition_applies_integer_sample_delays():
    """An angled plane wave shifts an impulse by exactly t0_delay * sampling_frequency samples."""
    n_el, n_ax, impulse_sample, active_transmit = 4, 64, 10, 2
    raw_data = np.zeros((1, n_el, n_ax, n_el, 1), dtype=np.float32)
    raw_data[0, active_transmit, impulse_sample] = 1.0
    # pitch=2, sound_speed=1, sin(30 degrees)=0.5 gives delays of exactly [0, 1, 2, 3] samples
    probe_geometry = linear_probe_geometry(n_el, pitch=2.0)

    constructed, t0_delays = synthetic_aperture_acquisition(
        raw_data, probe_geometry, polar_angle=np.pi / 6
    )

    np.testing.assert_allclose(t0_delays[0], np.arange(n_el), atol=1e-9)
    expected_sample = impulse_sample + active_transmit
    for element in range(n_el):
        signal = constructed[0, 0, :, element, 0]
        assert np.argmax(signal) == expected_sample
        np.testing.assert_allclose(signal[expected_sample], 1.0, atol=1e-4)


def test_construct_acquisition_applies_tx_apodization():
    """The tx_apodization scales the output per receive element."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n_el, n_ax = 4, 32
    raw_data = rng.standard_normal((1, n_el, n_ax, n_el, 1)).astype(np.float32)
    probe_geometry = linear_probe_geometry(n_el, pitch=1.0)
    apodization = np.array([[0.0, 1.0, 0.5, 2.0]], dtype=np.float32)

    unapodized, _ = synthetic_aperture_acquisition(raw_data, probe_geometry)
    apodized, _ = synthetic_aperture_acquisition(
        raw_data, probe_geometry, tx_apodization=keras.ops.convert_to_tensor(apodization)
    )

    expected = unapodized * apodization[0][None, None, None, :, None]
    np.testing.assert_allclose(apodized, expected, atol=1e-4)


def test_construct_acquisition_is_independent_of_chunk_size():
    """Processing transmits in chunks of 1 gives the same result as a single chunk."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n_el, n_ax = 4, 32
    raw_data = rng.standard_normal((2, n_el, n_ax, n_el, 1)).astype(np.float32)
    probe_geometry = linear_probe_geometry(n_el, pitch=2.0)

    single_chunk, _ = synthetic_aperture_acquisition(
        raw_data, probe_geometry, polar_angle=np.pi / 6, transmit_chunk_size=n_el
    )
    chunked, _ = synthetic_aperture_acquisition(
        raw_data, probe_geometry, polar_angle=np.pi / 6, transmit_chunk_size=1
    )

    np.testing.assert_allclose(chunked, single_chunk, atol=1e-4)


def test_construct_acquisition_focused_transmit_delays():
    """A finite focus distance uses focused transmit delays."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n_el, n_ax, focus_distance = 4, 32, 20.0
    raw_data = rng.standard_normal((1, n_el, n_ax, n_el, 1)).astype(np.float32)
    probe_geometry = linear_probe_geometry(n_el, pitch=1.0)

    constructed, t0_delays = synthetic_aperture_acquisition(
        raw_data, probe_geometry, focus_distance=focus_distance
    )

    expected_delays = compute_t0_delays_focused(
        transmit_origins=np.zeros((1, 3)),
        focus_distances=np.array([focus_distance]),
        probe_geometry=probe_geometry,
        polar_angles=np.array([0.0]),
        sound_speed=1.0,
    )
    assert constructed.shape == (1, 1, n_ax, n_el, 1)
    np.testing.assert_allclose(t0_delays, expected_delays, atol=1e-12)


def _complex_clutter_video(n_frames=40, n_z=16, n_x=16, seed=DEFAULT_TEST_SEED):
    """Complex IQ video: strong phase-rotating tissue clutter + weak blood.

    The tissue rotates in phase over time, which is what distinguishes the
    Hermitian Gram matrix from the plain-transpose one.

    Returns the video as a complex array (n_frames, n_z, n_x) and as real IQ
    channels (n_frames, n_z, n_x, 2).
    """
    rng = np.random.default_rng(seed)

    def _complex_normal(shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)

    spatial = _complex_normal((n_z, n_x)) * 10
    phase = np.exp(1j * 2 * np.pi * 0.02 * np.arange(n_frames)).astype(np.complex64)
    tissue = (phase[:, None, None] * spatial[None]).astype(np.complex64)
    blood = (_complex_normal((n_frames, n_z, n_x)) * 0.1).astype(np.complex64)
    data = (tissue + blood).astype(np.complex64)
    data_channels = np.stack([data.real, data.imag], axis=-1).astype(np.float32)
    return data, data_channels


@backend_equality_check(allow_none=True)
def test_tissue_suppression_complex():
    """TissueSuppression with filter_type='svd_complex' suppresses complex IQ clutter.

    The op takes IQ data as two real channels (n_frames, ..., 2), the zea
    convention, and converts to/from complex internally.
    """
    import keras

    from zea import ops

    _, data_channels = _complex_clutter_video()

    op = ops.TissueSuppression(cutoff=2, filter_type="svd_complex")
    output = keras.ops.convert_to_numpy(op(data=keras.ops.convert_to_tensor(data_channels))["data"])

    assert output.shape == data_channels.shape
    assert output.dtype == data_channels.dtype, (
        f"Expected dtype {data_channels.dtype}, got {output.dtype}"
    )

    # The clutter dominates the energy, so removing it must remove nearly all of it.
    energy_ratio = np.mean(output**2) / np.mean(data_channels**2)
    assert energy_ratio < 0.01, f"Expected clutter to be suppressed, energy ratio {energy_ratio}"


def test_tissue_suppression_complex_needs_conjugate():
    """The plain-transpose 'svd' filter fails on complex IQ; 'svd_complex' is required.

    On complex data the plain transpose builds X^T X rather than the temporal
    covariance X^H X, so the components it finds are not the tissue subspace.
    Checked against the reference implementation: this is maths, not backend behaviour.
    """
    import keras

    from zea.func.ultrasound import suppress_tissue

    data, _ = _complex_clutter_video()  # complex array, not the real-channel one

    def _energy_ratio(conjugate):
        output = keras.ops.convert_to_numpy(suppress_tissue(data, 2, conjugate=conjugate))
        return np.mean(np.abs(output) ** 2) / np.mean(np.abs(data) ** 2)

    assert _energy_ratio(conjugate=True) < 0.01
    # The plain transpose leaves the phase-rotating clutter essentially intact.
    assert _energy_ratio(conjugate=False) > 0.5


@backend_equality_check(allow_none=True)
def test_tissue_suppression_conjugate_matches_plain_on_zero_imag_data():
    """With a zero imaginary part, the conjugate transpose is a no-op."""
    import keras

    from zea import ops

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    real_data = rng.standard_normal((10, 8, 8)).astype(np.float32)
    channel_data = np.stack([real_data, np.zeros_like(real_data)], axis=-1)

    real_output = keras.ops.convert_to_numpy(
        ops.TissueSuppression(cutoff=3, filter_type="svd")(
            data=keras.ops.convert_to_tensor(real_data)
        )["data"]
    )
    complex_output = keras.ops.convert_to_numpy(
        ops.TissueSuppression(cutoff=3, filter_type="svd_complex")(
            data=keras.ops.convert_to_tensor(channel_data)
        )["data"]
    )

    np.testing.assert_allclose(real_output, complex_output[..., 0], atol=1e-2)
    np.testing.assert_allclose(complex_output[..., 1], 0, atol=1e-2)


def test_tissue_suppression_fractional_cutoff():
    """A float cutoff is resolved as a fraction of the number of frames."""
    from zea import ops

    assert ops.TissueSuppression(cutoff=0.05).resolve_cutoff(400) == 20
    assert ops.TissueSuppression(cutoff=0.1).resolve_cutoff(95) == 10
    # An int cutoff is used as-is, independent of the frame count.
    assert ops.TissueSuppression(cutoff=5).resolve_cutoff(400) == 5


def test_tissue_suppression_filter_type_autodetect():
    """filter_type=None picks svd_complex for IQ (n_ch=2) and svd otherwise."""
    import keras

    from zea import ops

    def _resolve(shape, filter_type=None):
        data = keras.ops.convert_to_tensor(np.zeros(shape, dtype=np.float32))
        return ops.TissueSuppression(cutoff=2, filter_type=filter_type).resolve_filter_type(data)

    assert _resolve((10, 8, 8, 2)) == "svd_complex"  # IQ
    assert _resolve((10, 8, 8, 1)) == "svd"  # RF
    assert _resolve((10, 8, 8)) == "svd"  # real, no channel axis
    assert _resolve((10,)) == "svd"  # 1-D, no channel axis to read

    # An explicit filter_type always wins over auto-detection.
    assert _resolve((10, 8, 8, 2), filter_type="svd") == "svd"
    assert _resolve((10, 8, 8), filter_type="svd_complex") == "svd_complex"


@backend_equality_check(allow_none=True)
def test_tissue_suppression_autodetect_matches_explicit():
    """Auto-detected IQ input gives the same result as filter_type='svd_complex'."""
    import keras

    from zea import ops

    _, data_channels = _complex_clutter_video()
    data_tensor = keras.ops.convert_to_tensor(data_channels)

    outputs = [
        keras.ops.convert_to_numpy(
            ops.TissueSuppression(cutoff=2, filter_type=filter_type)(data=data_tensor)["data"]
        )
        for filter_type in (None, "svd_complex")
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-5)


def test_tissue_suppression_invalid_arguments():
    """Invalid filter_type / cutoff are rejected at construction time."""
    import pytest

    from zea import ops

    with pytest.raises(ValueError, match="Unknown filter_type"):
        ops.TissueSuppression(filter_type="not_a_filter")

    with pytest.raises(ValueError, match="fraction of the frames"):
        ops.TissueSuppression(cutoff=1.5)
