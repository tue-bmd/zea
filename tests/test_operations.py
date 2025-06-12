"""Tests for different operations.

Note that in each test, we have to reimport all modules involving keras, such
that when the backend is switched, the functions inside are reimported with the
correct backend.
"""

import math

import keras
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

from zea import ops
from zea.ops import Pipeline, Simulate
from zea.probes import Probe
from zea.scan import Scan

from . import backend_equality_check


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

    for parameter_value in np.linspace(*parameter_value_range, 10):
        A = parameter_value if comp_type == "a" else 0
        mu = parameter_value if comp_type == "mu" else 0

        companding = ops.Companding(comp_type=comp_type, expand=False)

        signal = np.clip((np.random.random(size) - 0.5) * 2, -1, 1)
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

    data = np.random.random(size) * (_input_range[1] - _input_range[0]) + _input_range[0]
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
    data = np.random.random(size) * (input_range[1] - input_range[0]) + input_range[0]

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
    data = np.random.random(size) + 1j * np.random.random(size)
    _data = ops.complex_to_channels(data, axis=axis)
    __data = ops.channels_to_complex(_data)
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
    data = np.random.random(size)
    _data = ops.channels_to_complex(data)
    __data = ops.complex_to_channels(_data, axis=axis)
    np.testing.assert_almost_equal(data, __data)


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

    scan = Scan(
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
    probe = Probe(
        probe_geometry=probe_geometry,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
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
    parameters = simulator_pipeline.prepare_parameters(probe=probe, scan=scan)

    data = []
    _data = []
    for _ in range(batch_size):
        # Define scatterers with random variation
        scat_x_base, scat_z_base = np.meshgrid(
            np.linspace(-10e-3, 10e-3, 5),
            np.linspace(5e-3, 30e-3, 5),
            indexing="ij",
        )
        # Add random perturbations
        scat_x = np.ravel(scat_x_base) + np.random.uniform(-1e-3, 1e-3, 25)
        scat_z = np.ravel(scat_z_base) + np.random.uniform(-1e-3, 1e-3, 25)
        n_scat = len(scat_x)
        # Select random subset of scatterers
        idx = np.random.choice(n_scat, n_scat, replace=False)[:n_scat]
        scat_positions = np.stack(
            [
                scat_x[idx],
                np.zeros_like(scat_x[idx]),
                scat_z[idx],
            ],
            axis=1,
        )

        output = simulator_pipeline(
            **parameters,
            scatterer_positions=scat_positions.astype(np.float32),
            scatterer_magnitudes=np.ones(n_scat, dtype=np.float32),
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


@backend_equality_check(decimal=4)
def test_hilbert_transform():
    """Test hilbert transform"""

    import keras

    from zea import ops

    # create some dummy sinusoidal data of size (2, 500, 128, 1)
    # sinusoids on axis 1
    data = np.sin(np.linspace(0, 2 * math.e * np.pi, 500))
    data = data[np.newaxis, :, np.newaxis, np.newaxis]
    data = np.tile(data, (2, 1, 128, 1))

    data = data + np.random.random(data.shape) * 0.1

    data = keras.ops.convert_to_tensor(data)

    data_iq = ops.hilbert(data, axis=-3)
    assert keras.ops.dtype(data_iq) in [
        "complex64",
        "complex128",
    ], f"Data type should be complex, got {keras.ops.dtype(data_iq)} instead."

    data_iq = keras.ops.convert_to_numpy(data_iq)

    reference_data_iq = hilbert(data, axis=-3)
    np.testing.assert_almost_equal(reference_data_iq, data_iq, decimal=4)

    return data_iq


@pytest.fixture(scope="module")
def spiral_image():
    """
    Fixture for generating a synthetic spiral image and noisy variants.
    Returns:
        dict: {
            "spiral": clean spiral image,
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

    rng = np.random.default_rng(seed=42)
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
@backend_equality_check(decimal=4)
def test_gaussian_blur(sigma, spiral_image):
    """
    Test `ops.GaussianBlur against scipy.ndimage.gaussian_filter.`
    `GaussianBlur` with default args should be equivalent to scipy.
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

    np.testing.assert_allclose(blurred_scipy, blurred_zea, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("sigma", [1.0, 2.0])
@backend_equality_check(decimal=4)
def test_lee_filter(sigma, spiral_image):
    """
    Test `ops.LeeFilter`, only checks if variance is reduced.
    """
    import keras

    from zea import ops

    # Use spiral image for testing
    image = spiral_image["spiral"]

    lee = ops.LeeFilter(sigma=sigma, with_batch_dim=False)

    image_tensor = keras.ops.convert_to_tensor(image[..., None])
    filtered = lee(data=image_tensor)["data"][..., 0]

    assert keras.ops.var(filtered) < keras.ops.var(image_tensor), (
        "LeeFilter should reduce variance of the processed image"
    )


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
