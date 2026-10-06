"""Tests for Aperture Domain Model Image REconstruction (ADMIRE)."""

import dataclasses

import numpy as np
import pytest

from zea.beamform.admire import (
    ADMIREConfig,
    _aperture_growth,
    _resolve_calibration,
    _ica_basis,
    apply_admire,
    elastic_net_ccd,
    generate_admire_models,
)
from zea.internal.cache import cache_disabled

from . import backend_equality_check

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 5e6
WAVELENGTH = SOUND_SPEED / CENTER_FREQUENCY
PITCH = 0.3e-3
BANDWIDTH = 0.6


def _reference_ccd(X, y, alpha, lambda_scaling_factor, tolerance, max_iterations):
    """Line-by-line port of the reference ``ccd_double_precision.c``."""
    n_obs, n_predictors = X.shape
    lam = lambda_scaling_factor * np.sqrt(np.mean(y**2))
    std = np.sqrt(np.mean((y - y.mean()) ** 2))
    beta = np.zeros(n_predictors)
    if std == 0:
        return beta
    lam = lam / std
    residual = y / std
    max_change, iteration = np.inf, 0
    while max_change >= tolerance and iteration < max_iterations:
        max_change = 0.0
        for j in range(n_predictors):
            previous = beta[j]
            residual = residual + X[:, j] * previous
            p_j = X[:, j] @ residual / n_obs
            new = np.sign(p_j) * max(abs(p_j) - lam * alpha, 0.0)
            new = new / (1 / n_obs + lam * (1 - alpha))
            beta[j] = new
            residual = residual - X[:, j] * new
            max_change = max(max_change, (previous - new) ** 2 / n_obs)
        iteration += 1
    return beta * std


def _reference_fobi(X):
    """Port of the reference ``ica.m`` (FOBI) followed by ``pinv(W)``."""
    d, n = X.shape
    X = X - X.mean(axis=1, keepdims=True)
    covariance = X @ X.conj().T / (n - 1)
    eigenvalues, E = np.linalg.eigh(covariance)
    whitening = np.diag(1 / np.sqrt(eigenvalues)) @ E.conj().T
    X_w = whitening @ X
    U, _, _ = np.linalg.svd((np.sum(X_w * X_w, axis=0) * X_w) @ X_w.conj().T)
    return np.linalg.pinv(U @ whitening)


@pytest.fixture(scope="module")
def iq_models():
    """Small ADMIRE models for analytic data around 12 mm depth."""
    depths = np.arange(11e-3, 13e-3, WAVELENGTH / 8)
    with cache_disabled():
        return depths, generate_admire_models(
            depths,
            sound_speed=SOUND_SPEED,
            center_frequency=CENTER_FREQUENCY,
            pitch=PITCH,
            n_elements=16,
            f_number=2.0,
            analytic=True,
        )


def _point_channel_data(depths, n_elements, x, z, analytic=True):
    """Delayed channel data of a point scatterer under a 0 degree plane wave."""
    elements = (np.arange(n_elements) - (n_elements - 1) / 2) * PITCH
    delays = (depths[:, None] + np.hypot(elements, depths[:, None])) / SOUND_SPEED
    arrival = (z + np.hypot(elements - x, z)) / SOUND_SPEED
    dt = delays - arrival
    sigma_t = np.sqrt(8 * np.log(2)) / (2 * np.pi * BANDWIDTH * CENTER_FREQUENCY)
    signal = np.exp(-(dt**2) / (2 * sigma_t**2)) * np.exp(2j * np.pi * CENTER_FREQUENCY * dt)
    if analytic:
        channels = np.stack([signal.real, signal.imag], axis=-1)
    else:
        channels = signal.real[..., None]
    return channels[:, None].astype(np.float32)  # (n_z, 1 line, n_el, n_ch)


@pytest.mark.parametrize("max_iterations", [2, 50])
@backend_equality_check()
def test_elastic_net_ccd_matches_reference(max_iterations):
    """The batched solver matches the reference coordinate descent, also with masked rows."""
    rng = np.random.default_rng(0)
    n_obs, n_predictors, n_rhs, n_pad = 12, 20, 3, 4
    X = rng.standard_normal((n_obs, n_predictors))
    X /= np.linalg.norm(X, axis=0)
    y = rng.standard_normal((n_rhs, n_obs))
    settings = dict(alpha=0.9, lambda_scaling_factor=0.0189, tolerance=1e-10)
    expected = np.stack(
        [_reference_ccd(X, y_i, max_iterations=max_iterations, **settings) for y_i in y]
    )

    # Zero-padded rows that are masked out must not change the fit
    X_padded = np.concatenate([X, np.zeros((n_pad, n_predictors))]).astype(np.float32)
    y_padded = np.concatenate([y, np.zeros((n_rhs, n_pad))], axis=1).astype(np.float32)
    mask = np.r_[np.ones(n_obs), np.zeros(n_pad)].astype(np.float32)
    beta = elastic_net_ccd(X_padded, y_padded, mask, max_iterations=max_iterations, **settings)

    np.testing.assert_allclose(np.asarray(beta), expected, atol=1e-4)
    return beta


def test_elastic_net_ccd_zero_data():
    """All-zero observations give all-zero coefficients instead of NaNs."""
    X = np.eye(4, dtype=np.float32)
    beta = elastic_net_ccd(
        X, np.zeros((2, 4), np.float32), np.ones(4, np.float32), 0.9, 0.0189, 0.1, 10
    )
    np.testing.assert_array_equal(np.asarray(beta), 0)


@pytest.mark.parametrize("keep_chunks", [True, False])
def test_ica_basis_matches_reference(keep_chunks):
    """The chunked FOBI basis matches the reference ICA, up to column order and phase."""
    rng = np.random.default_rng(1)
    model = rng.standard_normal((6, 500)) + 1j * rng.standard_normal((6, 500))
    model = model * rng.uniform(0.5, 2, (6, 1))

    basis = _ica_basis(lambda: iter(np.split(model, 5, axis=1)), 6, keep_chunks=keep_chunks)
    expected = _reference_fobi(model)

    # Columns are only defined up to order and a unit-modulus factor: compare via correlations
    correlation = np.abs(
        (basis / np.linalg.norm(basis, axis=0)).conj().T
        @ (expected / np.linalg.norm(expected, axis=0))
    )
    np.testing.assert_allclose(np.sort(correlation.max(axis=1)), 1, atol=1e-6)


def test_generate_admire_models(iq_models):
    """Generated models are consistent: band, unit-norm columns, masks and windows."""
    depths, models = iq_models
    n_windows, n_freqs, n_el, n_predictors = models.models.shape

    assert n_el == 16
    assert n_windows == models.window_starts.size == models.aperture_mask.shape[0]
    assert models.fft_length == 2 * models.window_length
    np.testing.assert_array_equal(np.diff(models.window_starts), models.window_length)
    assert models.window_starts[-1] + models.window_length <= depths.size

    half_band = 0.5 * 1.2 * BANDWIDTH * CENTER_FREQUENCY
    assert np.all(np.abs(models.frequencies - CENTER_FREQUENCY) <= half_band)
    assert n_freqs >= 1

    # At f-number 2, 12 mm deep needs 20 elements: the full 16-element sub-aperture
    assert np.all(models.aperture_mask)

    # Active columns have unit norm and live on the active elements only
    norms = np.linalg.norm(models.models, axis=2)
    assert np.all((np.abs(norms - 1) < 1e-5) | (norms == 0))
    inactive = ~models.aperture_mask[:, None, :, None]
    assert np.all(np.where(inactive, models.models, 0) == 0)

    # With ICA, ROI and outer models each have at most one predictor per active element
    n_active = models.aperture_mask.sum(axis=1)[:, None]
    assert np.all(models.roi_mask.sum(axis=-1) <= n_active)
    assert np.all(models.roi_mask.sum(axis=-1) > 0)
    assert np.all((norms > 0).sum(axis=-1) <= 2 * n_active)


def test_generate_admire_models_parallel(iq_models):
    """Generating windows in worker processes gives the same models."""
    depths, models = iq_models
    with cache_disabled():
        parallel = generate_admire_models(
            depths,
            sound_speed=SOUND_SPEED,
            center_frequency=CENTER_FREQUENCY,
            pitch=PITCH,
            n_elements=16,
            f_number=2.0,
            analytic=True,
            n_workers=2,
        )
    np.testing.assert_array_equal(parallel.models, models.models)
    np.testing.assert_array_equal(parallel.roi_mask, models.roi_mask)


@pytest.mark.parametrize(
    "center_frequency, expected",
    [
        (7.8125e6, (1.295, 1.0, 0.855)),  # table key 7813000
        (25e6 / 12, (1.32, 1.0, 0.845)),  # table key 2083300
        (5e6, (1.29, 0.995, 0.85)),
        (4.2e6, (1.135, 0.995, 0.925)),  # not in the table: default values
    ],
)
def test_reference_wavenumber_calibration(center_frequency, expected):
    """The reference calibration is looked up by the nearest (rounded) table frequency."""
    np.testing.assert_allclose(_resolve_calibration("reference", center_frequency, 3), expected)


@pytest.mark.parametrize(
    "z, n_el, expected",
    [
        (5.7e-3, 32, np.arange(11, 21)),  # ceil(5.7 mm / 0.3 mm / 2) = 10 elements
        (1e-3, 32, np.arange(12, 20)),  # at least min_num_elements
        (30e-3, 32, np.arange(32)),  # at most all elements
    ],
)
def test_aperture_growth(z, n_el, expected):
    """Aperture growth keeps the central depth / f-number elements, like the reference."""
    mask = _aperture_growth(z, n_el, PITCH, f_number=2.0, min_num_elements=8)
    np.testing.assert_array_equal(np.flatnonzero(mask), expected)
    assert np.all(_aperture_growth(z, n_el, PITCH, f_number=0, min_num_elements=8))


def test_generate_admire_models_rejects_coarse_grid():
    """A grid too coarse for the signal band is rejected."""
    depths = np.arange(10e-3, 12e-3, WAVELENGTH)
    with cache_disabled(), pytest.raises(ValueError, match="axial"):
        generate_admire_models(depths, SOUND_SPEED, CENTER_FREQUENCY, PITCH, 8, analytic=True)
    with cache_disabled(), pytest.raises(ValueError, match="Nyquist"):
        generate_admire_models(depths, SOUND_SPEED, CENTER_FREQUENCY, PITCH, 8, analytic=False)


def test_apply_admire_suppresses_off_axis_clutter(iq_models):
    """An on-axis scatterer is kept, an off-axis one is suppressed much more."""
    depths, models = iq_models

    def retained_energy(x, z):
        data = _point_channel_data(depths, models.n_elements, x, z)
        out = np.asarray(apply_admire(data, models))
        das = data[..., 0].sum(-1) + 1j * data[..., 1].sum(-1)
        admire = out[..., 0].sum(-1) + 1j * out[..., 1].sum(-1)
        return np.sum(np.abs(admire) ** 2) / np.sum(np.abs(das) ** 2)

    on_axis = retained_energy(0.0, 12e-3)
    off_axis = retained_energy(3e-3, 12e-3)
    assert on_axis > 0.3
    assert off_axis < 0.5 * on_axis


def test_apply_admire_leaves_uncovered_samples(iq_models):
    """Samples after the last STFT window are passed through unchanged."""
    depths, models = iq_models
    rng = np.random.default_rng(2)
    n_extra = 3
    data = rng.standard_normal((depths.size + n_extra, 2, models.n_elements, 2)).astype(np.float32)
    out = np.asarray(apply_admire(data, models, window_batch_size=2))

    assert out.shape == data.shape
    end = models.window_starts[-1] + models.window_length
    np.testing.assert_array_equal(out[end:], data[end:])
    assert not np.allclose(out[:end], data[:end])


def test_apply_admire_window_batching(iq_models):
    """Fitting the windows in batches gives the same result as all at once."""
    depths, models = iq_models
    data = _point_channel_data(depths, models.n_elements, 1e-3, 12e-3)
    np.testing.assert_allclose(
        np.asarray(apply_admire(data, models, window_batch_size=3)),
        np.asarray(apply_admire(data, models)),
        atol=1e-5,
    )


def test_apply_admire_rf():
    """Real RF data is supported and stays real (one channel)."""
    depths = np.arange(11.5e-3, 12.5e-3, WAVELENGTH / 8)
    with cache_disabled():
        models = generate_admire_models(
            depths, SOUND_SPEED, CENTER_FREQUENCY, PITCH, 12, f_number=2.0, analytic=False
        )
    data = _point_channel_data(depths, 12, 0.0, 12e-3, analytic=False)
    out = np.asarray(apply_admire(data, models))
    assert out.shape == data.shape
    assert np.sum(out**2) > 0.3 * np.sum(data**2)


def test_apply_admire_checks_data(iq_models):
    """Mismatched element count or data type raise a clear error."""
    depths, models = iq_models
    with pytest.raises(ValueError, match="elements"):
        apply_admire(np.zeros((depths.size, 1, 8, 2), np.float32), models)
    with pytest.raises(ValueError, match="IQ"):
        apply_admire(np.zeros((depths.size, 1, 16, 1), np.float32), models)


def _admire_inputs(n_el=16, n_x=5, n_tx=2):
    """Grid, probe and parameters for an ADMIRE operation, with random aligned data."""
    probe_x = (np.arange(n_el) - (n_el - 1) / 2) * PITCH
    probe_geometry = np.stack([probe_x, 0 * probe_x, 0 * probe_x], axis=1).astype(np.float32)
    z = np.arange(11e-3, 12e-3, WAVELENGTH / 8)
    x = probe_x[n_el // 2 - n_x // 2 : n_el // 2 - n_x // 2 + n_x]
    X, Z = np.meshgrid(x, z)
    grid = np.stack([X, 0 * X, Z], axis=-1).astype(np.float32)
    rng = np.random.default_rng(3)
    data = rng.standard_normal((1, n_tx, grid.shape[0] * n_x, n_el, 2)).astype(np.float32)
    params = dict(
        grid=grid,
        probe_geometry=probe_geometry,
        sound_speed=SOUND_SPEED,
        center_frequency=0.0,  # as left by zea.ops.Demodulate
        demodulation_frequency=CENTER_FREQUENCY,
        f_number=2.0,
    )
    return data, params


def test_admire_operation():
    """The operation matches apply_admire on the compounded sub-aperture of each column."""
    from zea import ops

    data, params = _admire_inputs()
    n_z, n_x = params["grid"].shape[:2]
    with cache_disabled():
        operation = ops.ADMIRE(n_elements=9, with_batch_dim=True)
        output = np.asarray(operation(data=data, **params)["data"])
        eager_operation = ops.ADMIRE(n_elements=9, with_batch_dim=True, jit_compile=False)
        eager_output = np.asarray(eager_operation(data=data, **params)["data"])
    assert output.shape == (1, n_z * n_x, 2)

    # Columns sit on elements 6..10, so their 9-element sub-apertures are elements 2..14
    models = next(iter(operation._models_cache.values()))
    compounded = data[0].sum(axis=0).reshape(n_z, n_x, -1, 2)
    lines = np.stack([compounded[:, i, 2 + i : 11 + i] for i in range(n_x)], axis=1)
    expected = np.asarray(apply_admire(lines, models)).sum(axis=2).reshape(n_z * n_x, 2)
    np.testing.assert_allclose(eager_output[0], expected, rtol=1e-4, atol=1e-4)
    # Compiled reductions may round differently (e.g. on GPU), which can shift the
    # sweep at which the coarse CCD tolerance is met
    np.testing.assert_allclose(output[0], expected, rtol=0, atol=1e-2)

    # A second call reuses the models and the compiled fit
    operation(data=data, **params)
    assert len(operation._models_cache) == 1 and len(operation._fit_cache) == 1


def test_admire_in_beamform_requires_full_grid():
    """Patching splits the depth axis, which ADMIRE cannot handle."""
    from zea import ops

    data, params = _admire_inputs()
    operation = ops.ADMIRE(n_elements=9, with_batch_dim=True)
    with pytest.raises(ValueError, match="num_patches=1"):
        operation(data=data[:, :, :-5], **params)


def test_admire_registered_as_beamformer():
    """ADMIRE is available to Beamform and keeps its settings in a config."""
    from zea import ops

    beamform = ops.Beamform(beamformer="admire", num_patches=1, n_elements=9)
    admire = beamform.operations[1]
    assert isinstance(admire, ops.ADMIRE) and admire.n_elements == 9

    config = ADMIREConfig(alpha=0.5)
    operation = ops.ADMIRE(n_elements=9, config=config)
    params = operation.get_dict()["params"]
    assert params["n_elements"] == 9
    assert params["config"] == dataclasses.asdict(config)
    assert "models" not in params
    restored = ops.get_ops("admire")(**params)
    assert ADMIREConfig(**restored.config) == config
