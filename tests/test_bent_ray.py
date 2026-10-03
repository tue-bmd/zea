"""Tests for the refraction-aware (bent-ray) travel-time model in zea.beamform.bent_ray."""

import keras
import numpy as np
import pytest

from zea.backend.autograd import AutoGrad
from zea.beamform.beamformer import (
    calculate_delays_heterogeneous_medium,
    calculate_delays_travel_time_map,
    tof_correction,
)
from zea.beamform.bent_ray import (
    compute_bent_ray_travel_times,
    sample_travel_time_map,
    straight_ray_travel_times,
)

from . import backend_equality_check, run_in_backend

C0 = 1500.0  # m/s

# Speed-of-sound grid
SOS_X = np.linspace(-12e-3, 12e-3, 25).astype(np.float32)
SOS_Z = np.linspace(0.0, 30e-3, 31).astype(np.float32)

# Travel-time grid
GRID_X = np.linspace(-10e-3, 10e-3, 41).astype(np.float32)
GRID_Z = np.linspace(0.0, 28e-3, 57).astype(np.float32)

N_RAYS = 384


def _elements(n_el, half_width=9e-3):
    xs = np.linspace(-half_width, half_width, n_el)
    return np.stack([xs, np.zeros(n_el), np.zeros(n_el)], axis=-1).astype(np.float32)


def _mesh(grid_x, grid_z):
    gz, gx = np.meshgrid(grid_z, grid_x, indexing="ij")
    return gx, gz


def _inclusion_sos():
    """Smooth Gaussian inclusion of +40 m/s."""
    sx, sz = _mesh(SOS_X, SOS_Z)
    blob = np.exp(-(sx**2 + (sz - 15e-3) ** 2) / (2 * (4e-3) ** 2))
    return (C0 + 40.0 * blob).astype(np.float32)


def _linear_gradient_travel_times(elements, gx, gz, c0, gradient):
    """Exact first-arrival times for c(z) = c0 + gradient * z (circular-arc rays)."""
    times = []
    for e in elements:
        v1, v2 = c0 + gradient * e[2], c0 + gradient * gz
        r2 = (gx - e[0]) ** 2 + (gz - e[2]) ** 2
        times.append(np.arccosh(1 + gradient**2 * r2 / (2 * v1 * v2)) / gradient)
    return np.stack(times)


def _cone_mask(elements, gx, gz, min_distance=2e-3):
    """Grid nodes within 45 degrees of each element's normal, away from the element."""
    return np.stack(
        [(np.abs(gx - e[0]) < gz) & (np.hypot(gx - e[0], gz) > min_distance) for e in elements]
    )


def test_homogeneous_medium_matches_distance_over_speed():
    """In a homogeneous medium the rays are straight and t = |x - e| / c."""
    elements = _elements(3)
    sos = np.full((SOS_Z.size, SOS_X.size), C0, dtype=np.float32)
    travel_times = compute_bent_ray_travel_times(
        sos, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, n_rays=N_RAYS
    )
    travel_times = keras.ops.convert_to_numpy(travel_times)
    assert travel_times.shape == (3, GRID_Z.size, GRID_X.size)

    gx, gz = _mesh(GRID_X, GRID_Z)
    expected = np.stack([np.hypot(gx - e[0], gz - e[2]) / C0 for e in elements])
    error = np.abs(travel_times - expected)
    np.testing.assert_array_less(error, 1e-9)  # below 1 ns everywhere


def test_linear_gradient_more_accurate_than_straight_rays():
    """For c(z) = c0 + g z the bent rays match the analytic solution, straight rays do not."""
    c0, gradient = 1450.0, 5000.0
    sx, sz = _mesh(SOS_X, SOS_Z)
    sos = (c0 + gradient * sz).astype(np.float32)
    elements = _elements(4)
    gx, gz = _mesh(GRID_X, GRID_Z)

    bent = compute_bent_ray_travel_times(sos, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, n_rays=N_RAYS)
    nodes = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=-1).astype(np.float32)
    straight = straight_ray_travel_times(sos, SOS_X, SOS_Z, elements, nodes)
    bent = keras.ops.convert_to_numpy(bent)
    straight = keras.ops.convert_to_numpy(straight).reshape(bent.shape)

    exact = _linear_gradient_travel_times(elements, gx, gz, c0, gradient)
    mask = _cone_mask(elements, gx, gz)
    bent_error = np.abs(bent - exact)[mask].mean()
    straight_error = np.abs(straight - exact)[mask].mean()

    assert bent_error < 0.5e-9
    assert bent_error < 0.1 * straight_error


def test_unreached_nodes_use_straight_rays():
    """Nodes outside the fan of rays fall back to the straight-ray travel time."""
    elements = _elements(1, half_width=0.0)
    sos = _inclusion_sos()
    kwargs = dict(n_rays=64, max_angle=np.deg2rad(20.0))
    filled = compute_bent_ray_travel_times(sos, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, **kwargs)
    unfilled = compute_bent_ray_travel_times(
        sos, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, fill_unreached=False, **kwargs
    )
    filled = keras.ops.convert_to_numpy(filled)[0]
    unfilled = keras.ops.convert_to_numpy(unfilled)[0]

    unreached = unfilled == 0
    assert unreached.any() and not unreached.all()
    assert np.all(filled > 0)
    np.testing.assert_allclose(filled[~unreached], unfilled[~unreached])

    gx, gz = _mesh(GRID_X, GRID_Z)
    nodes = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=-1).astype(np.float32)
    straight = keras.ops.convert_to_numpy(
        straight_ray_travel_times(sos, SOS_X, SOS_Z, elements, nodes)
    ).reshape(filled.shape)
    np.testing.assert_allclose(filled[unreached], straight[unreached])


def _travel_time_gradient(sos, weights, elements, n_rays=N_RAYS):
    """Gradient and value of sum(weights * travel_times) * 1e6 w.r.t. the sos map."""

    def loss(sos_map):
        travel_times = compute_bent_ray_travel_times(
            sos_map, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, n_rays=n_rays
        )
        return keras.ops.sum(keras.ops.convert_to_tensor(weights) * travel_times) * 1e6

    autograd = AutoGrad()
    autograd.set_function(loss)
    grad, value = autograd.gradient_and_value(keras.ops.convert_to_tensor(sos))
    return keras.ops.convert_to_numpy(grad), float(keras.ops.convert_to_numpy(value)), loss


def test_gradient_homogeneous_scaling():
    """Scaling c by (1 + eps) scales every travel time by 1 / (1 + eps), so <grad, c> = -L."""
    elements = _elements(2)
    sos = np.full((SOS_Z.size, SOS_X.size), C0, dtype=np.float32)
    weights = np.ones((2, GRID_Z.size, GRID_X.size), dtype=np.float32)
    grad, value, _ = _travel_time_gradient(sos, weights, elements)
    np.testing.assert_allclose(np.sum(grad * sos), -value, rtol=1e-4)


def test_gradient_matches_finite_differences():
    """The replay adjoint matches central finite differences for smooth perturbations."""
    rng = np.random.default_rng(42)
    elements = _elements(3)
    sos = _inclusion_sos()
    weights = np.linspace(0.0, 1.0, 3 * GRID_Z.size * GRID_X.size, dtype=np.float32)
    weights = weights.reshape(3, GRID_Z.size, GRID_X.size)
    grad, _, loss = _travel_time_gradient(sos, weights, elements)

    sx, sz = _mesh(SOS_X, SOS_Z)
    for _ in range(2):
        center = rng.uniform([-8e-3, 5e-3], [8e-3, 25e-3])
        direction = np.exp(-((sx - center[0]) ** 2 + (sz - center[1]) ** 2) / (2 * (3e-3) ** 2))
        direction = direction.astype(np.float32)
        eps = 5.0
        plus = float(
            keras.ops.convert_to_numpy(loss(keras.ops.convert_to_tensor(sos + eps * direction)))
        )
        minus = float(
            keras.ops.convert_to_numpy(loss(keras.ops.convert_to_tensor(sos - eps * direction)))
        )
        finite_difference = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(np.sum(grad * direction), finite_difference, rtol=0.02)


@backend_equality_check(decimal=[9, 9, 9])
def test_bent_ray_travel_times_backends():
    """Forward travel times agree across backends."""
    from zea.beamform.bent_ray import compute_bent_ray_travel_times

    sos = _inclusion_sos()
    return compute_bent_ray_travel_times(
        sos, SOS_X, SOS_Z, _elements(2), GRID_X[::2], GRID_Z[::2], n_rays=64
    )


@backend_equality_check(gt_backend="jax", backends=["tensorflow", "torch"], decimal=[4, 4])
def test_bent_ray_gradient_backends():
    """The replay-adjoint gradient agrees across backends."""
    from zea.backend.autograd import AutoGrad
    from zea.beamform.bent_ray import compute_bent_ray_travel_times

    elements = _elements(2)
    weights = np.linspace(0.0, 1.0, 2 * 29 * 21, dtype=np.float32).reshape(2, 29, 21)

    def loss(sos_map):
        travel_times = compute_bent_ray_travel_times(
            sos_map, SOS_X, SOS_Z, elements, GRID_X[::2], GRID_Z[::2], n_rays=64
        )
        return keras.ops.sum(keras.ops.convert_to_tensor(weights) * travel_times) * 1e6

    autograd = AutoGrad()
    autograd.set_function(loss)
    return autograd.gradient(keras.ops.convert_to_tensor(_inclusion_sos()))


def test_sample_travel_time_map_exact_for_homogeneous_medium():
    """Interpolating a homogeneous map is exact, also close to the elements."""
    elements = _elements(3)
    gx, gz = _mesh(GRID_X, GRID_Z)
    travel_time_map = np.stack([np.hypot(gx - e[0], gz - e[2]) / C0 for e in elements])
    rng = np.random.default_rng(0)
    points = np.stack(
        [rng.uniform(-10e-3, 10e-3, 200), np.zeros(200), rng.uniform(0.1e-3, 28e-3, 200)],
        axis=-1,
    ).astype(np.float32)

    sampled = keras.ops.convert_to_numpy(
        sample_travel_time_map(travel_time_map.astype(np.float32), GRID_X, GRID_Z, elements, points)
    )
    expected = np.stack([np.hypot(points[:, 0] - e[0], points[:, 2] - e[2]) / C0 for e in elements])
    np.testing.assert_allclose(sampled, expected, atol=1e-11)


def _multistatic_tof_inputs(elements, flatgrid, n_ax=256):
    n_el = elements.shape[0]
    rng = np.random.default_rng(1)
    return dict(
        data=rng.standard_normal((n_el, n_ax, n_el, 2)).astype(np.float32),
        flatgrid=flatgrid,
        t0_delays=np.zeros((n_el, n_el), dtype=np.float32),
        tx_apodizations=np.eye(n_el, dtype=np.float32),
        sound_speed=C0,
        probe_geometry=elements,
        initial_times=np.zeros(n_el, dtype=np.float32),
        sampling_frequency=20e6,
        demodulation_frequency=5e6,
        f_number=0.5,
        polar_angles=np.zeros(n_el, dtype=np.float32),
        focus_distances=np.zeros(n_el, dtype=np.float32),
        t_peak=np.full(n_el, 1e-7, dtype=np.float32),
        transmit_origins=np.zeros((n_el, 3), dtype=np.float32),
    )


def _flatgrid():
    gx, gz = np.meshgrid(np.linspace(-6e-3, 6e-3, 7), np.linspace(4e-3, 20e-3, 9))
    return np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=-1).astype(np.float32)


def test_calculate_delays_travel_time_map_matches_straight_rays():
    """A straight-ray travel-time map gives the delays of the straight-ray model."""
    elements = _elements(6)
    flatgrid = _flatgrid()
    inputs = _multistatic_tof_inputs(elements, flatgrid)
    sos = _inclusion_sos()

    gx, gz = _mesh(GRID_X, GRID_Z)
    nodes = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], axis=-1).astype(np.float32)
    travel_time_map = keras.ops.reshape(
        straight_ray_travel_times(sos, SOS_X, SOS_Z, elements, nodes),
        (elements.shape[0], GRID_Z.size, GRID_X.size),
    )
    common = (
        inputs["t0_delays"],
        elements,
        inputs["initial_times"],
        inputs["sampling_frequency"],
        inputs["t_peak"],
    )
    tx_map, rx_map = calculate_delays_travel_time_map(
        flatgrid, travel_time_map, GRID_X, GRID_Z, *common
    )
    tx_ref, rx_ref = calculate_delays_heterogeneous_medium(flatgrid, sos, SOS_X, SOS_Z, *common)
    # Interpolating the map on a 0.5 mm grid costs well below a tenth of a sample.
    np.testing.assert_allclose(keras.ops.convert_to_numpy(rx_map), rx_ref, atol=0.05)
    np.testing.assert_allclose(keras.ops.convert_to_numpy(tx_map), tx_ref, atol=0.05)


def test_tof_correction_travel_time_map_homogeneous():
    """In a homogeneous medium bent-ray TOF correction equals the straight-ray one."""
    elements = _elements(6)
    inputs = _multistatic_tof_inputs(elements, _flatgrid())
    sos = np.full((SOS_Z.size, SOS_X.size), C0, dtype=np.float32)
    travel_time_map = compute_bent_ray_travel_times(
        sos, SOS_X, SOS_Z, elements, GRID_X, GRID_Z, n_rays=N_RAYS
    )

    straight = tof_correction(**inputs, sos_map=sos, sos_grid_x=SOS_X, sos_grid_z=SOS_Z)
    bent = tof_correction(
        **inputs,
        travel_time_map=travel_time_map,
        travel_time_grid_x=GRID_X,
        travel_time_grid_z=GRID_Z,
    )
    straight = keras.ops.convert_to_numpy(straight)
    bent = keras.ops.convert_to_numpy(bent)
    assert bent.shape == straight.shape
    np.testing.assert_allclose(bent, straight, atol=0.02 * np.abs(straight).max())


def test_travel_time_map_requires_multistatic():
    elements = _elements(4)
    travel_time_map = np.zeros((4, GRID_Z.size, GRID_X.size), dtype=np.float32)
    with pytest.raises(AssertionError, match="multistatic"):
        calculate_delays_travel_time_map(
            _flatgrid(),
            travel_time_map,
            GRID_X,
            GRID_Z,
            np.zeros((2, 4), dtype=np.float32),
            elements,
            np.zeros(2, dtype=np.float32),
            20e6,
            np.zeros(2, dtype=np.float32),
        )


def test_bent_ray_travel_times_operation():
    """The operation outputs a travel-time map on a refined copy of the sos grid."""
    from zea.ops import BentRayTravelTimes

    elements = _elements(3)
    op = BentRayTravelTimes(n_rays=64, grid_upsample=2)
    out = op(sos_map=_inclusion_sos(), sos_grid_x=SOS_X, sos_grid_z=SOS_Z, probe_geometry=elements)

    grid_x = keras.ops.convert_to_numpy(out["travel_time_grid_x"])
    grid_z = keras.ops.convert_to_numpy(out["travel_time_grid_z"])
    expected_x = np.linspace(SOS_X[0], SOS_X[-1], 2 * SOS_X.size - 1)
    expected_z = np.linspace(SOS_Z[0], SOS_Z[-1], 2 * SOS_Z.size - 1)
    np.testing.assert_allclose(grid_x, expected_x, atol=1e-8)
    np.testing.assert_allclose(grid_z, expected_z, atol=1e-8)
    assert out["travel_time_map"].shape == (3, grid_z.size, grid_x.size)
    assert op.get_dict()["params"] == {"n_rays": 64, "grid_upsample": 2}


@run_in_backend("jax")
def test_bent_ray_pipeline_differentiable():
    """BentRayTravelTimes -> PatchedGrid(TOFCorrection -> CMPE) is differentiable w.r.t. sos.

    Runs in JAX: differentiating the compiled autofocus pipeline is not supported with
    TensorFlow's XLA, also not for the straight-ray model.
    """
    from zea.backend.autograd import AutoGrad
    from zea.ops import (
        BentRayTravelTimes,
        CommonMidpointPhaseError,
        PatchedGrid,
        Pipeline,
        TOFCorrection,
    )

    elements = _elements(20, half_width=6e-3)
    inputs = _multistatic_tof_inputs(elements, _flatgrid())
    data = inputs.pop("data")
    pipeline = Pipeline(
        [
            BentRayTravelTimes(n_rays=64, grid_upsample=2),
            PatchedGrid([TOFCorrection(), CommonMidpointPhaseError()], num_patches=3),
        ],
        with_batch_dim=False,
    )

    def loss(sos_map):
        out = pipeline(
            data=data,
            sos_map=sos_map,
            sos_grid_x=SOS_X,
            sos_grid_z=SOS_Z,
            apply_lens_correction=False,
            **inputs,
        )
        return keras.ops.mean(keras.ops.nan_to_num(out["data"]))

    autograd = AutoGrad()
    autograd.set_function(loss)
    grad, value = autograd.gradient_and_value(keras.ops.convert_to_tensor(_inclusion_sos()))
    grad = keras.ops.convert_to_numpy(grad)
    assert grad.shape == (SOS_Z.size, SOS_X.size)
    assert np.all(np.isfinite(grad)) and np.any(grad != 0)
    assert np.isfinite(float(keras.ops.convert_to_numpy(value)))


@run_in_backend("jax")
def test_gradient_of_jit_with_traced_grids():
    """Differentiating a jitted function where the grids are traced arguments.

    Regression test: a custom gradient must not close over tensors traced outside of
    it, which fails when JAX linearizes the enclosing jit.
    """
    import jax

    from zea.beamform.bent_ray import compute_bent_ray_travel_times

    def total_time(sos_map, sos_grid_x, sos_grid_z, elements, grid_x, grid_z):
        travel_times = compute_bent_ray_travel_times(
            sos_map, sos_grid_x, sos_grid_z, elements, grid_x, grid_z, n_rays=32
        )
        return keras.ops.sum(travel_times)

    grids = (SOS_X, SOS_Z, _elements(2), GRID_X[::4], GRID_Z[::4])
    grads = jax.grad(lambda s: jax.jit(total_time)(s, *grids))(_inclusion_sos())
    assert np.all(np.isfinite(grads)) and np.any(grads != 0)
