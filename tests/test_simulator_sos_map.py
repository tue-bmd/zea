"""Sound speed maps in :func:`zea.simulator.simulate_rf`: the straight-ray travel times.

A map changes only the travel times of the element-scatterer paths, so a uniform map is the
homogeneous medium, a map at another speed is the homogeneous medium at that speed, and a
layered map delays each echo by the analytic straight-ray time. The FFT length, the record
gate, the gradients, the op and :class:`zea.Parameters` follow the map too.
"""

import keras
import numpy as np
import pytest
from keras import ops

import zea
from zea.ops import Pipeline, Simulate
from zea.simulator import fft_length, in_record, pressure_field, record_reach, simulate_rf

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
N_AX = 512
N_PERIOD = 4.0


def _linear_probe(n_el=16, pitch=0.3e-3):
    x = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    return np.stack([x, np.zeros(n_el), np.zeros(n_el)], -1).astype(np.float32)


def _matrix_probe(n_side=4, pitch=0.3e-3):
    x = (np.arange(n_side) - (n_side - 1) / 2) * pitch
    gx, gy = np.meshgrid(x, x, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(n_side**2)], -1).astype(np.float32)


def _phantom(n=24, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.01, 0.028, n)
    pos = np.stack([z * rng.uniform(-0.5, 0.5, n), rng.uniform(-1e-3, 1e-3, n), z], -1)
    return pos.astype(np.float32), rng.uniform(0.5, 1.0, n).astype(np.float32)


def _scan(geometry, n_tx=2):
    rng = np.random.default_rng(1)
    focus = np.array([[0.0, 0.0, 0.02], [0.005, 0.0, 0.025]])[:n_tx]
    dist = np.linalg.norm(focus[:, None] - geometry[None], axis=-1)
    t0 = ((dist.max(1, keepdims=True) - dist) / SOUND_SPEED).astype(np.float32)
    return {
        "probe_geometry": geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": N_AX,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "t0_delays": t0,
        "initial_times": np.zeros(n_tx, np.float32),
        "element_width": 0.27e-3,
        "attenuation_coef": 0.5,
        "tx_apodizations": rng.uniform(0.5, 1.0, (n_tx, len(geometry))).astype(np.float32),
        "t_peak": np.zeros(n_tx, np.float32),
        "scatter_exponent": 1.5,
    }


def _case(geometry, **overrides):
    positions, magnitudes = _phantom()
    return {
        "scatterer_positions": positions,
        "scatterer_magnitudes": magnitudes,
        **_scan(geometry),
        **overrides,
    }


def _map(speed, x=(-0.02, 0.02), z=(-0.005, 0.04), nx=41, nz=46, y=None, ny=None):
    """A uniform map with its grids; a 3D one when a ``y`` range is given."""
    grids = {
        "sos_grid_x": np.linspace(*x, nx).astype(np.float32),
        "sos_grid_z": np.linspace(*z, nz).astype(np.float32),
    }
    shape = (nz, nx)
    if y is not None:
        grids["sos_grid_y"] = np.linspace(*y, ny).astype(np.float32)
        shape = (nz, nx, ny)
    return {"sos_map": np.full(shape, speed, np.float32), **grids}


def _layered_map(c_top, c_bottom, z_interface, **kwargs):
    """Two horizontal layers: ``c_bottom`` from the map row at ``z_interface`` on. The bilinear
    map ramps over the row above it, so the interface is effectively half a row higher."""
    trio = _map(c_top, **kwargs)
    trio["sos_map"][trio["sos_grid_z"] >= z_interface] = c_bottom
    return trio


def _tensors(kwargs):
    return {
        k: ops.convert_to_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
    }


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


def _assert_close(reference, result, rel_tol=1e-3):
    reference, result = _np(reference), _np(result)
    assert result.shape == reference.shape
    rel = np.linalg.norm(reference - result) / np.linalg.norm(reference)
    assert rel < rel_tol, rel


def _rel(reference, result):
    reference, result = _np(reference), _np(result)
    return np.linalg.norm(reference - result) / np.linalg.norm(reference)


def _pulse(t):
    """Unit-peak Hann-windowed tone of ``simulate_rf``, centred at t = 0."""
    width = N_PERIOD / CENTER_FREQUENCY
    window = np.where(np.abs(t) < width / 2, np.cos(np.pi * t / width) ** 2, 0.0)
    return window * np.cos(2 * np.pi * CENTER_FREQUENCY * t)


def _echo_time(trace):
    """Arrival time of the one pulse in ``trace``: the peak of the envelope of its matched
    filter, to a fraction of a sample by a parabolic fit."""
    t = np.arange(N_AX) / SAMPLING_FREQUENCY
    pulse = _pulse(t - t[N_AX // 2])
    score = np.fft.fft(np.fft.ifft(np.fft.fft(trace) * np.conj(np.fft.fft(pulse))).real)
    score[N_AX // 2 + 1 :] = 0.0
    score[1 : N_AX // 2] *= 2.0
    envelope = np.abs(np.fft.ifft(score))
    k = int(np.argmax(envelope))
    a, b, c = envelope[k - 1], envelope[k], envelope[k + 1]
    return ((k + 0.5 * (a - c) / (a - 2 * b + c) + N_AX // 2) % N_AX) / SAMPLING_FREQUENCY


def _echo_times(rf):
    return np.array([_echo_time(rf[:, e]) for e in range(rf.shape[1])])


def _point_echo(position, trio, geometry=None, transmit=0, **overrides):
    """One transmit from element ``transmit``, one unit scatterer, elements without
    directivity, attenuation or scattering gain: the echo of every element is one clean pulse
    whose arrival time is the tx path plus the rx path."""
    geometry = _linear_probe() if geometry is None else geometry
    apod = np.zeros((1, len(geometry)), np.float32)
    apod[0, transmit] = 1.0
    kwargs = _case(geometry, **_scan(geometry, n_tx=1))
    kwargs.update(
        scatterer_positions=np.asarray([position], np.float32),
        scatterer_magnitudes=np.ones(1, np.float32),
        t0_delays=np.zeros((1, len(geometry)), np.float32),
        tx_apodizations=apod,
        element_width=1e-6,
        attenuation_coef=0.0,
        scatter_exponent=0.0,
        n_ax=N_AX,
        **trio,
        **overrides,
    )
    return _np(simulate_rf(**_tensors(kwargs)))[0, :, :, 0]


UNIFORM_CASES = {
    "matrix": _case(_matrix_probe()),
    "two_dimensional": _case(_linear_probe(), two_dimensional=True),
    "sub_elements": _case(_linear_probe(), n_sub_elements=(3, 2)),
    "lens": _case(_linear_probe(), apply_lens_correction=True),
    "matrix_3d_map": _case(_matrix_probe(), **_map(SOUND_SPEED, y=(-0.003, 0.003), ny=7)),
    "lens_3d_map": _case(
        _linear_probe(), apply_lens_correction=True, **_map(SOUND_SPEED, y=(-0.003, 0.003), ny=7)
    ),
}


@pytest.mark.parametrize("name", list(UNIFORM_CASES))
def test_uniform_map_is_the_homogeneous_medium(name):
    kwargs = dict(UNIFORM_CASES[name])
    if "sos_map" not in kwargs:
        kwargs.update(_map(SOUND_SPEED))
    homogeneous = {k: v for k, v in kwargs.items() if not k.startswith("sos_")}
    reference = simulate_rf(**_tensors(homogeneous))
    assert _np(reference).any()
    _assert_close(reference, simulate_rf(**_tensors(kwargs)))


def test_map_at_another_speed_is_the_homogeneous_medium_at_that_speed():
    """With point-like elements only the travel times see the speed, so a map at ``c2``
    covering every path is the homogeneous medium at ``c2``; a map that covers no path is the
    homogeneous medium at ``sound_speed``."""
    c2 = 1450.0
    kwargs = _case(_matrix_probe(), element_width=1e-6, n_fft=2048)
    reference = simulate_rf(**_tensors({**kwargs, "sound_speed": c2}))
    mapped = simulate_rf(**_tensors({**kwargs, **_map(c2)}))
    _assert_close(reference, mapped)
    assert _rel(simulate_rf(**_tensors(kwargs)), mapped) > 0.5

    aside = _map(c2, x=(0.1, 0.2))
    _assert_close(simulate_rf(**_tensors(kwargs)), simulate_rf(**_tensors({**kwargs, **aside})))


def test_layered_map_delays_each_echo_by_its_straight_ray_time():
    c_top, c_bottom, z_interface = 1540.0, 1400.0, 0.012
    trio = _layered_map(c_top, c_bottom, z_interface, z=(-0.004, 0.04), nz=45)
    assert z_interface in trio["sos_grid_z"]
    interface = z_interface - 0.5e-3  # the map ramps over the 1 mm row above
    position = np.array([0.002, 0.0, 0.025])
    geometry = _linear_probe()
    rf = _point_echo(position, trio, geometry, transmit=3, n_sos_ray_samples=1024)
    plain = _point_echo(position, {}, geometry, transmit=3)

    def one_way(element):
        dist = np.linalg.norm(position - geometry[element])
        top = interface / position[2]  # fraction of the ray above the interface
        return dist * (top / c_top + (1 - top) / c_bottom)

    expected = one_way(3) + np.array([one_way(e) for e in range(len(geometry))])
    dist = np.linalg.norm(position - geometry, axis=1)
    homogeneous = (dist[3] + dist) / c_top
    # The map moves the echoes by several samples; the fit resolves a tenth of one, its bias
    # cancelling between the two records.
    assert np.abs(expected - homogeneous).min() > 4 / SAMPLING_FREQUENCY
    delay = _echo_times(rf) - _echo_times(plain)
    assert np.abs(delay - (expected - homogeneous)).max() < 0.1 / SAMPLING_FREQUENCY


def test_3d_map_extruded_from_a_2d_one_gives_the_same_rf():
    kwargs = _case(_matrix_probe())
    planar = _layered_map(1540.0, 1450.0, 0.015)
    solid = {
        **planar,
        "sos_grid_y": np.linspace(-0.004, 0.004, 9).astype(np.float32),
        "sos_map": np.repeat(planar["sos_map"][..., None], 9, axis=-1),
    }
    reference = simulate_rf(**_tensors({**kwargs, **planar}))
    assert _rel(simulate_rf(**_tensors(kwargs)), reference) > 0.5
    _assert_close(reference, simulate_rf(**_tensors({**kwargs, **solid})))


def test_3d_map_varying_along_y_times_the_rays_in_elevation():
    """Layers in y are invisible to a 2D map; a 3D map times the ray through them."""
    c_near, c_far, y_interface = 1540.0, 1400.0, 0.004
    trio = _map(c_near, y=(-0.002, 0.012), ny=15)
    assert y_interface in trio["sos_grid_y"]
    trio["sos_map"][..., trio["sos_grid_y"] >= y_interface] = c_far
    interface = y_interface - 0.5e-3  # the map ramps over the 1 mm column before
    position = np.array([0.001, 0.01, 0.02])
    geometry = _linear_probe()
    rf = _point_echo(position, trio, geometry, transmit=5, n_sos_ray_samples=1024)
    plain = _point_echo(position, {}, geometry, transmit=5)

    def one_way(element):
        dist = np.linalg.norm(position - geometry[element])
        near = interface / position[1]
        return dist * (near / c_near + (1 - near) / c_far)

    expected = one_way(5) + np.array([one_way(e) for e in range(len(geometry))])
    dist = np.linalg.norm(position - geometry, axis=1)
    homogeneous = (dist[5] + dist) / c_near
    assert np.abs(expected - homogeneous).min() > 4 / SAMPLING_FREQUENCY
    delay = _echo_times(rf) - _echo_times(plain)
    assert np.abs(delay - (expected - homogeneous)).max() < 0.1 / SAMPLING_FREQUENCY
    # The same map without its y axis is a uniform map at c_near.
    planar = {k: v for k, v in trio.items() if k != "sos_grid_y"}
    planar["sos_map"] = trio["sos_map"][..., 0]
    flat = _point_echo(position, planar, geometry, transmit=5)
    _assert_close(plain, flat)


def test_map_and_grids_are_validated():
    kwargs = _case(_linear_probe())
    trio = _map(SOUND_SPEED)
    solid = _map(SOUND_SPEED, y=(-0.003, 0.003), ny=7)

    def run(**overrides):
        return simulate_rf(**_tensors({**kwargs, **overrides}))

    with pytest.raises(ValueError, match="without sos_map"):
        run(sos_grid_x=trio["sos_grid_x"], sos_grid_z=trio["sos_grid_z"])
    with pytest.raises(ValueError, match="sos_grid_x and sos_grid_z"):
        run(sos_map=trio["sos_map"], sos_grid_x=trio["sos_grid_x"])
    with pytest.raises(ValueError, match="does not match its grids"):
        run(**trio, sos_grid_y=solid["sos_grid_y"])
    with pytest.raises(ValueError, match="does not match its grids"):
        run(**{**solid, "sos_grid_y": None})
    with pytest.raises(ValueError, match="does not match its grids"):
        run(**{**trio, "sos_map": trio["sos_map"].T})
    with pytest.raises(ValueError, match="at least two points"):
        run(**_map(SOUND_SPEED, x=(0.0, 0.0), nx=1))
    with pytest.raises(ValueError, match="uniformly spaced"):
        run(**{**trio, "sos_grid_x": trio["sos_grid_x"] ** 3})
    with pytest.raises(ValueError, match="ascending"):
        run(**{**trio, "sos_grid_z": trio["sos_grid_z"][::-1].copy()})
    with pytest.raises(ValueError, match="positive"):
        run(**{**trio, "sos_map": -trio["sos_map"]})
    # Other float types are cast.
    reference = run(**trio)
    _assert_close(reference, run(**{k: v.astype(np.float64) for k, v in trio.items()}))
    _assert_close(reference, run(**{k: v.astype(np.float64) for k, v in solid.items()}))


def _slab_scene(n_el=48, speed=1400.0):
    """A wide probe over a slow slab, with scatterers right up to the record's reach, where
    the slow paths of the far elements stretch furthest past the record."""
    geometry = _linear_probe(n_el)
    scan = _scan(geometry, n_tx=1)
    trio = _map(speed, x=(-0.06, 0.06), z=(-0.005, 0.06), nx=121, nz=66)
    args = {
        k: scan[k]
        for k in (
            "probe_geometry",
            "sound_speed",
            "n_ax",
            "sampling_frequency",
            "center_frequency",
            "t0_delays",
            "initial_times",
            "t_peak",
        )
    }
    reach = record_reach(**args, sos_map=trio["sos_map"])
    rng = np.random.default_rng(4)
    x = rng.uniform(-0.012, 0.012, 200)
    z = rng.uniform(0.6, 1.0, 200) * reach
    positions = np.stack([x, np.zeros(200), z], -1).astype(np.float32)
    kwargs = {
        **scan,
        **trio,
        "scatterer_positions": positions,
        "scatterer_magnitudes": rng.uniform(0.5, 1.0, 200).astype(np.float32),
        "attenuation_coef": 0.0,
        "scatter_exponent": 0.0,
    }
    return kwargs, args, trio


def test_fft_length_grows_with_the_map_and_keeps_the_echoes_from_wrapping():
    """Scatterers at the reach of a slow slab, far to the side of a wide probe: their paths
    to the far elements are the longest a kept scatterer can have, and the slowest."""
    kwargs, args, trio = _slab_scene(speed=1000.0)
    rng = np.random.default_rng(6)
    x = rng.uniform(0.02, 0.05, 2000) * rng.choice([-1.0, 1.0], 2000)
    candidates = np.stack([x, np.zeros(2000), rng.uniform(0.002, 0.03, 2000)], -1)
    candidates = candidates.astype(np.float32)
    kept = _np(in_record(candidates, **args, **trio))
    positions = candidates[kept][:200]
    kwargs.update(
        scatterer_positions=positions, scatterer_magnitudes=np.ones(len(positions), np.float32)
    )
    shift = kwargs["t0_delays"]
    common = (N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, kwargs["probe_geometry"])
    homogeneous = fft_length(*common, shift.min(), shift.max())
    mapped = fft_length(*common, shift.min(), shift.max(), sos_map=trio["sos_map"])
    assert mapped > homogeneous
    # Only the extremes of the map matter, not its shape.
    assert mapped == fft_length(*common, shift.min(), shift.max(), sos_map=[[1000.0, 1540.0]])

    # The derived length holds every echo, up to the faint tail of the synthesis that a longer
    # record also shows for a homogeneous medium; the homogeneous bound wraps the late echoes.
    derived = simulate_rf(**_tensors(kwargs))
    reference = simulate_rf(**_tensors({**kwargs, "n_fft": 4 * mapped}))
    _assert_close(reference, derived, rel_tol=5e-4)
    assert _rel(reference, simulate_rf(**_tensors({**kwargs, "n_fft": homogeneous}))) > 5e-3


def test_short_explicit_fft_length_warns_with_a_map(caplog):
    kwargs, _, _ = _slab_scene(n_el=16)
    import logging

    with caplog.at_level(logging.WARNING, logger="zea"):
        simulate_rf(**_tensors({**kwargs, "n_fft": N_AX}))
    assert any("wrapping" in record.getMessage() for record in caplog.records)


def test_parameters_derive_n_fft_from_the_map():
    kwargs, _, trio = _slab_scene(n_el=16)
    parameters = zea.Parameters(
        n_tx=1,
        n_el=16,
        n_ax=N_AX,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
        probe_geometry=kwargs["probe_geometry"],
        t0_delays=kwargs["t0_delays"],
        initial_times=kwargs["initial_times"],
        t_peak=kwargs["t_peak"],
        tx_apodizations=kwargs["tx_apodizations"],
        sound_speed=SOUND_SPEED,
        selected_transmits="all",
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        element_width=0.27e-3,
        attenuation_coef=0.0,
        **trio,
    )
    shift = kwargs["t0_delays"]
    common = (N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, kwargs["probe_geometry"])
    expected = fft_length(*common, shift.min(), shift.max(), sos_map=trio["sos_map"])
    assert parameters.n_fft == expected
    assert expected > fft_length(*common, shift.min(), shift.max())
    parameters.sos_map = None
    parameters.sos_grid_x = None
    parameters.sos_grid_z = None
    assert parameters.n_fft == fft_length(*common, shift.min(), shift.max())
    parameters.sos_map = trio["sos_map"]
    parameters.sos_grid_x = trio["sos_grid_x"]
    parameters.sos_grid_z = trio["sos_grid_z"]

    pipeline = Pipeline([Simulate()], with_batch_dim=False, jit_options="pipeline")
    inputs = pipeline.prepare_parameters(parameters)
    assert inputs["n_fft"] == expected
    outputs = pipeline(
        **inputs,
        scatterer_positions=kwargs["scatterer_positions"],
        scatterer_magnitudes=kwargs["scatterer_magnitudes"],
        scatter_exponent=0.0,
    )
    _assert_close(simulate_rf(**_tensors(kwargs)), outputs["data"])


def test_record_helpers_gate_through_the_map():
    """The gate of a slow slab keeps fewer scatterers than the homogeneous one, and it is the
    simulator's: kept scatterers give the record, dropped ones give zeros."""
    kwargs, args, trio = _slab_scene(n_el=16)
    rng = np.random.default_rng(5)
    reach = record_reach(**args, sos_map=trio["sos_map"])
    positions = np.stack(
        [rng.uniform(-0.01, 0.01, 300), np.zeros(300), rng.uniform(0.5, 1.5, 300) * reach], -1
    ).astype(np.float32)
    magnitudes = rng.uniform(0.5, 1.0, 300).astype(np.float32)
    kwargs.update(scatterer_positions=positions, scatterer_magnitudes=magnitudes)
    mask = _np(in_record(positions, **args, **trio))
    homogeneous = _np(in_record(positions, **args))
    assert 0 < mask.sum() < homogeneous.sum()
    assert (positions[mask, 2] <= reach).all()

    reference = simulate_rf(**_tensors(kwargs))
    kept = {
        **kwargs,
        "scatterer_positions": positions[mask],
        "scatterer_magnitudes": magnitudes[mask],
    }
    dropped = {
        **kwargs,
        "scatterer_positions": positions[~mask],
        "scatterer_magnitudes": magnitudes[~mask],
    }
    _assert_close(reference, simulate_rf(**_tensors(kept)), rel_tol=1e-4)
    assert not _np(simulate_rf(**_tensors(dropped))).any()


def test_pressure_field_follows_the_map():
    kwargs = _case(_linear_probe())
    names = (
        "probe_geometry",
        "sound_speed",
        "center_frequency",
        "sampling_frequency",
        "t0_delays",
        "initial_times",
        "element_width",
        "tx_apodizations",
        "t_peak",
    )
    transmit = _tensors({k: kwargs[k] for k in names})
    x, z = np.meshgrid(np.linspace(-0.01, 0.01, 11), np.linspace(0.005, 0.03, 12), indexing="ij")
    grid = np.stack([x, np.zeros_like(x), z], -1).astype(np.float32)
    reference = pressure_field(grid, **transmit, n_ax=N_AX)
    uniform = pressure_field(grid, **transmit, n_ax=N_AX, **_tensors(_map(SOUND_SPEED)))
    _assert_close(reference, uniform)
    layered = _layered_map(1540.0, 1450.0, 0.015)
    field = pressure_field(grid, **transmit, n_ax=N_AX, output="time", **_tensors(layered))
    assert _rel(pressure_field(grid, **transmit, n_ax=N_AX, output="time"), field) > 0.1


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax gradients")
def test_gradients_with_respect_to_the_map_and_the_positions():
    import jax
    import jax.numpy as jnp

    kwargs = _case(_linear_probe(n_el=8), n_fft=1024)
    trio = _layered_map(1540.0, 1480.0, 0.015, nx=21, nz=23)
    positions = jnp.asarray(kwargs.pop("scatterer_positions"))
    sos_map = jnp.asarray(trio.pop("sos_map"))
    kwargs.update(trio)
    w = jnp.asarray(np.random.default_rng(2).normal(size=(2, N_AX, 8, 1)), jnp.float32)

    def loss(p, m):
        return jnp.sum(w * simulate_rf(scatterer_positions=p, sos_map=m, **kwargs))

    grads = jax.grad(loss, argnums=(0, 1))(positions, sos_map)
    assert np.isfinite(_np(grads[1])).all() and _np(grads[1]).any()

    # Pixels the rays cross. A 4 m/s step is exact in float32, moves the phases well clear of
    # their float32 round-off, and is small against the curvature.
    rng = np.random.default_rng(3)
    for i, j in ((10, 10), (15, 12)):
        bump = jnp.zeros_like(sos_map).at[i, j].set(4.0)
        fd = (loss(positions, sos_map + bump) - loss(positions, sos_map - bump)) / 8.0
        assert abs(grads[1][i, j] / fd - 1) < 1e-2, (i, j)

    v = jnp.asarray(rng.normal(size=positions.shape), jnp.float32)
    eps = 1e-6
    fd = (loss(positions + eps * v, sos_map) - loss(positions - eps * v, sos_map)) / (2 * eps)
    assert abs(jnp.sum(grads[0] * v) / fd - 1) < 1e-2


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_a_traced_map_needs_n_fft():
    import jax

    kwargs = _tensors(_case(_linear_probe()))
    trio = _map(SOUND_SPEED)
    sos_map = trio.pop("sos_map")
    kwargs.update(_tensors(trio))
    with pytest.raises(ValueError, match="n_fft"):
        jax.jit(lambda m: simulate_rf(sos_map=m, **kwargs))(sos_map)
    reference = simulate_rf(sos_map=sos_map, **kwargs)
    jitted = jax.jit(
        lambda m: simulate_rf(sos_map=m, n_fft=int(_np(reference).shape[1]) * 2, **kwargs)
    )
    _assert_close(reference, jitted(sos_map))


def test_simulate_op_takes_a_map():
    kwargs = _case(_matrix_probe())
    planar = _layered_map(1540.0, 1450.0, 0.015)
    solid = _map(1480.0, y=(-0.003, 0.003), ny=7)
    op = Simulate(jit_compile=True, with_batch_dim=False)
    for trio in (planar, solid):
        reference = simulate_rf(**_tensors({**kwargs, **trio}))
        _assert_close(reference, op(**_tensors({**kwargs, **trio}))[op.output_key])
    # The same op without the map is the homogeneous medium again.
    _assert_close(simulate_rf(**_tensors(kwargs)), op(**_tensors(kwargs))[op.output_key])
    assert _rel(simulate_rf(**_tensors(kwargs)), reference) > 0.1

    # Batched clouds, the second one straddling the footprint edge.
    batched = dict(kwargs)
    shifted = kwargs["scatterer_positions"] + np.array([0.015, 0.0, 0.0], np.float32)
    batched["scatterer_positions"] = np.stack([kwargs["scatterer_positions"], shifted])
    batched["scatterer_magnitudes"] = np.stack([kwargs["scatterer_magnitudes"]] * 2)
    op = Simulate(jit_compile=True, with_batch_dim=True)
    result = _np(op(**_tensors({**batched, **planar}))[op.output_key])
    for i, positions in enumerate(batched["scatterer_positions"]):
        expected = simulate_rf(**_tensors({**kwargs, **planar, "scatterer_positions": positions}))
        _assert_close(expected, result[i])

    op = Simulate(jit_compile=False, with_batch_dim=False)
    with pytest.raises(ValueError, match="frequency-domain"):
        op(**_tensors({**kwargs, **planar}), method="time_domain")


def test_n_sos_ray_samples_converges():
    kwargs = _case(_linear_probe(), **_layered_map(1540.0, 1450.0, 0.015), n_fft=2048)
    fine = simulate_rf(**_tensors({**kwargs, "n_sos_ray_samples": 512}))
    coarse = _rel(fine, simulate_rf(**_tensors({**kwargs, "n_sos_ray_samples": 4})))
    default = _rel(fine, simulate_rf(**_tensors(kwargs)))
    assert default < coarse
    assert default < 0.05
