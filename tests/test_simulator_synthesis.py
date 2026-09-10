"""Bookkeeping of :func:`zea.simulator.simulate_rf` against theoretical truths.

The physics is checked against SIMUS in ``test_simulator_simus.py`` and against analytic
models in ``test_simulator_physics.py``. Here the frequency-domain synthesis itself is checked:
the record gate, the FFT length, the band limit, the blocking over frequencies and transmits,
linearity, gradients, and the ``n_fft`` plumbing of the op and of :class:`zea.Parameters`.
"""

import keras
import numpy as np
import pytest
from keras import ops

import zea
from zea.ops import Pipeline, Simulate
from zea.probes import create_curved_probe_geometry, curved_probe_normals
from zea.simulator import (
    fft_length,
    in_record,
    record_bounds,
    record_reach,
    simulate_rf,
    smooth_size,
)
from zea.simulator_time_domain import simulate_rf_td

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


def _scan(geometry, n_tx=4):
    rng = np.random.default_rng(1)
    focus = np.array(
        [[0.0, 0.0, 0.02], [0.005, 0.0, 0.025], [-0.004, 0.002, 0.03], [0.002, -0.003, 0.018]]
    )[:n_tx]
    dist = np.linalg.norm(focus[:, None] - geometry[None], axis=-1)
    t0 = ((dist.max(1, keepdims=True) - dist) / SOUND_SPEED).astype(np.float32)
    apod = rng.uniform(0.5, 1.0, (n_tx, len(geometry))).astype(np.float32)
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
        "tx_apodizations": apod,
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


def _tensors(kwargs):
    """Array arguments as backend tensors; ``simulate_rf`` mixes them with numpy otherwise."""
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


def _correlation(a, b):
    a, b = _np(a).ravel(), _np(b).ravel()
    return a @ b / np.sqrt((a @ a) * (b @ b))


def _pulse(t):
    """Unit-peak Hann-windowed tone of ``simulate_rf``, centred at t = 0."""
    width = N_PERIOD / CENTER_FREQUENCY
    window = np.where(np.abs(t) < width / 2, np.cos(np.pi * t / width) ** 2, 0.0)
    return window * np.cos(2 * np.pi * CENTER_FREQUENCY * t)


def _single_element(**overrides):
    """One element at the origin, one unit scatterer, no attenuation or scattering gain."""
    kwargs = _case(np.zeros((1, 3), np.float32), **_scan(np.zeros((1, 3), np.float32), n_tx=1))
    kwargs.update(
        scatterer_magnitudes=np.ones(1, np.float32),
        attenuation_coef=0.0,
        scatter_exponent=0.0,
        tx_apodizations=np.ones((1, 1), np.float32),
        **overrides,
    )
    return kwargs


CASES = {
    "linear": _case(_linear_probe()),
    "matrix": _case(_matrix_probe()),
    "scatter_exponent_0": _case(_matrix_probe(), scatter_exponent=0.0),
    "soft_baffle": _case(_matrix_probe(), rigid_baffle=False),
    "transducer_bandwidth": _case(
        _linear_probe(), bandwidth_percent=60.0, probe_center_frequency=2.5e6
    ),
    "transducer_bandwidth_at_pulse_frequency": _case(_linear_probe(), bandwidth_percent=80.0),
    "chirp": _case(_linear_probe(), chirp_sweep=1.5e6, n_period=10.0),
    "chirp_with_bandwidth": _case(_matrix_probe(), chirp_sweep=1e6, bandwidth_percent=70.0),
    "convex_element_normals": _case(
        create_curved_probe_geometry(16, 0.3e-3, 15e-3),
        element_normals=curved_probe_normals(create_curved_probe_geometry(16, 0.3e-3, 15e-3)),
        rigid_baffle=False,
    ),
    "element_height": _case(_matrix_probe(), element_height=0.6e-3),
    "sub_elements": _case(_linear_probe(), n_sub_elements=(2, 3), element_height=2e-3),
    "auto_sub_elements_with_bandwidth": _case(
        _linear_probe(), n_sub_elements="auto", element_height=2e-3, bandwidth_percent=80.0
    ),
    "elevation_focus": _case(
        _linear_probe(), element_height=4e-3, elevation_focus=20e-3, apply_lens_correction=True
    ),
    "elevation_focus_convex": _case(
        create_curved_probe_geometry(16, 0.3e-3, 15e-3),
        element_normals=curved_probe_normals(create_curved_probe_geometry(16, 0.3e-3, 15e-3)),
        element_height=4e-3,
        elevation_focus=25e-3,
    ),
    "element_width_from_pitch": _case(_linear_probe(), element_width=None),
    "initial_times_and_t_peak": _case(
        _linear_probe(),
        initial_times=np.array([2e-6, -1e-6, 0.0, 1e-6], np.float32),
        t_peak=np.array([1, 2, 0.5, 1.5], np.float32) / CENTER_FREQUENCY,
    ),
    "lens_correction": _case(_linear_probe(), apply_lens_correction=True),
    "two_dimensional": _case(_linear_probe(), two_dimensional=True, element_height=1e-3),
    "noise_and_tgc": _case(
        _linear_probe(), noise_level_db=-40.0, tgc_max_db=20.0, noise_seed=3, noise_reference=1.0
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_every_feature_is_invariant_to_fft_length_and_frequency_blocks(name):
    """The FFT length and the block size are bookkeeping: any large enough length and any
    number of blocks give the same record."""
    kwargs = _tensors(CASES[name])
    reference = simulate_rf(**kwargs)
    assert np.isfinite(_np(reference)).all() and np.abs(_np(reference)).max() > 0
    _assert_close(reference, simulate_rf(**kwargs, n_fft=2048, max_chunk_gb=1e-4), rel_tol=1e-4)


def test_band_limit_drops_only_the_spectral_floor():
    kwargs = _tensors(CASES["transducer_bandwidth"])
    full = simulate_rf(**kwargs, band_db=None)
    _assert_close(full, simulate_rf(**kwargs), rel_tol=1e-4)
    assert _correlation(full, simulate_rf(**kwargs, band_db=-40.0)) > 0.999


def test_transmit_groups_and_subsets_give_the_same_rows():
    """More transmits than one irfft group; a subset of transmits is a subset of the rows."""
    n_tx, n_el, picks = 35, 8, [0, 32, 34]
    rng = np.random.default_rng(4)
    positions, magnitudes = _phantom(8, seed=5)
    kwargs = dict(CASES["linear"])
    kwargs.update(
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        probe_geometry=_linear_probe(n_el),
        t0_delays=rng.uniform(0, 2e-6, (n_tx, n_el)).astype(np.float32),
        initial_times=np.zeros(n_tx, np.float32),
        tx_apodizations=rng.uniform(0.5, 1.0, (n_tx, n_el)).astype(np.float32),
        t_peak=np.zeros(n_tx, np.float32),
    )
    per_tx = ("t0_delays", "tx_apodizations", "initial_times", "t_peak")
    subset = {**kwargs, **{key: kwargs[key][picks] for key in per_tx}}
    whole = simulate_rf(**_tensors(kwargs), n_fft=1024)
    assert _np(whole).shape[0] == n_tx
    _assert_close(simulate_rf(**_tensors(subset), n_fft=1024), _np(whole)[picks], rel_tol=1e-5)


def test_rf_is_a_superposition_of_the_scatterer_echoes():
    kwargs = dict(CASES["matrix"])
    positions, magnitudes = kwargs.pop("scatterer_positions"), kwargs.pop("scatterer_magnitudes")
    kwargs = _tensors(kwargs)
    half = len(positions) // 2

    def rf(pos, mag):
        return _np(simulate_rf(pos, mag, **kwargs, n_fft=1024))

    whole = rf(positions, magnitudes)
    _assert_close(
        whole,
        rf(positions[:half], magnitudes[:half]) + rf(positions[half:], magnitudes[half:]),
        1e-5,
    )
    _assert_close(2 * whole, rf(positions, 2 * magnitudes), 1e-6)


def test_rf_is_linear_in_the_transmit_apodization():
    kwargs = dict(CASES["linear"])
    apod = kwargs.pop("tx_apodizations")
    other = np.random.default_rng(6).uniform(-1, 1, apod.shape).astype(np.float32)
    kwargs = _tensors(kwargs)

    def rf(a):
        return _np(simulate_rf(**kwargs, tx_apodizations=a, n_fft=1024))

    _assert_close(rf(apod) + rf(other), rf(apod + other), 1e-5)


def test_record_prefix_does_not_depend_on_the_record_length():
    """Scatterers whose echo starts past a short record leave nothing in it, and those inside
    are not cut by the FFT length sized for that record."""
    kwargs = _tensors(CASES["lens_correction"])
    reach = (256 / SAMPLING_FREQUENCY + 0.5 * N_PERIOD / CENTER_FREQUENCY) * SOUND_SPEED / 2
    depths = np.linalg.norm(_np(kwargs["scatterer_positions"]), axis=1)
    assert depths.min() < reach < depths.max()
    long = _np(simulate_rf(**{**kwargs, "n_ax": 1024}))[:, :256]
    _assert_close(long, simulate_rf(**{**kwargs, "n_ax": 256}))


def _record_args(kwargs):
    """The arguments of the record helpers, out of a simulator call."""
    names = (
        "probe_geometry",
        "sound_speed",
        "n_ax",
        "sampling_frequency",
        "center_frequency",
        "t0_delays",
        "initial_times",
        "t_peak",
        "apply_lens_correction",
        "lens_thickness",
        "lens_sound_speed",
    )
    return {k: kwargs[k] for k in names if k in kwargs}


def test_gate_keeps_a_scatterer_inside_the_record_and_drops_one_past_it():
    kwargs = _single_element()
    reach = record_reach(**_record_args(kwargs))
    assert (
        abs(
            reach
            / ((N_AX / SAMPLING_FREQUENCY + 0.5 * N_PERIOD / CENTER_FREQUENCY) * SOUND_SPEED / 2)
            - 1
        )
        < 1e-12
    )
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, 0.98 * reach]], np.float32)
    inside = _np(simulate_rf(**_tensors(kwargs)))
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, 1.02 * reach]], np.float32)
    outside = _np(simulate_rf(**_tensors(kwargs)))
    peak = np.abs(inside).max()
    assert peak > 0
    # The pulse straddles the end of the record: energy in the last samples only.
    assert np.abs(inside[0, : N_AX // 2]).max() < 1e-4 * peak
    assert not outside.any()


def test_record_helpers_agree_with_the_gate():
    """``in_record`` is the gate: the record of a cloud is that of its kept scatterers, the
    dropped ones give zeros. ``record_bounds`` holds every kept scatterer, and the reach is a
    scatterer's distance from its nearest element."""
    rng = np.random.default_rng(2)
    kwargs = _case(_linear_probe(), lens_sound_speed=1000.0, apply_lens_correction=True)
    positions = rng.uniform([-0.05, -0.01, 0.0], [0.05, 0.01, 0.06], (300, 3)).astype(np.float32)
    magnitudes = rng.uniform(0.5, 1.0, len(positions)).astype(np.float32)
    kwargs.update(scatterer_positions=positions, scatterer_magnitudes=magnitudes)
    args = _record_args(kwargs)
    mask = _np(in_record(positions, **args))
    assert 0 < mask.sum() < len(mask)

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

    low, high = record_bounds(**args)
    assert (positions[mask] >= low).all() and (positions[mask] <= high).all()
    # A scatterer straight below an element is kept up to the reach and dropped past it.
    reach = record_reach(**args)
    probe = [kwargs["probe_geometry"][3]]
    on_axis = np.array([probe[0] + [0.0, 0.0, 0.99 * reach], probe[0] + [0.0, 0.0, 1.01 * reach]])
    assert _np(in_record(on_axis.astype(np.float32), **args)).tolist() == [True, False]

    # 2D collapses the box onto the plane and gates the projected scatterers.
    low_2d, high_2d = record_bounds(**args, two_dimensional=True)
    assert low_2d[1] == high_2d[1] == 0.0
    projected = positions * [1.0, 0.0, 1.0]
    mask_2d = _np(in_record(positions, **args, two_dimensional=True))
    assert (mask_2d == _np(in_record(projected, **args))).all()


def test_single_element_echo_is_the_delayed_and_spread_pulse():
    r = 0.6 * N_AX / SAMPLING_FREQUENCY * SOUND_SPEED / 2
    kwargs = _single_element()
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, r]], np.float32)
    rf = _np(simulate_rf(**_tensors(kwargs)))[0, :, 0, 0]
    t = np.arange(N_AX) / SAMPLING_FREQUENCY
    expected = _pulse(t - 2 * r / SOUND_SPEED) * (1e-3 / r) ** 2
    # The synthesis is band limited to the rfft grid, the analytic pulse is not.
    _assert_close(expected, rf, rel_tol=5e-3)


def test_t_peak_and_initial_times_shift_the_echo():
    r = 0.4 * N_AX / SAMPLING_FREQUENCY * SOUND_SPEED / 2
    shift = 40 / SAMPLING_FREQUENCY
    kwargs = _single_element()
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, r]], np.float32)
    plain = _np(simulate_rf(**_tensors(kwargs)))[0, :, 0, 0]
    late = _np(simulate_rf(**_tensors({**kwargs, "t_peak": np.full(1, shift, np.float32)})))
    early = _np(simulate_rf(**_tensors({**kwargs, "initial_times": np.full(1, shift, np.float32)})))
    _assert_close(plain[:-40], late[0, 40:, 0, 0])
    _assert_close(plain[40:], early[0, :-40, 0, 0])


def test_chirp_is_sampled_on_an_odd_fft_grid():
    # smooth_size lands on an odd length for some records; the chirp follows the same grid.
    assert smooth_size(1082) == 1125
    kwargs = _tensors(CASES["chirp"])
    _assert_close(simulate_rf(**kwargs, n_fft=1024), simulate_rf(**kwargs, n_fft=1125))


def test_empty_phantom_gives_zeros():
    kwargs = dict(CASES["linear"])
    kwargs["scatterer_positions"] = np.zeros((0, 3), np.float32)
    kwargs["scatterer_magnitudes"] = np.zeros(0, np.float32)
    result = _np(simulate_rf(**_tensors(kwargs)))
    assert result.shape == (4, N_AX, 16, 1)
    assert not result.any()


def test_n_period_changes_the_pulse():
    kwargs = _tensors(CASES["linear"])
    short = simulate_rf(**kwargs, n_period=2.0)
    default = simulate_rf(**kwargs)
    assert short.shape == default.shape
    assert _correlation(short, default) < 0.99


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_under_jit_with_traced_geometry_needs_n_fft():
    import jax

    kwargs = CASES["lens_correction"]
    static = (
        "apply_lens_correction",
        "n_ax",
        "center_frequency",
        "sampling_frequency",
        "scatter_exponent",
        "n_fft",
    )
    jitted = jax.jit(simulate_rf, static_argnames=static)
    with pytest.raises(ValueError, match="n_fft"):
        jitted(**kwargs)
    _assert_close(simulate_rf(**kwargs), jitted(**kwargs, n_fft=1024), rel_tol=1e-4)


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_under_jit_closed_over_geometry_derives_n_fft():
    import jax

    kwargs = dict(CASES["linear"])
    positions = kwargs.pop("scatterer_positions")
    magnitudes = kwargs.pop("scatterer_magnitudes")
    reference = simulate_rf(positions, magnitudes, **kwargs)
    result = jax.jit(lambda p, m: simulate_rf(p, m, **kwargs))(positions, magnitudes)
    _assert_close(reference, result, rel_tol=1e-4)


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="uses jax.grad")
def test_gradients_match_finite_differences():
    import jax
    import jax.numpy as jnp

    kwargs = dict(CASES["lens_correction"])
    positions = jnp.asarray(kwargs.pop("scatterer_positions"))
    magnitudes = jnp.asarray(kwargs.pop("scatterer_magnitudes"))
    kwargs.pop("sound_speed")
    w = jnp.asarray(np.random.default_rng(2).normal(size=(4, N_AX, 16, 1)), jnp.float32)

    def loss(p, m, c):
        return jnp.sum(w * simulate_rf(p, m, sound_speed=c, **kwargs, n_fft=1024))

    grads = jax.grad(loss, argnums=(0, 1, 2))(positions, magnitudes, 1540.0)
    # Linear in the magnitudes, so the gradient reproduces the loss.
    assert abs(jnp.dot(grads[1], magnitudes) / loss(positions, magnitudes, 1540.0) - 1) < 1e-4

    rng = np.random.default_rng(3)
    v = jnp.asarray(rng.normal(size=positions.shape), jnp.float32)
    eps = 1e-6
    fd = (
        loss(positions + eps * v, magnitudes, 1540.0)
        - loss(positions - eps * v, magnitudes, 1540.0)
    ) / (2 * eps)
    assert abs(jnp.sum(grads[0] * v) / fd - 1) < 1e-2

    # Exact in float32 (spacing 1.2e-4 at 1540 m/s) and small against the curvature in c.
    eps = 0.125
    fd = (loss(positions, magnitudes, 1540.0 + eps) - loss(positions, magnitudes, 1540.0 - eps)) / (
        2 * eps
    )
    assert abs(grads[2] / fd - 1) < 1e-2


def test_simulate_op_derives_n_fft_for_its_jitted_call():
    kwargs = _tensors(CASES["linear"])
    reference = simulate_rf(**kwargs)
    op = Simulate(jit_compile=True, with_batch_dim=False)
    _assert_close(reference, op(**kwargs)[op.output_key], rel_tol=1e-3)

    batched = dict(CASES["linear"])
    for key in ("scatterer_positions", "scatterer_magnitudes"):
        batched[key] = np.stack([batched[key], batched[key][::-1]])
    op = Simulate(jit_compile=True, with_batch_dim=True)
    result = _np(op(**_tensors(batched))[op.output_key])
    _assert_close(reference, result[0], rel_tol=1e-3)
    _assert_close(reference, result[1], rel_tol=1e-3)


def test_simulate_op_methods_and_deprecated_aliases():
    kwargs = _tensors(CASES["linear"])
    op = Simulate(jit_compile=False, with_batch_dim=False)
    frequency = op(**kwargs, method="frequency_domain")[op.output_key]
    _assert_close(frequency, op(**kwargs, method="exact")[op.output_key], rel_tol=1e-6)
    _assert_close(
        frequency, op(**kwargs, method="frequency_approximation")[op.output_key], rel_tol=1e-6
    )
    time = op(**kwargs, method="time_domain")[op.output_key]
    _assert_close(simulate_rf_td(**kwargs), time, rel_tol=1e-4)
    _assert_close(time, op(**kwargs, method="time_approximation")[op.output_key], rel_tol=1e-6)
    with pytest.raises(ValueError, match="method"):
        op(**kwargs, method="exact_slab")


def test_parameters_derive_n_fft_for_a_jitted_pipeline():
    """A whole-pipeline jit skips the op's eager derivation, so ``n_fft`` comes from the
    parameters."""
    scan = _scan(_linear_probe())
    parameters = zea.Parameters(
        n_tx=4,
        n_el=16,
        n_ax=N_AX,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
        probe_geometry=scan["probe_geometry"],
        t0_delays=scan["t0_delays"],
        initial_times=scan["initial_times"],
        t_peak=scan["t_peak"],
        tx_apodizations=scan["tx_apodizations"],
        sound_speed=SOUND_SPEED,
        selected_transmits="all",
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        element_width=0.27e-3,
        attenuation_coef=0.5,
    )
    shift = scan["t0_delays"]
    expected = fft_length(
        N_AX,
        SAMPLING_FREQUENCY,
        CENTER_FREQUENCY,
        SOUND_SPEED,
        scan["probe_geometry"],
        shift.min(),
        shift.max(),
    )
    assert parameters.n_fft == expected
    parameters.n_fft = 1024
    assert parameters.n_fft == 1024
    parameters.n_fft = None

    pipeline = Pipeline([Simulate()], with_batch_dim=False, jit_options="pipeline")
    inputs = pipeline.prepare_parameters(parameters)
    assert inputs["n_fft"] == expected
    positions, magnitudes = _phantom()
    outputs = pipeline(
        **inputs,
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        scatter_exponent=1.5,
    )
    kwargs = _tensors(CASES["linear"])
    # Tensorflow on GPU rounds to TF32 once torch is imported, as the test workers do.
    _assert_close(simulate_rf(**kwargs), outputs["data"], rel_tol=1e-3)


def test_smooth_size_and_fft_length():
    assert [smooth_size(n) for n in (1, 7, 100, 601, 1025)] == [1, 8, 100, 625, 1080]
    geometry = _linear_probe()
    n_fft = fft_length(N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, geometry, 0, 0)
    assert n_fft >= N_AX
    assert n_fft == smooth_size(n_fft)
    # The farthest scatterer bounds the length below the aperture bound when it is closer.
    near = np.array([[0.0, 0.0, 5e-3]])
    assert (
        fft_length(
            N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, geometry, 0, 0, 4.0, near
        )
        <= n_fft
    )
