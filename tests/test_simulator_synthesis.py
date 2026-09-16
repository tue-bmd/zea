"""Bookkeeping of :func:`zea.simulator.simulate_rf` against theoretical truths.

The physics is checked against SIMUS in ``test_simulator_simus.py`` and against analytic
models in ``test_simulator_physics.py``. Here the frequency-domain synthesis itself is checked:
the record gate, the FFT length, the band limit, the blocking over frequencies and transmits,
linearity, gradients, and the ``n_fft`` plumbing of the op and of :class:`zea.Parameters`.
"""

import logging

import keras
import numpy as np
import pytest

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
    transmit_pulse,
    transmit_pulses,
)
from zea.simulator_time_domain import simulate_rf_td

from .simulator_helpers import (
    CENTER_FREQUENCY,
    N_AX,
    SAMPLING_FREQUENCY,
    SOUND_SPEED,
    assert_close,
    case,
    correlation,
    hann_tone,
    hann_waveform,
    linear_probe,
    matrix_probe,
    phantom,
    scan,
    stack_padded,
    tensors,
    to_np,
)

# Support of the default pulse after its peak [s]: how far past the record an echo peak may be.
PULSE_TAIL = transmit_pulse(CENTER_FREQUENCY, SAMPLING_FREQUENCY).n_after / SAMPLING_FREQUENCY


def _single_element(**overrides):
    """One element at the origin, one unit scatterer, no attenuation or scattering gain."""
    kwargs = case(np.zeros((1, 3), np.float32), **scan(np.zeros((1, 3), np.float32), n_tx=1))
    kwargs.update(
        scatterer_magnitudes=np.ones(1, np.float32),
        attenuation_coef=0.0,
        scatter_exponent=0.0,
        tx_apodizations=np.ones((1, 1), np.float32),
        **overrides,
    )
    return kwargs


CASES = {
    "linear": case(linear_probe()),
    "matrix": case(matrix_probe()),
    "scatter_exponent_0": case(matrix_probe(), scatter_exponent=0.0),
    "soft_baffle": case(matrix_probe(), baffle_impedance_ratio=float("inf")),
    "transducer_bandwidth": case(
        linear_probe(),
        waveforms_two_way=hann_waveform(bandwidth_percent=60.0, probe_center_frequency=2.5e6),
    ),
    "transducer_bandwidth_at_pulse_frequency": case(
        linear_probe(), waveforms_two_way=hann_waveform(bandwidth_percent=80.0)
    ),
    "chirp": case(linear_probe(), waveforms_two_way=hann_waveform(10.0, chirp_sweep=1.5e6)),
    "chirp_with_bandwidth": case(
        matrix_probe(), waveforms_two_way=hann_waveform(chirp_sweep=1e6, bandwidth_percent=70.0)
    ),
    "per_transmit_waveforms": case(
        linear_probe(),
        waveforms_two_way=stack_padded(
            hann_waveform(),
            hann_waveform(2.0),
            hann_waveform(6.0, chirp_sweep=1e6),
            transmit_pulse(CENTER_FREQUENCY).waveform(),
        ),
    ),
    "convex_element_normals": case(
        create_curved_probe_geometry(16, 0.3e-3, 15e-3),
        element_normals=curved_probe_normals(create_curved_probe_geometry(16, 0.3e-3, 15e-3)),
        baffle_impedance_ratio=float("inf"),
    ),
    "element_height": case(matrix_probe(), element_height=0.6e-3),
    "sub_elements": case(linear_probe(), n_sub_elements=(2, 3), element_height=2e-3),
    "auto_sub_elements_with_bandwidth": case(
        linear_probe(),
        n_sub_elements="auto",
        element_height=2e-3,
        waveforms_two_way=hann_waveform(bandwidth_percent=80.0),
    ),
    "elevation_focus": case(
        linear_probe(), element_height=4e-3, elevation_focus=20e-3, apply_lens_correction=True
    ),
    "elevation_focus_convex": case(
        create_curved_probe_geometry(16, 0.3e-3, 15e-3),
        element_normals=curved_probe_normals(create_curved_probe_geometry(16, 0.3e-3, 15e-3)),
        element_height=4e-3,
        elevation_focus=25e-3,
    ),
    "element_width_from_pitch": case(linear_probe(), element_width=None),
    "initial_times_and_t_peak": case(
        linear_probe(),
        initial_times=np.array([2e-6, -1e-6, 0.0, 1e-6], np.float32),
        t_peak=np.array([1, 2, 0.5, 1.5], np.float32) / CENTER_FREQUENCY,
    ),
    "lens_correction": case(linear_probe(), apply_lens_correction=True),
    "two_dimensional": case(linear_probe(), two_dimensional=True, element_height=1e-3),
    "noise_and_tgc": case(
        linear_probe(), noise_level_db=-40.0, tgc_max_db=20.0, noise_seed=3, noise_reference=1.0
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_every_feature_is_invariant_to_fft_length_and_frequency_blocks(name):
    """The FFT length and the block size are bookkeeping: any large enough length and any
    number of blocks give the same record."""
    kwargs = tensors(CASES[name])
    reference = simulate_rf(**kwargs)
    assert np.isfinite(to_np(reference)).all() and np.abs(to_np(reference)).max() > 0
    assert_close(reference, simulate_rf(**kwargs, n_fft=2048, max_chunk_gb=1e-4), rel_tol=1e-4)


def test_band_limit_drops_only_the_spectral_floor():
    kwargs = tensors(CASES["transducer_bandwidth"])
    full = simulate_rf(**kwargs, band_db=None)
    assert_close(full, simulate_rf(**kwargs), rel_tol=1e-4)
    assert correlation(full, simulate_rf(**kwargs, band_db=-40.0)) > 0.999


def test_transmit_groups_and_subsets_give_the_same_rows():
    """More transmits than one irfft group; a subset of transmits is a subset of the rows."""
    n_tx, n_el, picks = 35, 8, [0, 32, 34]
    rng = np.random.default_rng(4)
    positions, magnitudes = phantom(8, seed=5)
    kwargs = dict(CASES["linear"])
    kwargs.update(
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        probe_geometry=linear_probe(n_el),
        t0_delays=rng.uniform(0, 2e-6, (n_tx, n_el)).astype(np.float32),
        initial_times=np.zeros(n_tx, np.float32),
        tx_apodizations=rng.uniform(0.5, 1.0, (n_tx, n_el)).astype(np.float32),
        t_peak=np.zeros(n_tx, np.float32),
    )
    per_tx = ("t0_delays", "tx_apodizations", "initial_times", "t_peak")
    subset = {**kwargs, **{key: kwargs[key][picks] for key in per_tx}}
    whole = simulate_rf(**tensors(kwargs), n_fft=1024)
    assert to_np(whole).shape[0] == n_tx
    assert_close(simulate_rf(**tensors(subset), n_fft=1024), to_np(whole)[picks], rel_tol=1e-5)


def test_rf_is_a_superposition_of_the_scatterer_echoes():
    kwargs = dict(CASES["matrix"])
    positions, magnitudes = kwargs.pop("scatterer_positions"), kwargs.pop("scatterer_magnitudes")
    kwargs = tensors(kwargs)
    half = len(positions) // 2

    def rf(pos, mag):
        return to_np(simulate_rf(pos, mag, **kwargs, n_fft=1024))

    whole = rf(positions, magnitudes)
    assert_close(
        whole,
        rf(positions[:half], magnitudes[:half]) + rf(positions[half:], magnitudes[half:]),
        1e-5,
    )
    assert_close(2 * whole, rf(positions, 2 * magnitudes), 1e-6)


def test_rf_is_linear_in_the_transmit_apodization():
    kwargs = dict(CASES["linear"])
    apod = kwargs.pop("tx_apodizations")
    other = np.random.default_rng(6).uniform(-1, 1, apod.shape).astype(np.float32)
    kwargs = tensors(kwargs)

    def rf(a):
        return to_np(simulate_rf(**kwargs, tx_apodizations=a, n_fft=1024))

    assert_close(rf(apod) + rf(other), rf(apod + other), 1e-5)


def test_record_prefix_does_not_depend_on_the_record_length():
    """Scatterers whose echo starts past a short record leave nothing in it, and those inside
    are not cut by the FFT length sized for that record."""
    kwargs = tensors(CASES["lens_correction"])
    reach = (256 / SAMPLING_FREQUENCY + PULSE_TAIL) * SOUND_SPEED / 2
    depths = np.linalg.norm(to_np(kwargs["scatterer_positions"]), axis=1)
    assert depths.min() < reach < depths.max()
    long = to_np(simulate_rf(**{**kwargs, "n_ax": 1024}))[:, :256]
    assert_close(long, simulate_rf(**{**kwargs, "n_ax": 256}))


def _record_args(kwargs, geometry=True):
    """The arguments of the record helpers, out of a simulator call; ``record_reach`` takes
    them without the geometry."""
    names = (
        *(("probe_geometry",) if geometry else ()),
        "sound_speed",
        "n_ax",
        "sampling_frequency",
        "center_frequency",
        "t0_delays",
        "initial_times",
        "t_peak",
        "waveforms_two_way",
        "waveform_sampling_frequency",
        "apply_lens_correction",
        "lens_thickness",
        "lens_sound_speed",
    )
    return {k: kwargs[k] for k in names if k in kwargs}


def test_gate_keeps_a_scatterer_inside_the_record_and_drops_one_past_it():
    reach = record_reach(**_record_args(_single_element(), geometry=False))
    assert abs(reach / ((N_AX / SAMPLING_FREQUENCY + PULSE_TAIL) * SOUND_SPEED / 2) - 1) < 1e-12
    # A Hann tone has a compact support, so the reach puts its peak just inside the record.
    waveform = hann_waveform()
    tail = transmit_pulses(1, CENTER_FREQUENCY, SAMPLING_FREQUENCY, waveform)[0].n_after
    kwargs = _single_element(waveforms_two_way=waveform)
    reach = record_reach(**_record_args(kwargs, geometry=False))
    expected = (N_AX / SAMPLING_FREQUENCY + tail / SAMPLING_FREQUENCY) * SOUND_SPEED / 2
    assert abs(reach / expected - 1) < 1e-12
    # An echo peaking a few samples before the end of the record straddles it: energy in the
    # last samples only.
    kwargs["scatterer_positions"] = np.array(
        [[0.0, 0.0, (N_AX - 4) / SAMPLING_FREQUENCY * SOUND_SPEED / 2]], np.float32
    )
    inside = to_np(simulate_rf(**tensors(kwargs)))
    peak = np.abs(inside).max()
    assert peak > 0
    assert np.abs(inside[0, : N_AX // 2]).max() < 1e-4 * peak
    # The gate: kept up to the reach, where only the foot of the pulse is left, dropped past it.
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, 0.99 * reach]], np.float32)
    assert to_np(simulate_rf(**tensors(kwargs))).any()
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, 1.01 * reach]], np.float32)
    assert not to_np(simulate_rf(**tensors(kwargs))).any()


def test_record_helpers_agree_with_the_gate():
    """``in_record`` is the gate: the record of a cloud is that of its kept scatterers, the
    dropped ones give zeros. ``record_bounds`` holds every kept scatterer, and the reach is a
    scatterer's distance from its nearest element."""
    rng = np.random.default_rng(2)
    kwargs = case(linear_probe(), lens_sound_speed=1000.0, apply_lens_correction=True)
    positions = rng.uniform([-0.05, -0.01, 0.0], [0.05, 0.01, 0.06], (300, 3)).astype(np.float32)
    magnitudes = rng.uniform(0.5, 1.0, len(positions)).astype(np.float32)
    kwargs.update(scatterer_positions=positions, scatterer_magnitudes=magnitudes)
    args = _record_args(kwargs)
    mask = to_np(in_record(positions, **args))
    assert 0 < mask.sum() < len(mask)

    reference = simulate_rf(**tensors(kwargs))
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
    assert_close(reference, simulate_rf(**tensors(kept)), rel_tol=1e-4)
    assert not to_np(simulate_rf(**tensors(dropped))).any()

    low, high = record_bounds(**args)
    assert (positions[mask] >= low).all() and (positions[mask] <= high).all()
    # A scatterer straight below an element is kept up to the reach and dropped past it.
    reach = record_reach(**_record_args(kwargs, geometry=False))
    probe = [kwargs["probe_geometry"][3]]
    on_axis = np.array([probe[0] + [0.0, 0.0, 0.99 * reach], probe[0] + [0.0, 0.0, 1.01 * reach]])
    assert to_np(in_record(on_axis.astype(np.float32), **args)).tolist() == [True, False]

    # 2D collapses the box onto the plane and gates the projected scatterers.
    low_2d, high_2d = record_bounds(**args, two_dimensional=True)
    assert low_2d[1] == high_2d[1] == 0.0
    projected = positions * [1.0, 0.0, 1.0]
    mask_2d = to_np(in_record(positions, **args, two_dimensional=True))
    assert (mask_2d == to_np(in_record(projected, **args))).all()


def test_single_element_echo_is_the_delayed_and_spreadhann_tone():
    r = 0.6 * N_AX / SAMPLING_FREQUENCY * SOUND_SPEED / 2
    kwargs = _single_element(waveforms_two_way=hann_waveform())
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, r]], np.float32)
    rf = to_np(simulate_rf(**tensors(kwargs)))[0, :, 0, 0]
    t = np.arange(N_AX) / SAMPLING_FREQUENCY
    expected = hann_tone(t - 2 * r / SOUND_SPEED) * (1e-3 / r) ** 2
    # The synthesis is band limited to the rfft grid, the analytic pulse is not.
    assert_close(expected, rf, rel_tol=5e-3)


def test_t_peak_and_initial_times_shift_the_echo():
    r = 0.4 * N_AX / SAMPLING_FREQUENCY * SOUND_SPEED / 2
    shift = 40 / SAMPLING_FREQUENCY
    kwargs = _single_element()
    kwargs["scatterer_positions"] = np.array([[0.0, 0.0, r]], np.float32)
    plain = to_np(simulate_rf(**tensors(kwargs)))[0, :, 0, 0]
    late = to_np(simulate_rf(**tensors({**kwargs, "t_peak": np.full(1, shift, np.float32)})))
    early = to_np(
        simulate_rf(**tensors({**kwargs, "initial_times": np.full(1, shift, np.float32)}))
    )
    assert_close(plain[:-40], late[0, 40:, 0, 0])
    assert_close(plain[40:], early[0, :-40, 0, 0])


def test_chirp_is_sampled_on_an_odd_fft_grid():
    # smooth_size lands on an odd length for some records; the pulse follows the same grid.
    assert smooth_size(1082) == 1125
    kwargs = tensors(CASES["chirp"])
    assert_close(simulate_rf(**kwargs, n_fft=1024), simulate_rf(**kwargs, n_fft=1125))


def test_empty_phantom_gives_zeros():
    kwargs = dict(CASES["linear"])
    kwargs["scatterer_positions"] = np.zeros((0, 3), np.float32)
    kwargs["scatterer_magnitudes"] = np.zeros(0, np.float32)
    result = to_np(simulate_rf(**tensors(kwargs)))
    assert result.shape == (4, N_AX, 16, 1)
    assert not result.any()


def test_waveforms_change_thehann_tone():
    kwargs = tensors(CASES["linear"])
    short = simulate_rf(**kwargs, waveforms_two_way=hann_waveform(2.0))
    default = simulate_rf(**kwargs)
    assert short.shape == default.shape
    assert correlation(short, default) < 0.99
    # The rows of a (n_tx, n_samples) array are the pulses of the transmits, in order.
    stacked = stack_padded(hann_waveform(2.0), hann_waveform(), hann_waveform(2.0), hann_waveform())
    mixed = to_np(simulate_rf(**kwargs, waveforms_two_way=stacked))
    assert_close(to_np(short)[0], mixed[0], 1e-4)
    assert_close(to_np(simulate_rf(**kwargs, waveforms_two_way=hann_waveform()))[1], mixed[1], 1e-4)
    with pytest.raises(ValueError, match="waveforms_two_way must have shape"):
        simulate_rf(**kwargs, waveforms_two_way=stacked[:3])


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
    assert_close(simulate_rf(**kwargs), jitted(**kwargs, n_fft=1024), rel_tol=1e-4)


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_under_jit_closed_over_geometry_derives_n_fft():
    import jax

    kwargs = dict(CASES["linear"])
    positions = kwargs.pop("scatterer_positions")
    magnitudes = kwargs.pop("scatterer_magnitudes")
    reference = simulate_rf(positions, magnitudes, **kwargs)
    result = jax.jit(lambda p, m: simulate_rf(p, m, **kwargs))(positions, magnitudes)
    assert_close(reference, result, rel_tol=1e-4)


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
    kwargs = tensors(CASES["linear"])
    reference = simulate_rf(**kwargs)
    op = Simulate(jit_compile=True, with_batch_dim=False)
    assert_close(reference, op(**kwargs)[op.output_key], rel_tol=1e-3)

    batched = dict(CASES["linear"])
    for key in ("scatterer_positions", "scatterer_magnitudes"):
        batched[key] = np.stack([batched[key], batched[key][::-1]])
    op = Simulate(jit_compile=True, with_batch_dim=True)
    result = to_np(op(**tensors(batched))[op.output_key])
    assert_close(reference, result[0], rel_tol=1e-3)
    assert_close(reference, result[1], rel_tol=1e-3)


def test_simulate_op_methods():
    """``time_domain`` reaches the time-domain simulator, and an unknown name is rejected.
    The default ``frequency_domain`` is checked against its function above."""
    kwargs = tensors(CASES["linear"])
    op = Simulate(jit_compile=False, with_batch_dim=False)
    assert_close(
        simulate_rf_td(**kwargs), op(**kwargs, method="time_domain")[op.output_key], rel_tol=1e-4
    )
    with pytest.raises(ValueError, match="method"):
        op(**kwargs, method="exact_slab")


def test_simulate_op_accepts_the_old_method_names(caplog, reset_warning_once):
    """The names from before the rename still select their simulator, with a deprecation
    warning naming the new one."""
    kwargs = tensors(CASES["linear"])
    op = Simulate(jit_compile=False, with_batch_dim=False)
    frequency_domain = op(**kwargs)[op.output_key]
    time_domain = op(**kwargs, method="time_domain")[op.output_key]
    aliases = {
        "exact": ("frequency_domain", frequency_domain),
        "frequency_approximation": ("frequency_domain", frequency_domain),
        "time_approximation": ("time_domain", time_domain),
    }
    for old, (new, reference) in aliases.items():
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="zea"):
            assert_close(reference, op(**kwargs, method=old)[op.output_key], rel_tol=1e-6)
        messages = [record.getMessage() for record in caplog.records]
        assert any(old in m and new in m and "deprecated" in m for m in messages), messages


def test_simulate_op_warns_about_options_the_time_domain_simulator_ignores(
    caplog, reset_warning_once
):
    """Element options that only the frequency-domain simulator models are named in a warning
    on the time-domain path; the +z normals of a flat probe and the defaults are not."""
    kwargs = tensors(CASES["linear"])
    n_el = kwargs["probe_geometry"].shape[0]
    op = Simulate(jit_compile=False, with_batch_dim=False)
    tilted = np.tile(np.array([0.5, 0.0, np.sqrt(0.75)], np.float32), (n_el, 1))
    with caplog.at_level(logging.WARNING, logger="zea"):
        op(**kwargs, method="time_domain", element_normals=tilted, elevation_focus=30e-3)
    messages = [record.getMessage() for record in caplog.records]
    assert any("element_normals" in m and "elevation_focus" in m for m in messages), messages

    caplog.clear()
    flat = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n_el, 1))
    with caplog.at_level(logging.WARNING, logger="zea"):
        op(**kwargs, method="time_domain", element_normals=flat, baffle_impedance_ratio=0.0)
    assert not any("ignores" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["+y", "-y"])
def test_element_normals_along_the_elevation_axis_are_rejected(sign):
    """The element frame projects +y onto the element plane, which vanishes for a normal along
    either direction of y (a probe with swapped y and z columns): an error, not NaN RF."""
    kwargs = tensors(CASES["linear"])
    n_el = kwargs["probe_geometry"].shape[0]
    normals = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (n_el, 1))
    normals[3] = [0.0, sign, 0.0]
    with pytest.raises(ValueError, match=r"element_normals of elements \[3\]"):
        simulate_rf(**kwargs, element_normals=normals)
    with pytest.raises(ValueError, match="zero length"):
        simulate_rf(**kwargs, element_normals=np.zeros((n_el, 3), np.float32))
    with pytest.raises(ValueError, match="shape"):
        simulate_rf(**kwargs, element_normals=normals[:, :2])


def test_scatter_exponent_accepts_a_python_list():
    """A list of one exponent per scatterer is the vector it converts to, so it takes the
    per-scatterer path and its length is checked, instead of failing as a float."""
    kwargs = tensors(CASES["linear"])
    n_scat = kwargs["scatterer_positions"].shape[0]
    exponent = kwargs.pop("scatter_exponent")
    reference = simulate_rf(**kwargs, scatter_exponent=np.full(n_scat, exponent, np.float32))
    assert_close(reference, simulate_rf(**kwargs, scatter_exponent=[exponent] * n_scat))
    with pytest.raises(ValueError, match="per scatterer"):
        simulate_rf(**kwargs, scatter_exponent=[exponent] * (n_scat + 1))


def test_parameters_derive_n_fft_for_a_jitted_pipeline():
    """A whole-pipeline jit skips the op's eager derivation, so ``n_fft`` comes from the
    parameters."""
    transmit = scan(linear_probe())
    parameters = zea.Parameters(
        n_tx=4,
        n_el=16,
        n_ax=N_AX,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=SAMPLING_FREQUENCY,
        probe_geometry=transmit["probe_geometry"],
        t0_delays=transmit["t0_delays"],
        initial_times=transmit["initial_times"],
        t_peak=transmit["t_peak"],
        tx_apodizations=transmit["tx_apodizations"],
        sound_speed=SOUND_SPEED,
        selected_transmits="all",
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        element_width=0.27e-3,
        attenuation_coef=0.5,
    )
    shift = transmit["t0_delays"]
    expected = fft_length(
        N_AX,
        SAMPLING_FREQUENCY,
        CENTER_FREQUENCY,
        SOUND_SPEED,
        transmit["probe_geometry"],
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
    positions, magnitudes = phantom()
    outputs = pipeline(
        **inputs,
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        scatter_exponent=1.5,
    )
    kwargs = tensors(CASES["linear"])
    # Tensorflow on GPU rounds to TF32 once torch is imported, as the test workers do.
    assert_close(simulate_rf(**kwargs), outputs["data"], rel_tol=1e-3)


def test_per_scatterer_scatter_exponent_matches_the_shared_one():
    """A constant vector of exponents is the scalar it repeats."""
    kwargs = CASES["matrix"]
    n_scat = kwargs["scatterer_positions"].shape[0]
    vector = np.full(n_scat, kwargs["scatter_exponent"], np.float32)
    assert_close(simulate_rf(**kwargs), simulate_rf(**{**kwargs, "scatter_exponent": vector}))


def test_per_scatterer_scatter_exponent_superposes():
    """Two exponents over disjoint halves of the phantom sum to the mixed-exponent frame,
    so each scatterer really carries its own backscatter coefficient."""
    kwargs = {**CASES["matrix"], "band_db": None}
    magnitudes = kwargs["scatterer_magnitudes"]
    first_half = np.arange(magnitudes.shape[0]) < magnitudes.shape[0] // 2
    mixed = simulate_rf(**{**kwargs, "scatter_exponent": np.where(first_half, 0.0, 2.0)})
    parts = [
        simulate_rf(
            **{
                **kwargs,
                "scatter_exponent": exponent,
                "scatterer_magnitudes": np.where(half, magnitudes, 0.0),
            }
        )
        for exponent, half in ((0.0, first_half), (2.0, ~first_half))
    ]
    assert_close(mixed, parts[0] + parts[1], rel_tol=1e-4)


def test_band_covers_both_scatter_exponent_extremes():
    """The trimmed band is the union over the exponents in play, so trimming it costs no
    more accuracy than it does for a single exponent."""
    kwargs = {**CASES["matrix"], "scatter_exponent": None}
    n_scat = kwargs["scatterer_positions"].shape[0]
    kwargs["scatter_exponent"] = np.where(np.arange(n_scat) % 2, 0.0, 2.0).astype(np.float32)
    assert_close(simulate_rf(**{**kwargs, "band_db": None}), simulate_rf(**kwargs))


def test_invalid_per_scatterer_scatter_exponent_raises():
    kwargs = CASES["matrix"]
    n_scat = kwargs["scatterer_positions"].shape[0]
    with pytest.raises(ValueError, match="one value per scatterer"):
        simulate_rf(**{**kwargs, "scatter_exponent": np.ones(n_scat + 1, np.float32)})
    with pytest.raises(ValueError, match="dimensions"):
        simulate_rf(**{**kwargs, "scatter_exponent": np.ones((n_scat, 1), np.float32)})
    with pytest.raises(ValueError, match="scatter_exponent"):
        simulate_rf(**{**kwargs, "scatter_exponent": -np.ones(n_scat, np.float32)})


def test_time_domain_rejects_per_scatterer_scatter_exponent():
    kwargs = {k: v for k, v in CASES["matrix"].items() if k != "scatter_exponent"}
    n_scat = kwargs["scatterer_positions"].shape[0]
    with pytest.raises(ValueError, match="only supported in the frequency domain"):
        simulate_rf_td(**kwargs, scatter_exponent=np.full(n_scat, 1.5, np.float32))


def _jitted_simulate_rf():
    """``simulate_rf`` under ``jax.jit`` with its static arguments declared, so that the
    scatterers, the delays and the scatter exponent are traced."""
    import jax

    static = (
        "n_ax",
        "center_frequency",
        "sampling_frequency",
        "n_fft",
        "band_db",
        "apply_lens_correction",
        "scatter_exponent_range",
    )
    return jax.jit(lambda **kw: simulate_rf(**kw), static_argnames=static)


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_traced_per_scatterer_scatter_exponent_needs_a_band():
    """The band is a static shape, so a traced exponent has to declare its range."""
    kwargs = {**CASES["matrix"], "n_fft": 1024}
    exponent = kwargs.pop("scatter_exponent")
    n_scat = kwargs["scatterer_positions"].shape[0]
    vector = np.full(n_scat, exponent, np.float32)
    jitted = _jitted_simulate_rf()
    with pytest.raises(ValueError, match="scatter_exponent_range"):
        jitted(**kwargs, scatter_exponent=vector)
    reference = simulate_rf(**kwargs, scatter_exponent=exponent)
    assert_close(reference, jitted(**kwargs, scatter_exponent=vector, band_db=None))
    assert_close(
        reference,
        jitted(**kwargs, scatter_exponent=vector, scatter_exponent_range=(exponent, exponent)),
    )


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_traced_shared_scatter_exponent():
    """A shared exponent may be traced too: only the band needs it concrete, not the gain.

    Tracing it keeps one compiled kernel across exponents and makes the exponent
    differentiable, at the price of declaring the band.
    """
    import jax
    import jax.numpy as jnp

    kwargs = {**CASES["matrix"], "n_fft": 1024}
    exponent = kwargs.pop("scatter_exponent")
    jitted = _jitted_simulate_rf()
    traced = jnp.float32(exponent)
    with pytest.raises(ValueError, match="scatter_exponent_range"):
        jitted(**kwargs, scatter_exponent=traced)
    reference = simulate_rf(**kwargs, scatter_exponent=exponent)
    assert_close(reference, jitted(**kwargs, scatter_exponent=traced, band_db=None))
    assert_close(
        reference,
        jitted(**kwargs, scatter_exponent=traced, scatter_exponent_range=(exponent, exponent)),
    )
    # One band covering a range serves every exponent in it without recompiling.
    for value in (0.5, exponent, 2.0):
        assert_close(
            simulate_rf(**kwargs, scatter_exponent=value),
            jitted(
                **kwargs, scatter_exponent=jnp.float32(value), scatter_exponent_range=(0.5, 2.0)
            ),
        )
    # A traced exponent carries a gradient, which a static one cannot.
    grad = jax.grad(
        lambda p: jnp.sum(
            jitted(**kwargs, scatter_exponent=p, scatter_exponent_range=(0.5, 2.0)) ** 2
        )
    )(traced)
    assert np.isfinite(grad) and grad != 0.0


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_op_traces_a_shared_scatter_exponent():
    """Under an outer jit the op cannot read the exponent, so the caller declares the band."""
    import jax
    import jax.numpy as jnp

    kwargs = {k: v for k, v in CASES["matrix"].items() if k != "scatter_exponent"}
    exponent = CASES["matrix"]["scatter_exponent"]
    op = Simulate(with_batch_dim=False, jit_compile=False)

    def run(p):
        return op(**kwargs, scatter_exponent=p, scatter_exponent_range=(exponent, exponent))["data"]

    reference = simulate_rf(**kwargs, scatter_exponent=exponent)
    assert_close(reference, jax.jit(run)(jnp.float32(exponent)))


def test_op_traces_a_per_scatterer_scatter_exponent():
    """The op derives the static band before the jitted call, so the vector can be traced."""
    kwargs = CASES["matrix"]
    n_scat = kwargs["scatterer_positions"].shape[0]
    op = Simulate(with_batch_dim=False)
    reference = op(**kwargs)["data"]
    vector = np.full(n_scat, kwargs["scatter_exponent"], np.float32)
    assert_close(reference, op(**{**kwargs, "scatter_exponent": vector})["data"])
    # And back to a scalar, which moves the argument between static and traced again.
    assert_close(reference, op(**kwargs)["data"])


def test_smooth_size_and_fft_length():
    assert [smooth_size(n) for n in (1, 7, 100, 601, 1025)] == [1, 8, 100, 625, 1080]
    geometry = linear_probe()
    n_fft = fft_length(N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, geometry, 0, 0)
    assert n_fft >= N_AX
    assert n_fft == smooth_size(n_fft)
    # The farthest scatterer bounds the length below the aperture bound when it is closer.
    near = np.array([[0.0, 0.0, 5e-3]])
    assert (
        fft_length(
            N_AX,
            SAMPLING_FREQUENCY,
            CENTER_FREQUENCY,
            SOUND_SPEED,
            geometry,
            0,
            0,
            scatterer_positions=near,
        )
        <= n_fft
    )
    # A longer pulse needs a longer FFT.
    longer = fft_length(
        N_AX,
        SAMPLING_FREQUENCY,
        CENTER_FREQUENCY,
        SOUND_SPEED,
        geometry,
        0,
        0,
        waveforms_two_way=hann_waveform(32.0),
    )
    assert longer > n_fft
