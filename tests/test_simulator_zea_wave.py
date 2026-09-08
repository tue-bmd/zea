"""``simulate_rf_zea_wave`` against ``simulate_rf``, feature by feature.

With the band limit off and the same FFT length the two evaluate the same physics and are
expected to agree to float32 precision, in value and in gradient.
"""

import keras
import numpy as np
import pytest
from keras import ops

from zea.ops import Simulate
from zea.probes import create_curved_probe_geometry, curved_probe_normals
from zea.simulator import fft_length, simulate_rf, simulate_rf_zea_wave, smooth_size

SOUND_SPEED = 1540.0
CENTER_FREQUENCY = 3e6
SAMPLING_FREQUENCY = 12e6
N_AX = 512


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
    "elevation_slab_2d_prunes": _case(_linear_probe(), elevation_slab_2d=True, element_height=1e-3),
    "elevation_slab_2d_all_inside": _case(_linear_probe(), elevation_slab_2d=True, element_height=5e-3),
    "noise_and_tgc": _case(
        _linear_probe(), noise_level_db=-40.0, tgc_max_db=20.0, noise_seed=3, noise_reference=1.0
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_matches_simulate_rf(name):
    kwargs = _tensors(CASES[name])
    n_fft = 1 << int(np.ceil(np.log2(N_AX)))
    reference = simulate_rf(**kwargs)
    result = simulate_rf_zea_wave(**kwargs, band_db=None, n_fft=n_fft)
    _assert_close(reference, result)


def test_default_band_and_derived_fft_length_are_close():
    kwargs = _tensors(CASES["linear"])
    assert _correlation(simulate_rf(**kwargs), simulate_rf_zea_wave(**kwargs)) > 0.9999


def test_scatterers_beyond_record_are_dropped_without_wrapping():
    kwargs = dict(CASES["linear"])
    positions = kwargs["scatterer_positions"].copy()
    positions[:4, 2] += 0.08
    kwargs["scatterer_positions"] = positions
    kwargs = _tensors(kwargs)
    reference = simulate_rf(**kwargs)
    _assert_close(reference, simulate_rf_zea_wave(**kwargs, band_db=None, n_fft=1024))
    _assert_close(reference, simulate_rf_zea_wave(**kwargs, band_db=None))


def test_empty_phantom_gives_zeros():
    kwargs = dict(CASES["linear"])
    kwargs["scatterer_positions"] = np.zeros((0, 3), np.float32)
    kwargs["scatterer_magnitudes"] = np.zeros(0, np.float32)
    result = _np(simulate_rf_zea_wave(**_tensors(kwargs)))
    assert result.shape == (4, N_AX, 16, 1)
    assert not result.any()


def test_n_period_changes_the_pulse():
    kwargs = _tensors(CASES["linear"])
    short = simulate_rf_zea_wave(**kwargs, n_period=2.0)
    default = simulate_rf_zea_wave(**kwargs)
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
        "band_db",
        "n_fft",
    )
    jitted = jax.jit(simulate_rf_zea_wave, static_argnames=static)
    with pytest.raises(ValueError, match="n_fft"):
        jitted(**kwargs, band_db=None)
    reference = simulate_rf(**kwargs)
    _assert_close(reference, jitted(**kwargs, band_db=None, n_fft=1024))
    # Default band limit under an outer jit: the band is still derived concretely.
    assert _correlation(reference, jitted(**kwargs, n_fft=1024)) > 0.9999


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="jax tracing semantics")
def test_under_jit_closed_over_geometry_derives_n_fft():
    import jax

    kwargs = dict(CASES["linear"])
    positions = kwargs.pop("scatterer_positions")
    magnitudes = kwargs.pop("scatterer_magnitudes")
    reference = simulate_rf(positions, magnitudes, **kwargs)
    result = jax.jit(lambda p, m: simulate_rf_zea_wave(p, m, **kwargs))(positions, magnitudes)
    assert _correlation(reference, result) > 0.9999


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="uses jax.grad")
def test_gradients_match_simulate_rf():
    import jax
    import jax.numpy as jnp

    kwargs = dict(CASES["lens_correction"])
    positions = kwargs.pop("scatterer_positions")
    magnitudes = kwargs.pop("scatterer_magnitudes")
    kwargs.pop("sound_speed")
    kwargs.pop("element_width")
    w = jnp.asarray(np.random.default_rng(2).normal(size=(4, N_AX, 16, 1)), jnp.float32)

    def loss(simulate, **extra):
        def fn(p, m, c, width):
            rf = simulate(p, m, sound_speed=c, element_width=width, **kwargs, **extra)
            return jnp.sum(w * rf)

        return fn

    args = (jnp.asarray(positions), jnp.asarray(magnitudes), 1540.0, 0.27e-3)
    reference = jax.grad(loss(simulate_rf), argnums=(0, 1, 2, 3))(*args)
    result = jax.grad(loss(simulate_rf_zea_wave, band_db=None, n_fft=1024), argnums=(0, 1, 2, 3))(
        *args
    )
    for a, b in zip(reference, result):
        assert _correlation(a, b) > 0.9999


def test_simulate_op():
    kwargs = _tensors(CASES["linear"])
    exact = Simulate(with_batch_dim=False)
    op = Simulate(with_batch_dim=False)
    reference = exact(**kwargs)[exact.output_key]
    result = op(**kwargs, method="zea_wave", band_db=None, n_fft=512)[op.output_key]
    _assert_close(reference, result)


def test_simulate_op_with_batch_dim():
    kwargs = dict(CASES["linear"])
    for key in ("scatterer_positions", "scatterer_magnitudes"):
        kwargs[key] = np.stack([kwargs[key], kwargs[key][::-1]])
    kwargs = _tensors(kwargs)
    exact = Simulate(with_batch_dim=True)
    op = Simulate(with_batch_dim=True)
    reference = exact(**kwargs)[exact.output_key]
    result = op(**kwargs, method="zea_wave", band_db=None, n_fft=512)[op.output_key]
    _assert_close(reference, result)


def test_smooth_size_and_fft_length():
    assert [smooth_size(n) for n in (1, 7, 100, 1025)] == [1, 8, 100, 1080]
    geometry = _linear_probe()
    n_fft = fft_length(N_AX, SAMPLING_FREQUENCY, CENTER_FREQUENCY, SOUND_SPEED, geometry, 0, 0)
    assert n_fft >= N_AX
    assert n_fft == smooth_size(n_fft)


def test_single_transmit_goes_to_simulate_rf():
    geometry = _linear_probe()
    kwargs = _tensors(_case(geometry, **_scan(geometry, n_tx=1)))
    np.testing.assert_array_equal(_np(simulate_rf(**kwargs)), _np(simulate_rf_zea_wave(**kwargs)))


@pytest.mark.parametrize("n_tx", [2, 3])
def test_few_transmits_match_simulate_rf(n_tx):
    geometry = _linear_probe()
    kwargs = _tensors(_case(geometry, **_scan(geometry, n_tx=n_tx)))
    _assert_close(simulate_rf(**kwargs), simulate_rf_zea_wave(**kwargs))
