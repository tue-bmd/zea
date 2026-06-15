"""Tests for the Refocus operation (REFoCUS pipeline operation)."""

import numpy as np
import pytest

from . import DEFAULT_TEST_SEED, backend_equality_check

N_EL = 8  # number of transducer elements
N_TX = 5  # number of transmit events
N_AX = 64  # number of axial samples
SAMPLING_FREQ = np.float32(40e6)  # Hz
SOUND_SPEED = 1540.0  # m/s
T_PEAK = np.float32(5e-7)  # transmit-waveform peak time (s)


@pytest.fixture
def probe_geometry():
    """Linear array with N_EL elements spanning ±10 mm in x."""
    xs = np.linspace(-10e-3, 10e-3, N_EL)
    return np.stack([xs, np.zeros(N_EL), np.zeros(N_EL)], axis=-1).astype(np.float32)


@pytest.fixture
def plane_wave_delays(probe_geometry):
    """Plane-wave transmit delays (n_tx, n_el) at a few steering angles."""
    from zea.beamform.delays import compute_t0_delays_planewave

    polar_angles = np.linspace(-0.2, 0.2, N_TX).astype(np.float32)
    return compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)


@pytest.fixture
def rf_data():
    """Random RF data: (n_tx, n_ax, n_el, 1)."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    return rng.standard_normal((N_TX, N_AX, N_EL, 1)).astype(np.float32)


@pytest.fixture
def iq_data():
    """Random IQ data: (n_tx, n_ax, n_el, 2)."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    return rng.standard_normal((N_TX, N_AX, N_EL, 2)).astype(np.float32)


def _call_refocus(op, data_np, probe_geometry_np, plane_wave_delays_np):
    """Helper to call Refocus with standard numpy inputs (no batch dim)."""
    import keras

    return op(
        data=keras.ops.convert_to_tensor(data_np),
        t0_delays=keras.ops.convert_to_tensor(plane_wave_delays_np),
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry_np),
        initial_times=np.zeros(N_TX, dtype=np.float32),
        t_peak=keras.ops.convert_to_tensor(np.full(N_TX, T_PEAK, dtype=np.float32)),
    )


def test_invalid_method_raises():
    """Constructing Refocus with an unknown method must raise ValueError."""
    from zea.ops import Refocus

    with pytest.raises(ValueError, match="method must be one of"):
        Refocus(method="unknown_method")


def test_valid_methods_construct():
    """All documented methods should construct without error."""
    from zea.ops import Refocus

    for method in ("adjoint", "tikhonov", "rsvd", "tsvd"):
        op = Refocus(method=method)
        assert op.method == method


@pytest.mark.parametrize("method", ["adjoint", "tikhonov", "rsvd", "tsvd"])
def test_output_shape_rf(method, probe_geometry, plane_wave_delays, rf_data):
    """Decoded RF output must have shape (n_el, n_ax, n_el, 1)."""
    import keras

    from zea.ops import Refocus

    op = Refocus(method=method, with_batch_dim=False)
    result = _call_refocus(op, rf_data, probe_geometry, plane_wave_delays)
    decoded = keras.ops.convert_to_numpy(result[op.output_key])
    assert decoded.shape == (N_EL, N_AX, N_EL, 1), (
        f"Expected ({N_EL}, {N_AX}, {N_EL}, 1), got {decoded.shape}"
    )


@pytest.mark.parametrize("method", ["adjoint", "tikhonov", "rsvd", "tsvd"])
def test_output_shape_iq(method, probe_geometry, plane_wave_delays, iq_data):
    """Decoded IQ output must have shape (n_el, n_ax, n_el, 2)."""
    import keras

    from zea.ops import Refocus

    op = Refocus(method=method, with_batch_dim=False)
    result = _call_refocus(op, iq_data, probe_geometry, plane_wave_delays)
    decoded = keras.ops.convert_to_numpy(result[op.output_key])
    assert decoded.shape == (N_EL, N_AX, N_EL, 2), (
        f"Expected ({N_EL}, {N_AX}, {N_EL}, 2), got {decoded.shape}"
    )


def test_sa_parameter_outputs(probe_geometry, plane_wave_delays, rf_data):
    """After decoding, synthetic-aperture parameters must have correct shapes and values."""
    import keras

    from zea.ops import Refocus

    op = Refocus(with_batch_dim=False)
    result = _call_refocus(op, rf_data, probe_geometry, plane_wave_delays)

    # t0_delays: zeros (n_el, n_el)
    t0 = keras.ops.convert_to_numpy(result["t0_delays"])
    assert t0.shape == (N_EL, N_EL)
    np.testing.assert_array_equal(t0, np.zeros((N_EL, N_EL), dtype=np.float32))

    # tx_apodizations: identity (n_el, n_el)
    apod = keras.ops.convert_to_numpy(result["tx_apodizations"])
    assert apod.shape == (N_EL, N_EL)
    np.testing.assert_array_equal(apod, np.eye(N_EL, dtype=np.float32))

    # polar_angles: zeros (n_el,)
    pa = keras.ops.convert_to_numpy(result["polar_angles"])
    assert pa.shape == (N_EL,)
    np.testing.assert_array_equal(pa, np.zeros(N_EL, dtype=np.float32))

    # focus_distances: zeros (n_el,)
    fd = keras.ops.convert_to_numpy(result["focus_distances"])
    assert fd.shape == (N_EL,)
    np.testing.assert_array_equal(fd, np.zeros(N_EL, dtype=np.float32))

    # initial_times: zeros (n_el,)
    it = keras.ops.convert_to_numpy(result["initial_times"])
    assert it.shape == (N_EL,)
    np.testing.assert_array_equal(it, np.zeros(N_EL, dtype=np.float32))

    # t_peak: shared transmit-waveform peak time, broadcast to (n_el,)
    tp = keras.ops.convert_to_numpy(result["t_peak"])
    assert tp.shape == (N_EL,)
    np.testing.assert_array_equal(tp, np.full(N_EL, T_PEAK, dtype=np.float32))

    # transmit_origins: equal to probe_geometry (n_el, 3)
    to = keras.ops.convert_to_numpy(result["transmit_origins"])
    np.testing.assert_array_equal(to, probe_geometry)

    # flat_pfield: None (resets pfield for downstream ops)
    assert result["flat_pfield"] is None


def test_default_apodization_matches_explicit_ones(probe_geometry, plane_wave_delays, rf_data):
    """Passing tx_apodizations=None must produce the same result as all-ones."""
    import keras

    from zea.ops import Refocus

    op = Refocus(with_batch_dim=False)
    data_t = keras.ops.convert_to_tensor(rf_data)
    t0_t = keras.ops.convert_to_tensor(plane_wave_delays)
    pg_t = keras.ops.convert_to_tensor(probe_geometry)
    it = np.zeros(N_TX, dtype=np.float32)
    apod_ones = np.ones((N_TX, N_EL), dtype=np.float32)

    result_none = op(
        data=data_t,
        t0_delays=t0_t,
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=pg_t,
        initial_times=it,
        tx_apodizations=None,
    )
    result_ones = op(
        data=data_t,
        t0_delays=t0_t,
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=pg_t,
        initial_times=it,
        tx_apodizations=keras.ops.convert_to_tensor(apod_ones),
    )

    dec_none = keras.ops.convert_to_numpy(result_none[op.output_key])
    dec_ones = keras.ops.convert_to_numpy(result_ones[op.output_key])
    np.testing.assert_allclose(dec_none, dec_ones, rtol=1e-5)


def test_adjoint_ramp_filter_differs_from_no_ramp(probe_geometry, plane_wave_delays, rf_data):
    """param=None (ramp) and param=0 (no ramp) must produce different outputs."""
    import keras

    from zea.ops import Refocus

    op_ramp = Refocus(method="adjoint", param=None, with_batch_dim=False)
    op_noramp = Refocus(method="adjoint", param=0, with_batch_dim=False)

    kwargs = dict(
        data=keras.ops.convert_to_tensor(rf_data),
        t0_delays=keras.ops.convert_to_tensor(plane_wave_delays),
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
        initial_times=np.zeros(N_TX, dtype=np.float32),
    )

    dec_ramp = keras.ops.convert_to_numpy(op_ramp(**kwargs)[op_ramp.output_key])
    dec_noramp = keras.ops.convert_to_numpy(op_noramp(**kwargs)[op_noramp.output_key])

    assert not np.allclose(dec_ramp, dec_noramp), (
        "Ramp-filtered and plain adjoint outputs should differ"
    )


def test_output_shape_with_batch_dim(probe_geometry, plane_wave_delays):
    """Refocus with with_batch_dim=True must handle a leading batch axis."""
    import keras

    from zea.ops import Refocus

    op = Refocus(with_batch_dim=True)
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    batch_size = 2
    data_batch = rng.standard_normal((batch_size, N_TX, N_AX, N_EL, 1)).astype(np.float32)

    result = op(
        data=keras.ops.convert_to_tensor(data_batch),
        t0_delays=keras.ops.convert_to_tensor(plane_wave_delays),
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
        initial_times=np.zeros(N_TX, dtype=np.float32),
    )
    decoded = keras.ops.convert_to_numpy(result[op.output_key])
    assert decoded.shape == (batch_size, N_EL, N_AX, N_EL, 1), (
        f"Expected ({batch_size}, {N_EL}, {N_AX}, {N_EL}, 1), got {decoded.shape}"
    )


def test_output_dtype_is_float32(probe_geometry, plane_wave_delays, rf_data):
    """Decoded output must always be float32 regardless of method."""
    import keras

    from zea.ops import Refocus

    for method in ("adjoint", "tikhonov", "rsvd", "tsvd"):
        op = Refocus(method=method, with_batch_dim=False)
        result = _call_refocus(op, rf_data, probe_geometry, plane_wave_delays)
        decoded = keras.ops.convert_to_numpy(result[op.output_key])
        assert decoded.dtype == np.float32, (
            f"method={method}: expected float32, got {decoded.dtype}"
        )


@pytest.mark.parametrize("method", ["adjoint", "tikhonov"])
@backend_equality_check(decimal=3)
def test_refocus_cross_backend(method):
    """Refocus output must be consistent across backends."""
    import keras
    import numpy as np

    from zea.beamform.delays import compute_t0_delays_planewave
    from zea.ops import Refocus

    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    probe_geometry = np.stack(
        [
            np.linspace(-10e-3, 10e-3, N_EL),
            np.zeros(N_EL),
            np.zeros(N_EL),
        ],
        axis=-1,
    ).astype(np.float32)

    polar_angles = np.linspace(-0.2, 0.2, N_TX).astype(np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)

    data = rng.standard_normal((N_TX, N_AX, N_EL, 1)).astype(np.float32)

    op = Refocus(method=method, with_batch_dim=False)
    result = op(
        data=keras.ops.convert_to_tensor(data),
        t0_delays=keras.ops.convert_to_tensor(t0_delays),
        sampling_frequency=np.float32(SAMPLING_FREQ),
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
        initial_times=np.zeros(N_TX, dtype=np.float32),
    )
    return keras.ops.convert_to_numpy(result[op.output_key])
