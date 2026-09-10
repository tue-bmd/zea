"""Tests for the Refocus operation (REFoCUS pipeline operation)."""

import numpy as np
import pytest

from . import DEFAULT_TEST_SEED, backend_equality_check

N_EL = 8  # number of transducer elements
N_TX = 5  # number of transmit events
N_AX = 64  # number of axial samples
SAMPLING_FREQ = np.float32(40e6)  # Hz
DEMODULATION_FREQ = np.float32(5e6)  # Hz
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


@pytest.fixture
def bandlimited_rf_data():
    """RF data as Gaussian-modulated pulses centred at DEMODULATION_FREQ."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    t = np.arange(N_AX) / SAMPLING_FREQ
    sigma = 3.0 / SAMPLING_FREQ  # ~1.5 cycles at DEMODULATION_FREQ
    data = np.zeros((N_TX, N_AX, N_EL, 1), dtype=np.float32)
    for tx in range(N_TX):
        for el in range(N_EL):
            for _ in range(3):  # a few echoes per trace
                t_c = rng.uniform(0.25, 0.75) * N_AX / SAMPLING_FREQ
                envelope = np.exp(-((t - t_c) ** 2) / (2 * sigma**2))
                phase = 2 * np.pi * DEMODULATION_FREQ * (t - t_c) + rng.uniform(0, 2 * np.pi)
                data[tx, :, el, 0] += envelope * np.cos(phase)
    return data


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
        demodulation_frequency=DEMODULATION_FREQ,
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


def test_svd_methods_are_non_jittable():
    """SVD-based Refocus methods must report jittable=False so JIT is skipped everywhere."""
    from zea import ops
    from zea.ops import Refocus

    # adjoint stays jittable
    assert Refocus(method="adjoint").jittable is True

    for method in ("tikhonov", "rsvd", "tsvd"):
        op = Refocus(method=method)
        assert op.jittable is False
        # A jit_options="ops" pipeline must not actually JIT-wrap a non-jittable op:
        # set_jit honours jittable, so _call stays the plain (un-jitted) call method.
        pipeline = ops.Pipeline([Refocus(method=method)], jit_options="ops", validate=False)
        refocus_op = pipeline.operations[0]
        assert refocus_op._call == refocus_op.call

    # A jit_options="pipeline" pipeline containing an SVD Refocus must fail fast and clearly,
    # rather than silently tracing the unsupported SVD op.
    with pytest.raises(ValueError, match="not all operations are jittable"):
        ops.Pipeline([Refocus(method="tsvd")], jit_options="pipeline", validate=False)


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


_IQ_EQUIV_TOL = {
    ("adjoint", None): 1e-3,
    ("adjoint", 0): 5e-2,
    ("tikhonov", None): 1e-2,
    ("rsvd", None): 1e-2,
    ("tsvd", None): 1e-2,
}


@pytest.mark.parametrize(
    ("method", "param"),
    [("adjoint", None), ("adjoint", 0), ("tikhonov", None), ("rsvd", None), ("tsvd", None)],
)
def test_iq_matches_demodulated_rf(
    method, param, probe_geometry, plane_wave_delays, bandlimited_rf_data
):
    """Decoding commutes with demodulation.
        demodulate(decode_RF(rf)) == decode_IQ(demodulate(rf))

    Asserting this pins down the carrier offset, the fftfreq sign convention,
    the even-length Nyquist bin and the inverse-FFT normalization of the IQ
    path all at once.
    """
    import keras

    from zea.func.ultrasound import demodulate
    from zea.ops import Refocus

    def _demodulate(array):
        return keras.ops.convert_to_numpy(
            demodulate(
                keras.ops.convert_to_tensor(array),
                DEMODULATION_FREQ,
                SAMPLING_FREQ,
                axis=-3,
            )
        )

    op = Refocus(method=method, param=param, with_batch_dim=False)

    iq_data = _demodulate(bandlimited_rf_data)
    assert iq_data.shape == (N_TX, N_AX, N_EL, 2)

    rf_decoded = keras.ops.convert_to_numpy(
        _call_refocus(op, bandlimited_rf_data, probe_geometry, plane_wave_delays)[op.output_key]
    )
    iq_decoded = keras.ops.convert_to_numpy(
        _call_refocus(op, iq_data, probe_geometry, plane_wave_delays)[op.output_key]
    )

    expected = _demodulate(rf_decoded)
    assert iq_decoded.shape == expected.shape

    error = np.abs(iq_decoded - expected).max() / np.abs(expected).max()
    tol = _IQ_EQUIV_TOL[(method, param)]
    assert error < tol, (
        f"method={method} param={param}: IQ decoding does not match the "
        f"demodulated RF decoding (relative max error {error:.3e} >= {tol:.3e})"
    )


def test_iq_requires_demodulation_frequency(probe_geometry, plane_wave_delays, iq_data):
    """IQ input without a demodulation frequency must raise, not silently decode."""
    import keras

    from zea.ops import Refocus

    op = Refocus(with_batch_dim=False)
    with pytest.raises(ValueError, match="demodulation_frequency"):
        op(
            data=keras.ops.convert_to_tensor(iq_data),
            t0_delays=keras.ops.convert_to_tensor(plane_wave_delays),
            sampling_frequency=SAMPLING_FREQ,
            probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
            initial_times=np.zeros(N_TX, dtype=np.float32),
            demodulation_frequency=None,
        )


def test_unsupported_n_ch_raises(probe_geometry, plane_wave_delays):
    """Only RF (n_ch=1) and IQ (n_ch=2) are supported."""
    from zea.ops import Refocus

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal((N_TX, N_AX, N_EL, 3)).astype(np.float32)

    op = Refocus(with_batch_dim=False)
    with pytest.raises(ValueError, match="n_ch=3"):
        _call_refocus(op, data, probe_geometry, plane_wave_delays)


def test_output_shape_iq_with_batch_dim(probe_geometry, plane_wave_delays):
    """IQ decoding must survive the vmap path used when with_batch_dim=True."""
    import keras

    from zea.ops import Refocus

    op = Refocus(with_batch_dim=True)
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    batch_size = 2
    data_batch = rng.standard_normal((batch_size, N_TX, N_AX, N_EL, 2)).astype(np.float32)

    result = op(
        data=keras.ops.convert_to_tensor(data_batch),
        t0_delays=keras.ops.convert_to_tensor(plane_wave_delays),
        sampling_frequency=SAMPLING_FREQ,
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
        initial_times=np.zeros(N_TX, dtype=np.float32),
        demodulation_frequency=DEMODULATION_FREQ,
    )
    decoded = keras.ops.convert_to_numpy(result[op.output_key])
    assert decoded.shape == (batch_size, N_EL, N_AX, N_EL, 2), (
        f"Expected ({batch_size}, {N_EL}, {N_AX}, {N_EL}, 2), got {decoded.shape}"
    )


def test_adjoint_ramp_uses_absolute_frequency():
    """The adjoint ramp filter must scale by |f|, not by signed f."""
    import keras

    from zea.ops import Refocus

    op = Refocus(method="adjoint", param=None)
    n_tx, n_el = 3, 4

    # With zero delays and unit apodization H == 1, so Hinv reduces to the
    # ramp itself and a sign flip is directly visible.
    delays = keras.ops.zeros((n_tx, n_el))
    apod = keras.ops.ones((n_tx, n_el))
    f_vec = keras.ops.convert_to_tensor(np.array([-0.25, 0.25], dtype=np.float32))

    hinv = keras.ops.convert_to_numpy(op._get_hinv(delays, f_vec, apod))

    assert hinv.shape == (2, n_el, n_tx)
    expected = 0.25 * np.ones((n_el, n_tx))
    np.testing.assert_allclose(
        hinv[0], expected, atol=1e-6, err_msg="negative f was not |f|-scaled"
    )
    np.testing.assert_allclose(hinv[1], expected, atol=1e-6)


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


@pytest.mark.parametrize("data_kind", ["rf", "iq"])
def test_output_dtype_is_float32(data_kind, probe_geometry, plane_wave_delays, rf_data, iq_data):
    """Decoded output must always be float32 regardless of method or n_ch."""
    import keras

    from zea.ops import Refocus

    data = rf_data if data_kind == "rf" else iq_data

    for method in ("adjoint", "tikhonov", "rsvd", "tsvd"):
        op = Refocus(method=method, with_batch_dim=False)
        result = _call_refocus(op, data, probe_geometry, plane_wave_delays)
        decoded = keras.ops.convert_to_numpy(result[op.output_key])
        assert decoded.dtype == np.float32, (
            f"method={method} data={data_kind}: expected float32, got {decoded.dtype}"
        )


@pytest.mark.parametrize("method", ["adjoint", "tikhonov"])
@pytest.mark.parametrize("n_ch", [1, 2])
@backend_equality_check(decimal=3)
def test_refocus_cross_backend(method, n_ch):
    """Refocus output must be consistent across backends, for RF and for IQ."""
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

    data = rng.standard_normal((N_TX, N_AX, N_EL, n_ch)).astype(np.float32)

    op = Refocus(method=method, with_batch_dim=False)
    result = op(
        data=keras.ops.convert_to_tensor(data),
        t0_delays=keras.ops.convert_to_tensor(t0_delays),
        sampling_frequency=np.float32(SAMPLING_FREQ),
        probe_geometry=keras.ops.convert_to_tensor(probe_geometry),
        initial_times=np.zeros(N_TX, dtype=np.float32),
        demodulation_frequency=np.float32(DEMODULATION_FREQ),
    )
    return keras.ops.convert_to_numpy(result[op.output_key])
