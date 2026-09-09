"""Tests for mixed-precision (bfloat16) and int16 beamforming.

zea keeps geometry and time-of-flight *delay* computation in ``float32`` while
running the bulk *signal* compute in a lower-precision "compute dtype" resolved
from the active Keras mixed-precision policy. These tests verify:

* the :mod:`zea.internal.precision` policy-resolution helpers,
* that the beamform signal path follows the policy (bfloat16 under
  ``mixed_bfloat16``) while delays stay float32,
* that mixed-precision reconstructions stay numerically faithful to float32, and
* that integer (int16) RF data is accepted natively.
"""

import keras
import numpy as np
import pytest
from keras import ops

from zea.beamform.beamformer import apply_delays, complex_rotate, tof_correction
from zea.beamform.delays import compute_t0_delays_planewave
from zea.beamform.pixelgrid import cartesian_pixel_grid
from zea.internal.precision import (
    LOW_PRECISION_DTYPES,
    is_mixed_precision,
    signal_compute_dtype,
)
from zea.ops import (
    CoherenceFactor,
    DelayAndSum,
    DelayMultiplyAndSum,
    GeneralizedCoherenceFactor,
    TOFCorrection,
)

N_EL = 16
SOUND_SPEED = 1540.0
SAMPLING_FREQ = 40e6
DEMOD_FREQ = 5e6


@pytest.fixture
def restore_policy():
    """Save and restore the global Keras mixed-precision policy around a test."""
    prev = keras.mixed_precision.dtype_policy().name
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(prev)


def _probe_geometry():
    xs = np.linspace(-10e-3, 10e-3, N_EL)
    return np.stack([xs, np.zeros(N_EL), np.zeros(N_EL)], axis=-1).astype(np.float32)


def _flatgrid():
    grid = cartesian_pixel_grid(
        xlims=(-5e-3, 5e-3), zlims=(5e-3, 20e-3), grid_size_x=12, grid_size_z=16
    )
    return grid.reshape(-1, 3).astype(np.float32)


def _tof_kwargs(n_ch=1, n_tx=3, n_ax=1200, seed=0, f_number=0.0):
    """Full set of inputs required by :func:`tof_correction` / :class:`TOFCorrection`.

    ``n_ax`` is large enough that the round-trip delays to a 5-20 mm grid land
    within the recorded samples, so the reconstruction carries real signal (and
    non-zero variance) rather than clipped/masked zeros.
    """
    probe_geometry = _probe_geometry()
    flatgrid = _flatgrid()
    rng = np.random.default_rng(seed)
    # Realistic RF amplitude scale so the int16 cast is meaningful (~ +/- 3000).
    data = (rng.standard_normal((n_tx, n_ax, N_EL, n_ch)) * 1000.0).astype(np.float32)
    polar_angles = np.zeros(n_tx, dtype=np.float32)
    t0_delays = compute_t0_delays_planewave(
        probe_geometry, polar_angles, sound_speed=SOUND_SPEED
    ).astype(np.float32)
    return dict(
        data=data,
        flatgrid=flatgrid,
        t0_delays=t0_delays,
        tx_apodizations=np.ones((n_tx, N_EL), dtype=np.float32),
        sound_speed=SOUND_SPEED,
        probe_geometry=probe_geometry,
        initial_times=np.zeros(n_tx, dtype=np.float32),
        sampling_frequency=SAMPLING_FREQ,
        demodulation_frequency=DEMOD_FREQ,
        f_number=f_number,
        polar_angles=polar_angles,
        focus_distances=np.zeros(n_tx, dtype=np.float32),
        t_peak=np.zeros(n_tx, dtype=np.float32),
        transmit_origins=np.zeros((n_tx, 3), dtype=np.float32),
    )


def _rel_l2(ref, test):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    return np.linalg.norm(test - ref) / (np.linalg.norm(ref) + 1e-30)


def _cosine_similarity(ref, test):
    """Cosine similarity of two signal vectors (robust to zero-mean signals)."""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    return float(np.dot(ref, test) / (np.linalg.norm(ref) * np.linalg.norm(test) + 1e-30))


# --------------------------------------------------------------------------- #
# Precision-helper unit tests
# --------------------------------------------------------------------------- #
def test_signal_compute_dtype_default(restore_policy):
    """Default policy resolves to floatx (float32)."""
    keras.mixed_precision.set_global_policy("float32")
    assert signal_compute_dtype() == keras.config.floatx()
    assert not is_mixed_precision()


def test_signal_compute_dtype_mixed(restore_policy):
    """A mixed-precision policy resolves to bfloat16."""
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    assert signal_compute_dtype() == "bfloat16"
    assert is_mixed_precision()


# --------------------------------------------------------------------------- #
# apply_delays: dtype preservation + int16 support
# --------------------------------------------------------------------------- #
def test_apply_delays_preserves_float_dtype():
    """apply_delays keeps a bfloat16 signal in bfloat16 (weights computed in float32)."""
    rng = np.random.default_rng(0)
    data = ops.cast(rng.standard_normal((64, N_EL, 1)) * 1000.0, "bfloat16")
    delays = ops.cast(rng.uniform(0, 63, size=(48, N_EL)), "float32")
    out = apply_delays(data, delays, clip_min=0, clip_max=63)
    assert keras.backend.standardize_dtype(out.dtype) == "bfloat16"


def test_apply_delays_int16_promotes_and_matches_float32(restore_policy):
    """int16 RF gathered directly is promoted and matches the float32 result."""
    keras.mixed_precision.set_global_policy("float32")
    rng = np.random.default_rng(1)
    data_f32 = (rng.standard_normal((80, N_EL, 1)) * 2000.0).astype(np.float32)
    data_i16 = data_f32.astype(np.int16)
    delays = rng.uniform(0, 79, size=(50, N_EL)).astype(np.float32)

    out_i16 = ops.convert_to_numpy(
        apply_delays(ops.convert_to_tensor(data_i16), ops.convert_to_tensor(delays), 0, 79)
    )
    out_f32 = ops.convert_to_numpy(
        apply_delays(
            ops.convert_to_tensor(data_i16.astype(np.float32)),
            ops.convert_to_tensor(delays),
            0,
            79,
        )
    )
    assert keras.backend.standardize_dtype(out_i16.dtype).startswith("float")
    # Same integer samples, same weights -> identical interpolation.
    np.testing.assert_allclose(out_i16, out_f32, rtol=1e-5, atol=1e-3)


# --------------------------------------------------------------------------- #
# complex_rotate stays accurate for large angles in mixed precision
# --------------------------------------------------------------------------- #
def test_complex_rotate_large_angle_bfloat16(restore_policy):
    """cos/sin of a large angle must be evaluated in float32 before down-casting."""
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    iq = ops.cast(np.array([[1.0, 0.0]], dtype=np.float32), "bfloat16")
    theta = ops.convert_to_tensor(np.array([1234.5], dtype=np.float32))  # large angle
    rotated = ops.convert_to_numpy(complex_rotate(iq, theta)).astype(np.float64)
    ref = np.array([np.cos(1234.5), np.sin(1234.5)])
    # bfloat16 output precision (~1e-2), but NOT the garbage a bf16 cos(1234.5) gives.
    np.testing.assert_allclose(rotated[0], ref, atol=3e-2)


# --------------------------------------------------------------------------- #
# tof_correction: dtype follows the policy, values stay faithful
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ch", [1, 2])
def test_tof_correction_dtype_follows_policy(restore_policy, n_ch):
    """TOF output is bfloat16 under a mixed policy, float32 otherwise."""
    kwargs = _tof_kwargs(n_ch=n_ch)

    keras.mixed_precision.set_global_policy("float32")
    out_f32 = tof_correction(**kwargs)
    assert keras.backend.standardize_dtype(out_f32.dtype) == "float32"

    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    out_bf16 = tof_correction(**kwargs)
    assert keras.backend.standardize_dtype(out_bf16.dtype) == "bfloat16"

    rel = _rel_l2(ops.convert_to_numpy(out_f32), ops.convert_to_numpy(out_bf16))
    assert rel < 5e-2, f"TOF mixed-precision rel L2 error too high: {rel}"


# --------------------------------------------------------------------------- #
# Full DAS reconstruction: mixed precision & int16 stay faithful to float32
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ch", [1, 2])
def test_das_mixed_precision_matches_float32(restore_policy, n_ch):
    """A TOFCorrection + DelayAndSum reconstruction agrees across precisions."""
    kwargs = _tof_kwargs(n_ch=n_ch, seed=3)

    def beamform():
        aligned = TOFCorrection(with_batch_dim=False)(**kwargs)
        return ops.convert_to_numpy(DelayAndSum(with_batch_dim=False)(**aligned)["data"])

    keras.mixed_precision.set_global_policy("float32")
    ref = beamform()
    assert keras.backend.standardize_dtype(ref.dtype) == "float32"  # DAS accumulates in float32

    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    mixed = beamform()
    # Output stays float32 (accumulation), even though the aligned data was bfloat16.
    assert keras.backend.standardize_dtype(mixed.dtype) == "float32"

    assert np.std(ref) > 0, "degenerate (constant) reference reconstruction"
    rel = _rel_l2(ref, mixed)
    cos = _cosine_similarity(ref, mixed)
    assert rel < 5e-2, f"DAS mixed-precision rel L2 error too high: {rel}"
    assert cos > 0.999, f"DAS mixed-precision cosine similarity too low: {cos}"


@pytest.mark.parametrize("dtype", LOW_PRECISION_DTYPES)
@pytest.mark.parametrize(
    "op_cls",
    [DelayMultiplyAndSum, CoherenceFactor, GeneralizedCoherenceFactor],
)
def test_low_precision_beamformer_upcasts_and_matches_float32(op_cls, dtype):
    """bfloat16/float16 aligned data is up-cast to float32 internally, so the
    result stays numerically close to running the same op on float32 data."""
    n_tx, n_pix, n_el, n_ch = 3, 5, 8, 2
    rng = np.random.default_rng(7)
    data = (rng.standard_normal((1, n_tx, n_pix, n_el, n_ch)) * 10.0).astype(np.float32)
    data_f32 = ops.convert_to_tensor(data)
    data_low = ops.cast(data_f32, dtype)
    assert keras.backend.standardize_dtype(data_low.dtype) in LOW_PRECISION_DTYPES

    operation = op_cls(with_batch_dim=True)
    out_f32 = ops.convert_to_numpy(operation(data=data_f32)["data"])
    out_low = ops.convert_to_numpy(operation(data=data_low)["data"])

    rel = _rel_l2(out_f32, out_low)
    assert rel < 5e-2, f"{op_cls.__name__} {dtype} up-cast rel L2 error too high: {rel}"


def test_das_int16_input(restore_policy):
    """int16 RF input is accepted natively and matches the float32-input result."""
    kwargs = _tof_kwargs(n_ch=1, seed=5)
    data_f32 = kwargs["data"]

    keras.mixed_precision.set_global_policy("float32")

    def beamform(data):
        kw = {**kwargs, "data": data}
        aligned = TOFCorrection(with_batch_dim=False)(**kw)
        return ops.convert_to_numpy(DelayAndSum(with_batch_dim=False)(**aligned)["data"])

    ref = beamform(ops.convert_to_tensor(data_f32))
    got = beamform(ops.convert_to_tensor(data_f32.astype(np.int16)))

    # int16 quantizes the RF (~1 LSB on a ~1000 scale); reconstruction stays close.
    rel = _rel_l2(ref, got)
    assert rel < 5e-3, f"int16 vs float32 rel L2 error too high: {rel}"
