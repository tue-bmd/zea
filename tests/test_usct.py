"""Regression tests for the USCT reflectivity DAS reconstruction.

USCT is not covered by zea's standard beamformer, so its reconstruction lives in
:func:`zea.func.usct.usct_reflectivity_das` (wrapped by the
:class:`zea.ops.USCTReflectivityDAS` operation). These tests pin that
implementation two ways:

1. **Numerical oracle.** An independent, plain-NumPy round-trip time-of-flight
   DAS (``_numpy_usct_das`` below) recomputes the exact same quantity from first
   principles. We assert the keras/zea implementation matches it on a small
   deterministic case across both interpolation modes and with
   transmission-rejection / backscatter-apodization on and off. This guards the
   gather/apodization/summation arithmetic against regressions.

2. **Physical correctness.** A synthetic point scatterer on a ring array
   (analytic RF built from the round-trip delays) must reconstruct to a peak at
   the scatterer's true location — the same "does the reflector land where it
   should" check we used to validate against the Dartmouth ground truth.

The DAS algorithm (round-trip ToF delay-and-sum with through-transmission
rejection) matches the reflection-USCT baseline described for OpenPros
(arXiv:2505.12261) and the RUCT DAS reference of Lafci et al. (pyruct); the
NumPy oracle here is an independent re-derivation, not vendored code.
"""

import numpy as np
import pytest
from keras import ops

from zea.func.usct import channels_to_analytic, usct_reflectivity_das


def _numpy_usct_das(
    analytic,
    tx,
    rx,
    pixels,
    fs,
    t0,
    c,
    *,
    reject_transmission,
    transmission_guard_s,
    backscatter_apodization,
    interpolation,
):
    """Independent NumPy reference for :func:`usct_reflectivity_das`.

    Shared receive aperture, constant sound speed (no SoS map). Deliberately
    written as a naive double loop over transmit/receive pairs so it is obviously
    correct rather than fast.
    """
    analytic = np.asarray(analytic)
    tx = np.asarray(tx, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    n_tx, n_ax, n_el = analytic.shape
    P = pixels.shape[0]

    accum = np.zeros(P, dtype=np.complex128)
    hits = np.zeros(P, dtype=np.float64)

    for i in range(n_tx):
        s = tx[i]
        tx_dist = np.linalg.norm(pixels - s, axis=1)
        tx_unit = (s - pixels) / (tx_dist[:, None] + 1e-9)  # pixel -> source
        for j in range(n_el):
            e = rx[j]
            rx_dist = np.linalg.norm(pixels - e, axis=1)
            rx_unit = (e - pixels) / (rx_dist[:, None] + 1e-9)  # pixel -> element
            t_round = (tx_dist + rx_dist) / c
            sample_pos = (t_round - t0[i]) * fs

            valid = (sample_pos >= 0) & (sample_pos < n_ax)
            if reject_transmission:
                direct = np.linalg.norm(s - e) / c
                valid = valid & (t_round > direct + transmission_guard_s)

            cos = np.sum(tx_unit * rx_unit, axis=1)
            if backscatter_apodization:
                weight = valid.astype(np.float64) * np.where(cos > 0.0, cos, 0.0)
            else:
                weight = valid.astype(np.float64)

            trace = analytic[i, :, j]
            if interpolation == "nearest":
                idx = np.clip(np.round(sample_pos), 0, n_ax - 1).astype(int)
                amp = trace[idx]
            else:
                i0 = np.floor(sample_pos)
                frac = sample_pos - i0
                i0c = np.clip(i0, 0, n_ax - 1).astype(int)
                i1c = np.clip(i0 + 1, 0, n_ax - 1).astype(int)
                amp = trace[i0c] * (1.0 - frac) + trace[i1c] * frac

            accum += amp * weight
            hits += weight

    return np.abs(accum) / (hits + 1e-6)


def _small_scene(seed=0):
    """A tiny deterministic USCT scene: partial ring geometry + random analytic.

    The geometry, ``fs`` and ``n_ax`` are chosen together so the round-trip
    delays land *inside* the trace (sample positions ~30-50 of 64) with non-zero
    fractional parts. If they fell outside, every gather would be masked out and
    the oracle comparison would be trivially satisfied by two all-zero images.
    """
    rng = np.random.default_rng(seed)
    n_tx, n_ax, n_el = 4, 64, 6
    r = 0.02
    ang_tx = np.linspace(0.2, 2.4, n_tx)
    ang_rx = np.linspace(np.pi - 0.3, np.pi + 1.6, n_el)
    tx = np.stack([r * np.cos(ang_tx), r * np.sin(ang_tx)], axis=-1).astype(np.float32)
    rx = np.stack([r * np.cos(ang_rx), r * np.sin(ang_rx)], axis=-1).astype(np.float32)

    n = 5
    g = np.linspace(-0.005, 0.005, n).astype(np.float32)
    gx, gy = np.meshgrid(g, g)
    pixels = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    analytic = (
        rng.standard_normal((n_tx, n_ax, n_el)) + 1j * rng.standard_normal((n_tx, n_ax, n_el))
    ).astype(np.complex64)

    return dict(
        analytic=analytic, tx=tx, rx=rx, pixels=pixels, fs=1.5e6,
        t0=np.zeros(n_tx, np.float32), c=1500.0, grid_shape=(n, n),
    )


@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
@pytest.mark.parametrize("reject", [True, False])
@pytest.mark.parametrize("apod", [True, False])
def test_usct_das_matches_numpy_oracle(interpolation, reject, apod):
    """The keras implementation reproduces the independent NumPy DAS bit-for-bit
    (to float32 tolerance) across interpolation modes and options."""
    s = _small_scene()
    guard = 1.0e-6

    got = usct_reflectivity_das(
        s["analytic"], s["tx"], s["rx"], s["pixels"], s["fs"], s["t0"], s["c"],
        tx_chunk=2, reject_transmission=reject,
        transmission_guard_s=guard, backscatter_apodization=apod,
        interpolation=interpolation,
    )
    got = np.asarray(ops.convert_to_numpy(got), dtype=np.float64)

    ref = _numpy_usct_das(
        s["analytic"], s["tx"], s["rx"], s["pixels"], s["fs"], s["t0"], s["c"],
        reject_transmission=reject, transmission_guard_s=guard,
        backscatter_apodization=apod, interpolation=interpolation,
    )

    # Guard against a degenerate scene: if the delays fell outside the trace both
    # images would be all-zero and the comparison below would pass vacuously.
    assert np.any(ref > 1e-6), "reference image is all zeros - scene is degenerate"

    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


def test_usct_das_linear_differs_from_nearest():
    """Sanity check that the interpolation modes are actually distinct (guards
    against the linear branch silently collapsing to nearest)."""
    s = _small_scene()
    common = dict(
        tx_chunk=2, reject_transmission=False, backscatter_apodization=False,
    )
    lin = np.asarray(ops.convert_to_numpy(usct_reflectivity_das(
        s["analytic"], s["tx"], s["rx"], s["pixels"], s["fs"], s["t0"], s["c"],
        interpolation="linear", **common)))
    near = np.asarray(ops.convert_to_numpy(usct_reflectivity_das(
        s["analytic"], s["tx"], s["rx"], s["pixels"], s["fs"], s["t0"], s["c"],
        interpolation="nearest", **common)))
    assert not np.allclose(lin, near, atol=1e-3)


def test_usct_das_point_scatterer_peaks_at_true_location():
    """A single point scatterer on a full ring reconstructs to a peak at its
    true location (validates the RF->analytic Hilbert path and the physics).

    Backscatter apodization is deliberately **off** here: it weights pairs by the
    pixel-referred tx/rx cosine, which models a *specular* reflector (the skin
    boundary of a phantom). An ideal point scatterer radiates omnidirectionally,
    so that weighting is the wrong model for this target and suppresses the very
    peak we are checking. Apodization correctness is covered by the oracle test.
    """
    n_el = 24
    r = 0.05
    ang = np.linspace(0, 2 * np.pi, n_el, endpoint=False)
    elems = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1).astype(np.float32)
    tx = elems  # full synthetic-transmit-aperture: every element transmits
    rx = elems

    c, fs = 1500.0, 20e6
    scat = np.array([0.012, -0.006], dtype=np.float32)  # true scatterer location

    # Round-trip delays for the scatterer, over all tx/rx pairs.
    tx_d = np.linalg.norm(scat[None, :] - tx, axis=1)  # (n_tx,)
    rx_d = np.linalg.norm(scat[None, :] - rx, axis=1)  # (n_el,)
    tau = (tx_d[:, None] + rx_d[None, :]) / c           # (n_tx, n_el)
    centres = tau * fs
    n_ax = int(centres.max()) + 40

    # Build RF: a short Gaussian-modulated cosine pulse at each round-trip delay.
    t_idx = np.arange(n_ax)[None, None, :]              # (1,1,n_ax)
    ctr = centres[:, :, None]                           # (n_tx,n_el,1)
    sigma, f0 = 2.5, 3e6
    env = np.exp(-((t_idx - ctr) ** 2) / (2 * sigma ** 2))
    rf = env * np.cos(2 * np.pi * f0 * (t_idx - ctr) / fs)
    rf = np.transpose(rf, (0, 2, 1))[..., None].astype(np.float32)  # (n_tx,n_ax,n_el,1)

    analytic = channels_to_analytic(rf, axis=1)

    n = 64
    half = 0.03
    g = np.linspace(-half, half, n).astype(np.float32)
    gx, gy = np.meshgrid(g, g)
    pixels = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    img = usct_reflectivity_das(
        analytic, tx, rx, pixels, fs, np.zeros(n_el, np.float32), c,
        tx_chunk=6, reject_transmission=False, backscatter_apodization=False,
        interpolation="linear",
    )
    # The DAS returns one value per pixel; reshape to the grid to locate the peak.
    img = np.asarray(ops.convert_to_numpy(img)).reshape(n, n)

    peak = np.unravel_index(np.argmax(img), img.shape)
    peak_xy = np.array([g[peak[1]], g[peak[0]]])  # (col->x, row->y)
    err_mm = np.linalg.norm(peak_xy - scat) * 1e3
    # Grid pitch is ~0.94 mm; require the peak within ~2 pixels of truth.
    assert err_mm < 2.0, f"peak at {peak_xy} is {err_mm:.2f} mm from {scat}"


def _to_xz(points, y=0.0):
    """Embed in-plane (x, z) points into zea's (x, y, z) convention."""
    out = np.zeros((*points.shape[:-1], 3), np.float32)
    out[..., 0] = points[..., 0]
    out[..., 1] = y
    out[..., 2] = points[..., 1]
    return out


def test_usct_operation_matches_functional():
    """The registered USCTReflectivityDAS operation produces the same image as a
    direct call to the functional core (guards the op wrapper / key plumbing).

    The op takes zea's standard 3-D ``grid`` / ``probe_geometry`` /
    ``transmit_origins`` and projects them onto the XZ imaging plane itself, so
    the same scene expressed in-plane must go through the functional core
    unchanged.
    """
    from zea.ops import USCTReflectivityDAS

    s = _small_scene(seed=3)
    # RF input (n_ch == 1) so the op exercises channels_to_analytic internally.
    rng = np.random.default_rng(7)
    n_tx, n_ax, n_el = 4, 60, 6
    rf = rng.standard_normal((n_tx, n_ax, n_el, 1)).astype(np.float32)

    op = USCTReflectivityDAS(
        tx_chunk=2, reject_transmission=True, transmission_guard_s=1e-6,
        backscatter_apodization=True, interpolation="linear",
    )
    out = op(
        data=rf,
        flatgrid=_to_xz(s["pixels"]),
        probe_geometry=_to_xz(s["rx"]),
        transmit_origins=_to_xz(s["tx"]),
        sampling_frequency=s["fs"], initial_times=s["t0"], sound_speed=s["c"],
    )[op.output_key]
    out = np.asarray(ops.convert_to_numpy(out))

    analytic = channels_to_analytic(rf, axis=1)
    ref = usct_reflectivity_das(
        analytic, s["tx"], s["rx"], s["pixels"], s["fs"], s["t0"], s["c"],
        tx_chunk=2, reject_transmission=True,
        transmission_guard_s=1e-6, backscatter_apodization=True,
        interpolation="linear",
    )
    ref = np.asarray(ops.convert_to_numpy(ref))
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_usct_operation_ignores_elevation():
    """``y`` is the out-of-plane axis: shifting the whole scene along it must not
    change the image. Pins the XZ imaging-plane convention (a probe stored in the
    XY plane would silently reconstruct nothing)."""
    from zea.ops import USCTReflectivityDAS

    s = _small_scene(seed=5)
    rng = np.random.default_rng(11)
    rf = rng.standard_normal((4, 60, 6, 1)).astype(np.float32)
    op = USCTReflectivityDAS(tx_chunk=2, transmission_guard_s=1e-6)

    def run(y):
        return np.asarray(ops.convert_to_numpy(op(
            data=rf,
            flatgrid=_to_xz(s["pixels"], y=y),
            probe_geometry=_to_xz(s["rx"], y=y),
            transmit_origins=_to_xz(s["tx"], y=y),
            sampling_frequency=s["fs"], initial_times=s["t0"], sound_speed=s["c"],
        )[op.output_key]))

    np.testing.assert_allclose(run(0.0), run(0.03), rtol=1e-5, atol=1e-5)
