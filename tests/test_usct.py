"""Regression tests for the USCT reflectivity DAS reconstruction.

USCT is not covered by zea's standard beamformer, so its reconstruction lives in
:func:`zea.func.usct.usct_reflectivity_das` (wrapped by the
:class:`zea.ops.USCTReflectivityDAS` operation). The DAS algorithm (round-trip
time-of-flight delay-and-sum with through-transmission rejection and backscatter
apodization) follows the reflection ultrasound computed tomography (RUCT)
approach described in :mod:`zea.ops.usct`.

These tests cover the functional core and the operation wrapper directly: a
synthetic point scatterer must reconstruct at its true location, each
reconstruction option (interpolation mode, transmission rejection, backscatter
apodization, per-transmit receive apertures, speed-of-sound maps) is checked
for the effect it is documented to have, and the operation wrapper is checked
against the functional core it wraps.
"""

import numpy as np
import pytest
from keras import ops

from zea.func.ultrasound import channels_to_analytic
from zea.func.usct import usct_reflectivity_das


def _small_scene(seed=0):
    """A tiny deterministic USCT scene: partial ring geometry + random analytic.

    The geometry, ``fs`` and ``n_ax`` are chosen together so the round-trip
    delays land *inside* the trace (sample positions ~30-50 of 64) with non-zero
    fractional parts. If they fell outside, every gather would be masked out and
    tests comparing two reconstructions would pass vacuously on two all-zero
    images.
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
        analytic=analytic,
        tx=tx,
        rx=rx,
        pixels=pixels,
        fs=1.5e6,
        t0=np.zeros(n_tx, np.float32),
        c=1500.0,
        grid_shape=(n, n),
    )


def test_usct_das_reject_transmission_masks_near_direct_path():
    """A pixel that sits on the straight line between transmitter and receiver
    has a round-trip delay equal to the direct-arrival delay; with
    ``reject_transmission`` that pair must be masked out entirely."""
    c, fs = 1500.0, 5e6
    tx = np.array([[-0.005, 0.0]], dtype=np.float32)
    rx = np.array([[0.005, 0.0]], dtype=np.float32)
    pixels = np.array([[0.0, 0.0]], dtype=np.float32)  # on the tx-rx segment
    analytic = np.ones((1, 64, 1), dtype=np.complex64)
    t0 = np.zeros(1, dtype=np.float32)
    common = dict(
        tx_chunk=1,
        backscatter_apodization=False,
        interpolation="linear",
        transmission_guard_s=1e-6,
    )

    kept = usct_reflectivity_das(
        analytic, tx, rx, pixels, fs, t0, c, reject_transmission=False, **common
    )
    rejected = usct_reflectivity_das(
        analytic, tx, rx, pixels, fs, t0, c, reject_transmission=True, **common
    )
    kept = float(np.asarray(ops.convert_to_numpy(kept))[0])
    rejected = float(np.asarray(ops.convert_to_numpy(rejected))[0])
    assert kept == pytest.approx(1.0, rel=1e-3)
    assert rejected == pytest.approx(0.0, abs=1e-8)


def test_usct_das_backscatter_apodization_effect():
    """``backscatter_apodization`` weights pairs by the pixel-referred tx/rx
    cosine: it must suppress a pass-through geometry (tx and rx on opposite
    sides of the pixel, cos ~= -1) while keeping a genuine backscatter
    geometry (tx and rx on the same side, cos > 0), and have no effect when
    disabled."""
    c, fs = 1500.0, 5e6
    analytic = np.ones((1, 64, 1), dtype=np.complex64)
    t0 = np.zeros(1, dtype=np.float32)
    pixels = np.array([[0.0, 0.0]], dtype=np.float32)
    common = dict(tx_chunk=1, reject_transmission=False, interpolation="linear")

    # tx and rx on the same side of the pixel: backscatter geometry (cos > 0).
    tx_back = np.array([[-0.004, 0.001]], dtype=np.float32)
    rx_back = np.array([[-0.004, -0.001]], dtype=np.float32)
    # tx and rx on opposite sides of the pixel: pass-through geometry (cos < 0).
    tx_fwd = np.array([[-0.005, 0.0]], dtype=np.float32)
    rx_fwd = np.array([[0.005, 0.0]], dtype=np.float32)

    def image_value(tx, rx, apod):
        out = usct_reflectivity_das(
            analytic, tx, rx, pixels, fs, t0, c, backscatter_apodization=apod, **common
        )
        return float(np.asarray(ops.convert_to_numpy(out))[0])

    assert image_value(tx_back, rx_back, apod=True) == pytest.approx(1.0, rel=1e-2)
    assert image_value(tx_fwd, rx_fwd, apod=True) == pytest.approx(0.0, abs=1e-8)
    # With apodization off, both geometries contribute with full weight.
    assert image_value(tx_back, rx_back, apod=False) == pytest.approx(1.0, rel=1e-2)
    assert image_value(tx_fwd, rx_fwd, apod=False) == pytest.approx(1.0, rel=1e-2)


def test_usct_das_incoherent_compounding_avoids_cross_transmit_cancellation():
    """``compounding="incoherent"`` takes the magnitude per transmit and
    averages those magnitudes across transmits, instead of summing complex
    amplitudes across transmits. A transmit-to-transmit phase flip that
    perfectly cancels the coherent sum must leave the incoherent one at full
    amplitude."""
    c, fs = 1500.0, 5e6
    tx = np.array([[-0.006, 0.0], [-0.006, 0.002]], dtype=np.float32)
    rx = np.array([[0.006, 0.0]], dtype=np.float32)
    pixels = np.array([[0.0, 0.0]], dtype=np.float32)
    t0 = np.zeros(2, dtype=np.float32)

    analytic = np.ones((2, 64, 1), dtype=np.complex64)
    analytic[1] *= -1.0  # second transmit's trace is exactly out of phase

    common = dict(
        tx_chunk=1,
        reject_transmission=False,
        backscatter_apodization=False,
        interpolation="linear",
    )
    coherent = usct_reflectivity_das(
        analytic, tx, rx, pixels, fs, t0, c, compounding="coherent", **common
    )
    incoherent = usct_reflectivity_das(
        analytic, tx, rx, pixels, fs, t0, c, compounding="incoherent", **common
    )
    coherent = float(np.asarray(ops.convert_to_numpy(coherent))[0])
    incoherent = float(np.asarray(ops.convert_to_numpy(incoherent))[0])
    assert coherent == pytest.approx(0.0, abs=1e-6)
    assert incoherent == pytest.approx(1.0, rel=1e-3)


def test_usct_das_invalid_compounding_raises():
    """An unrecognized ``compounding`` value must fail loudly."""
    s = _small_scene()
    with pytest.raises(ValueError, match="compounding"):
        usct_reflectivity_das(
            s["analytic"],
            s["tx"],
            s["rx"],
            s["pixels"],
            s["fs"],
            s["t0"],
            s["c"],
            tx_chunk=2,
            compounding="banana",
        )


def test_usct_das_linear_differs_from_nearest():
    """Sanity check that the interpolation modes are actually distinct (guards
    against the linear branch silently collapsing to nearest)."""
    s = _small_scene()
    common = dict(
        tx_chunk=2,
        reject_transmission=False,
        backscatter_apodization=False,
    )
    lin = np.asarray(
        ops.convert_to_numpy(
            usct_reflectivity_das(
                s["analytic"],
                s["tx"],
                s["rx"],
                s["pixels"],
                s["fs"],
                s["t0"],
                s["c"],
                interpolation="linear",
                **common,
            )
        )
    )
    near = np.asarray(
        ops.convert_to_numpy(
            usct_reflectivity_das(
                s["analytic"],
                s["tx"],
                s["rx"],
                s["pixels"],
                s["fs"],
                s["t0"],
                s["c"],
                interpolation="nearest",
                **common,
            )
        )
    )
    assert not np.allclose(lin, near, atol=1e-3)


def test_usct_das_invalid_interpolation_raises():
    """An unrecognized ``interpolation`` value must fail loudly rather than
    silently falling back to a default."""
    s = _small_scene()
    with pytest.raises(ValueError, match="interpolation"):
        usct_reflectivity_das(
            s["analytic"],
            s["tx"],
            s["rx"],
            s["pixels"],
            s["fs"],
            s["t0"],
            s["c"],
            tx_chunk=2,
            interpolation="cubic",
        )


def test_usct_channels_to_analytic_invalid_n_ch_raises():
    """A channel count other than 1 (RF) or 2 (I/Q) must fail loudly."""
    with pytest.raises(ValueError, match="n_ch"):
        channels_to_analytic(np.zeros((2, 8, 3), dtype=np.float32), axis=1)


def test_usct_channels_to_analytic_iq_path():
    """Two-channel I/Q data is read directly as real/imaginary parts, with no
    Hilbert transform involved (the RF path is exercised elsewhere)."""
    rng = np.random.default_rng(21)
    iq = rng.standard_normal((3, 8, 4, 2)).astype(np.float32)
    got = np.asarray(ops.convert_to_numpy(channels_to_analytic(iq, axis=1)))
    expected = iq[..., 0] + 1j * iq[..., 1]
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_usct_das_per_tx_rx_matches_shared_aperture():
    """A per-transmit receive aperture (``receive_positions`` of shape
    ``(n_tx, n_el, 2)``) that happens to be identical for every transmit must
    reproduce the shared-aperture (``(n_el, 2)``) result. Exercises the
    per_tx_rx branch, which the other tests never touch since they all pass a
    shared receive aperture."""
    s = _small_scene(seed=9)
    n_tx = s["tx"].shape[0]
    rx_per_tx = np.broadcast_to(s["rx"], (n_tx, *s["rx"].shape)).copy()

    common = dict(
        tx_chunk=2,
        reject_transmission=True,
        transmission_guard_s=1e-6,
        backscatter_apodization=True,
        interpolation="linear",
    )
    shared = usct_reflectivity_das(
        s["analytic"],
        s["tx"],
        s["rx"],
        s["pixels"],
        s["fs"],
        s["t0"],
        s["c"],
        **common,
    )
    per_tx = usct_reflectivity_das(
        s["analytic"],
        s["tx"],
        rx_per_tx,
        s["pixels"],
        s["fs"],
        s["t0"],
        s["c"],
        **common,
    )
    shared = np.asarray(ops.convert_to_numpy(shared))
    per_tx = np.asarray(ops.convert_to_numpy(per_tx))
    np.testing.assert_allclose(shared, per_tx, rtol=1e-5, atol=1e-5)


def test_usct_straight_ray_times_matches_constant_speed():
    """Straight-ray integration through a uniform SoS map (matching the
    background speed) must reduce to the plain straight-line travel time,
    both inside and outside the map footprint."""
    from zea.func.usct import straight_ray_times

    c = 1500.0
    x_axis = np.linspace(-0.01, 0.01, 21).astype(np.float32)
    z_axis = np.linspace(-0.01, 0.01, 21).astype(np.float32)
    sos_map = np.full((21, 21), c, dtype=np.float32)

    # One position inside the map footprint, one outside (exercises the
    # background_c fallback).
    positions = np.array([[0.003, -0.004], [0.02, 0.0]], dtype=np.float32)
    pixels = np.array([[0.0, 0.0], [0.003, -0.002]], dtype=np.float32)

    times = straight_ray_times(positions, pixels, sos_map, x_axis, z_axis, c, n_samples=8)
    times = np.asarray(ops.convert_to_numpy(times))

    dist = np.linalg.norm(positions[:, None, :] - pixels[None, :, :], axis=-1)
    np.testing.assert_allclose(times, dist / c, rtol=1e-5, atol=1e-8)


def test_usct_das_sos_map_matches_constant_speed_when_uniform():
    """A uniform SoS map exactly matching the background speed must give the
    same image as the plain constant-speed path, validating the straight-ray
    wiring inside ``usct_reflectivity_das`` end to end."""
    s = _small_scene(seed=13)
    c = s["c"]
    margin = 0.01
    x_axis = np.linspace(-margin, margin, 33).astype(np.float32)
    z_axis = np.linspace(-margin, margin, 33).astype(np.float32)
    sos_map = np.full((33, 33), c, dtype=np.float32)

    common = dict(
        tx_chunk=2,
        reject_transmission=True,
        transmission_guard_s=1e-6,
        backscatter_apodization=True,
        interpolation="linear",
    )
    no_sos = usct_reflectivity_das(
        s["analytic"],
        s["tx"],
        s["rx"],
        s["pixels"],
        s["fs"],
        s["t0"],
        c,
        **common,
    )
    with_sos = usct_reflectivity_das(
        s["analytic"],
        s["tx"],
        s["rx"],
        s["pixels"],
        s["fs"],
        s["t0"],
        c,
        **common,
        sos_map=sos_map,
        sos_grid_x=x_axis,
        sos_grid_z=z_axis,
        n_sos_ray_samples=32,
    )
    no_sos = np.asarray(ops.convert_to_numpy(no_sos))
    with_sos = np.asarray(ops.convert_to_numpy(with_sos))
    np.testing.assert_allclose(no_sos, with_sos, rtol=1e-3, atol=1e-3)


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
    tau = (tx_d[:, None] + rx_d[None, :]) / c  # (n_tx, n_el)
    centres = tau * fs
    n_ax = int(centres.max()) + 40

    # Build RF: a short Gaussian-modulated cosine pulse at each round-trip delay.
    t_idx = np.arange(n_ax)[None, None, :]  # (1,1,n_ax)
    ctr = centres[:, :, None]  # (n_tx,n_el,1)
    sigma, f0 = 2.5, 3e6
    env = np.exp(-((t_idx - ctr) ** 2) / (2 * sigma**2))
    rf = env * np.cos(2 * np.pi * f0 * (t_idx - ctr) / fs)
    rf = np.transpose(rf, (0, 2, 1))[..., None].astype(np.float32)  # (n_tx,n_ax,n_el,1)

    analytic = channels_to_analytic(rf, axis=1)

    n = 64
    half = 0.03
    g = np.linspace(-half, half, n).astype(np.float32)
    gx, gy = np.meshgrid(g, g)
    pixels = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    img = usct_reflectivity_das(
        analytic,
        tx,
        rx,
        pixels,
        fs,
        np.zeros(n_el, np.float32),
        c,
        tx_chunk=6,
        reject_transmission=False,
        backscatter_apodization=False,
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
        tx_chunk=2,
        reject_transmission=True,
        transmission_guard_s=1e-6,
        backscatter_apodization=True,
        interpolation="linear",
        compounding="incoherent",
    )
    out = op(
        data=rf,
        flatgrid=_to_xz(s["pixels"]),
        probe_geometry=_to_xz(s["rx"]),
        transmit_origins=_to_xz(s["tx"]),
        sampling_frequency=s["fs"],
        initial_times=s["t0"],
        sound_speed=s["c"],
    )[op.output_key]
    out = np.asarray(ops.convert_to_numpy(out))

    analytic = channels_to_analytic(rf, axis=1)
    ref = usct_reflectivity_das(
        analytic,
        s["tx"],
        s["rx"],
        s["pixels"],
        s["fs"],
        s["t0"],
        s["c"],
        tx_chunk=2,
        reject_transmission=True,
        transmission_guard_s=1e-6,
        backscatter_apodization=True,
        interpolation="linear",
        compounding="incoherent",
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

    def run(y_grid, y_probe, y_tx):
        return np.asarray(
            ops.convert_to_numpy(
                op(
                    data=rf,
                    flatgrid=_to_xz(s["pixels"], y=y_grid),
                    probe_geometry=_to_xz(s["rx"], y=y_probe),
                    transmit_origins=_to_xz(s["tx"], y=y_tx),
                    sampling_frequency=s["fs"],
                    initial_times=s["t0"],
                    sound_speed=s["c"],
                )[op.output_key]
            )
        )

    # Distinct, non-matching elevation offsets per entity: a bug that fails to
    # drop the y-axis (e.g. computing 3-D instead of in-plane distances) would
    # change the pairwise geometry here, unlike a single shared offset applied
    # to every entity (which cancels out in all pairwise differences).
    np.testing.assert_allclose(run(0.0, 0.0, 0.0), run(0.02, -0.05, 0.03), rtol=1e-5, atol=1e-5)
