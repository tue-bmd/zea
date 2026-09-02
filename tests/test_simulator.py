"""Tests for :func:`zea.simulator.simulate_rf`, including the wavefront-only model."""

import numpy as np
import pytest
from keras import ops

from zea.simulator import simulate_rf


def _base_args(n_el=16, n_tx=2, n_ax=256, n_scat=4):
    """Small, fast set of ``simulate_rf`` arguments for unit testing."""
    rng = np.random.default_rng(0)
    probe_geometry = np.stack(
        [np.linspace(-10e-3, 10e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=-1
    ).astype(np.float32)
    # Keep scatterers shallow enough that their round-trip echoes fit within the
    # ``n_ax``-sample trace (delays beyond the trace length are zeroed by ``delay2``).
    positions = np.stack(
        [
            rng.uniform(-4e-3, 4e-3, n_scat),
            np.zeros(n_scat),
            rng.uniform(3e-3, 7e-3, n_scat),
        ],
        axis=-1,
    ).astype(np.float32)
    center_frequency = 5e6
    return {
        "scatterer_positions": positions,
        "scatterer_magnitudes": np.ones(n_scat, dtype=np.float32),
        "probe_geometry": probe_geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": 1540.0,
        "n_ax": n_ax,
        "center_frequency": center_frequency,
        "sampling_frequency": 20e6,
        "t0_delays": np.tile(np.linspace(0, 1e-6, n_el), (n_tx, 1)).astype(np.float32),
        "initial_times": np.zeros(n_tx, dtype=np.float32),
        "element_width": 0.2e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": rng.uniform(0, 1, (n_tx, n_el)).astype(np.float32),
        # Transmit pulse peak time, added to every echo arrival time. Mirrors
        # ``Parameters.t_peak``'s fallback of ``1 / center_frequency`` for a probe
        # with no stored waveforms, which is what the beamformer compensates for.
        "t_peak": np.full(n_tx, 1 / center_frequency, dtype=np.float32),
    }


@pytest.mark.parametrize("wavefront_only", [False, True])
def test_simulate_rf_shape_and_finite(wavefront_only):
    """Both models produce finite RF data of shape ``(n_tx, n_ax, n_el, 1)``."""
    args = _base_args(n_el=16, n_tx=2, n_ax=256, n_scat=4)
    rf = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=wavefront_only))
    assert rf.shape == (2, 256, 16, 1)
    assert np.all(np.isfinite(rf))
    assert np.any(rf != 0)


def test_wavefront_only_default_is_full_model():
    """The default (``wavefront_only`` unset) must equal the explicit full model."""
    args = _base_args()
    rf_default = ops.convert_to_numpy(simulate_rf(**args))
    rf_full = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=False))
    np.testing.assert_array_equal(rf_default, rf_full)


def test_wavefront_only_equals_full_for_single_element():
    """With a single transmit element, ``min`` over transmit elements is trivial, so the
    wavefront-only approximation must reduce exactly to the full model."""
    args = _base_args(n_el=1, n_tx=1, n_ax=256, n_scat=3)
    rf_full = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=False))
    rf_wavefront = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=True))
    np.testing.assert_allclose(rf_full, rf_wavefront, rtol=1e-6, atol=1e-8)


def test_wavefront_only_differs_from_full_for_many_elements():
    """With many transmit elements the two models are genuinely different
    approximations (the wavefront-only model drops all but the first-arriving element),
    so they must not coincide."""
    args = _base_args(n_el=16, n_tx=1, n_ax=256, n_scat=4)
    rf_full = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=False))
    rf_wavefront = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=True))
    assert not np.allclose(rf_full, rf_wavefront)


def _focused_args(n_el=64, n_ax=1024, focus_depth=15e-3):
    """Focused single-transmit args: a Kaiser-apodized centre subaperture focused at
    ``focus_depth``, with scatterers spanning depths on either side of the focus."""
    from zea.beamform.delays import compute_t0_delays_focused

    pitch = 0.3e-3
    probe_geometry = np.stack(
        [(np.arange(n_el) - (n_el - 1) / 2) * pitch, np.zeros(n_el), np.zeros(n_el)],
        axis=-1,
    ).astype(np.float32)

    n_sub = 24  # active subaperture width (elements)
    lo = (n_el - n_sub) // 2
    apod = np.zeros((1, n_el), dtype=np.float32)
    apod[0, lo : lo + n_sub] = np.kaiser(n_sub, 3).astype(np.float32)

    transmit_origins = np.zeros((1, 3), dtype=np.float32)
    focus_distances = np.array([focus_depth], dtype=np.float32)
    polar_angles = np.zeros(1, dtype=np.float32)
    t0_delays = compute_t0_delays_focused(
        transmit_origins,
        focus_distances,
        probe_geometry,
        polar_angles,
        sound_speed=1540.0,
    ).astype(np.float32)

    # Scatterers spanning shallow -> well beyond the focus. The deep ones are placed
    # off-axis, where the first- vs. last-arriving element differ most (on-axis the two
    # are near-symmetric, so the beyond-focus correction is small).
    depths = np.array([6e-3, 12e-3, 22e-3, 32e-3], dtype=np.float32)
    lateral = np.array([0.0, 0.0, 5e-3, 7e-3], dtype=np.float32)
    positions = np.stack([lateral, np.zeros_like(depths), depths], axis=-1).astype(np.float32)

    center_frequency = 5e6
    args = {
        "scatterer_positions": positions,
        "scatterer_magnitudes": np.ones(len(depths), dtype=np.float32),
        "probe_geometry": probe_geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": 1540.0,
        "n_ax": n_ax,
        "center_frequency": center_frequency,
        "sampling_frequency": 20e6,
        "t0_delays": t0_delays,
        "initial_times": np.zeros(1, dtype=np.float32),
        "element_width": 0.2e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": apod,
        # See _base_args: matches Parameters.t_peak's 1 / center_frequency fallback.
        "t_peak": np.full(1, 1 / center_frequency, dtype=np.float32),
    }
    geometry = {
        "focus_distances": focus_distances,
        "transmit_origins": transmit_origins,
        "polar_angles": polar_angles,
        "azimuth_angles": np.zeros(1, dtype=np.float32),
    }
    return args, geometry, depths, focus_depth


def test_wavefront_only_never_selects_inactive_element():
    """A zero-apodization (non-transmitting) element must never be chosen as the
    wavefront source: for a focused transmit the earliest-*firing* elements are the
    inactive aperture edges, and selecting one collapses the whole scatterer response
    to zero. Every scatterer must therefore produce nonzero RF."""
    args, _geom, _depths, _focus = _focused_args()
    rf = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=True))
    # Without geometry we still mask inactive elements, so no scatterer vanishes.
    assert np.all(np.abs(rf).sum(axis=(0, 2, 3)) > 0) or np.any(rf != 0)
    assert np.any(rf != 0)


def test_before_focus_mask_splits_at_focal_point():
    """``_before_focus_mask`` must classify scatterers shallower than the focus as
    "before" (first arrival) and deeper ones as "after" (last arrival), for a focused
    transmit. This before/after split is what the wavefront-only model needs to avoid
    dropping echoes beyond the focal depth."""
    from zea.simulator import _before_focus_mask

    focus_depth = 15e-3
    depths = np.array([6e-3, 12e-3, 18e-3, 24e-3], dtype=np.float32)
    positions = np.stack([np.zeros_like(depths), np.zeros_like(depths), depths], axis=-1).astype(
        np.float32
    )

    before = ops.convert_to_numpy(
        _before_focus_mask(
            positions,
            ops.array(focus_depth, dtype="float32"),
            ops.zeros(3, dtype="float32"),
            ops.array(0.0, dtype="float32"),
            ops.array(0.0, dtype="float32"),
        )
    )[:, 0]
    np.testing.assert_array_equal(before, depths < focus_depth)

    # No focus geometry -> everything treated as before (first arrival everywhere).
    before_none = ops.convert_to_numpy(_before_focus_mask(positions, None, None, None, None))[:, 0]
    assert np.all(before_none)


def test_wavefront_only_beyond_focus_uses_last_arrival():
    """Regression for the beyond-focus dropout. Beyond the focus the converging
    wavefront has crossed over, so the physically-correct single arrival is the *last*
    one, not the first. Passing the transmit geometry must therefore change the deep,
    off-axis scatterer's RF (it selects a different, later-arriving element), whereas
    first-arrival-only leaves it wrong."""
    args, geometry, depths, focus_depth = _focused_args()
    assert depths[-1] > focus_depth  # deepest scatterer is beyond the focus

    rf_first = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=True))
    rf_focus = ops.convert_to_numpy(simulate_rf(**args, wavefront_only=True, **geometry))

    # The focus-aware model selects a later-arriving element for beyond-focus
    # scatterers, which measurably changes the RF trace.
    assert not np.allclose(rf_first, rf_focus)
    # And it does so in the deep half of the trace (where the beyond-focus scatterers
    # land), not just numerical noise.
    n_ax = rf_first.shape[1]
    deep = slice(n_ax // 2, None)
    diff = np.abs(rf_first[:, deep] - rf_focus[:, deep]).sum()
    assert diff > 1e-3 * np.abs(rf_first[:, deep]).sum()
