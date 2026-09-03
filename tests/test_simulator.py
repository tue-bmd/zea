"""
Simulate and beamform the fish phantom with a planewave, then check that every scatterer shows
up as a bright dot. For approximations, check that the image is close to the exact mode results.
"""

import keras
import numpy as np
import pytest

import zea
from zea import Parameters, Probe, display
from zea.beamform import phantoms
from zea.beamform.delays import compute_t0_delays_planewave
from zea.metrics import psnr
from zea.simulator import (
    elevation_slab_bucket,
    select_elevation_slab,
    simulate_rf,
)
from zea.ops import Simulate
from zea.simulator_time_domain import simulate_rf_td

N_EL = 80
APERTURE = 32e-3
N_TX = 5
CENTER_FREQUENCY = 3e6  # Hz
SOUND_SPEED = 1540.0  # m/s
XLIMS = (-24e-3, 24e-3)
ZLIMS = (10e-3, 35e-3)
DYNAMIC_RANGE = (-50.0, 0.0)


def _parameters(probe_geometry):
    angles = np.linspace(-15, 15, N_TX) * np.pi / 180
    wavelength = SOUND_SPEED / CENTER_FREQUENCY
    return Parameters(
        n_tx=N_TX,
        n_el=N_EL,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=CENTER_FREQUENCY * 4,
        probe_geometry=probe_geometry,
        t0_delays=compute_t0_delays_planewave(
            probe_geometry=probe_geometry, polar_angles=angles, sound_speed=SOUND_SPEED
        ),
        tx_apodizations=np.ones((N_TX, N_EL)) * np.hanning(N_EL)[None],
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=np.ones(N_TX) * np.inf,
        polar_angles=angles,
        initial_times=np.ones(N_TX) * 1e-6,
        n_ax=1024,
        xlims=XLIMS,
        zlims=ZLIMS,
        grid_size_x=int((XLIMS[1] - XLIMS[0]) / (0.5 * wavelength)) + 1,
        grid_size_z=int((ZLIMS[1] - ZLIMS[0]) / (0.5 * wavelength)) + 1,
        lens_sound_speed=1000,
        lens_thickness=1e-3,
        n_ch=1,
        selected_transmits="all",
        sound_speed=SOUND_SPEED,
        apply_lens_correction=False,
        attenuation_coef=0.0,
    )


@pytest.fixture(scope="module")
def fish_scan():
    probe_geometry = np.stack(
        [np.linspace(-APERTURE / 2, APERTURE / 2, N_EL), np.zeros(N_EL), np.zeros(N_EL)], axis=1
    )
    probe = Probe(probe_geometry=probe_geometry, probe_center_frequency=CENTER_FREQUENCY)
    parameters = _parameters(probe_geometry)
    positions = phantoms.fish()

    simulation_args = {
        "scatterer_positions": positions,
        "scatterer_magnitudes": np.ones(len(positions), dtype=np.float32),
        "probe_geometry": probe.probe_geometry,
        "apply_lens_correction": parameters.apply_lens_correction,
        "lens_thickness": parameters.lens_thickness,
        "lens_sound_speed": parameters.lens_sound_speed,
        "sound_speed": parameters.sound_speed,
        "n_ax": parameters.n_ax,
        "center_frequency": probe.probe_center_frequency,
        "sampling_frequency": parameters.sampling_frequency,
        "t0_delays": parameters.t0_delays,
        "initial_times": parameters.initial_times,
        "element_width": parameters.element_width,
        "attenuation_coef": parameters.attenuation_coef,
        "tx_apodizations": parameters.tx_apodizations,
        "t_peak": parameters.t_peak,
    }

    pipeline = zea.Pipeline.from_default(enable_pfield=False, with_batch_dim=False, baseband=False)
    inputs = pipeline.prepare_parameters(parameters, dynamic_range=DYNAMIC_RANGE)

    def beamform(rf_data):
        """Beamformed 8-bit B-mode image, as in the simulation example notebook."""
        kwargs = {**inputs, pipeline.key: rf_data}
        image = pipeline(**kwargs)[pipeline.output_key]
        return np.asarray(display.to_8bit(image, dynamic_range=DYNAMIC_RANGE), dtype=np.float32)

    return positions, simulation_args, beamform


@pytest.fixture(scope="module")
def images(fish_scan):
    _, simulation_args, beamform = fish_scan
    return {
        "exact": beamform(simulate_rf(**simulation_args)),
        "fast": beamform(simulate_rf_td(**simulation_args)),
    }


def _dot_brightness(image, positions):
    z = np.linspace(ZLIMS[0], ZLIMS[1], image.shape[0])
    x = np.linspace(XLIMS[0], XLIMS[1], image.shape[1])

    values = []
    for pos_x, _, pos_z in positions:
        values.append(image[np.argmin(np.abs(z - pos_z)), np.argmin(np.abs(x - pos_x))])
    return np.array(values)


@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_simulated_fish_dots_visible(fish_scan, images, mode):
    positions, _, _ = fish_scan
    image = images[mode]
    image_norm = image / image.mean()
    values = _dot_brightness(image_norm, positions)

    darkest = values.min()
    assert darkest > 2.0, (
        f"{mode} simulator: dimmest scatterer is only {darkest:.1f}x the average brightness"
    )


@pytest.mark.parametrize("mode", ["fast"])
def test_simulator_mode_psnr_against_exact(images, mode):
    """The approximate modes stay close to the exact simulator after beamforming."""
    value = float(psnr(images["exact"][..., None], images[mode][..., None], max_val=255))
    min_psnr = 20.0
    assert value > min_psnr, (
        f"{mode} mode: PSNR against `exact` is low! {value:.1f} dB, expected {min_psnr:.0f} dB"
    )


def test_elevation_lens_prunes_out_of_plane_scatterers():
    n_el = 16
    probe_geometry = np.stack(
        [np.linspace(-8e-3, 8e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
    element_height = 5e-3
    args = {
        "scatterer_magnitudes": np.ones(1, dtype=np.float32),
        "probe_geometry": probe_geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": 1024,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": CENTER_FREQUENCY * 4,
        "t0_delays": np.zeros((1, n_el), dtype=np.float32),
        "initial_times": np.zeros(1, dtype=np.float32),
        "element_width": 1e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": np.ones((1, n_el), dtype=np.float32),
        "t_peak": np.full(1, 1 / CENTER_FREQUENCY, dtype=np.float32),
        "elevation_lens": True,
        "element_height": element_height,
    }

    inside = np.array([[0.0, 0.5 * element_height, 30e-3]], dtype=np.float32)
    outside = np.array([[0.0, 1.5 * element_height, 30e-3]], dtype=np.float32)
    assert (
        np.abs(keras.ops.convert_to_numpy(simulate_rf(scatterer_positions=inside, **args))).max()
        > 0
    )
    assert (
        np.abs(keras.ops.convert_to_numpy(simulate_rf(scatterer_positions=outside, **args))).max()
        == 0
    )

    both = np.concatenate([inside, outside], axis=0)
    positions, magnitudes = select_elevation_slab(
        both, np.ones(2, dtype=np.float32), probe_geometry, element_height
    )
    assert positions.shape == (1, 3)
    assert magnitudes.shape == (1,)

    # Check if pruning and masking results in the same rf.
    args["scatterer_magnitudes"] = np.ones(2, dtype=np.float32)
    pruned = keras.ops.convert_to_numpy(simulate_rf(scatterer_positions=both, **args))
    args["scatterer_magnitudes"] = np.ones(1, dtype=np.float32)
    reference = keras.ops.convert_to_numpy(simulate_rf(scatterer_positions=inside, **args))
    assert np.allclose(pruned, reference)


def _slab_cloud(n_inside, n_outside, element_height, seed=0):
    """A cloud split into scatterers inside and outside the elevation slab."""
    rng = np.random.default_rng(seed)
    n = n_inside + n_outside
    y = np.concatenate(
        [
            rng.uniform(-0.4, 0.4, n_inside) * element_height,
            rng.uniform(1.5, 3.0, n_outside) * element_height,
        ]
    )
    positions = np.stack(
        [rng.uniform(-5e-3, 5e-3, n), y, rng.uniform(15e-3, 30e-3, n)], axis=1
    ).astype(np.float32)
    return positions, rng.uniform(0.5, 1.5, n).astype(np.float32)


def test_elevation_slab_bucket_rounds_up_and_is_a_noop_when_inapplicable():
    n_el = 16
    probe_geometry = np.stack(
        [np.linspace(-8e-3, 8e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
    element_height = 5e-3
    kwargs = {
        "probe_geometry": probe_geometry,
        "element_height": element_height,
        "elevation_lens": True,
    }

    for n_inside in (5000, 7000):
        positions, magnitudes = _slab_cloud(n_inside, 20000 - n_inside, element_height)
        out = elevation_slab_bucket(
            scatterer_positions=positions, scatterer_magnitudes=magnitudes, **kwargs
        )
        assert out["scatterer_positions"].shape == (8192, 3)
        assert int((out["scatterer_magnitudes"] > 0).sum()) == n_inside

    positions, magnitudes = _slab_cloud(100, 900, element_height)
    no_lens = {**kwargs, "elevation_lens": False}
    no_height = {**kwargs, "element_height": None}
    lensless_bucket = elevation_slab_bucket(
        scatterer_positions=positions, scatterer_magnitudes=magnitudes, **no_lens
    )
    heightless_bucket = elevation_slab_bucket(
        scatterer_positions=positions, scatterer_magnitudes=magnitudes, **no_height
    )
    assert lensless_bucket == {}
    assert heightless_bucket == {}
    # Check that passing irrelevant simulator params do not raise errors.
    assert elevation_slab_bucket(
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        sound_speed=SOUND_SPEED,
        n_ax=1024,
        **kwargs,
    )


def test_elevation_slab_bucket_matches_unpruned_simulation():
    """Pruning to a padded bucket must not change the RF."""
    n_el = 16
    probe_geometry = np.stack(
        [np.linspace(-8e-3, 8e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
    element_height = 5e-3
    positions, magnitudes = _slab_cloud(20, 300, element_height)

    args = {
        "probe_geometry": probe_geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": 1024,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": CENTER_FREQUENCY * 4,
        "t0_delays": np.zeros((1, n_el), dtype=np.float32),
        "initial_times": np.zeros(1, dtype=np.float32),
        "element_width": 1e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": np.ones((1, n_el), dtype=np.float32),
        "t_peak": np.full(1, 1 / CENTER_FREQUENCY, dtype=np.float32),
        "elevation_lens": True,
        "element_height": element_height,
    }

    pruned = elevation_slab_bucket(
        scatterer_positions=positions, scatterer_magnitudes=magnitudes, **args
    )
    assert pruned["scatterer_positions"].shape[0] == 32

    reference = keras.ops.convert_to_numpy(
        simulate_rf(scatterer_positions=positions, scatterer_magnitudes=magnitudes, **args)
    )
    bucketed = keras.ops.convert_to_numpy(simulate_rf(**pruned, **args))
    assert np.allclose(reference, bucketed, atol=1e-3 * np.abs(reference).max())


def test_simulate_op_prunes_elevation_slab_without_leaking_pruned_cloud():
    """The `Simulate` op prunes before its jitted `call`, but must hand the full cloud on."""
    n_el = 16
    probe_geometry = np.stack(
        [np.linspace(-8e-3, 8e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
    element_height = 5e-3
    positions, magnitudes = _slab_cloud(20, 300, element_height)

    op = zea.ops.Simulate(jit_compile=True, with_batch_dim=False)
    outputs = op(
        scatterer_positions=positions,
        scatterer_magnitudes=magnitudes,
        probe_geometry=probe_geometry,
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        sound_speed=SOUND_SPEED,
        n_ax=1024,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=CENTER_FREQUENCY * 4,
        t0_delays=np.zeros((1, n_el), dtype=np.float32),
        initial_times=np.zeros(1, dtype=np.float32),
        element_width=1e-3,
        attenuation_coef=0.0,
        tx_apodizations=np.ones((1, n_el), dtype=np.float32),
        t_peak=np.full(1, 1 / CENTER_FREQUENCY, dtype=np.float32),
        elevation_lens=True,
        element_height=element_height,
    )

    assert np.abs(keras.ops.convert_to_numpy(outputs[op.output_key])).max() > 0
    assert outputs["scatterer_positions"].shape == positions.shape
    assert outputs["scatterer_magnitudes"].shape == magnitudes.shape


def test_record_length_gate_keeps_in_record_pairs_without_aliasing():
    """Scatterers that fit in the record must not be zeroed, but no aliasing may happen."""
    probe_geometry = np.array([[-8e-3, 0.0, 0.0], [8e-3, 0.0, 0.0]], dtype=np.float32)
    scatterer_positions = np.array([[9.375e-3, 0.0, 9.905e-3]], dtype=np.float32)

    args = {
        "scatterer_positions": scatterer_positions,
        "scatterer_magnitudes": np.ones(1, dtype=np.float32),
        "probe_geometry": probe_geometry,
        "apply_lens_correction": False,
        "lens_thickness": 1e-3,
        "lens_sound_speed": 1000.0,
        "sound_speed": SOUND_SPEED,
        "n_ax": 256,
        "center_frequency": CENTER_FREQUENCY,
        "sampling_frequency": CENTER_FREQUENCY * 4,
        "t0_delays": np.zeros((1, 2), dtype=np.float32),
        "initial_times": np.zeros(1, dtype=np.float32),
        "element_width": 1e-3,
        "attenuation_coef": 0.0,
        "tx_apodizations": np.ones((1, 2), dtype=np.float32),
        "t_peak": np.zeros(1, dtype=np.float32),
    }

    rf = keras.ops.convert_to_numpy(simulate_rf(**args))[0, :, :, 0]
    # Four times the record gates nothing, so it is the un-truncated ground truth.
    reference = keras.ops.convert_to_numpy(simulate_rf(**{**args, "n_ax": 1024}))[0, :256, :, 0]

    near = 1  # element 8 mm from the scatterer, so its own round trip is the 13.0 us pair
    peak = np.abs(rf[:, near]).max()
    assert np.abs(rf[:, near]).argmax() == np.abs(reference[:, near]).argmax(), (
        "In-record pair was gated out or moved."
    )
    assert np.abs(rf[:, near] - reference[:, near]).max() < 1e-3 * peak

    # The 26.0 us pair would land near sample 56; the earliest real arrival is the pulse
    # around sample 156.
    quiet = np.abs(rf[:140]).max()
    assert quiet < 1e-3 * peak, (
        f"Aliased energy detected: {quiet:.3g} should be much less than peak ({peak:.3g})"
    )


def _receive_chain_image(fish_scan, simulator, **receive_chain_kwargs):
    _, simulation_args, beamform = fish_scan
    return beamform(simulator(**simulation_args, noise_seed=0, **receive_chain_kwargs))


@pytest.mark.parametrize("simulator", [simulate_rf, simulate_rf_td], ids=["exact", "fast"])
def test_tgc_brightens_the_deepest_scatterers(fish_scan, simulator):
    """TGC compensates spreading loss, so the deep scatterers gain on the shallow ones."""
    positions, _, _ = fish_scan
    by_depth = np.argsort(positions[:, 2])
    quartile = len(positions) // 4
    deepest, shallowest = positions[by_depth[-quartile:]], positions[by_depth[:quartile]]

    without = _receive_chain_image(fish_scan, simulator, noise_level_db=None, tgc_max_db=0.0)
    with_tgc = _receive_chain_image(fish_scan, simulator, noise_level_db=None, tgc_max_db=50.0)

    dim = _dot_brightness(without, deepest).mean()
    bright = _dot_brightness(with_tgc, deepest).mean()
    assert dim < bright, (
        f"Deepest scatterers are not brighter with TGC: {dim:.1f} without, {bright:.1f} with"
    )

    # Depth ratio isolates the gain ramp from any global brightness shift.
    without_ratio = dim / _dot_brightness(without, shallowest).mean()
    with_ratio = bright / _dot_brightness(with_tgc, shallowest).mean()
    assert without_ratio < 1.0 < with_ratio, (
        f"TGC did not invert the deep/shallow brightness ratio: {without_ratio:.2f} without, "
        f"{with_ratio:.2f} with"
    )


@pytest.mark.parametrize("simulator", [simulate_rf, simulate_rf_td], ids=["exact", "fast"])
def test_noise_lowers_relative_scatterer_amplitude(fish_scan, simulator):
    """Electronic noise lifts the background, so scatterers stand out less above the mean."""
    positions, _, _ = fish_scan

    noiseless = _receive_chain_image(fish_scan, simulator, noise_level_db=None, tgc_max_db=50.0)
    noisy = _receive_chain_image(fish_scan, simulator, noise_level_db=-30.0, tgc_max_db=50.0)

    clean = _dot_brightness(noiseless / noiseless.mean(), positions).mean()
    degraded = _dot_brightness(noisy / noisy.mean(), positions).mean()
    assert degraded < clean, (
        f"Noise did not lower the relative scatterer amplitude: {clean:.1f}x noiseless, "
        f"{degraded:.1f}x at -30 dB"
    )


def _batched_rf(simulation_args, batch, **receive_chain_kwargs):
    """RF for `batch` identical copies of a few scatterers, through the batched op path."""
    args = dict(simulation_args)
    positions = np.asarray(args["scatterer_positions"], dtype=np.float32)[:16]
    args["scatterer_positions"] = np.repeat(positions[None], batch, axis=0)
    args["scatterer_magnitudes"] = np.ones((batch, len(positions)), dtype=np.float32)
    op = Simulate(with_batch_dim=True)
    outputs = op(**args, **receive_chain_kwargs)
    return np.asarray(keras.ops.convert_to_numpy(outputs[op.output_key]))


def test_batched_noise_is_independent_across_items(fish_scan):
    """A stateless seed must not repeat the same noise realisation for every batch item."""
    _, simulation_args, _ = fish_scan

    noiseless = _batched_rf(simulation_args, 3, noise_level_db=None, noise_seed=0)
    noisy = _batched_rf(simulation_args, 3, noise_level_db=-20.0, noise_seed=0)
    noise = noisy - noiseless

    assert np.allclose(noiseless[0], noiseless[1]), "Identical scatterers gave different RF"
    for other in (1, 2):
        assert not np.allclose(noise[0], noise[other]), (
            f"Batch item {other} got the same noise realisation as item 0"
        )


def test_batched_noise_is_reproducible(fish_scan):
    """Same seed, same batch: identical noise. Different seed: different noise."""
    _, simulation_args, _ = fish_scan
    kwargs = {"noise_level_db": -20.0}

    first = _batched_rf(simulation_args, 2, noise_seed=3, **kwargs)
    again = _batched_rf(simulation_args, 2, noise_seed=3, **kwargs)
    other = _batched_rf(simulation_args, 2, noise_seed=4, **kwargs)

    assert np.array_equal(first, again), "Same seed did not reproduce the batched noise"
    assert not np.allclose(first, other), "Different seeds gave the same batched noise"


def test_batched_receive_chain_matches_unbatched(fish_scan):
    """TGC and the default noise reference are per item, so batching must not change them."""
    _, simulation_args, _ = fish_scan
    positions = np.asarray(simulation_args["scatterer_positions"], dtype=np.float32)[:16]

    batched = _batched_rf(simulation_args, 2, noise_level_db=None, tgc_max_db=50.0)
    single = keras.ops.convert_to_numpy(
        simulate_rf(
            **{
                **simulation_args,
                "scatterer_positions": positions,
                "scatterer_magnitudes": np.ones(len(positions), dtype=np.float32),
            },
            noise_level_db=None,
            tgc_max_db=50.0,
        )
    )

    # ops.map reduces in a different order, so compare against the RF peak.
    scale = np.abs(single).max()
    np.testing.assert_allclose(batched[0] / scale, single / scale, atol=1e-4)


def test_batched_noise_reference_is_per_item(fish_scan):
    """The default reference is each item's own peak, not one maximum shared by the batch."""
    _, simulation_args, _ = fish_scan
    args = dict(simulation_args)
    positions = np.asarray(args["scatterer_positions"], dtype=np.float32)[:16]
    magnitudes = np.ones(len(positions), dtype=np.float32)

    args["scatterer_positions"] = np.repeat(positions[None], 2, axis=0)
    args["scatterer_magnitudes"] = np.stack([magnitudes, magnitudes * 100.0])

    op = Simulate(with_batch_dim=True)
    noiseless = np.asarray(
        keras.ops.convert_to_numpy(op(**args, noise_level_db=None)[op.output_key])
    )
    noisy = np.asarray(
        keras.ops.convert_to_numpy(op(**args, noise_level_db=-20.0, noise_seed=0)[op.output_key])
    )
    noise = noisy - noiseless

    ratio = noise[1].std() / noise[0].std()
    assert 90.0 < ratio < 110.0, (
        f"Noise did not track the per-item peak: 100x brighter item got {ratio:.1f}x the noise"
    )
