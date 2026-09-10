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
from zea.simulator import simulate_rf
from zea.ops import Simulate
from zea.ops.ultrasound import simulator_settings
from zea.probes import create_curved_probe_geometry, create_probe_geometry, curved_probe_normals
from zea.simulator_time_domain import get_pulse_waveform, simulate_rf_td

N_EL = 80
APERTURE = 32e-3
N_TX = 5
CENTER_FREQUENCY = 3e6  # Hz
SOUND_SPEED = 1540.0  # m/s
XLIMS = (-24e-3, 24e-3)
ZLIMS = (10e-3, 35e-3)
DYNAMIC_RANGE = (-50.0, 0.0)


def test_time_domain_scatter_exponent_weights_pulse_spectrum():
    """The time-domain approximation applies scatter frequency dependence to its pulse."""
    n_samples = 129
    unweighted = np.asarray(
        keras.ops.convert_to_numpy(
            get_pulse_waveform(CENTER_FREQUENCY, CENTER_FREQUENCY * 4, n_samples=n_samples)
        )
    )
    weighted = np.asarray(
        keras.ops.convert_to_numpy(
            get_pulse_waveform(
                CENTER_FREQUENCY,
                CENTER_FREQUENCY * 4,
                n_samples=n_samples,
                scatter_exponent=2.0,
            )
        )
    )
    frequencies = np.fft.rfftfreq(n_samples, 1 / (CENTER_FREQUENCY * 4))
    expected = np.fft.rfft(unweighted) * (frequencies / CENTER_FREQUENCY) ** 2
    np.testing.assert_allclose(np.fft.rfft(weighted), expected, rtol=2e-5, atol=2e-5)


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


@pytest.mark.parametrize("method", list(simulator_settings))
def test_simulate_op_runs_every_method(method):
    """The op hands options that only the frequency-domain simulators take to those alone."""
    n_el = 8
    probe_geometry = np.stack(
        [np.linspace(-4e-3, 4e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
    op = Simulate(jit_compile=False, with_batch_dim=False)
    outputs = op(
        scatterer_positions=np.array([[0.0, 0.0, 15e-3]], dtype=np.float32),
        scatterer_magnitudes=np.ones(1, dtype=np.float32),
        probe_geometry=probe_geometry,
        apply_lens_correction=False,
        lens_thickness=1e-3,
        lens_sound_speed=1000.0,
        sound_speed=SOUND_SPEED,
        n_ax=512,
        center_frequency=CENTER_FREQUENCY,
        sampling_frequency=CENTER_FREQUENCY * 4,
        t0_delays=np.zeros((1, n_el), dtype=np.float32),
        initial_times=np.zeros(1, dtype=np.float32),
        element_width=1e-3,
        attenuation_coef=0.0,
        tx_apodizations=np.ones((1, n_el), dtype=np.float32),
        t_peak=np.zeros(1, dtype=np.float32),
        method=method,
    )
    assert np.abs(keras.ops.convert_to_numpy(outputs[op.output_key])).max() > 0


def test_parameters_derive_element_normals_from_probe_geometry():
    curved = create_curved_probe_geometry(N_EL, 0.4e-3, 40e-3)
    # One-sided differences tilt the end elements by half the angular pitch (5 mrad here).
    np.testing.assert_allclose(
        Parameters(probe_geometry=curved).element_normals, curved_probe_normals(curved), atol=1e-2
    )
    flat = create_probe_geometry(N_EL, 0.4e-3)
    z_normals = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (N_EL, 1))
    np.testing.assert_array_equal(Parameters(probe_geometry=flat).element_normals, z_normals)
    # A virtual apex behind a flat array does not tilt its elements.
    with_apex = Parameters(probe_geometry=flat, distance_to_apex=20e-3)
    np.testing.assert_array_equal(with_apex.element_normals, z_normals)
    tilted = np.tile(np.array([0.5, 0.0, np.sqrt(0.75)], np.float32), (N_EL, 1))
    np.testing.assert_array_equal(
        Parameters(probe_geometry=flat, element_normals=tilted).element_normals, tilted
    )


def test_pipeline_simulates_curved_probe_in_its_element_frames():
    """The Simulate op gets a curved probe's normals from Parameters, not the flat +z frame."""
    probe_geometry = create_curved_probe_geometry(N_EL, APERTURE / N_EL, 40e-3)
    pipeline = zea.Pipeline([Simulate()], with_batch_dim=False, jit_options=None)
    inputs = pipeline.prepare_parameters(_parameters(probe_geometry))
    normals = keras.ops.convert_to_numpy(inputs["element_normals"])
    np.testing.assert_allclose(normals, curved_probe_normals(probe_geometry), atol=1e-2)

    inputs["scatterer_positions"] = np.array([[-12e-3, 0, 20e-3], [10e-3, 0, 25e-3]], np.float32)
    inputs["scatterer_magnitudes"] = np.ones(2, np.float32)
    curved = pipeline(**inputs)[pipeline.output_key]
    flat = pipeline(**{**inputs, "element_normals": None})[pipeline.output_key]
    curved, flat = keras.ops.convert_to_numpy(curved), keras.ops.convert_to_numpy(flat)
    assert np.linalg.norm(curved - flat) > 0.05 * np.linalg.norm(curved)


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
    op = Simulate(with_batch_dim=False)
    single = keras.ops.convert_to_numpy(
        op(
            **{
                **simulation_args,
                "scatterer_positions": positions,
                "scatterer_magnitudes": np.ones(len(positions), dtype=np.float32),
            },
            noise_level_db=None,
            tgc_max_db=50.0,
        )[op.output_key]
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


def _td_args(n_el=16, **overrides):
    """Minimal single-transmit argument set for the time-domain simulator."""
    probe_geometry = np.stack(
        [np.linspace(-8e-3, 8e-3, n_el), np.zeros(n_el), np.zeros(n_el)], axis=1
    ).astype(np.float32)
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
    }
    return {**args, **overrides}


def test_time_domain_two_dimensional_moves_scatterers_into_the_plane():
    """In 2D an off-plane scatterer echoes as its projection onto the imaging plane."""
    args = _td_args(two_dimensional=True, element_height=5e-3)
    args["scatterer_magnitudes"] = np.ones(1, dtype=np.float32)
    in_plane = np.array([[2e-3, 0.0, 30e-3]], dtype=np.float32)
    off_plane = np.array([[2e-3, 8e-3, 30e-3]], dtype=np.float32)

    rf_in = keras.ops.convert_to_numpy(simulate_rf_td(scatterer_positions=in_plane, **args))
    rf_off = keras.ops.convert_to_numpy(simulate_rf_td(scatterer_positions=off_plane, **args))
    assert np.abs(rf_in).max() > 0
    assert np.allclose(rf_in, rf_off)
    args["two_dimensional"] = False
    rf_3d = keras.ops.convert_to_numpy(simulate_rf_td(scatterer_positions=off_plane, **args))
    assert not np.allclose(rf_in, rf_3d)


def test_time_domain_lens_correction_delays_arrivals():
    """A slow lens lengthens the round trip."""
    args = _td_args()
    args["scatterer_positions"] = np.array([[0.0, 0.0, 30e-3]], dtype=np.float32)
    args["scatterer_magnitudes"] = np.ones(1, dtype=np.float32)

    uncorrected = keras.ops.convert_to_numpy(simulate_rf_td(**args))[0, :, :, 0]
    corrected = keras.ops.convert_to_numpy(
        simulate_rf_td(**{**args, "apply_lens_correction": True})
    )[0, :, :, 0]

    center = args["probe_geometry"].shape[0] // 2
    delay = np.abs(corrected[:, center]).argmax() - np.abs(uncorrected[:, center]).argmax()
    # Two lens crossings at 1000 m/s instead of 1540 m/s: ~0.7 us, ~8 samples at 4x fc.
    expected = (
        2
        * args["lens_thickness"]
        * (1 / args["lens_sound_speed"] - 1 / args["sound_speed"])
        * args["sampling_frequency"]
    )
    assert delay == pytest.approx(expected, abs=2), (
        f"Lens correction shifted the echo by {delay} samples, expected ~{expected:.1f}"
    )


@pytest.mark.parametrize("simulator", [simulate_rf, simulate_rf_td], ids=["exact", "fast"])
@pytest.mark.parametrize("scatter_exponent", [-1.0, np.nan, np.inf])
def test_invalid_scatter_exponent_raises(simulator, scatter_exponent):
    """A bad exponent must fail loudly, not silently return an all-NaN RF frame."""
    args = _td_args()
    args["scatterer_positions"] = np.array([[0.0, 0.0, 30e-3]], dtype=np.float32)
    args["scatterer_magnitudes"] = np.ones(1, dtype=np.float32)

    with pytest.raises(ValueError, match="scatter_exponent"):
        simulator(**args, scatter_exponent=scatter_exponent)


@pytest.mark.parametrize("simulator", [simulate_rf, simulate_rf_td], ids=["exact", "fast"])
@pytest.mark.parametrize("scatter_exponent", [0.0, 0.6, 2.0])
def test_valid_scatter_exponent_gives_finite_rf(simulator, scatter_exponent):
    """Physical exponents, including the unweighted 0, stay accepted and finite."""
    args = _td_args()
    args["scatterer_positions"] = np.array([[0.0, 0.0, 30e-3]], dtype=np.float32)
    args["scatterer_magnitudes"] = np.ones(1, dtype=np.float32)

    rf = keras.ops.convert_to_numpy(simulator(**args, scatter_exponent=scatter_exponent))
    assert np.isfinite(rf).all()
    assert np.abs(rf).max() > 0
