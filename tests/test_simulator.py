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
from zea.simulator import apply_receive_chain, simulate_rf, transmit_pulse
from zea.ops import Simulate
from zea.ops.ultrasound import simulator_settings
from zea.probes import create_curved_probe_geometry, create_probe_geometry, curved_probe_normals
from zea.simulator_time_domain import _scattered_waveform, simulate_rf_td

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
    pulse = transmit_pulse(CENTER_FREQUENCY, CENTER_FREQUENCY * 4)
    unweighted, weighted = (
        np.asarray(keras.ops.convert_to_numpy(_scattered_waveform(pulse, CENTER_FREQUENCY, e)))
        for e in (0.0, 2.0)
    )
    frequencies = np.fft.rfftfreq(len(unweighted), 1 / (CENTER_FREQUENCY * 4))
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


def _np(x):
    return np.asarray(keras.ops.convert_to_numpy(x))


def test_multi_plane_transmit_is_the_sum_of_its_delay_sets(fish_scan):
    _, args, _ = fish_scan
    separate = _np(simulate_rf(**args))
    together = {k: args[k][:1] for k in ("tx_apodizations", "initial_times", "t_peak")}
    mpt = _np(simulate_rf(**{**args, **together, "t0_delays": args["t0_delays"][None]}))
    np.testing.assert_allclose(mpt[0], separate.sum(0), atol=1e-4 * np.abs(separate).max())


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


def _rf_block(shape=(2, 256, 8, 1), seed=0):
    """Random RF of shape (n_tx, n_ax, n_el, n_ch), as the simulators return it."""
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def test_receive_chain_tgc_is_an_exponential_ramp_over_depth():
    """TGC gains each axial sample by tgc_max_db * n / (n_ax - 1) dB, and 0 dB leaves the RF."""
    rf = _rf_block()
    n_ax = rf.shape[1]
    gain = 10.0 ** (40.0 * np.arange(n_ax) / (n_ax - 1) / 20.0)
    with_tgc = _np(apply_receive_chain(rf, noise_level_db=None, tgc_max_db=40.0))
    np.testing.assert_allclose(with_tgc, rf * gain[None, :, None, None], rtol=1e-5)
    assert np.array_equal(_np(apply_receive_chain(rf, noise_level_db=None, tgc_max_db=0.0)), rf)


def test_receive_chain_noise_is_gaussian_at_the_level_below_the_reference():
    """The noise is white Gaussian with sigma = reference * 10^(noise_level_db / 20), the
    reference defaulting to the peak of the RF. A seed fixes the realisation."""
    rf = _rf_block()
    noise = _np(apply_receive_chain(rf, noise_level_db=-20.0, noise_seed=0)) - rf
    sigma = np.abs(rf).max() * 10.0 ** (-20.0 / 20.0)
    assert noise.std() == pytest.approx(sigma, rel=0.05)
    again = _np(apply_receive_chain(rf, noise_level_db=-20.0, noise_seed=0)) - rf
    other = _np(apply_receive_chain(rf, noise_level_db=-20.0, noise_seed=1)) - rf
    assert np.array_equal(noise, again)
    assert not np.allclose(noise, other)
    # A fixed reference scales the same realisation, and None disables the noise.
    doubled = apply_receive_chain(
        rf, noise_level_db=-20.0, noise_seed=0, noise_reference=2.0 * np.abs(rf).max()
    )
    np.testing.assert_allclose(_np(doubled) - rf, 2.0 * noise, rtol=1e-5, atol=1e-6 * sigma)
    assert np.array_equal(_np(apply_receive_chain(rf, noise_level_db=None)), rf)


def test_receive_chain_noise_reference_is_per_batch_item():
    """On a batch the default reference is each item's own peak, not the peak of the batch,
    and every item draws its own realisation."""
    rf = np.stack([_rf_block(), 100.0 * _rf_block(seed=1)])
    noise = _np(apply_receive_chain(rf, noise_level_db=-20.0, noise_seed=0)) - rf
    sigma = np.abs(rf).max(axis=(1, 2, 3, 4)) * 10.0 ** (-20.0 / 20.0)
    np.testing.assert_allclose(noise.std(axis=(1, 2, 3, 4)), sigma, rtol=0.05)
    assert not np.allclose(noise[0] / sigma[0], noise[1] / sigma[1])


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
