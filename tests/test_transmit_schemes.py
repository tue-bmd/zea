"""Tests the pipeline for different transmit schemes."""

import keras
import numpy as np
import pytest

from zea import ops
from zea.beamform.delays import compute_t0_delays_focused, compute_t0_delays_planewave
from zea.beamform.phantoms import fish
from zea.probes import Probe
from zea.scan import Scan


def _get_flatgrid(extent, shape):
    """Helper function to get a flat grid corresponding to an image."""
    x = np.linspace(extent[0], extent[1], shape[0])
    y = np.linspace(extent[2], extent[3], shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.vstack((X.flatten(), Y.flatten())).T


def _get_pixel_size(extent, shape):
    """Helper function to get the pixel size of an image.

    Returns:
        np.ndarray: The pixel size (width, height).
    """

    width, height = extent[1] - extent[0], extent[3] - extent[2]
    if shape[0] == 1:
        pixel_width = width
    else:
        pixel_width = width / (shape[0] - 1)

    if shape[1] == 1:
        pixel_height = height
    else:
        pixel_height = height / (shape[1] - 1)

    return np.array([pixel_width, pixel_height])


def _find_peak_location(image, extent, position, max_diff=0.6e-3):
    """Find the point with the maximum intensity within a certain distance of a given point.

    Args:
    image (np.ndarray): The image to search in.
    extent (tuple): The extent of the image.
    position (np.array): The position to search around.
    max_diff (float): The maximum distance from the position to search.

    Returns:
    np.array: The corrected position which is at most `max_diff` away from the original
        position.
    """

    position = np.array(position)

    if max_diff == 0.0:
        return position

    flatgrid = _get_flatgrid(extent, image.shape)

    # Compute the distances between the points and the position
    distances = np.linalg.norm(flatgrid - position, axis=1)

    # Mask the points that are within the maximum distance
    mask = distances <= max_diff
    candidate_intensities = np.ravel(image)[mask]
    candidate_points = flatgrid[mask]

    no_points_to_consider = candidate_intensities.size == 0
    if no_points_to_consider:
        raise ValueError("No candidate points found.")

    highest_intensity_pixel_idx = np.argmax(candidate_intensities)
    highest_intensity_pixel_location = candidate_points[highest_intensity_pixel_idx]

    return highest_intensity_pixel_location


# module scope is used to avoid recompiling the pipeline for each test
@pytest.fixture(scope="module")
def default_pipeline():
    """Returns a default pipeline for ultrasound simulation."""
    pipeline = ops.Pipeline.from_default(num_patches=10, jit_options="ops")
    pipeline.prepend(ops.Simulate())
    pipeline.append(ops.Normalize(input_range=ops.DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))
    return pipeline


def _get_linear_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 128
    aperture = 30e-3
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=2.5e6,
        sampling_frequency=10e6,
    )


def _get_phased_array_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 80
    aperture = 20e-3
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=3.12e6,
        sampling_frequency=12.5e6,
    )


def _get_n_ax(ultrasound_probe):
    """Returns the number of ax for ultrasound simulation tests based on the center
    frequency. A probe with a higher center frequency needs more samples to cover
    the image depth.
    """
    is_low_frequency_probe = ultrasound_probe.center_frequency < 4e6

    if is_low_frequency_probe:
        return 510

    return 1024


def _get_probe(kind):
    if kind == "linear":
        return _get_linear_probe()
    elif kind == "phased_array":
        return _get_phased_array_probe()
    else:
        raise ValueError(f"Unknown probe kind: {kind}")


def _get_constant_scan_kwargs():
    return {
        "lens_sound_speed": 1000,
        "lens_thickness": 1e-3,
        "n_ch": 1,
        "selected_transmits": "all",
        "sound_speed": 1540.0,
        "apply_lens_correction": False,
        "attenuation_coef": 0.0,
    }


def _get_lims_and_gridsize(center_frequency, sound_speed):
    """Returns the limits and gridsize for ultrasound simulation tests."""
    xlims, zlims = (-20e-3, 20e-3), (0, 35e-3)
    width, height = xlims[1] - xlims[0], zlims[1] - zlims[0]
    wavelength = sound_speed / center_frequency
    gridsize = (
        int(width / (0.5 * wavelength)) + 1,
        int(height / (0.5 * wavelength)) + 1,
    )
    return {"xlims": xlims, "zlims": zlims, "grid_size_x": gridsize[0], "grid_size_z": gridsize[1]}


def _get_planewave_scan(ultrasound_probe, grid_type):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = ultrasound_probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * np.inf
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry, polar_angles=angles, sound_speed=sound_speed
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
    )


def _get_multistatic_scan(ultrasound_probe, grid_type):
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.zeros((n_tx, n_el))
    for n, idx in enumerate(np.linspace(0, n_el - 1, n_tx, dtype=int)):
        tx_apodizations[n, idx] = 1
    probe_geometry = ultrasound_probe.probe_geometry

    focus_distances = np.zeros(n_tx)
    t0_delays = np.zeros((n_tx, n_el))

    constant_scan_kwargs = _get_constant_scan_kwargs()

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        focus_distances=focus_distances,
        polar_angles=np.zeros(n_tx),
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(
            ultrasound_probe.center_frequency, constant_scan_kwargs["sound_speed"]
        ),
        **constant_scan_kwargs,
    )


def _get_diverging_scan(ultrasound_probe, grid_type):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(10, -10, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * -15e-3
    t0_delays = compute_t0_delays_focused(
        origins=np.zeros((n_tx, 3)),
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
    )


def _get_focused_scan(ultrasound_probe, grid_type):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]

    angles = np.linspace(30, -30, n_tx) * np.pi / 180

    sound_speed = constant_scan_kwargs["sound_speed"]
    focus_distances = np.ones(n_tx) * 15e-3
    t0_delays = compute_t0_delays_focused(
        origins=np.zeros((n_tx, 3)),
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
    )


def _get_linescan_scan(ultrasound_probe, grid_type):
    """Returns a scan for ultrasound simulation tests."""
    constant_scan_kwargs = _get_constant_scan_kwargs()
    n_el = ultrasound_probe.n_el
    n_tx = 8

    center_elements = np.linspace(0, n_el + 1, n_tx + 2, dtype=int)
    center_elements = center_elements[1:-1]
    tx_apodizations = np.zeros((n_tx, n_el))
    aperture_size_elements = 24

    # Define subapertures
    origins = []
    for n, idx in enumerate(center_elements):
        el0 = np.clip(idx - aperture_size_elements // 2, 0, n_el)
        el1 = np.clip(idx + aperture_size_elements // 2, 0, n_el)
        tx_apodizations[n, el0:el1] = np.hanning(el1 - el0)[None]
        origins.append(ultrasound_probe.probe_geometry[idx])
    origins = np.stack(origins, axis=0)

    # All angles should be zero because each line fires straight ahead
    angles = np.zeros(n_tx)

    sound_speed = constant_scan_kwargs["sound_speed"]

    focus_distances = np.ones(n_tx) * 15e-3
    t0_delays = compute_t0_delays_focused(
        origins=origins,
        focus_distances=focus_distances,
        probe_geometry=ultrasound_probe.probe_geometry,
        polar_angles=angles,
        sound_speed=sound_speed,
    )
    element_width = np.linalg.norm(
        ultrasound_probe.probe_geometry[1] - ultrasound_probe.probe_geometry[0]
    )

    return Scan(
        n_tx=n_tx,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=ultrasound_probe.probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=element_width,
        focus_distances=focus_distances,
        polar_angles=angles,
        initial_times=np.ones(n_tx) * 1e-6,
        n_ax=_get_n_ax(ultrasound_probe),
        grid_type=grid_type,
        **_get_lims_and_gridsize(ultrasound_probe.center_frequency, sound_speed),
        **constant_scan_kwargs,
    )


def _get_scan(ultrasound_probe, kind, grid_type="cartesian"):
    if kind == "planewave":
        return _get_planewave_scan(ultrasound_probe, grid_type)
    elif kind == "multistatic":
        return _get_multistatic_scan(ultrasound_probe, grid_type)
    elif kind == "diverging":
        return _get_diverging_scan(ultrasound_probe, grid_type)
    elif kind == "focused":
        return _get_focused_scan(ultrasound_probe, grid_type)
    elif kind == "linescan":
        return _get_linescan_scan(ultrasound_probe, grid_type)
    else:
        raise ValueError(f"Unknown scan kind: {kind}")


def _test_location(image, extent, true_position):
    """Tests the peak location function."""

    if true_position.shape[0] == 3:
        true_position = np.array([true_position[0], true_position[2]])
    start_position = true_position
    new_position = _find_peak_location(image, extent, start_position, max_diff=1.5e-3)

    pixel_size = _get_pixel_size(extent, image.shape)

    difference = np.abs(new_position - true_position)
    assert np.all(difference <= pixel_size * 3.0)


@pytest.fixture
def ultrasound_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests."""
    scat_positions = fish()
    n_scat = scat_positions.shape[0]

    return {
        "positions": scat_positions.astype(np.float32),
        "magnitudes": np.ones(n_scat, dtype=np.float32),
        "n_scat": n_scat,
    }


@pytest.mark.parametrize(
    "probe_kind, scan_kind",
    [
        ("linear", "planewave"),
        ("linear", "multistatic"),
        ("linear", "diverging"),
        ("linear", "focused"),
        ("linear", "linescan"),
        ("phased_array", "planewave"),
        ("phased_array", "multistatic"),
        ("phased_array", "diverging"),
        ("phased_array", "focused"),
    ],
)
@pytest.mark.heavy
def test_transmit_schemes(
    default_pipeline,
    probe_kind,
    scan_kind,
    ultrasound_scatterers,
):
    """Tests the default ultrasound pipeline."""

    ultrasound_probe = _get_probe(probe_kind)
    ultrasound_scan = _get_scan(ultrasound_probe, scan_kind)

    parameters = default_pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)

    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    output_default = default_pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )

    image = output_default["data"][0]

    # Convert to numpy
    image = keras.ops.convert_to_numpy(image)
    extent = [
        ultrasound_scan.xlims[0],
        ultrasound_scan.xlims[1],
        ultrasound_scan.zlims[0],
        ultrasound_scan.zlims[1],
    ]

    # Target the scatterer that forms the eye
    target_scatterer_index = -4

    # Check if the scatterer is in the right location in the image
    _test_location(
        image.T,
        extent=extent,
        true_position=ultrasound_scatterers["positions"][target_scatterer_index],
    )
    # Check that the pipeline produced the expected outputs
    assert output_default["data"].shape[0] == 1  # Batch dimension
    # Verify the normalized image has values between 0 and 255
    assert np.nanmin(output_default["data"]) >= 0.0
    assert np.nanmax(output_default["data"]) <= 255.0


@pytest.mark.heavy
def test_polar_grid(default_pipeline: ops.Pipeline, ultrasound_scatterers):
    """Tests the polar grid generation."""
    ultrasound_probe = _get_linear_probe()
    ultrasound_scan = _get_scan(ultrasound_probe, "focused", grid_type="polar")

    # Check if the grid type is set correctly
    assert ultrasound_scan.grid_type == "polar"

    default_pipeline.append(ops.ScanConvert(order=3))

    parameters = default_pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)

    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    output_default = default_pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )

    image = output_default["data"][0]

    # Convert to numpy
    image = keras.ops.convert_to_numpy(image)

    assert ultrasound_scan.zlims[0] == 0.0

    # xlims for polar grid can be computed as follows, think about the unit circle :)
    radius = ultrasound_scan.zlims[1]
    xlims = (
        radius * np.cos(-np.pi / 2 + ultrasound_scan.theta_range[0]),
        radius * np.cos(-np.pi / 2 + ultrasound_scan.theta_range[1]),
    )
    extent = [*xlims, *ultrasound_scan.zlims]

    # Target the scatterer that forms the eye
    target_scatterer_index = -4

    # Check if the scatterer is in the right location in the image
    _test_location(
        image.T,
        extent=extent,
        true_position=ultrasound_scatterers["positions"][target_scatterer_index],
    )
