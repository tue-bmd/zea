"""Tests the pipeline for different transmit schemes."""

import keras
import numpy as np
import pytest

from zea import ops
from zea.beamform.phantoms import fish
from zea.internal.core import DEFAULT_DYNAMIC_RANGE
from zea.internal.dummy_scan import _get_probe, _get_scan


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
    pipeline.append(ops.Normalize(input_range=DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))
    return pipeline


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
    ultrasound_probe = _get_probe("linear")
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
