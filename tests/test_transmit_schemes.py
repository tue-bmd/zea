"""Tests the pipeline for different transmit schemes."""

import keras
import numpy as np
import pytest

from zea import ops
from zea.beamform.phantoms import fish, rose, fibonacci, lissajous, golden_ratio
from zea.internal.core import DEFAULT_DYNAMIC_RANGE
from zea.internal.dummy_scan import _get_probe, _get_scan


def _get_flatgrid(extent, shape):
    """Helper function to get a flat grid corresponding to an image."""
    xmin, xmax, zmax, zmin = extent
    x = np.linspace(xmin, xmax, shape[0])
    y = np.linspace(zmin, zmax, shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.vstack((X.flatten(), Y.flatten())).T


def _get_pixel_size(extent, shape):
    """Helper function to get the pixel size of an image.

    Returns:
        np.ndarray: The pixel size (width, height).
    """
    xmin, xmax, zmax, zmin = extent
    width, height = xmax - xmin, zmax - zmin
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
    scat_positions = np.expand_dims(fish(), axis=0)
    n_scat = scat_positions.shape[1]

    return {
        "positions": scat_positions.astype(np.float32),
        "magnitudes": np.ones((1, n_scat), dtype=np.float32),
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
    default_pipeline: ops.Pipeline,
    probe_kind,
    scan_kind,
    ultrasound_scatterers,
):
    """Tests the default ultrasound pipeline."""

    probe = _get_probe(probe_kind)
    scan = _get_scan(probe, scan_kind)

    parameters = default_pipeline.prepare_parameters(probe, scan)

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

    # Target the scatterer that forms the eye
    target_scatterer_index = -4

    # Check if the scatterer is in the right location in the image
    _test_location(
        image.T,
        extent=scan.extent,
        true_position=ultrasound_scatterers["positions"][0, target_scatterer_index],
    )
    # Check that the pipeline produced the expected outputs
    assert output_default["data"].shape[0] == 1  # Batch dimension
    # Verify the normalized image has values between 0 and 255
    assert np.nanmin(output_default["data"]) >= 0.0
    assert np.nanmax(output_default["data"]) <= 255.0

    # Additional test for planewave: verify focus_distance=0 gives same result
    if scan_kind == "planewave":
        scan_zero_focus = _get_scan(probe, scan_kind, focus_distances=np.zeros(scan.n_tx))
        parameters_zero = default_pipeline.prepare_parameters(probe, scan_zero_focus)

        output_zero_focus = default_pipeline(
            **parameters_zero,
            scatterer_positions=ultrasound_scatterers["positions"],
            scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
        )

        image_zero = keras.ops.convert_to_numpy(output_zero_focus["data"][0])

        # The images should be identical (or very close due to numerical precision)
        np.testing.assert_allclose(
            image,
            image_zero,
            rtol=1e-5,
            atol=1e-3,
            err_msg="Planewave with focus_distance=inf and "
            + "focus_distance=0 should give same result",
        )


@pytest.mark.heavy
def test_polar_grid(default_pipeline: ops.Pipeline, ultrasound_scatterers):
    """Tests the polar grid generation."""
    probe = _get_probe("linear")
    scan = _get_scan(probe, "focused", grid_type="polar")

    # Check if the grid type is set correctly
    assert scan.grid_type == "polar"

    default_pipeline.append(ops.ScanConvert(order=3))

    parameters = default_pipeline.prepare_parameters(probe, scan)

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

    # Target the scatterer that forms the eye
    target_scatterer_index = -4

    # Check if the scatterer is in the right location in the image
    _test_location(
        image.T,
        extent=scan.extent,
        true_position=ultrasound_scatterers["positions"][0, target_scatterer_index],
    )


def test_phantoms():
    """Tests the phantom generation functions."""
    fish_scat = fish()
    rose_scat = rose(num_scatterers=50)
    fibonacci_scat = fibonacci(num_scatterers=50)
    lissajous_scat = lissajous(num_scatterers=50)
    golden_ratio_scat = golden_ratio(num_scatterers=50)

    assert fish_scat.shape == (104, 3)
    assert rose_scat.shape == (50, 3)
    assert fibonacci_scat.shape == (50, 3)
    assert lissajous_scat.shape == (50, 3)
    assert golden_ratio_scat.shape == (50, 3)
