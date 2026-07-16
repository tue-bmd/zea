"""Tests the pipeline for different transmit schemes."""

import keras
import numpy as np
import pytest

from zea import ops
from zea.beamform.phantoms import fibonacci, fish, golden_ratio, lissajous, rose
from zea.internal.core import DEFAULT_DYNAMIC_RANGE
from zea.internal.dummy_scan import _get_parameters, _get_probe


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
    parameters = _get_parameters(probe, scan_kind)

    inputs = default_pipeline.prepare_parameters(parameters)

    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    output_default = default_pipeline(
        **inputs,
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
        extent=parameters.extent_imshow,
        true_position=ultrasound_scatterers["positions"][0, target_scatterer_index],
    )
    # Check that the pipeline produced the expected outputs
    assert output_default["data"].shape[0] == 1  # Batch dimension
    # Verify the normalized image has values between 0 and 255
    assert np.nanmin(output_default["data"]) >= 0.0
    assert np.nanmax(output_default["data"]) <= 255.0

    # Additional test for planewave: verify focus_distance=0 gives same result
    if scan_kind == "planewave":
        parameters_zero_focus = _get_parameters(
            probe, scan_kind, focus_distances=np.zeros(parameters.n_tx)
        )
        inputs_zero = default_pipeline.prepare_parameters(parameters_zero_focus)

        output_zero_focus = default_pipeline(
            **inputs_zero,
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
    parameters = _get_parameters(probe, "focused", grid_type="polar")

    # Check if the grid type is set correctly
    assert parameters.grid_type == "polar"

    default_pipeline.append(ops.ScanConvert(order=3))

    inputs = default_pipeline.prepare_parameters(parameters)

    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    output_default = default_pipeline(
        **inputs,
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
        extent=parameters.extent_imshow,
        true_position=ultrasound_scatterers["positions"][0, target_scatterer_index],
    )


@pytest.mark.heavy
def test_scanline_grid(ultrasound_scatterers):
    """Scanline imaging (``enable_scanline=True``) is the special case of pixel-based DAS
    with one grid column per transmit and a receive apodization mask that keeps only
    each pixel's owning transmit, wired through the exact same Beamform / TOFCorrection
    / PatchedGrid machinery as the regular pixel-based pipeline.

    Note: with only ``n_tx=8`` lines (this fixture's coarse synthetic sub-aperture
    scan), lateral sampling is far coarser than depth sampling, and sidelobes from
    neighbouring scatterers can dominate the local peak. That makes a
    scatterer-localization check (as used for the cartesian/polar grids above)
    unreliable here, so this test instead checks numerical properties: patch-wise
    beamforming must match unpatched beamforming, and each column must equal an
    independent single-transmit reference built from the same `Parameters`.
    """
    probe = _get_probe("linear")
    parameters = _get_parameters(probe, "linescan", grid_type="cartesian", enable_scanline=True)

    assert parameters.enable_scanline is True
    assert parameters.grid_type == "cartesian"

    n_tx = parameters.n_tx
    num_scanline_pixels = int(parameters.grid_size_z)

    def run(params, num_patches):
        """Beamform-only pipeline (no envelope/normalize/log): raw complex IQ
        pixel values, so results across differently-sized transmit sets are
        directly comparable (Normalize would rescale each image by its own
        max and break that comparison).
        """
        pipeline = ops.Pipeline(
            operations=[
                ops.Simulate(),
                ops.Cast(dtype="float32"),
                ops.ApplyWindow(),
                ops.Demodulate(),
                ops.Beamform(
                    beamformer="delay_and_sum",
                    num_patches=num_patches,
                    enable_aligned_apodization=True,
                ),
            ],
            jit_options=None,
        )
        inputs = pipeline.prepare_parameters(params)
        output = pipeline(
            **inputs,
            scatterer_positions=ultrasound_scatterers["positions"],
            scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
        )
        return keras.ops.convert_to_numpy(output["data"][0])

    image_unpatched = run(parameters, num_patches=1)
    image_patched = run(parameters, num_patches=10)

    assert image_unpatched.shape == (num_scanline_pixels, n_tx, 2)
    assert np.all(np.isfinite(image_unpatched))
    assert np.any(image_unpatched != 0)

    # Patch-wise beamforming (PatchedGrid chunks `flatgrid` and `flat_aligned_apodization`
    # together) must give the same result as beamforming the whole grid at once.
    np.testing.assert_allclose(image_patched, image_unpatched, rtol=1e-4, atol=1e-4)

    # Column n is beamformed from transmit n only: it must equal an independent
    # single-transmit, single-column reference built from the same `Parameters`.
    column = n_tx // 2
    single_tx_parameters = parameters.copy()
    single_tx_parameters.set_transmits([column])
    image_single_tx = run(single_tx_parameters, num_patches=1)

    assert image_single_tx.shape == (num_scanline_pixels, 1, 2)
    np.testing.assert_allclose(
        image_patched[:, column], image_single_tx[:, 0], rtol=1e-4, atol=1e-4
    )


def test_scanline_grid_polar_style():
    """``enable_scanline=True`` combined with ``grid_type="polar"`` builds steered rays
    from each transmit's own origin (the old ``scanline_sector=True`` behavior),
    reusing the same ``grid_type`` that also drives the regular (non-scanline)
    grid, instead of a separate scanline-only flag.
    """
    from zea.beamform.pixelgrid import scanline_pixel_grid
    from zea.parameters import Parameters

    n_tx, n_el = 4, 16
    probe_geometry = np.zeros((n_el, 3), np.float32)
    probe_geometry[:, 0] = np.linspace(-10e-3, 10e-3, n_el)
    transmit_origins = np.zeros((n_tx, 3), np.float32)
    focus_distances = np.full(n_tx, 30e-3, np.float32)
    polar_angles = np.linspace(-0.2, 0.2, n_tx).astype(np.float32)
    zlims = (0.0, 40e-3)
    grid_size_z = 12

    parameters = Parameters(
        n_tx=n_tx,
        n_el=n_el,
        probe_geometry=probe_geometry,
        transmit_origins=transmit_origins,
        focus_distances=focus_distances,
        polar_angles=polar_angles,
        zlims=zlims,
        grid_size_z=grid_size_z,
        center_frequency=5e6,
        sound_speed=1540.0,
        sampling_frequency=20e6,
        grid_type="polar",
        enable_scanline=True,
        selected_transmits="all",
    )

    expected = scanline_pixel_grid(
        transmit_origins,
        focus_distances,
        polar_angles,
        zlims,
        grid_size_z,
        grid_type="polar",
    )
    np.testing.assert_allclose(np.asarray(parameters.grid), expected, atol=1e-6)
    assert parameters.grid_size_x == n_tx
    assert parameters.flat_aligned_apodization.shape == (grid_size_z * n_tx, n_tx)


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
