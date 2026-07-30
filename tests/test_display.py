"""Tests for the display module."""

import numpy as np
import pytest

from zea import display

from . import DEFAULT_TEST_SEED, backend_equality_check


@pytest.mark.parametrize(
    "size, resolution, order",
    [
        ((128, 32), None, 1),
        ((512, 512), 0.1, 1),
        ((40, 20, 20), None, 1),
        ((40, 20, 20), 0.5, 1),
        ((112, 112), None, 3),  # will use scipy ndimage for order > 1
    ],
)
@backend_equality_check(decimal=0, backends=["torch", "jax", "tensorflow"])
def test_scan_conversion(size, resolution, order):
    """Tests the scan_conversion function with random data."""
    import keras
    from keras import ops

    from zea import display

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal(size).astype(np.float32)

    rho_range = (0, 100)
    theta_range = (-45, 45)
    theta_range = np.deg2rad(theta_range)

    if len(size) == 3:
        phi_range = (-20, 20)
        phi_range = np.deg2rad(phi_range)
        out, params = display.scan_convert_3d(
            data,
            rho_range,
            theta_range,
            phi_range,
            resolution=resolution,
            order=order,
        )
    else:
        out, params = display.scan_convert_2d(
            data,
            rho_range,
            theta_range,
            resolution=resolution,
            order=order,
        )

    assert isinstance(params, dict), "params is not a dict"

    # Check that dtype was not changed
    assert ops.dtype(out) == ops.dtype(data), "output dtype is not the same as input dtype"

    out = ops.convert_to_numpy(out)

    # make sure outputs are not all nans or zeros
    assert not np.all(np.isnan(out)), "scan conversion is all nans"
    assert not np.all(out == 0), (
        f"scan conversion is all zeros for backend {keras.backend.backend()}"
    )
    out = np.nan_to_num(out, nan=0)
    return out


def test_scan_convert_in_process_smoke():
    """In-process smoke test for scan_convert_2d / scan_convert_3d.

    The parametrized ``test_scan_conversion`` runs inside per-backend subprocesses
    (via ``backend_equality_check``), which coverage tooling does not observe, so
    this plain in-process test ensures the scan-conversion code paths are covered.
    """
    rng = np.random.default_rng(DEFAULT_TEST_SEED)

    img2d = rng.standard_normal((32, 16)).astype(np.float32)
    out2d, params2d = display.scan_convert_2d(img2d, rho_range=(0, 1), theta_range=(-0.5, 0.5))
    assert isinstance(params2d, dict)
    assert np.asarray(out2d).ndim == 2

    # Exercise the alternative interpolation branches: order > 1 (scipy/CPU path)
    # and vectorize=False (ops.map path), in addition to the default vectorized path.
    out2d_order2, _ = display.scan_convert_2d(
        img2d, rho_range=(0, 1), theta_range=(-0.5, 0.5), order=2
    )
    assert np.asarray(out2d_order2).ndim == 2
    out2d_novec, _ = display.scan_convert_2d(
        img2d, rho_range=(0, 1), theta_range=(-0.5, 0.5), vectorize=False
    )
    assert np.asarray(out2d_novec).ndim == 2

    vol3d = rng.standard_normal((16, 10, 10)).astype(np.float32)
    out3d, params3d = display.scan_convert_3d(
        vol3d, rho_range=(0, 1), theta_range=(-0.4, 0.4), phi_range=(-0.4, 0.4)
    )
    assert isinstance(params3d, dict)
    assert np.asarray(out3d).ndim == 3


def create_radial_pattern(size):
    """Creates a radial pattern for testing scan conversion."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    r = np.sqrt(x**2 + y**2)
    image = np.exp(-(r**2))
    return image.astype("float32")


def create_concentric_rings(size):
    """Creates a ring pattern for testing scan conversion."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    r = np.sqrt(x**2 + y**2)
    image = np.sin(10 * r) ** 2
    return image.astype("float32")


@pytest.mark.parametrize(
    "size, pattern_creator, allowed_error, angle",
    [
        ((200, 200), "create_radial_pattern", 0.001, None),
        ((100, 333), "create_radial_pattern", 0.001, None),
        ((200, 200), "create_concentric_rings", 0.1, None),
        ((100, 333), "create_concentric_rings", 0.1, None),
        ((200, 200), "create_radial_pattern", 0.001, np.deg2rad(30)),
    ],
)
@backend_equality_check(decimal=2)
def test_scan_conversion_and_inverse(size, pattern_creator, allowed_error, angle):
    """Tests the scan_conversion function with structured test patterns and
    inverts the data with inverse_scan_convert_2d.

    Note:
        The allowed_error is set to 0.1 for concentric rings because the MSE is
        expected to be higher due to the nature of the pattern.
    """
    from keras import ops

    from zea import display

    # data range is [0, 1] and type is float32
    if pattern_creator == "create_radial_pattern":
        polar_data = create_radial_pattern(size)
    elif pattern_creator == "create_concentric_rings":
        polar_data = create_concentric_rings(size)
    else:
        raise ValueError("Unknown pattern creator")

    rho_range = (0, 100)

    theta_range = np.deg2rad((-45, 45))

    # Scan convert
    cartesian_data, _ = display.scan_convert_2d(polar_data, rho_range, theta_range)

    # Inverse scan convert
    cartesian_data_inv = display.inverse_scan_convert_2d(
        cartesian_data,
        output_size=polar_data.shape,
        find_scan_cone=False,
        theta_range=theta_range,
    )
    cartesian_data_inv = ops.convert_to_numpy(cartesian_data_inv)

    mean_squared_error = ((polar_data - cartesian_data_inv) ** 2).mean()

    assert mean_squared_error < allowed_error, f"MSE is too high: {mean_squared_error:.4f}"

    return cartesian_data_inv


@pytest.mark.parametrize(
    "size, pattern_creator, allowed_error",
    [
        ((200, 200), "create_radial_pattern", 0.0015),
        ((100, 333), "create_radial_pattern", 0.0015),
        ((200, 200), "create_concentric_rings", 0.1),
        ((100, 333), "create_concentric_rings", 0.1),
    ],
)
@backend_equality_check(decimal=2)
def test_scan_conversion_and_inverse_padded(size, pattern_creator, allowed_error):
    """Tests the scan_conversion function with structured test patterns and
    inverts the data with inverse_scan_convert_2d. In this case, the scan cone is
    padded such that it is no longer centered and cropped. find_scan_cone=True is
    used to automatically crop and center the scan cone.
    """
    from keras import ops

    from zea import display

    if pattern_creator == "create_radial_pattern":
        polar_data = create_radial_pattern(size)
    elif pattern_creator == "create_concentric_rings":
        polar_data = create_concentric_rings(size)
    else:
        raise ValueError("Unknown pattern creator")

    rho_range = (0, 100)
    theta_range = np.deg2rad((-45, 45))

    cartesian_data, _ = display.scan_convert_2d(polar_data, rho_range, theta_range)
    cartesian_data = ops.convert_to_numpy(cartesian_data)

    # now pad the cartesian image and test with find_scan_cone=True
    left_padding = ops.zeros((ops.shape(cartesian_data)[0], 20))
    cartesian_data_padded = ops.concatenate([left_padding, cartesian_data], axis=1)
    top_padding = ops.zeros((20, ops.shape(cartesian_data_padded)[1]))
    cartesian_data_padded = ops.concatenate([top_padding, cartesian_data_padded], axis=0)
    cartesian_data_inv = display.inverse_scan_convert_2d(
        cartesian_data_padded, output_size=polar_data.shape, find_scan_cone=True, image_range=(0, 1)
    )
    cartesian_data_inv = ops.convert_to_numpy(cartesian_data_inv)
    mean_squared_error = ((polar_data - cartesian_data_inv) ** 2).mean()

    assert mean_squared_error < allowed_error, f"MSE is too high: {mean_squared_error:.4f}"

    return cartesian_data_inv


@backend_equality_check()
def test_polar_to_cartesian_matrix_roundtrip():
    """polar_to_cartesian_matrix is a faithful inverse of cartesian_to_polar_matrix.

    Unlike scan_convert_2d (which fits the cone bounding box into the output), it pins the
    apex at ``tip`` on a full-size canvas, so a forward/inverse round-trip reconstructs the
    original frame at its original position and scale. Uses an *asymmetric* theta_range to
    guard against an angular shift: the inverse is obtained by passing theta_range in the
    reversed order (matching cartesian_to_polar_matrix's rot90 column ordering, which is what
    polar_geometry_from_coords_for_interp returns).
    """
    from keras import ops

    from zea import display

    height, width = 220, 300
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    apex_x, apex_y = 170.0, 30.0
    # cartesian_to_polar_matrix convention: theta = arctan2(-(x - apex_x), y - apex_y).
    r = np.sqrt((xx - apex_x) ** 2 + (yy - apex_y) ** 2)
    theta = np.arctan2(-(xx - apex_x), yy - apex_y)
    theta_min, theta_max = -0.3, 0.6  # asymmetric cone
    mask = (theta > theta_min) & (theta < theta_max) & (r < 180)
    blob = np.exp(-(((xx - 150) / 12) ** 2 + ((yy - 150) / 12) ** 2))  # off-centre marker
    image = np.where(mask, 0.3 + blob, 0.0).astype("float32")

    polar = display.cartesian_to_polar_matrix(
        ops.convert_to_tensor(image),
        tip=(apex_x, apex_y),
        r_max=180.0,
        theta_range=(theta_min, theta_max),
    )
    # Invert: theta_range reversed to match the polar image's column order.
    back = display.polar_to_cartesian_matrix(
        polar,
        (height, width),
        tip=(apex_x, apex_y),
        r_max=180.0,
        theta_range=(theta_max, theta_min),
    )
    back = np.nan_to_num(ops.convert_to_numpy(back))

    assert back.shape == (height, width)
    # Faithful reconstruction inside the cone (residual is double-resampling blur).
    assert np.abs(back - image)[mask].mean() < 0.1
    # The off-centre marker returns to its location (an angular shift would move it).
    by, bx = np.unravel_index(np.argmax(back), back.shape)
    assert abs(by - 150) < 12 and abs(bx - 150) < 12

    return back


@pytest.mark.parametrize(
    "size, dynamic_range",
    [
        ((2, 1, 128, 32), (-30, -5)),
        ((512, 512), None),
        ((1, 128, 32), None),
    ],
)
def test_converting_to_image(size, dynamic_range):
    """Test converting to image functions"""
    # create random data between dynamic range
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    if dynamic_range is None:
        _dynamic_range = (-60, 0)
    else:
        _dynamic_range = dynamic_range

    data = rng.standard_normal(size) * (_dynamic_range[1] - _dynamic_range[0]) + _dynamic_range[0]
    _data = display.to_8bit(data, dynamic_range, pillow=False)
    assert np.all(np.logical_and(_data >= 0, _data <= 255))
    assert _data.dtype == "uint8"


@pytest.mark.parametrize(
    "dtype, order",
    [
        ("float16", 0),
        ("float16", 1),
        ("float16", 2),
        ("float32", 0),
        ("float32", 1),
        ("float32", 2),
    ],
)
def test_map_coordinates_dtype(dtype, order):
    """Test map_coordinates with different data types and interpolation orders.

    This test verifies that map_coordinates works correctly with float16 and float32
    inputs across different interpolation orders.
    """
    from keras import ops

    from zea import display

    # Create a simple 2D test image
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image = rng.random((32, 32)).astype(dtype)

    # Create simple coordinates for interpolation
    # Sample points at fractional positions to test interpolation
    coords = np.array(
        [
            [15.5, 16.2, 20.1, 10.3],  # y coordinates
            [15.5, 14.8, 18.7, 12.9],  # x coordinates
        ],
        dtype="float32",
    )

    # Convert to ops tensors
    image_tensor = ops.convert_to_tensor(image)
    coords_tensor = ops.convert_to_tensor(coords)

    # Perform map_coordinates
    result = display.map_coordinates(
        image_tensor, coords_tensor, order=order, fill_mode="constant", fill_value=0.0
    )

    # Convert result to numpy for assertions
    result_np = ops.convert_to_numpy(result)

    # Basic sanity checks
    assert result_np.shape == (4,), f"Expected shape (4,), got {result_np.shape}"
    assert not np.any(np.isnan(result_np)), "Result contains NaN values"
    assert not np.all(result_np == 0), "Result is all zeros (likely failed)"

    # The output dtype should always match the input dtype
    assert result_np.dtype == np.dtype(dtype), (
        f"Expected output dtype {dtype}, got {result_np.dtype}"
    )

    # Verify interpolated values are within reasonable range
    assert np.all(result_np >= 0) and np.all(result_np <= 1), (
        f"Interpolated values out of expected range [0, 1]: {result_np}"
    )


@pytest.mark.parametrize(
    "image_mode, num_masks, alpha, use_colors",
    [
        ("L", 1, 0.5, False),  # grayscale image, single mask
        ("RGB", 2, 0.3, False),  # RGB image, two masks, default colors
        ("L", 3, 1.0, True),  # custom colors for all masks
        ("L", 1, 0.0, False),  # alpha=0 → no overlay painted
    ],
)
def test_overlay_masks(image_mode, num_masks, alpha, use_colors):
    """Test overlay_masks composites masks onto an image correctly."""
    from PIL import Image

    from zea.display import overlay_masks

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    h, w = 64, 64

    img_arr = rng.integers(0, 255, (h, w) if image_mode == "L" else (h, w, 3), dtype=np.uint8)
    image = Image.fromarray(img_arr, mode=image_mode)

    # Build binary masks — each mask covers a distinct quarter of the image
    masks = []
    for i in range(num_masks):
        m = np.zeros((h, w), dtype=np.uint8)
        m[: h // 2, : w // 2] = 255
        masks.append(Image.fromarray(m, mode="L"))

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][:num_masks] if use_colors else None

    result = overlay_masks(image, masks, alpha=alpha, colors=colors)

    assert isinstance(result, Image.Image), "Result should be a PIL Image"
    assert result.mode == "RGB", f"Result mode should be RGB, got {result.mode}"
    assert result.size == (w, h), f"Result size mismatch: {result.size} != {(w, h)}"

    result_arr = np.array(result)

    # The unmasked region (bottom-right quarter) should be unchanged from the base image
    base_rgb = np.array(image.convert("RGB"))
    np.testing.assert_array_equal(
        result_arr[h // 2 :, w // 2 :],
        base_rgb[h // 2 :, w // 2 :],
        err_msg="Unmasked region was modified",
    )

    if alpha > 0:
        # At least some pixels in the masked region should differ from the base
        masked_region_changed = not np.array_equal(
            result_arr[: h // 2, : w // 2], base_rgb[: h // 2, : w // 2]
        )
        assert masked_region_changed, "Masked region should differ from base image when alpha > 0"


def test_overlay_masks_ndarray_inputs():
    """Test overlay_masks accepts ndarray image and ndarray masks (non-PIL inputs)."""
    from zea.display import overlay_masks

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    h, w = 64, 64

    # Pass raw ndarrays instead of PIL Images to exercise the conversion branches
    image_arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    mask_arr = np.zeros((h, w), dtype=np.uint8)
    mask_arr[: h // 2, : w // 2] = 255

    result = overlay_masks(image_arr, [mask_arr], alpha=0.5)

    assert isinstance(result, __import__("PIL").Image.Image), "Result should be a PIL Image"
    assert result.mode == "RGB"
    assert result.size == (w, h)


def test_overlay_masks_non_L_mask():
    """Test overlay_masks converts non-'L' mode PIL masks to 'L' mode."""
    from PIL import Image

    from zea.display import overlay_masks

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    h, w = 64, 64

    image = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), mode="RGB")

    # Create an RGB mask (mode != "L") to trigger the mask.convert("L") branch
    mask_arr = np.zeros((h, w, 3), dtype=np.uint8)
    mask_arr[: h // 2, : w // 2] = 255
    mask_rgb = Image.fromarray(mask_arr, mode="RGB")

    result = overlay_masks(image, [mask_rgb], alpha=0.5)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (w, h)


def _speckle(rng, size=(128, 128), scale=1.0):
    """Log-compressed fully developed speckle, i.e. a log-Rayleigh distributed image (dB)."""
    return 20 * np.log10(rng.rayleigh(scale, size))


def _quantile_error(image, reference, n_levels=100):
    """Largest difference between the quantile functions of two images (dB)."""
    levels = np.linspace(0, 1, n_levels)
    return np.abs(np.quantile(image, levels) - np.quantile(reference, levels)).max()


# The matching variations of the paper, as keyword arguments to `histogram_match`.
MATCHES = {"partial": {"mode": "partial"}, "full": {}, "exact": {"n_levels": "all"}}
MONOTONE_MATCHES = {name: MATCHES[name] for name in ("full", "exact")}


def test_histogram_match_all_levels_matches_skimage():
    """One level per pixel is rank transport, i.e. skimage's match_histograms.

    ``match_histograms`` bins with ``np.unique`` for anything but integer dtype, which on dB
    floats is one bin per pixel, i.e. the limit ``n_levels="all"`` asks for.
    """
    from skimage.exposure import match_histograms

    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)

    matched = histogram_match(image, reference, n_levels="all")

    np.testing.assert_allclose(matched, match_histograms(image, reference))
    # Rank transport reproduces the reference distribution exactly, not just closely.
    np.testing.assert_allclose(np.sort(matched.ravel()), np.sort(reference.ravel()))


@pytest.mark.parametrize("kwargs", MATCHES.values(), ids=MATCHES)
def test_histogram_match_is_monotone(kwargs):
    """Every variation is a monotone map, so the ranking of pixel values is preserved."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)

    matched = histogram_match(image, reference, **kwargs)

    assert matched.shape == image.shape
    order = np.argsort(image.ravel())
    assert np.all(np.diff(matched.ravel()[order]) >= 0)


@pytest.mark.parametrize("kwargs", MONOTONE_MATCHES.values(), ids=MONOTONE_MATCHES)
def test_histogram_match_reproduces_distribution(kwargs):
    """A full match reproduces the reference distribution, at any number of levels."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image = _speckle(rng, size=(128, 128), scale=0.1)
    reference = _speckle(rng, size=(64, 96), scale=1.0)

    matched = histogram_match(image, reference, **kwargs)

    assert _quantile_error(image, reference) > 10, "Unmatched images should differ a lot"
    assert _quantile_error(matched, reference) < 0.5


def test_histogram_match_partial_recovers_known_transform():
    """A partial (affine) match undoes a scale and offset applied to the reference."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    reference = _speckle(rng)
    image = (reference - 3.0) / 2.0  # e.g. a square law detector, doubled after compression

    matched = histogram_match(image, reference, mode="partial")

    np.testing.assert_allclose(matched, reference)


def test_histogram_match_partial_preserves_ssnr():
    """The partial match matches mean and variance, hence the sSNR, of the fitted region."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)
    # Compounding-like: a different distribution shape, not just a shifted or scaled one.
    image = 0.5 * (image + _speckle(rng, scale=0.1))

    roi = np.zeros(image.shape, dtype=bool)
    roi[32:96, 32:96] = True
    matched = histogram_match(image, reference, mode="partial", roi=roi)

    assert matched[roi].mean() == pytest.approx(reference[roi].mean())
    assert matched[roi].std() == pytest.approx(reference[roi].std())
    # Shape differences remain: that is what a full match is for.
    assert _quantile_error(matched[roi], reference[roi]) > 1


@pytest.mark.parametrize("kwargs", MONOTONE_MATCHES.values(), ids=MONOTONE_MATCHES)
def test_histogram_match_roi_is_fit_on_roi_and_extends_linearly(kwargs):
    """With an ROI the fit uses only that region, and brighter pixels stay brighter."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)

    roi = np.zeros(image.shape, dtype=bool)
    roi[32:96, 32:96] = True
    # A point target outside the ROI, brighter than any speckle pixel in it.
    image[10, 10] = image[roi].max() + 20

    matched = histogram_match(image, reference, roi=roi, **kwargs)

    assert _quantile_error(matched[roi], reference[roi]) < 0.5
    # Linear extension rather than clipping: the point target remains the brightest pixel.
    assert matched[10, 10] > matched[roi].max()


@pytest.mark.parametrize("kwargs", MATCHES.values(), ids=MATCHES)
def test_histogram_match_ignores_dead_pixels(kwargs):
    """The log(0) sentinel neither anchors the mapping nor turns into a bright pixel."""
    from zea.display import DEAD_PIXEL_DB, histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)

    # Two images with the same valid pixels, differing only in how the invalid ones are
    # marked: NaN (always excluded) versus zea's log_compress sentinel for zeros.
    image[:2] = np.nan
    dead = image.copy()
    dead[:2] = 20 * np.log10(1e-16)

    matched = histogram_match(image, reference, **kwargs)
    matched_dead = histogram_match(dead, reference, **kwargs)

    np.testing.assert_allclose(matched_dead[2:], matched[2:])
    assert np.all(np.isnan(matched[:2])), "Non-finite pixels should be passed through"
    assert np.all(matched_dead[:2] < DEAD_PIXEL_DB), "Sentinel pixels should stay below the floor"


@pytest.mark.parametrize("kwargs", MATCHES.values(), ids=MATCHES)
def test_histogram_match_ignores_dead_reference_pixels(kwargs):
    """Dead pixels in the reference do not enter the transform being fitted either."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)

    dead = reference.copy()
    dead[:2] = np.nan  # e.g. the fill value of a scan conversion
    dead[2:4] = 20 * np.log10(1e-16)

    matched = histogram_match(image, reference[4:], **kwargs)
    matched_dead = histogram_match(image, dead, **kwargs)

    np.testing.assert_allclose(matched_dead, matched)


def test_histogram_match_extension_is_bounded_for_clipped_images():
    """An image piled up on its clipping floor collapses knots; the extension stays sane."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image = np.clip(_speckle(rng, scale=0.1), -20, None)  # ~40% of the pixels on the floor
    reference = _speckle(rng, scale=1.0)

    roi = np.zeros(image.shape, dtype=bool)
    roi[32:96, 32:96] = True
    assert (image[roi] == -20).mean() > 0.1, "Test premise: an atom at the bottom of the ROI"
    image[0, 0] = -30.0  # a dark pixel 10 dB below the atom the bottom knot collapses onto

    matched = histogram_match(image, reference, roi=roi)

    assert np.isfinite(matched).all()
    # Extending with the secant across the atom would send this pixel hundreds of dB down
    # instead of the ~10 dB the input is below the floor it was clipped to.
    floor = matched[image == -20].min()
    assert floor - 50 < matched[0, 0] < floor


def test_histogram_match_accepts_tensors():
    """Tensor images and masks are accepted, like elsewhere in this module."""
    from keras import ops

    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, scale=0.1), _speckle(rng, scale=1.0)
    roi = np.zeros(image.shape, dtype=bool)
    roi[32:96, 32:96] = True

    expected = histogram_match(image, reference, roi=roi)
    matched = histogram_match(
        ops.convert_to_tensor(image),
        ops.convert_to_tensor(reference),
        roi=ops.convert_to_tensor(roi),
    )

    np.testing.assert_allclose(matched, expected, rtol=1e-6)


def test_histogram_match_invalid_arguments():
    """Unknown modes, levels and too small fitting regions raise informative errors."""
    from zea.display import histogram_match

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    image, reference = _speckle(rng, size=(8, 8)), _speckle(rng, size=(8, 8))

    with pytest.raises(ValueError, match="Unknown mode"):
        histogram_match(image, reference, mode="adaptive")

    with pytest.raises(ValueError, match="Invalid n_levels"):
        histogram_match(image, reference, n_levels="every")

    with pytest.raises(ValueError, match="Too few valid pixels"):
        histogram_match(image, reference, n_levels=256)

    # Fitting a transform needs something to fit: an image of one value has no distribution.
    flat, speckle = np.zeros((32, 32)), _speckle(rng, size=(32, 32))
    for kwargs in MATCHES.values():
        with pytest.raises(ValueError, match="the same value"):
            histogram_match(flat, speckle, **kwargs)
