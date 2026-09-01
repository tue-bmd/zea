"""Tests for :mod:`zea.tools.selection_tool`.

The tool is interactive by nature, so the matplotlib widget that collects the user's
mouse input is replaced by :class:`_FakeSelector` (see ``fake_selector``). Everything
below that -- mask building, interpolation, file handling, prompts and the
``zea tools select`` orchestration -- is exercised for real.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from zea.tools import selection_tool  # noqa: E402
from zea.tools.selection_tool import (  # noqa: E402
    SELECTORS,
    annotate_sequence,
    ask_for_num_selections,
    ask_for_selection_tool,
    ask_for_title,
    ask_save_animation_with_fps,
    ask_for_files,
    crop_array,
    equalize_polygons,
    extract_polygon_from_mask,
    extract_rectangle_from_mask,
    interactive_selector,
    interpolate_masks,
    interpolate_polygons,
    interpolate_rectangles,
    load_input_files,
    match_polygons,
    normalize_title,
    reconstruct_mask_from_polygon,
    reconstruct_mask_from_rectangle,
    remove_masks_from_axs,
    run_selection_tool,
    save_mask_animation,
    save_masks,
    update_imshow_with_mask,
    wait_for_key,
)


# ── helpers ───────────────────────────────────────────────────────────────────


class _Event:
    """Stand-in for the matplotlib mouse events a RectangleSelector passes on."""

    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata


def _make_fake_selector(regions):
    """Build a selector class that immediately 'draws' ``regions`` and returns.

    Args:
        regions: list of ``((x0, y0), (x1, y1))`` corner pairs, one per selection.
    """

    class _FakeSelector:
        def __init__(self, ax, onselect, **kwargs):
            self.ax = ax
            for (x0, y0), (x1, y1) in regions:
                if getattr(self, "_is_lasso", False):
                    onselect([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
                else:
                    onselect(_Event(x0, y0), _Event(x1, y1))

        def disconnect_events(self):
            pass

        def set_visible(self, visible):
            pass

        def update(self):
            pass

    return _FakeSelector


@pytest.fixture
def fake_selector(monkeypatch):
    """Replace the matplotlib selector widget with one that selects fixed regions.

    Yields a callable that installs the fake for a list of ``((x0, y0), (x1, y1))``
    regions, so a test can pick the boxes the 'user' draws.
    """

    def install(regions, selector="rectangle"):
        cls = _make_fake_selector(regions)
        if selector == "lasso":
            cls._is_lasso = True
            monkeypatch.setattr(selection_tool, "LassoSelector", cls)
        else:
            monkeypatch.setattr(selection_tool, "RectangleSelector", cls)
        return cls

    return install


@pytest.fixture
def gif_path(tmp_path):
    """Write a small 8-frame gif and return its path."""
    frames = [Image.fromarray(np.full((24, 32), 40 + 10 * i, dtype=np.uint8)) for i in range(8)]
    path = tmp_path / "clip.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
    return path


@pytest.fixture
def image_paths(tmp_path):
    """Write two small png images and return their paths."""
    paths = []
    for i in range(2):
        path = tmp_path / f"frame_{i}.png"
        Image.fromarray(np.full((24, 32), 60 + 40 * i, dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ── array helpers ─────────────────────────────────────────────────────────────


def test_crop_array_removes_empty_rows_and_columns():
    array = np.zeros((6, 5))
    array[2:4, 1:3] = 7
    np.testing.assert_array_equal(crop_array(array, value=0), np.full((2, 2), 7.0))


def test_crop_array_default_value_is_a_no_op():
    """The default ``value=None`` matches nothing, so the array comes back unchanged."""
    array = np.zeros((3, 4))
    np.testing.assert_array_equal(crop_array(array), array)


def test_crop_array_rejects_non_2d():
    with pytest.raises(AssertionError):
        crop_array(np.zeros((2, 3, 4)), value=0)


# ── rectangles and polygons ───────────────────────────────────────────────────


def test_rectangles():
    """Test rectangle extraction / reconstruction."""
    # create random rectangle mask
    mask = np.zeros((120, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    # extract rectangle
    rect = extract_rectangle_from_mask(mask)
    # reconstruct mask
    mask_reconstructed = reconstruct_mask_from_rectangle(rect, mask.shape)
    assert np.all(mask == mask_reconstructed)


def test_extract_rectangle_from_empty_mask_returns_none():
    assert extract_rectangle_from_mask(np.zeros((10, 10))) is None


def test_polygon():
    """Test polygon extraction / reconstruction."""
    # create random polygon mask
    mask = np.zeros((120, 101))
    mask[10:20, 10:20] = 1
    mask[20:30, 20:30] = 1
    mask[30:40, 30:40] = 1

    # extract polygon
    poly = extract_polygon_from_mask(mask, 0.0)
    # reconstruct mask
    mask_reconstructed = reconstruct_mask_from_polygon(poly, mask.shape)
    np.testing.assert_array_almost_equal(mask, mask_reconstructed, 0.1)


def test_extract_polygon_from_empty_mask_returns_none():
    assert extract_polygon_from_mask(np.zeros((10, 10)), verbose=False) is None


def test_extract_polygon_picks_largest_contour():
    """Two disjoint blobs: only the biggest one is returned."""
    mask = np.zeros((60, 60))
    mask[5:10, 5:10] = 1
    mask[20:50, 20:50] = 1

    poly = extract_polygon_from_mask(mask, tolerance=0.0, verbose=False)
    # the polygon must live inside the bounding box of the large blob
    assert poly[:, 0].min() >= 19 and poly[:, 0].max() <= 50
    assert poly[:, 1].min() >= 19 and poly[:, 1].max() <= 50


@pytest.mark.parametrize(
    "mode",
    ["min", "max"],
)
def test_equalize_polygons(mode):
    """Test polygon equalization."""
    # make some random polygons
    poly1 = np.array([[1, 1], [2, 2], [3, 3]])
    poly2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    poly3 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    # equalize
    polygons = (poly1, poly2, poly3)
    polygons = equalize_polygons(polygons, mode=mode)
    assert len(polygons) == 3
    # same length for all elements in list
    assert len(set(len(poly) for poly in polygons)) == 1
    if mode == "min":
        assert len(polygons[0]) == 3
    elif mode == "max":
        assert len(polygons[0]) == 5


def test_equalize_polygons_rejects_unknown_mode():
    poly = np.array([[1, 1], [2, 2], [3, 3]])
    with pytest.raises(AssertionError):
        equalize_polygons([poly, poly], mode="median")


def test_match_polygons():
    """Test polygon matching."""
    # make some random polygons
    poly1 = np.array([[1, 1], [2, 2], [3, 3]])
    poly2 = np.array([[1, 1], [2, 2], [3, 3]])

    # match
    poly1, poly2 = match_polygons(poly1, poly2)
    assert np.all(poly1 == poly2)

    poly1 = np.array([[1, 1], [2, 2], [3, 3]])
    poly2 = np.array([[2, 2], [3, 3], [1, 1]])

    poly1, poly2 = match_polygons(poly1, poly2)
    assert np.all(poly1 == poly2)


def test_interpolate_polygons_endpoints_and_midpoint():
    poly1 = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    poly2 = poly1 + 10.0

    np.testing.assert_allclose(interpolate_polygons(poly1, poly2, 0.0), poly1)
    np.testing.assert_allclose(interpolate_polygons(poly1, poly2, 1.0), poly2)
    np.testing.assert_allclose(interpolate_polygons(poly1, poly2, 0.5), poly1 + 5.0)


def test_interpolate_polygons_requires_equal_vertex_count():
    with pytest.raises(ValueError):
        interpolate_polygons(np.zeros((3, 2)), np.zeros((4, 2)), 0.5)


def test_interpolate_rectangles():
    rectangles = [((0, 0), (10, 10)), ((10, 10), (20, 20))]
    interpolated = interpolate_rectangles(rectangles, np.array([0, 4]), np.arange(5))

    assert len(interpolated) == 5
    assert interpolated[0] == ((0, 0), (10, 10))
    assert interpolated[-1] == ((10, 10), (20, 20))
    # strictly moving from the first to the last rectangle
    assert interpolated[2] == ((5, 5), (15, 15))


# ── mask interpolation ────────────────────────────────────────────────────────


def test_interpolate_masks_rectangle():
    mask1 = reconstruct_mask_from_rectangle(((2, 2), (8, 8)), (40, 40))
    mask2 = reconstruct_mask_from_rectangle(((12, 12), (18, 18)), (40, 40))

    masks = interpolate_masks([mask1, mask2], num_frames=6, rectangle=True)

    assert len(masks) == 6
    assert all(mask.shape == (40, 40) for mask in masks)
    np.testing.assert_array_equal(masks[0], mask1)
    np.testing.assert_array_equal(masks[-1], mask2)
    # the box travels monotonically from the first to the second position
    tops = [extract_rectangle_from_mask(mask)[0][1] for mask in masks]
    assert tops == sorted(tops)


def test_interpolate_masks_polygon():
    mask1 = np.zeros((60, 60), dtype=np.uint8)
    mask1[10:25, 10:25] = 1
    mask2 = np.zeros((60, 60), dtype=np.uint8)
    mask2[30:45, 30:45] = 1

    masks = interpolate_masks([mask1, mask2], num_frames=5, rectangle=False)

    assert len(masks) == 5
    assert all(mask.shape == (60, 60) for mask in masks)
    assert all(mask.sum() > 0 for mask in masks)
    # the centre of mass moves down-right across the sequence
    centres = [np.argwhere(mask).mean(axis=0)[0] for mask in masks]
    assert centres == sorted(centres)


@pytest.mark.parametrize("rectangle", [True, False])
def test_interpolate_masks_honours_positions(rectangle):
    """Masks land on the frames they belong to, and the last one is held afterwards."""
    mask1 = reconstruct_mask_from_rectangle(((2, 2), (8, 8)), (40, 40))
    mask2 = reconstruct_mask_from_rectangle(((2, 22), (8, 28)), (40, 40))

    # both masks are annotated early on, so frames 2-5 have no key frame to move towards
    masks = interpolate_masks([mask1, mask2], num_frames=6, rectangle=rectangle, positions=[0, 1])

    # the polygon round trip is not pixel exact, so allow a pixel of slack
    tops = [extract_rectangle_from_mask(mask)[0][1] for mask in masks]
    assert abs(tops[0] - 2) <= 1
    # frame 1 carries the second mask, and every later frame holds on to it
    assert all(abs(top - 22) <= 1 for top in tops[1:])
    assert len(set(tops[1:])) == 1


def test_interpolate_masks_rejects_bad_positions():
    mask = reconstruct_mask_from_rectangle(((2, 2), (8, 8)), (40, 40))

    with pytest.raises(AssertionError, match="One position per mask"):
        interpolate_masks([mask, mask], num_frames=4, rectangle=True, positions=[0])
    with pytest.raises(AssertionError, match="strictly increasing"):
        interpolate_masks([mask, mask], num_frames=4, rectangle=True, positions=[2, 2])
    # only frames 0..num_frames-1 are rendered, so num_frames itself is out of range
    with pytest.raises(AssertionError, match="frame indices"):
        interpolate_masks([mask, mask], num_frames=4, rectangle=True, positions=[0, 4])
    with pytest.raises(AssertionError, match="frame indices"):
        interpolate_masks([mask, mask], num_frames=4, rectangle=True, positions=[-1, 2])
    # only whole frames are rendered, so a mask cannot sit between two of them
    with pytest.raises(AssertionError, match="frame indices"):
        interpolate_masks([mask, mask], num_frames=4, rectangle=True, positions=[0.5, 2.5])


@pytest.mark.parametrize(
    "masks,num_frames",
    [
        ([np.zeros((4, 4))], 4),  # a single mask cannot be interpolated
        ([np.zeros((4, 4)), np.zeros((4, 4))], 1),  # at least two frames required
        ([np.zeros((4, 4)), np.zeros((5, 5))], 4),  # shapes must match
    ],
)
def test_interpolate_masks_validates_inputs(masks, num_frames):
    with pytest.raises(AssertionError):
        interpolate_masks(masks, num_frames=num_frames, rectangle=True)


# ── plotting helpers ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("selector", SELECTORS)
def test_update_imshow_with_mask(selector):
    images = np.random.default_rng(0).integers(0, 255, size=(3, 20, 20)).astype(np.uint8)
    masks = np.zeros((3, 20, 20), dtype=bool)
    for i in range(3):
        masks[i, i : i + 5, i : i + 5] = True

    _, axs = plt.subplots()
    imshow_obj = axs.imshow(images[0], cmap="gray")

    updated, mask_obj = update_imshow_with_mask(2, axs, imshow_obj, images, masks, selector)

    assert updated is imshow_obj
    assert mask_obj is not None
    np.testing.assert_array_equal(updated.get_array(), images[2])


def test_remove_masks_from_axs():
    from zea.visualize import plot_rectangle_from_mask

    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    _, axs = plt.subplots()
    axs.imshow(np.zeros((20, 20)), cmap="gray")
    plot_rectangle_from_mask(axs, mask)
    assert len(axs.patches) == 1

    remove_masks_from_axs(axs)
    assert len(axs.patches) == 0


# ── prompts ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [("Left Ventricle", "left_ventricle"), ("  Septum  ", "septum"), ("ROI 1", "roi_1")],
)
def test_normalize_title(raw, expected):
    assert normalize_title(raw) == expected


@pytest.mark.parametrize("raw", ["", "   "])
def test_normalize_title_rejects_empty(raw):
    with pytest.raises(ValueError):
        normalize_title(raw)


def _answers(monkeypatch, *replies):
    """Feed ``replies`` to successive ``input()`` calls."""
    it = iter(replies)
    monkeypatch.setattr("builtins.input", lambda *_: next(it))


def test_ask_for_title_retries_until_non_empty(monkeypatch):
    _answers(monkeypatch, "", "  ", "Left Ventricle")
    assert ask_for_title() == "left_ventricle"


def test_ask_for_selection_tool_retries_until_valid(monkeypatch):
    _answers(monkeypatch, "circle", "", "lasso")
    assert ask_for_selection_tool() == "lasso"


def test_ask_for_num_selections_rejects_non_positive(monkeypatch):
    _answers(monkeypatch, "not-a-number", "0", "-1", "3")
    assert ask_for_num_selections() == 3


def test_ask_save_animation_with_fps(monkeypatch):
    _answers(monkeypatch, "abc", "0", "20")
    assert ask_save_animation_with_fps() == 20


# ── input handling ────────────────────────────────────────────────────────────


def test_load_input_files_images(image_paths):
    inputs = load_input_files(image_paths)

    assert inputs.is_sequence is False
    assert len(inputs.images) == 2
    assert inputs.file_names == [path.name for path in image_paths]
    assert all(image.shape == (24, 32) for image in inputs.images)
    assert inputs.source is None


def test_load_input_files_video(gif_path):
    inputs = load_input_files([gif_path])

    assert inputs.is_sequence is True
    assert len(inputs.images) == 8
    assert set(inputs.file_names) == {gif_path.name}


def test_load_input_files_rejects_video_mixed_with_images(gif_path, image_paths):
    with pytest.raises(ValueError, match="single video"):
        load_input_files([image_paths[0], gif_path])


def test_load_input_files_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "scan.mat"
    path.touch()
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_input_files([path])


def test_load_input_files_rejects_empty():
    with pytest.raises(ValueError, match="No input files"):
        load_input_files([])


def test_ask_for_files_stops_on_an_empty_line(monkeypatch, image_paths):
    """Paths are typed one per line; an empty line ends the loop."""
    _answers(monkeypatch, *[str(path) for path in image_paths], "")
    assert ask_for_files() == image_paths


def test_ask_for_files_stops_after_a_sequence(monkeypatch, gif_path):
    """A video ends the loop right away; no second path is asked for."""
    _answers(monkeypatch, str(gif_path))
    assert ask_for_files() == [gif_path]


def test_ask_for_files_rejects_missing_paths(monkeypatch, image_paths):
    """A typo is reported and re-asked, not silently accepted."""
    _answers(monkeypatch, "does/not/exist.png", str(image_paths[0]), "")
    assert ask_for_files() == [image_paths[0]]


def test_ask_for_files_strips_quotes(monkeypatch, gif_path):
    """Dragging a file into a terminal often quotes the path."""
    _answers(monkeypatch, f'"{gif_path}"')
    assert ask_for_files() == [gif_path]


def test_ask_for_files_raises_without_selection(monkeypatch):
    _answers(monkeypatch, "")
    with pytest.raises(ValueError, match="No files selected"):
        ask_for_files()


# ── interactive selector ──────────────────────────────────────────────────────


@pytest.mark.parametrize("selector", SELECTORS)
def test_interactive_selector_returns_masks_and_patches(fake_selector, selector):
    fake_selector([((4, 6), (12, 14))], selector=selector)

    data = np.arange(20 * 20, dtype=float).reshape(20, 20)
    _, ax = plt.subplots()
    ax.imshow(data, cmap="gray")

    patches, masks = interactive_selector(
        data, ax, selector=selector, num_selections=1, confirm_selection=False, verbose=False
    )

    assert len(masks) == len(patches) == 1
    mask = masks[0]
    assert mask.dtype == bool
    assert mask.shape == data.shape
    # the mask covers (roughly) the requested box and nothing outside it
    ys, xs = np.where(mask)
    assert 3 <= xs.min() and xs.max() <= 13
    assert 5 <= ys.min() and ys.max() <= 15
    # the patch is the cropped selection, so it is smaller than the full image
    assert patches[0].shape[0] < data.shape[0]


def test_interactive_selector_multiple_selections(fake_selector):
    fake_selector([((1, 1), (5, 5)), ((10, 10), (15, 15))])

    data = np.ones((20, 20))
    _, ax = plt.subplots()
    ax.imshow(data, cmap="gray")

    _, masks = interactive_selector(
        data, ax, selector="rectangle", num_selections=2, confirm_selection=False, verbose=False
    )

    assert len(masks) == 2
    # the two selections are disjoint
    assert not np.any(masks[0] & masks[1])


def test_interactive_selector_rejects_non_2d_data():
    _, ax = plt.subplots()
    with pytest.raises(AssertionError):
        interactive_selector(np.zeros((2, 3, 4)), ax, confirm_selection=False)


def test_interactive_selector_rejects_unknown_selector():
    _, ax = plt.subplots()
    with pytest.raises(AssertionError):
        interactive_selector(np.zeros((4, 4)), ax, selector="circle", confirm_selection=False)


def test_interactive_selector_with_extent(fake_selector):
    """With an extent, the drawn coordinates are translated back to pixel indices."""
    # the axis spans 0..2 in x and 0..2 in y, while the data is 20x20 pixels
    fake_selector([((0.5, 0.5), (1.0, 1.0))])

    data = np.ones((20, 20))
    extent = [0, 2, 0, 2]
    _, ax = plt.subplots()
    ax.imshow(data, cmap="gray", extent=extent)

    _, masks = interactive_selector(
        data,
        ax,
        selector="rectangle",
        extent=extent,
        num_selections=1,
        confirm_selection=False,
        verbose=False,
    )

    ys, xs = np.where(masks[0])
    # 0.5..1.0 of a 0..2 axis maps to pixels 5..10
    assert 4 <= xs.min() <= 6 and 9 <= xs.max() <= 11
    assert 4 <= ys.min() <= 6 and 9 <= ys.max() <= 11


def test_interactive_selector_with_plot_and_metric_requires_two_selections(fake_selector):
    from zea.tools.selection_tool import interactive_selector_with_plot_and_metric

    # three selections where the wrapper expects exactly two
    fake_selector([((1, 1), (5, 5)), ((6, 6), (9, 9)), ((10, 10), (15, 15))])

    with pytest.raises(ValueError, match="exactly 2 patches"):
        interactive_selector_with_plot_and_metric(
            np.ones((20, 20)), metric=None, confirm_selection=False, verbose=False
        )


def test_interactive_selector_with_plot_and_metric_without_metric(fake_selector):
    from zea.tools.selection_tool import interactive_selector_with_plot_and_metric

    fake_selector([((1, 1), (5, 5)), ((10, 10), (15, 15))])

    _, axes = plt.subplots(1, 2)
    for axis in axes:
        axis.imshow(np.ones((20, 20)), cmap="gray")

    scores = interactive_selector_with_plot_and_metric(
        [np.ones((20, 20)), np.ones((20, 20))],
        list(axes),
        metric=None,
        confirm_selection=False,
        verbose=False,
    )

    assert scores == []
    # without a metric there is nothing to title, but the selections are still drawn
    assert all(len(axis.patches) == 2 for axis in axes)


# ── saving ────────────────────────────────────────────────────────────────────


def test_save_masks_writes_a_readable_zea_file(tmp_path):
    from zea import File

    images = np.zeros((2, 5, 5), dtype=np.uint8)
    masks = [np.eye(5, dtype=bool), np.tri(5, dtype=bool)]

    path = save_masks(masks, tmp_path / "nested" / "annotations", images=images, label="lv")

    assert path == tmp_path / "nested" / "annotations.hdf5"
    with File(path) as file:
        np.testing.assert_array_equal(file.data.image.values[:], images)
        np.testing.assert_array_equal(file.data.segmentation.values[..., 0], np.asarray(masks))
        assert list(file.data.segmentation.labels[:]) == ["lv"]


def test_save_masks_carries_source_metadata_over(tmp_path):
    """Coordinates, frame timing and file-level metadata come along; bulk data does not."""
    from zea import File
    from zea.tools.selection_tool import SourceMetadata

    images = np.zeros((2, 4, 6), dtype=np.uint8)
    masks = np.zeros((2, 4, 6), dtype=bool)
    masks[:, 1:3, 2:4] = True
    coordinates = np.zeros((4, 6, 3), dtype=np.float32)
    coordinates[..., 0] = np.arange(6, dtype=np.float32)

    source = SourceMetadata(
        file_fields={"probe": {"name": "GE M5S", "type": "phased"}, "us_machine": "GE Vivid"},
        map_fields={
            "coordinates": coordinates,
            "timestamps": np.array([0.0, 0.02], dtype=np.float32),
            "start_time_offset": np.float32(0.0),
        },
    )

    path = save_masks(masks, tmp_path / "annotations", images=images, label="roi", source=source)

    with File(path) as file:
        # the pixel grid and the frame timing describe both maps
        np.testing.assert_allclose(file.data.image.coordinates[:], coordinates)
        np.testing.assert_allclose(file.data.segmentation.coordinates[:], coordinates)
        np.testing.assert_allclose(file.data.image.timestamps[:], [0.0, 0.02])
        np.testing.assert_allclose(file.data.segmentation.timestamps[:], [0.0, 0.02])
        # file-level metadata survives too
        assert file.probe.name == "GE M5S"
        assert file.us_machine == "GE Vivid"


def test_save_masks_requires_matching_shapes(tmp_path):
    with pytest.raises(AssertionError):
        save_masks(
            np.zeros((2, 4, 4), dtype=bool), tmp_path / "a", images=np.zeros((3, 4, 4), np.uint8)
        )


@pytest.mark.parametrize("selector", SELECTORS)
def test_save_mask_animation(tmp_path, selector):
    images = np.zeros((4, 16, 16), dtype=np.uint8)
    masks = np.zeros((4, 16, 16), dtype=bool)
    for i in range(4):
        masks[i, i : i + 4, i : i + 4] = True

    path = save_mask_animation(images, masks, tmp_path / "anim.gif", selector=selector, fps=5)

    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as gif:
        assert gif.n_frames == 4


def test_save_mask_animation_requires_matching_lengths(tmp_path):
    with pytest.raises(AssertionError):
        save_mask_animation(np.zeros((3, 8, 8)), np.zeros((2, 8, 8)), tmp_path / "a.gif")


# ── high level routines ───────────────────────────────────────────────────────


def test_annotate_sequence_returns_one_mask_per_frame(fake_selector):
    fake_selector([((2, 2), (10, 10))])

    images = [np.ones((20, 20)) for _ in range(6)]
    masks = annotate_sequence(
        images, selector="rectangle", num_selections=2, confirm_selection=False
    )

    assert len(masks) == len(images)
    assert all(mask.shape == (20, 20) for mask in masks)
    assert all(mask.sum() > 0 for mask in masks)


def test_annotate_sequence_caps_selections_at_frame_count(fake_selector):
    """Asking for more key frames than there are frames must not crash."""
    fake_selector([((2, 2), (10, 10))])

    images = [np.ones((20, 20)) for _ in range(3)]
    masks = annotate_sequence(
        images, selector="rectangle", num_selections=10, confirm_selection=False
    )
    assert len(masks) == 3


def test_annotate_sequence_stops_when_the_plot_is_closed(monkeypatch, fake_selector):
    """Closing the window keeps the key frames already done instead of raising."""
    fake_selector([((2, 2), (10, 10))])

    calls = []
    real = selection_tool.interactive_selector

    def _close_on_third(*args, **kwargs):
        calls.append(1)
        # the user closes the window on the third key frame without selecting
        if len(calls) >= 3:
            return [], []
        return real(*args, **kwargs)

    monkeypatch.setattr(selection_tool, "interactive_selector", _close_on_third)

    images = [np.ones((20, 20)) for _ in range(6)]
    masks = annotate_sequence(
        images, selector="rectangle", num_selections=4, confirm_selection=False
    )

    assert len(masks) == len(images)
    assert all(mask.sum() > 0 for mask in masks)


def test_annotate_sequence_keeps_masks_at_their_key_frames(monkeypatch):
    """Distinct selections stay on the key frames they were drawn on, early stop or not."""

    def _box(top):
        mask = np.zeros((40, 40))
        mask[top : top + 6, 2:8] = 1
        return mask

    tops_drawn = [0, 20]
    calls = []

    def _draw_then_close(*args, **kwargs):
        calls.append(1)
        if len(calls) > len(tops_drawn):
            return [], []  # the user closes the window on the third key frame
        return [None], [_box(tops_drawn[len(calls) - 1])]

    monkeypatch.setattr(selection_tool, "interactive_selector", _draw_then_close)

    # 6 frames and 4 key frames puts the key frames at 0, 1, 3 and 5
    images = [np.ones((40, 40)) for _ in range(6)]
    masks = annotate_sequence(
        images, selector="rectangle", num_selections=4, confirm_selection=False
    )

    tops = [extract_rectangle_from_mask(mask)[0][1] for mask in masks]
    assert len(masks) == len(images)
    # the two selections sit on key frames 0 and 1, not spread over the whole sequence
    assert tops[0] == tops_drawn[0]
    assert tops[1] == tops_drawn[1]
    # nothing was annotated past key frame 1, so those frames keep the last mask
    assert tops[2:] == [tops_drawn[1]] * 4


def test_annotate_sequence_raises_when_nothing_was_selected(monkeypatch):
    """Closing the very first window leaves nothing to interpolate."""
    monkeypatch.setattr(selection_tool, "interactive_selector", lambda *a, **k: ([], []))

    with pytest.raises(ValueError, match="nothing to interpolate"):
        annotate_sequence(
            [np.ones((20, 20)) for _ in range(4)], num_selections=2, confirm_selection=False
        )


def test_annotate_sequence_requires_multiple_frames(fake_selector):
    fake_selector([((2, 2), (10, 10))])
    with pytest.raises(AssertionError):
        annotate_sequence([np.ones((20, 20))], confirm_selection=False)


def test_run_selection_tool_on_video(fake_selector, gif_path, tmp_path):
    """End-to-end video mode: a zea file and a preview animation land in ``output_dir``."""
    from zea import File

    fake_selector([((4, 4), (12, 12))])

    masks = run_selection_tool(
        files=[gif_path],
        selector="rectangle",
        title="Left Ventricle",
        num_selections=2,
        fps=5,
        output_dir=tmp_path / "out",
        confirm_selection=False,
    )

    assert len(masks) == 8
    stem = tmp_path / "out" / f"{gif_path.stem}_left_ventricle_annotations"
    assert stem.with_suffix(".gif").exists()
    with File(stem.with_suffix(".hdf5")) as file:
        np.testing.assert_array_equal(file.data.segmentation.values[..., 0], np.asarray(masks))
        assert list(file.data.segmentation.labels[:]) == ["left_ventricle"]
        assert file.data.image.values.shape == (8, 24, 32)


def test_run_selection_tool_keeps_dots_in_the_source_name(fake_selector, gif_path, tmp_path):
    """A dotted stem must not swallow the title: `clip.720p.gif` -> `clip.720p_roi_...`."""
    dotted = gif_path.parent / "clip.720p.gif"
    gif_path.rename(dotted)
    fake_selector([((4, 4), (12, 12))])

    run_selection_tool(
        files=[dotted],
        selector="rectangle",
        title="roi",
        num_selections=2,
        save_animation=False,
        output_dir=tmp_path,
        confirm_selection=False,
    )

    assert (tmp_path / "clip.720p_roi_annotations.hdf5").exists()


def test_run_selection_tool_can_skip_the_animation(fake_selector, gif_path, tmp_path):
    fake_selector([((4, 4), (12, 12))])

    run_selection_tool(
        files=[gif_path],
        selector="rectangle",
        title="septum",
        num_selections=2,
        save_animation=False,
        output_dir=tmp_path,
        confirm_selection=False,
    )

    stem = tmp_path / f"{gif_path.stem}_septum_annotations"
    assert stem.with_suffix(".hdf5").exists()
    assert not stem.with_suffix(".gif").exists()


def test_run_selection_tool_defaults_output_next_to_input(fake_selector, gif_path):
    fake_selector([((4, 4), (12, 12))])

    run_selection_tool(
        files=[gif_path],
        selector="rectangle",
        title="roi",
        num_selections=2,
        save_animation=False,
        confirm_selection=False,
    )

    assert (gif_path.parent / f"{gif_path.stem}_roi_annotations.hdf5").exists()


def test_run_selection_tool_prompts_for_missing_options(monkeypatch, fake_selector, gif_path):
    """Nothing but the input file: selector, title, key frames and fps are asked for."""
    fake_selector([((4, 4), (12, 12))])
    _answers(monkeypatch, "rectangle", "Apex", "2", "10")

    masks = run_selection_tool(files=[gif_path], confirm_selection=False)

    assert len(masks) == 8
    assert (gif_path.parent / f"{gif_path.stem}_apex_annotations.gif").exists()


def test_run_selection_tool_asks_for_files_when_given_none(monkeypatch, fake_selector, gif_path):
    fake_selector([((4, 4), (12, 12))])
    _answers(monkeypatch, str(gif_path))

    masks = run_selection_tool(
        selector="rectangle",
        title="roi",
        num_selections=2,
        save_animation=False,
        confirm_selection=False,
    )
    assert len(masks) == 8


def test_run_selection_tool_rejects_unknown_selector(gif_path):
    with pytest.raises(AssertionError):
        run_selection_tool(files=[gif_path], selector="circle")


def test_run_selection_tool_on_images(fake_selector, image_paths):
    """Image mode returns one metric score per image."""
    fake_selector([((2, 2), (8, 8)), ((12, 12), (20, 20))])

    scores = run_selection_tool(
        files=image_paths,
        selector="rectangle",
        metric="gcnr",
        confirm_selection=False,
    )

    assert len(scores) == len(image_paths)
    assert all(0.0 <= float(score) <= 1.0 for score in scores)


# ── zea files in and out ──────────────────────────────────────────────────────


@pytest.fixture
def zea_path(tmp_path):
    """Write a minimal zea file with a 6-frame image map and return its path."""
    from zea.data.file import File

    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(6, 20, 24)).astype(np.uint8)
    coordinates = np.zeros((20, 24, 3), dtype=np.float32)
    coordinates[..., 0] = np.arange(24, dtype=np.float32) * 1e-4
    coordinates[..., 2] = np.arange(20, dtype=np.float32)[:, None] * 1e-4

    path = tmp_path / "scan.hdf5"
    File.create(
        path=path,
        data={"image": {"values": images, "coordinates": coordinates}},
        probe={"name": "GE M5S", "type": "phased"},
        us_machine="GE Vivid",
        metadata={"subject": {"id": "patient0401", "type": "human"}},
        description="A source recording.",
        ignore_warnings=True,
    )
    return path


def test_load_input_files_zea(zea_path):
    inputs = load_input_files([zea_path])

    assert inputs.is_sequence is True
    assert len(inputs.images) == 6
    assert all(image.shape == (20, 24) for image in inputs.images)
    assert inputs.source is not None
    assert inputs.source.map_fields["coordinates"].shape == (20, 24, 3)
    # metadata from the source file is picked up for the annotation file
    assert inputs.source.file_fields["probe"]["name"] == "GE M5S"


def test_load_input_files_zea_without_image_data(tmp_path):
    from tests.data import generate_example_dataset

    path = tmp_path / "raw.hdf5"
    generate_example_dataset(path, n_frames=1, n_ax=64, n_el=8, n_tx=1)

    with pytest.raises(ValueError, match="no 'data/image' group"):
        load_input_files([path])


def test_run_selection_tool_on_zea_file(fake_selector, zea_path, tmp_path):
    """A zea file in gives a zea file out, with the source coordinates carried over."""
    from zea import File

    fake_selector([((4, 4), (12, 12))])

    masks = run_selection_tool(
        files=[zea_path],
        selector="rectangle",
        title="LV endo",
        num_selections=2,
        save_animation=False,
        output_dir=tmp_path / "out",
        confirm_selection=False,
    )

    assert len(masks) == 6
    out = tmp_path / "out" / f"{zea_path.stem}_lv_endo_annotations.hdf5"
    with File(zea_path) as source, File(out) as annotated:
        np.testing.assert_array_equal(annotated.data.image.values[:], source.data.image.values[:])
        np.testing.assert_allclose(
            annotated.data.segmentation.coordinates[:], source.data.image.coordinates[:]
        )
    with File(out) as annotated:
        assert list(annotated.data.segmentation.labels[:]) == ["lv_endo"]
        assert annotated.data.segmentation.values.shape == (6, 20, 24, 1)


def test_run_selection_tool_refuses_to_clobber(fake_selector, gif_path, tmp_path):
    """The guard fires before any annotating happens, so no selections are lost."""
    fake_selector([((4, 4), (12, 12))])
    (tmp_path / f"{gif_path.stem}_roi_annotations.hdf5").write_bytes(b"stale")

    with pytest.raises(FileExistsError):
        run_selection_tool(
            files=[gif_path],
            selector="rectangle",
            title="roi",
            num_selections=2,
            save_animation=False,
            output_dir=tmp_path,
            confirm_selection=False,
        )


def test_run_selection_tool_overwrites_when_asked(fake_selector, gif_path, tmp_path):
    from zea import File

    fake_selector([((4, 4), (12, 12))])
    out = tmp_path / f"{gif_path.stem}_roi_annotations.hdf5"
    out.write_bytes(b"stale")

    run_selection_tool(
        files=[gif_path],
        selector="rectangle",
        title="roi",
        num_selections=2,
        save_animation=False,
        output_dir=tmp_path,
        confirm_selection=False,
        overwrite=True,
    )

    with File(out) as file:
        assert file.data.segmentation.values.shape == (8, 24, 32, 1)


def test_interactive_selector_stops_when_the_plot_is_closed(monkeypatch):
    """Closing the window before finishing must return, not spin forever."""

    class _NeverSelects:
        def __init__(self, ax, onselect, **kwargs):
            self.ax = ax

        def disconnect_events(self):
            pass

        def set_visible(self, visible):
            pass

        def update(self):
            pass

    monkeypatch.setattr(selection_tool, "RectangleSelector", _NeverSelects)
    monkeypatch.setattr(plt, "fignum_exists", lambda _number: False)

    _, ax = plt.subplots()
    patches, masks = interactive_selector(
        np.ones((10, 10)), ax, num_selections=2, confirm_selection=False, verbose=False
    )
    assert patches == [] and masks == []


# ── in-figure confirmation (no tkinter) ───────────────────────────────────────


def _press_keys(monkeypatch, fig, *keys):
    """Deliver `keys` to `fig` one per `plt.pause` call, as a user typing would."""
    from matplotlib.backend_bases import KeyEvent

    pending = list(keys)

    def fake_pause(_interval):
        if pending:
            KeyEvent("key_press_event", fig.canvas, pending.pop(0))._process()

    monkeypatch.setattr(plt, "pause", fake_pause)


def test_wait_for_key_accepts(monkeypatch):
    fig, _ = plt.subplots()
    _press_keys(monkeypatch, fig, "y")

    assert wait_for_key(fig, "press y", accept=("y",), redo=("n",)) is True
    # the instruction is cleaned up again
    assert all("press y" not in text.get_text() for text in fig.texts)


def test_wait_for_key_ignores_unrelated_keys(monkeypatch):
    fig, _ = plt.subplots()
    _press_keys(monkeypatch, fig, "a", "0", "n")

    assert wait_for_key(fig, "press n", accept=("y",), redo=("n",)) is False


def test_wait_for_key_returns_on_a_closed_window(monkeypatch):
    """A user who closes the plot is done, not stuck."""
    fig, _ = plt.subplots()
    monkeypatch.setattr(plt, "fignum_exists", lambda _number: False)

    assert wait_for_key(fig, "press enter") is True


def test_interactive_selector_confirmation_redoes_the_selection(monkeypatch):
    """A redo key runs the selector again; an accept key returns the second round."""
    rounds = []

    class _CountingSelector:
        def __init__(self, ax, onselect, **kwargs):
            rounds.append(len(rounds))
            size = 4 + 8 * len(rounds)
            onselect(_Event(1, 1), _Event(size, size))

        def disconnect_events(self):
            pass

        def set_visible(self, visible):
            pass

        def update(self):
            pass

    monkeypatch.setattr(selection_tool, "RectangleSelector", _CountingSelector)

    data = np.ones((30, 30))
    fig, ax = plt.subplots()
    ax.imshow(data, cmap="gray")
    _press_keys(monkeypatch, fig, "n", "enter")

    patches, masks = interactive_selector(
        data, ax, num_selections=1, confirm_selection=True, verbose=False
    )

    assert len(rounds) == 2  # first round rejected, second accepted
    assert len(masks) == 1
    # the second (larger) box is what comes back
    assert patches[0].shape[0] > 10


def test_redo_keys_avoid_matplotlibs_own_keymap():
    """'r' resets the view and 'q' closes the window, so they cannot mean "redo"."""
    reserved = {key for keys in matplotlib.rcParams.find_all("keymap").values() for key in keys}
    assert not reserved & set(selection_tool.REDO_KEYS)
    assert not reserved & set(selection_tool.ACCEPT_KEYS)
