"""Tests for the selection tool module."""

import numpy as np
import pytest

from usbmd.utils.selection_tool import (
    equalize_polygons,
    extract_polygon_from_mask,
    extract_rectangle_from_mask,
    match_polygons,
    reconstruct_mask_from_polygon,
    reconstruct_mask_from_rectangle,
)


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
    assert np.testing.assert_array_almost_equal(mask, mask_reconstructed, 0.1)

@pytest.mark.parametrize(
    "mode", ["min", "max"],
)
def test_equalize_polygons(mode):
    """Test polygon equalization."""
    # make some random polygons
    poly1 = np.array([[1, 1], [2, 2], [3, 3]])
    poly2 = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    poly3 = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])

    # equalize
    poly1, poly2, poly3 = equalize_polygons((poly1, poly2, poly3), mode=mode)
    assert len(poly1) == len(poly2) == len(poly3)
    if mode == "min":
        assert len(poly1) == 3
    elif mode == "max":
        assert len(poly1) == 5

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
