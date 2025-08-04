"""Test for the fnumber_mask function."""

import numpy as np
import pytest
from keras import ops

from zea.beamform.beamformer import (
    fnum_window_fn_hann,
    fnum_window_fn_rect,
    fnum_window_fn_tukey,
    fnumber_mask,
)
from zea.beamform.pixelgrid import cartesian_pixel_grid


@pytest.fixture
def probe_geometry():
    n_el = 5
    return np.stack(
        [np.linspace(-0.05, 0.05, n_el), np.zeros(n_el), np.zeros(n_el)], axis=-1
    ).astype(np.float32)


@pytest.fixture
def flatgrid():
    return (
        cartesian_pixel_grid(
            xlims=(-10e-3, 10e-3), zlims=(0, 20e-3), grid_size_x=65, grid_size_z=65
        )
        .reshape(-1, 3)
        .astype(np.float32)
    )


@pytest.mark.parametrize(
    "fnum_window_fn", [fnum_window_fn_hann, fnum_window_fn_rect, fnum_window_fn_tukey]
)
def test_fnumber_mask(probe_geometry, flatgrid, fnum_window_fn):
    """Runs the fnumber_mask function with different window functions."""
    mask = fnumber_mask(
        flatgrid, probe_geometry=probe_geometry, f_number=0.5, fnum_window_fn=fnum_window_fn
    )

    assert mask.shape == (flatgrid.shape[0], probe_geometry.shape[0], 1)

    mask_middle_element = ops.reshape(mask[:, probe_geometry.shape[0] // 2, 0], (65, 65))

    # Mask should not be zero in front of the element
    idx_x = 32
    idx_z = 32
    assert mask_middle_element[idx_z, idx_x] > 0.0

    # Mask should be zero all the way to the right of the element
    idx_x = 64
    idx_z = 1
    assert mask_middle_element[idx_z, idx_x] == 0.0

    # Mask should be zero just right of the f-number cone boundary
    idx_x = 32 + 16
    idx_z = 16
    assert mask_middle_element[idx_z, idx_x] == 0.0

    idx_x = 32 + 15
    idx_z = 16
    assert mask_middle_element[idx_z, idx_x] > 0.0
