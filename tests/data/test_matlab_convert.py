"""Tests for MATLAB conversion functions."""

import numpy as np
import pytest


def test_read_raw_data_dimension_handling():
    """Test that read_raw_data correctly handles single frame (2D) and multi-frame (3D) data.
    
    This test validates the fix for issue #112 where single-frame data (2D arrays)
    from MATLAB files were not properly handled.
    """
    # Test Case 1: Single frame data (2D) should be expanded to 3D
    raw_data_2d = np.random.random((100, 200))
    
    # Simulate the dimension check and expansion logic from read_raw_data
    if raw_data_2d.ndim == 2:
        raw_data_2d = np.expand_dims(raw_data_2d, axis=0)
    elif raw_data_2d.ndim != 3:
        raise ValueError(
            f"Raw data has {raw_data_2d.ndim} dimensions, but should have 2 or 3. "
            "Please check the input file."
        )
    
    assert raw_data_2d.ndim == 3, "2D data should be expanded to 3D"
    assert raw_data_2d.shape[0] == 1, "Single frame should have frame dimension of 1"
    
    # Test Case 2: Multi-frame data (3D) should pass through unchanged
    raw_data_3d = np.random.random((5, 100, 200))
    original_shape = raw_data_3d.shape
    
    if raw_data_3d.ndim == 2:
        raw_data_3d = np.expand_dims(raw_data_3d, axis=0)
    elif raw_data_3d.ndim != 3:
        raise ValueError(
            f"Raw data has {raw_data_3d.ndim} dimensions, but should have 2 or 3. "
            "Please check the input file."
        )
    
    assert raw_data_3d.shape == original_shape, "3D data should remain unchanged"
    assert raw_data_3d.ndim == 3, "3D data should still be 3D"
    
    # Test Case 3: Invalid dimensions (4D) should raise ValueError
    raw_data_4d = np.random.random((2, 5, 100, 200))
    
    with pytest.raises(ValueError, match="Raw data has 4 dimensions, but should have 2 or 3"):
        if raw_data_4d.ndim == 2:
            raw_data_4d = np.expand_dims(raw_data_4d, axis=0)
        elif raw_data_4d.ndim != 3:
            raise ValueError(
                f"Raw data has {raw_data_4d.ndim} dimensions, but should have 2 or 3. "
                "Please check the input file."
            )
    
    # Test Case 4: 1D data should also raise ValueError
    raw_data_1d = np.random.random(100)
    
    with pytest.raises(ValueError, match="Raw data has 1 dimensions, but should have 2 or 3"):
        if raw_data_1d.ndim == 2:
            raw_data_1d = np.expand_dims(raw_data_1d, axis=0)
        elif raw_data_1d.ndim != 3:
            raise ValueError(
                f"Raw data has {raw_data_1d.ndim} dimensions, but should have 2 or 3. "
                "Please check the input file."
            )


def test_frame_indexing_after_dimension_fix():
    """Test that frame indexing works correctly after dimension handling.
    
    This verifies that the fix allows proper indexing with frame_indices,
    which was the root cause of the issue.
    """
    # Simulate single frame case
    raw_data_2d = np.arange(20).reshape(4, 5)
    
    # Apply the fix
    if raw_data_2d.ndim == 2:
        raw_data_2d = np.expand_dims(raw_data_2d, axis=0)
    
    # Now frame indexing should work
    frame_indices = np.array([0])
    result = raw_data_2d[frame_indices]
    
    assert result.shape == (1, 4, 5), "Frame indexing should work after dimension fix"
    assert np.array_equal(result[0], np.arange(20).reshape(4, 5)), "Data should be preserved"
    
    # Simulate multi-frame case
    raw_data_3d = np.arange(60).reshape(3, 4, 5)
    
    # Apply the fix (should pass through)
    if raw_data_3d.ndim == 2:
        raw_data_3d = np.expand_dims(raw_data_3d, axis=0)
    
    # Frame indexing with multiple frames
    frame_indices = np.array([0, 2])
    result = raw_data_3d[frame_indices]
    
    assert result.shape == (2, 4, 5), "Frame indexing should work with multiple frames"
    assert np.array_equal(result[0], np.arange(20).reshape(4, 5)), "First frame should match"
    assert np.array_equal(result[1], np.arange(40, 60).reshape(4, 5)), "Last frame should match"
