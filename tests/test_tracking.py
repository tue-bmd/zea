"""Tests tracking module."""

import numpy as np
import pytest
from keras import ops

from zea.tracking import LucasKanadeTracker, SegmentationTracker


@pytest.fixture
def simple_2d_frames():
    """Create simple 2D frames with a moving blob."""
    frames = []
    for t in range(5):
        frame = np.zeros((100, 100), dtype=np.float32)
        # Create a Gaussian blob that moves diagonally
        y_center = 30 + t * 5
        x_center = 30 + t * 5
        y, x = np.ogrid[:100, :100]
        blob = np.exp(-((y - y_center) ** 2 + (x - x_center) ** 2) / (2 * 5**2))
        frame = blob.astype(np.float32)
        frames.append(frame)
    return frames


@pytest.fixture
def simple_3d_frames():
    """Create simple 3D frames with a moving blob."""
    frames = []
    for t in range(5):
        frame = np.zeros((40, 40, 40), dtype=np.float32)
        # Create a 3D Gaussian blob that moves diagonally
        z_center = 15 + t * 2
        y_center = 15 + t * 2
        x_center = 15 + t * 2
        z, y, x = np.ogrid[:40, :40, :40]
        blob = np.exp(
            -((z - z_center) ** 2 + (y - y_center) ** 2 + (x - x_center) ** 2) / (2 * 3**2)
        )
        frame = blob.astype(np.float32)
        frames.append(frame)
    return frames


@pytest.fixture
def circular_2d_frames():
    """Create 2D frames with circular motion."""
    frames = []
    n_frames = 15
    for t in range(n_frames):
        frame = np.zeros((120, 120), dtype=np.float32)
        # Circular motion: blob moves in a circle
        y_center = 60 + 20 * np.sin(2 * np.pi * t / n_frames)
        x_center = 60 + 20 * np.cos(2 * np.pi * t / n_frames)
        y, x = np.ogrid[:120, :120]
        blob = np.exp(-((y - y_center) ** 2 + (x - x_center) ** 2) / (2 * 5**2))
        frame = blob.astype(np.float32)
        frames.append(frame)
    return frames


@pytest.fixture
def stationary_2d_frames():
    """Create stationary 2D frames (no motion)."""
    frame = np.random.rand(100, 100).astype(np.float32)
    # Add a clear feature
    y, x = np.ogrid[:100, :100]
    blob = np.exp(-((y - 50) ** 2 + (x - 50) ** 2) / (2 * 5**2))
    frame = frame * 0.2 + blob * 0.8
    return [frame.copy() for _ in range(3)]


@pytest.fixture
def mock_seg_model():
    """Mock segmentation model for testing."""

    class MockSegModel:
        def call(self, frame):
            # Return input as segmentation (simple identity for testing)
            return frame

    return MockSegModel()


@pytest.fixture
def seg_preprocess_fn():
    """Preprocessing function for segmentation tracker."""

    def preprocess(frame):
        return ops.convert_to_tensor(frame, dtype="float32")

    return preprocess


@pytest.fixture
def seg_postprocess_fn():
    """Postprocessing function for segmentation tracker."""

    def postprocess(output, original_shape):
        # Threshold to get binary mask
        mask = ops.where(output >= 0.5, 1.0, 0.0)
        return mask

    return postprocess


@pytest.fixture
def lk_tracker_2d():
    """Create a 2D Lucas-Kanade tracker."""
    return LucasKanadeTracker(win_size=(21, 21), max_level=2)


@pytest.fixture
def lk_tracker_3d():
    """Create a 3D Lucas-Kanade tracker."""
    return LucasKanadeTracker(win_size=(11, 11, 11), max_level=1)


@pytest.fixture
def seg_tracker_2d(mock_seg_model, seg_preprocess_fn, seg_postprocess_fn):
    """Create a 2D segmentation tracker."""
    return SegmentationTracker(
        model=mock_seg_model,
        preprocess_fn=seg_preprocess_fn,
        postprocess_fn=seg_postprocess_fn,
    )


@pytest.mark.parametrize(
    "tracker_fixture,expected_ndim",
    [
        ("lk_tracker_2d", 2),
        ("lk_tracker_3d", 3),
        ("seg_tracker_2d", 2),
    ],
)
def test_tracker_initialization(tracker_fixture, expected_ndim, request):
    """Test that all trackers initialize correctly."""
    tracker = request.getfixturevalue(tracker_fixture)
    assert tracker.ndim == expected_ndim


@pytest.mark.parametrize(
    "tracker_fixture,frames_fixture,initial_points,expected_motion",
    [
        # 2D trackers with linear motion
        (
            "lk_tracker_2d",
            "simple_2d_frames",
            [[30.0, 30.0]],
            {"per_frame": (5, 5), "total": (20, 20)},
        ),
        (
            "seg_tracker_2d",
            "simple_2d_frames",
            [[30.0, 30.0]],
            {"per_frame": (5, 5), "total": (20, 20)},
        ),
        # 3D tracker with linear motion
        (
            "lk_tracker_3d",
            "simple_3d_frames",
            [[15.0, 15.0, 15.0]],
            {"per_frame": (2, 2, 2), "total": (8, 8, 8)},
        ),
    ],
)
def test_track_single_point_linear_motion(
    tracker_fixture,
    frames_fixture,
    initial_points,
    expected_motion,
    request,
    tolerance=0.4,
):
    """Test tracking a single point with linear motion."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = request.getfixturevalue(frames_fixture)

    initial_points_array = np.array(initial_points, dtype=np.float32)
    trajectories = tracker.track_sequence(frames, initial_points_array)

    assert len(trajectories) == 1

    n_frames = len(frames)
    n_dims = len(initial_points[0])
    assert trajectories[0].shape == (n_frames, n_dims)

    traj = trajectories[0]
    total_motion = traj[-1] - traj[0]
    total_expected = expected_motion["total"]

    is_seg_tracker = "seg" in tracker_fixture

    for dim in range(n_dims):
        expected = total_expected[dim]
        assert expected * (1 - tolerance) < total_motion[dim] < expected * (1 + tolerance), (
            f"Total motion dim {dim}: expected ~{expected}, got {total_motion[dim]:.2f}"
        )

    # For non-segmentation trackers, also check per-frame motion
    if not is_seg_tracker:
        per_frame_expected = expected_motion["per_frame"]
        for t in range(1, n_frames):
            motion = traj[t] - traj[t - 1]
            for dim in range(n_dims):
                expected = per_frame_expected[dim]
                assert expected * (1 - tolerance) < motion[dim] < expected * (1 + tolerance), (
                    f"Frame {t}, dim {dim}: expected ~{expected}, got {motion[dim]:.2f}"
                )


@pytest.mark.parametrize(
    "tracker_fixture,frames_fixture,initial_points",
    [
        ("lk_tracker_2d", "simple_2d_frames", [[30.0, 30.0], [28.0, 32.0], [32.0, 28.0]]),
        ("seg_tracker_2d", "simple_2d_frames", [[30.0, 30.0], [28.0, 32.0]]),
        ("lk_tracker_3d", "simple_3d_frames", [[15.0, 15.0, 15.0], [14.0, 16.0, 15.0]]),
    ],
)
def test_track_multiple_points(tracker_fixture, frames_fixture, initial_points, request):
    """Test tracking multiple points simultaneously."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = request.getfixturevalue(frames_fixture)

    initial_points_array = np.array(initial_points, dtype=np.float32)
    trajectories = tracker.track_sequence(frames, initial_points_array)

    # Should have trajectory for each point
    assert len(trajectories) == len(initial_points)

    # All trajectories should have correct shape
    n_frames = len(frames)
    n_dims = len(initial_points[0])

    for traj in trajectories:
        assert traj.shape == (n_frames, n_dims)

        # Check that points moved in a consistent direction
        # (all moving diagonally in positive direction)
        total_motion = traj[-1] - traj[0]
        for dim in range(n_dims):
            assert total_motion[dim] > 0, f"Point should move in positive direction in dim {dim}"


@pytest.mark.parametrize(
    "tracker_fixture,frames_fixture",
    [
        ("lk_tracker_2d", "circular_2d_frames"),
        ("seg_tracker_2d", "circular_2d_frames"),
    ],
)
def test_track_nonlinear_motion(tracker_fixture, frames_fixture, request):
    """Test tracking with non-linear (circular) motion."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = request.getfixturevalue(frames_fixture)

    # Start at initial position of circular motion
    y_start = 60 + 20 * np.sin(0)
    x_start = 60 + 20 * np.cos(0)
    initial_points = np.array([[y_start, x_start]], dtype=np.float32)

    trajectories = tracker.track_sequence(frames, initial_points)

    n_frames = len(frames)
    traj = np.array(trajectories[0])
    assert traj.shape == (n_frames, 2)

    center = np.array([60.0, 60.0])
    distances = np.linalg.norm(traj - center, axis=1)
    avg_dist = np.mean(distances)

    assert 12 < avg_dist < 28, f"Average distance from center should be ~20, got {avg_dist:.1f}"


@pytest.mark.parametrize(
    "tracker_fixture",
    [
        "lk_tracker_2d",
        "seg_tracker_2d",
    ],
)
def test_track_stationary(tracker_fixture, stationary_2d_frames, request):
    """Test tracking on stationary images."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = stationary_2d_frames

    initial_points = np.array([[50.0, 50.0]], dtype=np.float32)
    trajectories = tracker.track_sequence(frames, initial_points)

    # Point should stay approximately in the same place
    # Segmentation tracker is less precise, so use more lenient threshold
    is_seg_tracker = "seg" in tracker_fixture
    max_motion = 5.0 if is_seg_tracker else 2.0

    traj = trajectories[0]
    for t in range(1, len(frames)):
        motion = np.linalg.norm(traj[t] - traj[0])
        assert motion < max_motion, (
            f"Point should not move much in stationary image, moved {motion:.2f}"
        )


@pytest.mark.parametrize(
    "tracker_fixture,frames_fixture",
    [
        ("lk_tracker_2d", "simple_2d_frames"),
        ("seg_tracker_2d", "simple_2d_frames"),
        ("lk_tracker_3d", "simple_3d_frames"),
    ],
)
def test_track_single_frame_pair(tracker_fixture, frames_fixture, request):
    """Test tracking between just two consecutive frames."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = request.getfixturevalue(frames_fixture)

    frame1 = ops.convert_to_tensor(frames[0], dtype="float32")
    frame2 = ops.convert_to_tensor(frames[1], dtype="float32")

    # Initial point
    if tracker.ndim == 2:
        points = ops.convert_to_tensor([[30.0, 30.0]], dtype="float32")
    else:
        points = ops.convert_to_tensor([[15.0, 15.0, 15.0]], dtype="float32")

    new_points = tracker.track(frame1, frame2, points)

    assert new_points.shape == (1, tracker.ndim)
    assert new_points.dtype == "float32"

    motion = ops.convert_to_numpy(new_points - points)
    motion_magnitude = np.linalg.norm(motion)
    assert motion_magnitude > 1.0, f"Point should move, moved {motion_magnitude:.2f}"


@pytest.mark.parametrize(
    "tracker_fixture,frames_fixture",
    [
        ("lk_tracker_2d", "simple_2d_frames"),
        ("seg_tracker_2d", "simple_2d_frames"),
    ],
)
def test_track_near_boundary(tracker_fixture, frames_fixture, request):
    """Test tracking with points near image boundaries."""
    tracker = request.getfixturevalue(tracker_fixture)
    frames = request.getfixturevalue(frames_fixture)

    initial_points = np.array([[5.0, 5.0]], dtype=np.float32)

    trajectories = tracker.track_sequence(frames, initial_points)

    assert len(trajectories) == 1


def test_segmentation_tracker_requires_postprocess_fn(mock_seg_model):
    """Test that segmentation tracker requires postprocess_fn."""
    with pytest.raises(ValueError, match="postprocess_fn must be provided"):
        SegmentationTracker(model=mock_seg_model)


def test_segmentation_tracker_no_contour_fallback(mock_seg_model, seg_preprocess_fn):
    """Test segmentation tracker fallback when no contour is found."""

    def postprocess_empty(output, original_shape):
        return ops.zeros(original_shape, dtype="float32")

    tracker = SegmentationTracker(
        model=mock_seg_model,
        preprocess_fn=seg_preprocess_fn,
        postprocess_fn=postprocess_empty,
    )

    frame1 = ops.zeros((100, 100), dtype="float32")
    frame2 = ops.zeros((100, 100), dtype="float32")
    points = ops.convert_to_tensor([[50.0, 50.0]], dtype="float32")

    new_points = tracker.track(frame1, frame2, points)

    np.testing.assert_array_equal(ops.convert_to_numpy(new_points), ops.convert_to_numpy(points))


@pytest.mark.parametrize(
    "max_level",
    [0, 1, 2, 3],
)
def test_lk_tracker_pyramid_levels(max_level, simple_2d_frames):
    """Test LK tracker with different pyramid levels."""
    tracker = LucasKanadeTracker(win_size=(21, 21), max_level=max_level)
    initial_points = np.array([[30.0, 30.0]], dtype=np.float32)

    trajectories = tracker.track_sequence(simple_2d_frames, initial_points)

    assert len(trajectories) == 1
    assert trajectories[0].shape == (5, 2)
