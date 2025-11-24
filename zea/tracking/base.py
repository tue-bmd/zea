"""Base tracker class for point tracking algorithms."""

from abc import ABC, abstractmethod
from typing import List

from keras import ops


class BaseTracker(ABC):
    """Abstract base class for point tracking algorithms.

    This class defines the interface for tracking algorithms in the zea package.
    Implementations should handle both 2D and 3D tracking where applicable.

    Args:
        ndim: Number of dimensions (2 for 2D, 3 for 3D).
        **kwargs: Tracker-specific parameters.
    """

    def __init__(self, ndim: int = 2, **kwargs):
        """Initialize the tracker with parameters."""
        self.ndim = ndim

        if self.ndim not in [2, 3]:
            raise ValueError(f"Only 2D and 3D tracking supported, got {ndim}D")

    @abstractmethod
    def track(
        self,
        prev_frame,
        next_frame,
        points,
    ):
        """
        Track points from prev_frame to next_frame.

        Args:
            prev_frame: Previous frame/volume of shape (H, W) or (D, H, W).
            next_frame: Next frame/volume of shape (H, W) or (D, H, W).
            points: Points to track, shape (N, ndim) in (y, x) or (z, y, x) format.

        Returns:
            new_points: Tracked point locations, shape (N, ndim).
        """
        pass

    def track_sequence(
        self,
        frames: List,
        initial_points,
    ) -> List:
        """
        Track points through a sequence of frames.

        Args:
            frames: List of frames/volumes to track through.
            initial_points: Starting points in first frame, shape (N, ndim).

        Returns:
            List of N arrays, where each array has shape (T, ndim) containing
            the trajectory of one point through all T frames.

        """

        n_frames = len(frames)
        n_points = int(ops.shape(initial_points)[0])

        frames_t = [ops.convert_to_tensor(f, dtype="float32") for f in frames]
        current_points = ops.convert_to_tensor(initial_points, dtype="float32")

        trajectories = [ops.zeros((n_frames, self.ndim), dtype="float32") for _ in range(n_points)]

        # Set initial positions
        for i in range(n_points):
            trajectories[i] = ops.scatter_update(
                trajectories[i], [[0]], ops.expand_dims(current_points[i], 0)
            )

        # Track frame by frame
        for t in range(n_frames - 1):
            new_points = self.track(frames_t[t], frames_t[t + 1], current_points)

            for i in range(n_points):
                trajectories[i] = ops.scatter_update(
                    trajectories[i], [[t + 1]], ops.expand_dims(new_points[i], 0)
                )

            current_points = new_points

        return trajectories

    def __repr__(self):
        """String representation of the tracker."""
        return f"{self.__class__.__name__}(ndim={self.ndim})"
