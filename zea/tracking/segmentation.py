"""Segmentation-based tracker using contour matching.

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/speckle_tracking_example`.

"""

from keras import ops

from zea.func.tensor import find_contour

from .base import BaseTracker


class SegmentationTracker(BaseTracker):
    """Segmentation-based tracker.

    This tracker segments each frame independently and finds the closest points
    on the segmented contour to the previous frame's points.

    Args:
        model: Segmentation model with a `call` method.
        preprocess_fn: Optional preprocessing function to apply to frames before segmentation.
        postprocess_fn: Optional postprocessing function to apply to segmentation output, which
            should return a binary mask of the target structure.

    """

    def __init__(
        self,
        model,
        preprocess_fn: callable = None,
        postprocess_fn: callable = None,
    ):
        """Initialize segmentation-based tracker."""
        super().__init__(ndim=2)
        self.model = model
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

        if self.preprocess_fn is None:
            self.preprocess_fn = lambda frame: frame

        if self.postprocess_fn is None:
            raise ValueError("A postprocess_fn must be provided to extract binary masks.")

    def track(
        self,
        prev_frame,  # noqa F821
        next_frame,
        points,
    ):
        """
        Track points by segmenting next_frame and finding closest contour points.

        Args:
            prev_frame: Previous frame (not used, kept for interface compatibility).
            next_frame: Next frame to segment, shape (H, W).
            points: Points from previous frame, shape (N, 2) in (row, col) format.

        Returns:
            new_points: Closest points on next frame's contour, shape (N, 2).
        """
        orig_shape = ops.shape(next_frame)

        frame_input = self.preprocess_fn(next_frame)

        outputs = self.model.call(frame_input)

        mask = self.postprocess_fn(outputs, orig_shape)

        contour_points = find_contour(mask)

        if ops.shape(contour_points)[0] > 0:
            new_points = self._find_closest_points(points, contour_points)
        else:
            new_points = points

        return new_points

    def _find_closest_points(self, query_points, target_points):
        """Find closest target points to each query point.

        Args:
            query_points: Points to match, shape (N, 2).
            target_points: Points to match to, shape (M, 2).

        Returns:
            Closest target points, shape (N, 2).
        """
        # Compute pairwise squared distances
        # query_points: (N, 2), target_points: (M, 2)
        # Expand dims: (N, 1, 2) and (1, M, 2)
        query_expanded = ops.expand_dims(query_points, axis=1)  # (N, 1, 2)
        target_expanded = ops.expand_dims(target_points, axis=0)  # (1, M, 2)

        # Compute squared distances: (N, M)
        diff = query_expanded - target_expanded
        sq_distances = ops.sum(diff * diff, axis=2)

        closest_indices = ops.argmin(sq_distances, axis=1)

        closest_points = ops.take(target_points, closest_indices, axis=0)

        return closest_points

    def __repr__(self):
        """String representation."""
        return f"SegmentationTracker(model={self.model.__class__.__name__})"
