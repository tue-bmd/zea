"""Block matching tracker.

Tracks points by searching, for each point, the displacement within a search
window that best matches a reference block extracted from the previous frame.
Supports 2D and 3D data, multiple matching metrics, and subpixel refinement.
"""

from typing import Tuple, Union

import numpy as np
from keras import ops

from zea.func.tensor import translate

from .base import BaseTracker


class BlockMatchingTracker(BaseTracker):
    """Block matching tracker.

    For each point a reference block is extracted from the previous frame and
    compared against candidate blocks in the next frame over an exhaustive
    search window. The best matching displacement is selected and optionally
    refined to subpixel accuracy with a parabolic fit.

    Args:
        block_size: Block size (height, width) for 2D or (depth, height, width) for 3D.
        search_range: Maximum displacement searched per dimension. An int is used
            for all dimensions, or a tuple matching ``block_size``.
        metric: Matching cost, one of ``"ssd"``, ``"sad"`` or ``"ncc"``.
        subpixel: Whether to refine the best match to subpixel accuracy.
        **kwargs: Additional parameters.

    Example:
        .. doctest::

            >>> from zea.tracking import BlockMatchingTracker
            >>> import numpy as np

            >>> tracker = BlockMatchingTracker(block_size=(21, 21), search_range=10)
            >>> frame1 = np.random.rand(100, 100).astype("float32")
            >>> frame2 = np.random.rand(100, 100).astype("float32")
            >>> points = np.array([[50.5, 55.2], [60.1, 65.8]], dtype="float32")
            >>> new_points = tracker.track(frame1, frame2, points)
            >>> new_points.shape
            (2, 2)
    """

    def __init__(
        self,
        block_size: Tuple[int, ...] = (21, 21),
        search_range: Union[int, Tuple[int, ...]] = 10,
        metric: str = "ssd",
        subpixel: bool = True,
        **kwargs,
    ):
        """Initialize the block matching tracker."""
        self.ndim = len(block_size)
        super().__init__(ndim=self.ndim, **kwargs)

        if metric not in ("ssd", "sad", "ncc"):
            raise ValueError(f"Unknown metric '{metric}', expected 'ssd', 'sad' or 'ncc'")

        self.block_size = block_size
        self.metric = metric
        self.subpixel = subpixel
        self.half_block = tuple(size // 2 for size in block_size)
        self.search_range = self._normalize_search_range(search_range)
        self.search_shape = tuple(2 * radius + 1 for radius in self.search_range)

    def _normalize_search_range(self, search_range) -> Tuple[int, ...]:
        """Expand a scalar search range to one radius per dimension."""
        if isinstance(search_range, int):
            return (search_range,) * self.ndim
        return tuple(search_range)

    def track(self, prev_frame, next_frame, points):
        """Track points from prev_frame to next_frame.

        Args:
            prev_frame: Previous frame/volume, shape (H, W) for 2D or (D, H, W) for 3D.
            next_frame: Next frame/volume, shape (H, W) for 2D or (D, H, W) for 3D.
            points: Points to track, shape (N, ndim) in (y, x) or (z, y, x) format.

        Returns:
            new_points: Tracked points as tensor, shape (N, ndim).
        """
        prev_norm = translate(prev_frame, range_to=(0, 1))
        next_norm = translate(next_frame, range_to=(0, 1))

        n_points = int(points.shape[0])
        tracked = [self._track_point(prev_norm, next_norm, points[i]) for i in range(n_points)]
        return ops.stack(tracked)

    def _track_point(self, prev_frame, next_frame, point):
        """Find the best matching displacement for a single point."""
        template = self._sample_blocks(prev_frame, ops.reshape(point, (1, self.ndim)))[0]
        offsets = self._candidate_offsets()
        candidate_centers = ops.expand_dims(point, 0) + offsets
        blocks = self._sample_blocks(next_frame, candidate_centers)

        costs = self._match_costs(template, blocks)
        best = int(ops.argmin(costs))

        displacement = offsets[best]
        if self.subpixel:
            displacement = displacement + self._subpixel_offset(costs, best)
        return point + displacement

    def _candidate_offsets(self):
        """Integer search offsets, shape (num_candidates, ndim)."""
        axes = [ops.arange(-radius, radius + 1, dtype="float32") for radius in self.search_range]
        grids = ops.meshgrid(*axes, indexing="ij")
        return ops.stack([ops.reshape(grid, [-1]) for grid in grids], axis=1)

    def _block_grid(self):
        """Within-block coordinate offsets, shape (ndim, *block_size)."""
        axes = [ops.arange(2 * half + 1, dtype="float32") - half for half in self.half_block]
        grids = ops.meshgrid(*axes, indexing="ij")
        return ops.stack(grids, axis=0)

    def _sample_blocks(self, image, centers):
        """Sample a block around each center, shape (num_centers, *block_size)."""
        block_grid = ops.expand_dims(self._block_grid(), 1)
        centers_per_dim = ops.transpose(centers)
        centers_per_dim = ops.reshape(centers_per_dim, (self.ndim, -1) + (1,) * self.ndim)
        coords = block_grid + centers_per_dim
        return ops.image.map_coordinates(image, coords, order=1, fill_mode="constant")

    def _match_costs(self, template, blocks):
        """Matching cost per candidate block (lower is better)."""
        block_axes = tuple(range(1, self.ndim + 1))
        if self.metric == "sad":
            return ops.sum(ops.abs(blocks - template), axis=block_axes)
        if self.metric == "ncc":
            return -self._normalized_cross_correlation(template, blocks, block_axes)
        return ops.sum(ops.square(blocks - template), axis=block_axes)

    def _normalized_cross_correlation(self, template, blocks, block_axes):
        """Normalized cross-correlation between the template and each block."""
        centered_template = template - ops.mean(template)
        centered_blocks = blocks - ops.mean(blocks, axis=block_axes, keepdims=True)
        numerator = ops.sum(centered_template * centered_blocks, axis=block_axes)
        template_energy = ops.sum(ops.square(centered_template))
        block_energy = ops.sum(ops.square(centered_blocks), axis=block_axes)
        return numerator / (ops.sqrt(template_energy * block_energy) + 1e-8)

    def _subpixel_offset(self, costs, best):
        """Parabolic subpixel refinement of the best displacement."""
        cost_grid = np.asarray(ops.convert_to_numpy(ops.reshape(costs, self.search_shape)))
        best_index = np.unravel_index(best, self.search_shape)
        deltas = [self._parabolic_delta(cost_grid, best_index, axis) for axis in range(self.ndim)]
        return ops.convert_to_tensor(deltas, dtype="float32")

    def _parabolic_delta(self, cost_grid, best_index, axis):
        """Subpixel shift along one axis from a parabola through three samples."""
        position = best_index[axis]
        if position <= 0 or position >= self.search_shape[axis] - 1:
            return 0.0
        center = float(cost_grid[best_index])
        lower = float(cost_grid[self._neighbor_index(best_index, axis, -1)])
        upper = float(cost_grid[self._neighbor_index(best_index, axis, 1)])
        denominator = lower - 2 * center + upper
        if abs(denominator) < 1e-8:
            return 0.0
        return 0.5 * (lower - upper) / denominator

    def _neighbor_index(self, best_index, axis, step):
        """Index tuple shifted by step along one axis."""
        neighbor = list(best_index)
        neighbor[axis] += step
        return tuple(neighbor)

    def __repr__(self):
        """String representation."""
        return (
            f"BlockMatchingTracker(block_size={self.block_size}, "
            f"search_range={self.search_range}, metric='{self.metric}', "
            f"subpixel={self.subpixel})"
        )
