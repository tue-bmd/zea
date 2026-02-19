"""Lucas-Kanade optical flow tracker.

.. seealso::
    A tutorial notebook where this model is used:
    :doc:`../notebooks/models/speckle_tracking_example`.

"""

from typing import Tuple

from keras import ops

from zea.func.tensor import gaussian_filter, translate

from .base import BaseTracker


class LucasKanadeTracker(BaseTracker):
    """Lucas-Kanade optical flow tracker.

    Implements pyramidal Lucas-Kanade optical flow tracking.

    Args:
        win_size: Window size (height, width) for 2D or (depth, height, width) for 3D.
        max_level: Number of pyramid levels (0 means no pyramid).
        max_iterations: Maximum iterations per pyramid level.
        epsilon: Convergence threshold for iterative solver.
        **kwargs: Additional parameters.

    Example:
        .. doctest::

            >>> from zea.tracking import LucasKanadeTracker
            >>> import numpy as np

            >>> tracker = LucasKanadeTracker(win_size=(32, 32), max_level=3)
            >>> frame1 = np.random.rand(100, 100).astype("float32")
            >>> frame2 = np.random.rand(100, 100).astype("float32")
            >>> points = np.array([[50.5, 55.2], [60.1, 65.8]], dtype="float32")
            >>> new_points = tracker.track(frame1, frame2, points)
            >>> new_points.shape
            (2, 2)
    """

    def __init__(
        self,
        win_size: Tuple[int, ...] = (32, 32),
        max_level: int = 3,
        max_iterations: int = 30,
        epsilon: float = 0.01,
        **kwargs,
    ):
        """Initialize custom Lucas-Kanade tracker."""
        self.ndim = len(win_size)

        super().__init__(ndim=self.ndim, **kwargs)

        self.win_size = win_size
        self.max_level = max_level
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        self.half_win = tuple(w // 2 for w in win_size)

    def track(
        self,
        prev_frame,
        next_frame,
        points,
    ) -> Tuple:
        """
        Track points using custom pyramidal Lucas-Kanade.

        Args:
            prev_frame: Previous frame/volume (tensor), shape (H, W) for 2D or (D, H, W) for 3D.
            next_frame: Next frame/volume (tensor), shape (H, W) for 2D or (D, H, W) for 3D.
            points: Points to track (tensor), shape (N, ndim) in (y, x) or (z, y, x) format.

        Returns:
            new_points: Tracked points as tensor, shape (N, ndim).
        """
        if self.ndim not in [2, 3]:
            raise NotImplementedError(f"Only 2D and 3D tracking supported, got {self.ndim}D")

        # Normalize frames to [0, 1]
        prev_norm = translate(prev_frame, range_to=(0, 1))
        next_norm = translate(next_frame, range_to=(0, 1))

        # Build pyramids
        if self.max_level > 0:
            prev_pyr = self._build_pyramid(prev_norm, self.max_level + 1)
            next_pyr = self._build_pyramid(next_norm, self.max_level + 1)
        else:
            prev_pyr = [prev_norm]
            next_pyr = [next_norm]

        n_levels = len(prev_pyr)
        n_points = int(points.shape[0])

        # Start at coarsest level
        scale = 2 ** (n_levels - 1)
        curr_points = points / scale
        flows = ops.zeros((n_points, self.ndim), dtype="float32")

        # Track through pyramid levels
        for level in range(n_levels):
            prev_img = prev_pyr[level]
            next_img = next_pyr[level]

            flows = self._track_points_batched(prev_img, next_img, curr_points, flows)

            # Scale for next level (if not at finest)
            if level < n_levels - 1:
                flows = flows * 2.0
                curr_points = curr_points * 2.0

        # Final points at full resolution
        new_points = points + flows

        return new_points

    def _build_pyramid(self, image, n_levels: int) -> list:
        """Build Gaussian pyramid."""
        pyramid = [image]
        for _ in range(1, n_levels):
            curr = pyramid[-1]
            shape = ops.shape(curr)

            # Check minimum size based on dimensionality
            if self.ndim == 2:
                h, w = shape[0], shape[1]
                min_size = ops.minimum(h, w)
                if min_size < 4:
                    break
            else:  # 3D
                d, h, w = shape[0], shape[1], shape[2]
                min_size = ops.minimum(ops.minimum(d, h), w)
                if min_size < 4:
                    break

            blurred = gaussian_filter(curr, sigma=0.849, mode="reflect")

            # Downsample by 2x using map_coordinates
            if self.ndim == 2:
                new_h, new_w = h // 2, w // 2
                # Create downsampled coordinate grid
                y_coords = ops.linspace(0, h - 1, new_h)
                x_coords = ops.linspace(0, w - 1, new_w)
                grid_y, grid_x = ops.meshgrid(y_coords, x_coords, indexing="ij")
                coords = ops.stack([grid_y, grid_x], axis=0)
                downsampled = ops.image.map_coordinates(blurred, coords, order=1)
            else:  # 3D
                new_d, new_h, new_w = d // 2, h // 2, w // 2
                # Create downsampled coordinate grid
                z_coords = ops.linspace(0, d - 1, new_d)
                y_coords = ops.linspace(0, h - 1, new_h)
                x_coords = ops.linspace(0, w - 1, new_w)
                grid_z, grid_y, grid_x = ops.meshgrid(z_coords, y_coords, x_coords, indexing="ij")
                coords = ops.stack([grid_z, grid_y, grid_x], axis=0)
                downsampled = ops.image.map_coordinates(blurred, coords, order=1)

            pyramid.append(downsampled)
        return pyramid[::-1]

    def _track_points_batched(
        self,
        prev_img,
        next_img,
        points,
        flow_guesses,
    ):
        """Track all points using batched iterative Lucas-Kanade.

        Args:
            prev_img: Previous image/volume
            next_img: Next image/volume
            points: All points to track, shape (N, ndim)
            flow_guesses: Initial flow estimates, shape (N, ndim)

        Returns:
            flows: Refined flows for all points, shape (N, ndim)
        """
        n_points = int(points.shape[0])

        templates, valid_templates = self._extract_windows_batched(prev_img, points)
        gradients = self._sobel_gradients_batched(templates)

        # Pre-compute structure tensors for all points
        if self.ndim == 2:
            Iy, Ix = gradients  # (N, H, W)
            Ix_flat = ops.reshape(Ix, (n_points, -1))  # (N, H*W)
            Iy_flat = ops.reshape(Iy, (n_points, -1))

            IxIx = ops.sum(Ix_flat * Ix_flat, axis=1)  # (N,)
            IxIy = ops.sum(Ix_flat * Iy_flat, axis=1)
            IyIy = ops.sum(Iy_flat * Iy_flat, axis=1)

            structure = ops.stack(
                [
                    ops.stack([IxIx, IxIy], axis=1),
                    ops.stack([IxIy, IyIy], axis=1),
                ],
                axis=1,
            )
            eye = ops.eye(2, dtype=structure.dtype)
            structure = structure + ops.expand_dims(eye, 0) * 1e-5

        else:  # 3D
            Iz, Iy, Ix = gradients  # (N, D, H, W)
            # Flatten spatial dimensions: (N, D*H*W)
            Ix_flat = ops.reshape(Ix, (n_points, -1))
            Iy_flat = ops.reshape(Iy, (n_points, -1))
            Iz_flat = ops.reshape(Iz, (n_points, -1))

            # Structure tensor components: (N,)
            IxIx = ops.sum(Ix_flat * Ix_flat, axis=1)
            IxIy = ops.sum(Ix_flat * Iy_flat, axis=1)
            IxIz = ops.sum(Ix_flat * Iz_flat, axis=1)
            IyIy = ops.sum(Iy_flat * Iy_flat, axis=1)
            IyIz = ops.sum(Iy_flat * Iz_flat, axis=1)
            IzIz = ops.sum(Iz_flat * Iz_flat, axis=1)

            # Build structure tensor matrices: (N, 3, 3)
            structure = ops.stack(
                [
                    ops.stack([IxIx, IxIy, IxIz], axis=1),
                    ops.stack([IxIy, IyIy, IyIz], axis=1),
                    ops.stack([IxIz, IyIz, IzIz], axis=1),
                ],
                axis=1,
            )
            # Add regularization
            eye = ops.eye(3, dtype=structure.dtype)
            structure = structure + ops.expand_dims(eye, 0) * 1e-5

        # Iterative refinement for all points
        flows = flow_guesses
        active_mask = valid_templates  # Track which points are still being tracked

        for iteration in range(self.max_iterations):
            # Extract warped windows for all points
            warped_pts = points + flows
            warped_windows, valid_warped = self._extract_windows_batched(next_img, warped_pts)

            # Update active mask
            active_mask = ops.logical_and(active_mask, valid_warped)

            # Compute image differences
            diffs = templates - warped_windows  # (N, H, W) or (N, D, H, W)
            diffs_flat = ops.reshape(diffs, (n_points, -1))  # (N, H*W) or (N, D*H*W)

            # Compute right-hand side vectors
            if self.ndim == 2:
                b_x = ops.sum(Ix_flat * diffs_flat, axis=1)  # (N,)
                b_y = ops.sum(Iy_flat * diffs_flat, axis=1)  # (N,)
                rhs = ops.stack([b_x, b_y], axis=1)  # (N, 2)
                rhs = ops.expand_dims(rhs, axis=2)  # (N, 2, 1)

                # Solve batched linear systems: (N, 2, 2) @ (N, 2, 1) -> (N, 2, 1)
                delta_xy = ops.matmul(ops.linalg.inv(structure), rhs)
                delta_xy = ops.squeeze(delta_xy, axis=2)  # (N, 2)

                # Reorder to (y, x)
                delta = ops.stack([delta_xy[:, 1], delta_xy[:, 0]], axis=1)  # (N, 2)

            else:  # 3D
                b_x = ops.sum(Ix_flat * diffs_flat, axis=1)  # (N,)
                b_y = ops.sum(Iy_flat * diffs_flat, axis=1)  # (N,)
                b_z = ops.sum(Iz_flat * diffs_flat, axis=1)  # (N,)
                rhs = ops.stack([b_x, b_y, b_z], axis=1)  # (N, 3)
                rhs = ops.expand_dims(rhs, axis=2)  # (N, 3, 1)

                # Solve batched linear systems: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
                delta_xyz = ops.matmul(ops.linalg.inv(structure), rhs)
                delta_xyz = ops.squeeze(delta_xyz, axis=2)  # (N, 3)

                # Reorder to (z, y, x)
                delta = ops.stack(
                    [delta_xyz[:, 2], delta_xyz[:, 1], delta_xyz[:, 0]], axis=1
                )  # (N, 3)

            # Apply updates only to active points
            delta = ops.where(ops.expand_dims(active_mask, 1), delta, ops.zeros_like(delta))
            flows = flows + delta

            # Check convergence for all points
            delta_norms = ops.sqrt(ops.sum(delta * delta, axis=1))  # (N,)
            converged = delta_norms < self.epsilon
            active_mask = ops.logical_and(active_mask, ops.logical_not(converged))

            # Early exit if all points converged
            if not ops.any(active_mask):
                break

        return flows

    def _extract_windows_batched(self, image, points):
        """Extract windows around all points with subpixel interpolation."""
        if self.ndim == 2:
            return self._extract_windows_batched_2d(image, points)
        elif self.ndim == 3:
            return self._extract_windows_batched_3d(image, points)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _extract_window(self, image, point):
        """Extract window around point with subpixel interpolation."""
        if self.ndim == 2:
            return self._extract_window_2d(image, point)
        elif self.ndim == 3:
            return self._extract_window_3d(image, point)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _extract_windows_batched_2d(self, image, points):
        """Extract 2D windows for all points with bilinear interpolation."""
        hy, hx = self.half_win
        h, w = ops.shape(image)[0], ops.shape(image)[1]
        n_points = int(points.shape[0])

        win_h, win_w = 2 * hy + 1, 2 * hx + 1

        py = points[:, 0]  # (N,)
        px = points[:, 1]

        # Bounds check for all points
        h_float = ops.cast(h, py.dtype)
        w_float = ops.cast(w, px.dtype)
        valid_mask = ops.logical_and(
            ops.logical_and(py >= hy + 1, py < h_float - hy - 1),
            ops.logical_and(px >= hx + 1, px < w_float - hx - 1),
        )

        y_offsets = ops.arange(win_h, dtype="float32") - hy  # (win_h,)
        x_offsets = ops.arange(win_w, dtype="float32") - hx  # (win_w,)

        # Broadcast to create coordinate grids for all points
        py_expanded = ops.reshape(py, (n_points, 1, 1))
        px_expanded = ops.reshape(px, (n_points, 1, 1))
        y_offsets_expanded = ops.reshape(y_offsets, (1, win_h, 1))
        x_offsets_expanded = ops.reshape(x_offsets, (1, 1, win_w))

        grid_y = py_expanded + y_offsets_expanded  # (N, win_h, 1)
        grid_x = px_expanded + x_offsets_expanded  # (N, 1, win_w)

        grid_y = ops.broadcast_to(grid_y, (n_points, win_h, win_w))
        grid_x = ops.broadcast_to(grid_x, (n_points, win_h, win_w))

        coords = ops.stack([grid_y, grid_x], axis=0)
        coords_flat = ops.reshape(coords, (2, n_points * win_h * win_w))
        windows_flat = ops.image.map_coordinates(image, coords_flat, order=1)
        windows = ops.reshape(windows_flat, (n_points, win_h, win_w))

        valid_mask_expanded = ops.reshape(valid_mask, (n_points, 1, 1))
        windows = ops.where(valid_mask_expanded, windows, ops.zeros_like(windows))

        return windows, valid_mask

    def _extract_window_2d(self, image, point):
        """Extract 2D window with bilinear interpolation using map_coordinates."""
        hy, hx = self.half_win
        h, w = ops.shape(image)[0], ops.shape(image)[1]

        py, px = point[0], point[1]

        # Bounds check
        if ops.any(
            ops.stack(
                [
                    py < hy + 1,
                    py >= ops.cast(h, py.dtype) - hy - 1,
                    px < hx + 1,
                    px >= ops.cast(w, px.dtype) - hx - 1,
                ]
            )
        ):
            return ops.zeros((2 * hy + 1, 2 * hx + 1), dtype="float32"), False

        # Create coordinate grid for the window
        # Grid centered at point location
        y_coords = ops.arange(2 * hy + 1, dtype="float32") + py - hy
        x_coords = ops.arange(2 * hx + 1, dtype="float32") + px - hx
        grid_y, grid_x = ops.meshgrid(y_coords, x_coords, indexing="ij")

        # Stack coordinates for map_coordinates
        coords = ops.stack([grid_y, grid_x], axis=0)

        # Extract window using bilinear interpolation
        window = ops.image.map_coordinates(image, coords, order=1)

        return window, True

    def _extract_windows_batched_3d(self, image, points):
        """Extract 3D windows for all points with trilinear interpolation."""
        hz, hy, hx = self.half_win
        d, h, w = ops.shape(image)[0], ops.shape(image)[1], ops.shape(image)[2]
        n_points = int(points.shape[0])

        win_d, win_h, win_w = 2 * hz + 1, 2 * hy + 1, 2 * hx + 1

        # Extract point coordinates
        pz = points[:, 0]  # (N,)
        py = points[:, 1]
        px = points[:, 2]

        # Bounds check for all points
        d_float = ops.cast(d, pz.dtype)
        h_float = ops.cast(h, py.dtype)
        w_float = ops.cast(w, px.dtype)
        valid_mask = ops.logical_and(
            ops.logical_and(
                ops.logical_and(pz >= hz + 1, pz < d_float - hz - 1),
                ops.logical_and(py >= hy + 1, py < h_float - hy - 1),
            ),
            ops.logical_and(px >= hx + 1, px < w_float - hx - 1),
        )

        # Create coordinate offsets
        z_offsets = ops.arange(win_d, dtype="float32") - hz
        y_offsets = ops.arange(win_h, dtype="float32") - hy
        x_offsets = ops.arange(win_w, dtype="float32") - hx

        pz_expanded = ops.reshape(pz, (n_points, 1, 1, 1))
        py_expanded = ops.reshape(py, (n_points, 1, 1, 1))
        px_expanded = ops.reshape(px, (n_points, 1, 1, 1))
        z_offsets_expanded = ops.reshape(z_offsets, (1, win_d, 1, 1))
        y_offsets_expanded = ops.reshape(y_offsets, (1, 1, win_h, 1))
        x_offsets_expanded = ops.reshape(x_offsets, (1, 1, 1, win_w))

        grid_z = pz_expanded + z_offsets_expanded
        grid_y = py_expanded + y_offsets_expanded
        grid_x = px_expanded + x_offsets_expanded

        grid_z = ops.broadcast_to(grid_z, (n_points, win_d, win_h, win_w))
        grid_y = ops.broadcast_to(grid_y, (n_points, win_d, win_h, win_w))
        grid_x = ops.broadcast_to(grid_x, (n_points, win_d, win_h, win_w))

        coords = ops.stack([grid_z, grid_y, grid_x], axis=0)
        coords_flat = ops.reshape(coords, (3, n_points * win_d * win_h * win_w))
        windows_flat = ops.image.map_coordinates(image, coords_flat, order=1)
        windows = ops.reshape(windows_flat, (n_points, win_d, win_h, win_w))

        valid_mask_expanded = ops.reshape(valid_mask, (n_points, 1, 1, 1))
        windows = ops.where(valid_mask_expanded, windows, ops.zeros_like(windows))

        return windows, valid_mask

    def _extract_window_3d(self, image, point):
        """Extract 3D window with trilinear interpolation using map_coordinates."""
        hz, hy, hx = self.half_win
        d, h, w = ops.shape(image)[0], ops.shape(image)[1], ops.shape(image)[2]

        pz, py, px = point[0], point[1], point[2]

        # Bounds check
        if ops.any(
            ops.stack(
                [
                    pz < hz + 1,
                    pz >= ops.cast(d, pz.dtype) - hz - 1,
                    py < hy + 1,
                    py >= ops.cast(h, py.dtype) - hy - 1,
                    px < hx + 1,
                    px >= ops.cast(w, px.dtype) - hx - 1,
                ]
            )
        ):
            return ops.zeros((2 * hz + 1, 2 * hy + 1, 2 * hx + 1), dtype="float32"), False

        # Create coordinate grid for the window
        # Grid centered at point location
        z_coords = ops.arange(2 * hz + 1, dtype="float32") + pz - hz
        y_coords = ops.arange(2 * hy + 1, dtype="float32") + py - hy
        x_coords = ops.arange(2 * hx + 1, dtype="float32") + px - hx
        grid_z, grid_y, grid_x = ops.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

        # Stack coordinates for map_coordinates
        coords = ops.stack([grid_z, grid_y, grid_x], axis=0)

        # Extract window using trilinear interpolation
        window = ops.image.map_coordinates(image, coords, order=1)

        return window, True

    def _sobel_gradients_batched(self, images):
        """batched version of _sobel_gradients"""
        if self.ndim == 2:
            return self._sobel_gradients_2d_batched(images)
        elif self.ndim == 3:
            return self._sobel_gradients_3d_batched(images)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _sobel_gradients(self, image):
        """Compute Sobel gradients for 2D or 3D."""
        if self.ndim == 2:
            return self._sobel_gradients_2d(image)
        elif self.ndim == 3:
            return self._sobel_gradients_3d(image)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

    def _sobel_gradients_2d_batched(self, images):
        """batched version of _sobel_gradients_2d"""
        # Standard Sobel kernels
        sobel_y = ops.convert_to_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float32") / 8.0
        sobel_x = ops.convert_to_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32") / 8.0

        n_points = int(images.shape[0])
        h, w = ops.shape(images)[1], ops.shape(images)[2]

        # Pad all images: (N, H, W) -> (N, H+2, W+2)
        padded = ops.pad(images, [[0, 0], [1, 1], [1, 1]], mode="reflect")

        # Reshape for conv: (N, H+2, W+2, 1)
        img_4d = ops.reshape(padded, [n_points, h + 2, w + 2, 1])
        sobel_y_4d = ops.reshape(sobel_y, [3, 3, 1, 1])
        sobel_x_4d = ops.reshape(sobel_x, [3, 3, 1, 1])

        # Apply convolution to all images in batch
        Iy_4d = ops.conv(img_4d, sobel_y_4d, padding="valid")
        Ix_4d = ops.conv(img_4d, sobel_x_4d, padding="valid")

        # Reshape back to (N, H, W)
        Iy = ops.reshape(Iy_4d, [n_points, h, w])
        Ix = ops.reshape(Ix_4d, [n_points, h, w])

        return Iy, Ix

    def _sobel_gradients_2d(self, image):
        """Compute 2D Sobel gradients using keras.ops."""
        # Standard Sobel kernels
        sobel_y = ops.convert_to_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype="float32") / 8.0
        sobel_x = ops.convert_to_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype="float32") / 8.0

        h, w = ops.shape(image)[0], ops.shape(image)[1]

        padded = ops.pad(image, [[1, 1], [1, 1]], mode="reflect")

        # Reshape for conv: image needs (batch, height, width, channels)
        img_4d = ops.reshape(padded, [1, h + 2, w + 2, 1])
        sobel_y_4d = ops.reshape(sobel_y, [3, 3, 1, 1])
        sobel_x_4d = ops.reshape(sobel_x, [3, 3, 1, 1])

        Iy_4d = ops.conv(img_4d, sobel_y_4d, padding="valid")
        Ix_4d = ops.conv(img_4d, sobel_x_4d, padding="valid")

        # Reshape back to 2D
        Iy = ops.reshape(Iy_4d, [h, w])
        Ix = ops.reshape(Ix_4d, [h, w])

        return Iy, Ix

    def _sobel_gradients_3d_batched(self, images):
        """batched version of _sobel_gradients_3d"""
        sobel_z = (
            ops.convert_to_tensor(
                [
                    [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )
        sobel_y = (
            ops.convert_to_tensor(
                [
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                    [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )
        sobel_x = (
            ops.convert_to_tensor(
                [
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )

        n_points = int(images.shape[0])
        d, h, w = ops.shape(images)[1], ops.shape(images)[2], ops.shape(images)[3]

        # Pad all images: (N, D, H, W) -> (N, D+2, H+2, W+2)
        padded = ops.pad(images, [[0, 0], [1, 1], [1, 1], [1, 1]], mode="reflect")

        # Reshape for conv: (N, D+2, H+2, W+2, 1)
        img_5d = ops.reshape(padded, [n_points, d + 2, h + 2, w + 2, 1])
        sobel_z_5d = ops.reshape(sobel_z, [3, 3, 3, 1, 1])
        sobel_y_5d = ops.reshape(sobel_y, [3, 3, 3, 1, 1])
        sobel_x_5d = ops.reshape(sobel_x, [3, 3, 3, 1, 1])

        # Apply 3D convolution to all images in batch
        Iz_5d = ops.conv(img_5d, sobel_z_5d, padding="valid")
        Iy_5d = ops.conv(img_5d, sobel_y_5d, padding="valid")
        Ix_5d = ops.conv(img_5d, sobel_x_5d, padding="valid")

        # Reshape back to (N, D, H, W)
        Iz = ops.reshape(Iz_5d, [n_points, d, h, w])
        Iy = ops.reshape(Iy_5d, [n_points, d, h, w])
        Ix = ops.reshape(Ix_5d, [n_points, d, h, w])

        return (Iz, Iy, Ix)

    def _sobel_gradients_3d(self, image):
        """Compute 3D Sobel gradients using keras.ops."""
        # 3D Sobel kernels (separable: smooth in 2 dims, gradient in 1 dim)
        # Gradient in z-direction
        sobel_z = (
            ops.convert_to_tensor(
                [
                    [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )

        # Gradient in y-direction
        sobel_y = (
            ops.convert_to_tensor(
                [
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                    [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )

        # Gradient in x-direction
        sobel_x = (
            ops.convert_to_tensor(
                [
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                ],
                dtype="float32",
            )
            / 32.0
        )

        d, h, w = ops.shape(image)[0], ops.shape(image)[1], ops.shape(image)[2]

        padded = ops.pad(image, [[1, 1], [1, 1], [1, 1]], mode="reflect")

        # Reshape for conv: image needs (batch, depth, height, width, channels)
        img_5d = ops.reshape(padded, [1, d + 2, h + 2, w + 2, 1])
        sobel_z_5d = ops.reshape(sobel_z, [3, 3, 3, 1, 1])
        sobel_y_5d = ops.reshape(sobel_y, [3, 3, 3, 1, 1])
        sobel_x_5d = ops.reshape(sobel_x, [3, 3, 3, 1, 1])

        # Apply 3D convolution with 'valid' padding (we pre-padded)
        Iz_5d = ops.conv(img_5d, sobel_z_5d, padding="valid")
        Iy_5d = ops.conv(img_5d, sobel_y_5d, padding="valid")
        Ix_5d = ops.conv(img_5d, sobel_x_5d, padding="valid")

        # Reshape back to 3D
        Iz = ops.reshape(Iz_5d, [d, h, w])
        Iy = ops.reshape(Iy_5d, [d, h, w])
        Ix = ops.reshape(Ix_5d, [d, h, w])

        return (Iz, Iy, Ix)

    def __repr__(self):
        """String representation."""
        return (
            f"LucasKanadeTracker(win_size={self.win_size}, max_level={self.max_level}, "
            f"max_iterations={self.max_iterations}, epsilon={self.epsilon})"
        )
