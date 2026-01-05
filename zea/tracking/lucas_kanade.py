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

            # Track each point
            new_flows = []

            for i in range(n_points):
                pt = curr_points[i]
                flow_guess = flows[i]

                flow = self._track_point(prev_img, next_img, pt, flow_guess)
                new_flows.append(flow)

            flows = ops.stack(new_flows)

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

    def _track_point(
        self,
        prev_img,
        next_img,
        point,
        flow_guess,
    ):
        """Track a single point using iterative Lucas-Kanade."""
        # Extract template window
        template, valid_template = self._extract_window(prev_img, point)
        if not valid_template:
            return flow_guess

        # Compute template gradients (Sobel) - returns tensors
        gradients = self._sobel_gradients(template)

        # Flatten gradients for 2D or 3D
        if self.ndim == 2:
            Iy, Ix = gradients
            Ix_flat = ops.reshape(Ix, [-1])
            Iy_flat = ops.reshape(Iy, [-1])

            # Structure tensor 2D components
            IxIx = ops.sum(Ix_flat * Ix_flat)
            IxIy = ops.sum(Ix_flat * Iy_flat)
            IyIy = ops.sum(Iy_flat * Iy_flat)

        else:  # 3D
            Iz, Iy, Ix = gradients
            Ix_flat = ops.reshape(Ix, [-1])
            Iy_flat = ops.reshape(Iy, [-1])
            Iz_flat = ops.reshape(Iz, [-1])

            # Structure tensor 3D components
            IxIx = ops.sum(Ix_flat * Ix_flat)
            IxIy = ops.sum(Ix_flat * Iy_flat)
            IxIz = ops.sum(Ix_flat * Iz_flat)
            IyIy = ops.sum(Iy_flat * Iy_flat)
            IyIz = ops.sum(Iy_flat * Iz_flat)
            IzIz = ops.sum(Iz_flat * Iz_flat)

        # Iterative refinement (keep as tensors)
        flow = flow_guess

        for iteration in range(self.max_iterations):
            # Extract warped window from next image
            warped_pt = point + flow
            warped, valid_warped = self._extract_window(next_img, warped_pt)

            if not valid_warped:
                break

            # Image difference
            diff = template - warped
            diff_flat = ops.reshape(diff, [-1])

            # Solve for flow update
            if self.ndim == 2:
                # Build structure tensor matrix (2x2)
                structure = ops.stack(
                    [
                        ops.stack([IxIx, IxIy]),
                        ops.stack([IxIy, IyIy]),
                    ],
                    axis=0,
                )
                # Add regularization to diagonal
                structure = structure + ops.eye(2, dtype=structure.dtype) * 1e-5

                # Right-hand side vector
                b_x = ops.sum(Ix_flat * diff_flat)
                b_y = ops.sum(Iy_flat * diff_flat)
                rhs = ops.reshape(ops.stack([b_x, b_y]), (2, 1))

                # Solve: structure * delta_xy = rhs
                delta_xy = ops.matmul(ops.linalg.inv(structure), rhs)
                delta_xy = ops.reshape(delta_xy, (2,))

                # Reorder to (y, x)
                delta = ops.stack([delta_xy[1], delta_xy[0]])

            else:  # 3D
                # Build structure tensor matrix (3x3)
                structure = ops.stack(
                    [
                        ops.stack([IxIx, IxIy, IxIz]),
                        ops.stack([IxIy, IyIy, IyIz]),
                        ops.stack([IxIz, IyIz, IzIz]),
                    ],
                    axis=0,
                )
                # Add regularization to diagonal
                structure = structure + ops.eye(3, dtype=structure.dtype) * 1e-5

                # Right-hand side vector
                b_x = ops.sum(Ix_flat * diff_flat)
                b_y = ops.sum(Iy_flat * diff_flat)
                b_z = ops.sum(Iz_flat * diff_flat)
                rhs = ops.reshape(ops.stack([b_x, b_y, b_z]), (3, 1))

                # Solve: structure * delta_xyz = rhs
                delta_xyz = ops.matmul(ops.linalg.inv(structure), rhs)
                delta_xyz = ops.reshape(delta_xyz, (3,))

                # Reorder to (z, y, x)
                delta = ops.stack([delta_xyz[2], delta_xyz[1], delta_xyz[0]])

            # Update flow
            flow = flow + delta

            # Check convergence
            delta_norm = ops.sqrt(ops.sum(delta * delta))
            if delta_norm < self.epsilon:
                break

        return flow

    def _extract_window(self, image, point):
        """Extract window around point with subpixel interpolation."""
        if self.ndim == 2:
            return self._extract_window_2d(image, point)
        elif self.ndim == 3:
            return self._extract_window_3d(image, point)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

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

    def _sobel_gradients(self, image):
        """Compute Sobel gradients for 2D or 3D."""
        if self.ndim == 2:
            return self._sobel_gradients_2d(image)
        elif self.ndim == 3:
            return self._sobel_gradients_3d(image)
        else:
            raise ValueError(f"Unsupported ndim: {self.ndim}")

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
