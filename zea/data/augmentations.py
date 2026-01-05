"""Augmentation layers for ultrasound data."""

import keras
import numpy as np
from keras import layers, ops

from zea.func.tensor import is_jax_prng_key, split_seed


class RandomCircleInclusion(layers.Layer):
    """
    Adds a circular inclusion to the image, optionally at random locations.

    Since this can accept N-dimensional inputs, you'll need to specify your
    ``circle_axes`` -- these are the axes onto which a circle will be drawn.
    This circle will then be broadcast along the remaining dimensions.

    You can then optionally specify whether there is a batch dim,
    and whether the circles should be located randomly across that batch.

    For example, if you have a batch of videos, e.g. of shape [batch, frame, height, width],
    then you might want to specify ``circle_axes=(2, 3)``, and
    ``randomize_location_across_batch=True``. This would result in a circle that is located
    in the same place per video, but different locations for different videos.

    Once your method has recovered the circles, you can evaluate them using
    the ``evaluate_recovered_circle_accuracy()`` method, which will expect an input
    shape matching your inputs to ``call()``.
    """

    def __init__(
        self,
        radius: int | tuple[int, int],
        fill_value: float = 1.0,
        circle_axes: tuple[int, int] = (1, 2),
        with_batch_dim=True,
        return_centers=False,
        recovery_threshold=0.1,
        randomize_location_across_batch=True,
        seed=None,
        width_range: tuple[int, int] = None,
        height_range: tuple[int, int] = None,
        **kwargs,
    ):
        """
        Initialize RandomCircleInclusion.

        Args:
            radius (int or tuple[int, int]): Radius of the circle/ellipse to include.
            fill_value (float): Value to fill inside the circle.
            circle_axes (tuple[int, int]): Axes along which to draw the circle
                (height, width).
            with_batch_dim (bool): Whether input has a batch dimension.
            return_centers (bool): Whether to return circle centers along with images.
            recovery_threshold (float): Threshold for considering a pixel as recovered.
            randomize_location_across_batch (bool): If True (and with_batch_dim=True),
                each batch element gets a different random center. If False, all batch
                elements share the same center.
            seed (Any): Optional random seed for reproducibility.
            width_range (tuple[int, int], optional): Range (min, max) for circle
                center x (width axis).
            height_range (tuple[int, int], optional): Range (min, max) for circle
                center y (height axis).
            **kwargs: Additional keyword arguments for the parent Layer.

        Example:
            .. doctest::

                >>> from zea.data.augmentations import RandomCircleInclusion
                >>> from keras import ops

                >>> layer = RandomCircleInclusion(
                ...     radius=5,
                ...     circle_axes=(1, 2),
                ...     with_batch_dim=True,
                ... )
                >>> image = ops.zeros((1, 28, 28), dtype="float32")
                >>> out = layer(image)  # doctest: +SKIP

        """
        super().__init__(**kwargs)

        # Validate randomize_location_across_batch only makes sense with batch dim
        if not with_batch_dim and not randomize_location_across_batch:
            raise ValueError(
                "randomize_location_across_batch=False is only applicable when "
                "with_batch_dim=True. When with_batch_dim=False, there is no batch "
                "to randomize across."
            )
        # Convert radius to tuple if int, else validate tuple
        if isinstance(radius, int):
            if radius <= 0:
                raise ValueError(f"radius must be a positive integer, got {radius}.")
            self.radius = (radius, radius)
        elif isinstance(radius, tuple) and len(radius) == 2:
            rx, ry = radius
            if not all(isinstance(r, int) for r in (rx, ry)):
                raise TypeError(f"radius tuple must contain two integers, got {radius!r}.")
            if rx <= 0 or ry <= 0:
                raise ValueError(f"radius components must be positive, got {radius!r}.")
            self.radius = (rx, ry)
        else:
            raise TypeError("radius must be an int or a tuple of two ints")

        self.fill_value = fill_value
        self.circle_axes = circle_axes
        self.with_batch_dim = with_batch_dim
        self.return_centers = return_centers
        self.recovery_threshold = recovery_threshold
        self.randomize_location_across_batch = randomize_location_across_batch
        self.seed = seed
        self.width_range = width_range
        self.height_range = height_range
        self._axis1 = None
        self._axis2 = None
        self._perm = None
        self._inv_perm = None
        self._static_shape = None
        self._static_batch = None
        self._static_h = None
        self._static_w = None
        self._static_flat_batch = 1

    def build(self, input_shape):
        """
        Build the layer and compute static shape and permutation info.

        Args:
            input_shape (tuple): Shape of the input tensor.
        """
        rank = len(input_shape) - 1 if self.with_batch_dim else len(input_shape)
        a1, a2 = self.circle_axes
        if self.with_batch_dim and (a1 == 0 or a2 == 0):
            raise ValueError("The circle axes should not be a batch dim")
        if a1 < 0:
            a1 += rank
        elif a1 > 0 and self.with_batch_dim:
            a1 -= 1
        if a2 < 0:
            a2 += rank
        elif a2 > 0 and self.with_batch_dim:
            a2 -= 1
        if not (0 <= a1 < rank and 0 <= a2 < rank):
            raise ValueError(f"circle_axes {self.circle_axes} out of range for rank {rank}")
        if a1 == a2:
            raise ValueError("circle_axes must be two distinct axes")
        self._axis1, self._axis2 = a1, a2

        all_axes = list(range(rank))
        other_axes = [ax for ax in all_axes if ax not in (a1, a2)]
        self._perm = other_axes + [a1, a2]
        inv = [0] * rank
        for i, ax in enumerate(self._perm):
            inv[ax] = i
        self._inv_perm = inv

        if self.with_batch_dim:
            input_shape = input_shape[1:]  # ignore batch dim
        permuted_shape = [input_shape[ax] for ax in self._perm]
        if len(permuted_shape) > 2:
            self._static_flat_batch = int(np.prod(permuted_shape[:-2]))
        self._static_h = int(permuted_shape[-2])
        self._static_w = int(permuted_shape[-1])
        self._static_shape = tuple(permuted_shape)

        # Validate that ellipse can fit within image bounds
        rx, ry = self.radius
        min_required_width = 2 * rx + 1
        min_required_height = 2 * ry + 1

        if self._static_w < min_required_width:
            raise ValueError(
                f"Image width ({self._static_w}) is too small for radius {rx}. "
                f"Minimum required width: {min_required_width}"
            )
        if self._static_h < min_required_height:
            raise ValueError(
                f"Image height ({self._static_h}) is too small for radius {ry}. "
                f"Minimum required height: {min_required_height}"
            )

        # Validate width_range and height_range if provided
        if self.width_range is not None:
            min_x, max_x = self.width_range
            if min_x >= max_x:
                raise ValueError(f"width_range must have min < max, got {self.width_range}")
            if min_x < rx or max_x > self._static_w - rx:
                raise ValueError(
                    f"width_range {self.width_range} would place circle outside image bounds. "
                    f"Valid range: [{rx}, {self._static_w - rx})"
                )

        if self.height_range is not None:
            min_y, max_y = self.height_range
            if min_y >= max_y:
                raise ValueError(f"height_range must have min < max, got {self.height_range}")
            if min_y < ry or max_y > self._static_h - ry:
                raise ValueError(
                    f"height_range {self.height_range} would place circle outside image bounds. "
                    f"Valid range: [{ry}, {self._static_h - ry})"
                )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape for the layer.

        Args:
            input_shape (tuple): Shape of the input tensor.

        Returns:
            tuple: The output shape (same as input).
        """
        return input_shape

    def _permute_axes_to_circle_last(self, x):
        """
        Permute axes so that circle axes are last.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Tensor with circle axes as the last two dimensions.
        """
        return ops.transpose(x, axes=self._perm)

    def _flatten_batch_and_other_dims(self, x):
        """
        Flatten all axes except the last two (circle axes).

        Args:
            x (Tensor): Input tensor with circle axes last.

        Returns:
            tuple: (reshaped tensor, flat batch size, height, width).
        """
        shape = x.shape
        flat_batch = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        h, w = shape[-2], shape[-1]
        return ops.reshape(x, [flat_batch, h, w]), flat_batch, h, w

    def _make_circle_mask(self, centers, h, w, radius, dtype):
        """
        Create a mask for each center (batch, h, w) using Keras ops.

        Args:
            centers (Tensor): Tensor of shape (batch, 2) with circle centers.
            h (int): Height of the image.
            w (int): Width of the image.
            radius (tuple[int, int]): Radii of the ellipse (rx, ry).
            dtype (str or dtype): Data type for the mask.

        Returns:
            Tensor: Mask of shape (batch, h, w).
        """
        Y = ops.arange(h)
        X = ops.arange(w)
        Y, X = ops.meshgrid(Y, X, indexing="ij")
        Y = ops.expand_dims(Y, 0)  # (1, h, w)
        X = ops.expand_dims(X, 0)  # (1, h, w)
        cx = centers[:, 0][:, None, None]
        cy = centers[:, 1][:, None, None]
        rx, ry = radius
        # Ellipse equation: ((X-cx)/rx)^2 + ((Y-cy)/ry)^2 <= 1
        dist = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2
        mask = ops.cast(dist <= 1, dtype)
        return mask

    def call(self, x, seed=None):
        """
        Apply the random circle inclusion augmentation.

        Args:
            x (Tensor): Input tensor.
            seed (Any, optional): Optional random seed for reproducibility.

        Returns:
            Tensor or tuple: Augmented images, and optionally the circle
                centers if return_centers is True.
        """
        if keras.backend.backend() == "jax" and not is_jax_prng_key(seed):
            if isinstance(seed, keras.random.SeedGenerator):
                raise ValueError(
                    "When using JAX backend, please provide a jax.random.PRNGKey as seed, "
                    "instead of keras.random.SeedGenerator."
                )
            else:
                raise TypeError(
                    f"When using JAX backend, seed must be a JAX PRNG key (created with "
                    f"jax.random.PRNGKey()), but got {type(seed)}. Note: jax.random.key() "
                    f"keys are not currently supported."
                )
        seed = seed if seed is not None else self.seed

        if self.with_batch_dim:
            x_is_symbolic_tensor = not isinstance(ops.shape(x)[0], int)
            if x_is_symbolic_tensor:
                if self.randomize_location_across_batch:
                    imgs, centers = ops.map(lambda arg: self._call(arg, seed), x)
                else:
                    raise NotImplementedError(
                        "You cannot fix circle locations across batch while using "
                        + "RandomCircleInclusion as a dataset augmentation, "
                        + "since samples in a batch are handled independently."
                    )
            else:
                batch_size = ops.shape(x)[0]
                if self.randomize_location_across_batch:
                    seeds = split_seed(seed, batch_size)
                    if all(s is seeds[0] for s in seeds):
                        imgs, centers = ops.map(lambda arg: self._call(arg, seeds[0]), x)
                    else:
                        imgs, centers = ops.map(
                            lambda args: self._call(args[0], args[1]), (x, seeds)
                        )
                else:
                    # Generate one random center that will be used for all batch elements
                    img0, center0 = self._call(x[0], seed)

                    # Apply the same center to all batch elements
                    imgs_list, centers_list = [img0], [center0]
                    for i in range(1, batch_size):
                        img_aug, center_out = self._call_with_fixed_center(x[i], center0)
                        imgs_list.append(img_aug)
                        centers_list.append(center_out)

                    imgs = ops.stack(imgs_list, axis=0)
                    centers = ops.stack(centers_list, axis=0)
        else:
            imgs, centers = self._call(x, seed)

        if self.return_centers:
            return imgs, centers
        else:
            return imgs

    def _call(self, x, seed):
        """
        Internal method to apply the augmentation to a single image.

        Args:
            x (Tensor): Input image tensor with circle axes last.
            seed (Any): Random seed for circle location.

        Returns:
            tuple: (augmented image, center coordinates).
        """
        x = self._permute_axes_to_circle_last(x)
        flat, flat_batch_size, h, w = self._flatten_batch_and_other_dims(x)

        def _draw_circle_2d(img2d):
            rx, ry = self.radius
            # Determine allowed ranges for center
            if self.width_range is not None:
                min_x, max_x = self.width_range
            else:
                min_x, max_x = rx, w - rx
            if self.height_range is not None:
                min_y, max_y = self.height_range
            else:
                min_y, max_y = ry, h - ry
            # Ensure the ellipse fits within the allowed region
            cx = ops.cast(
                keras.random.uniform((), min_x, max_x, seed=seed),
                "int32",
            )
            new_seed, _ = split_seed(seed, 2)  # ensure that cx and cy are independent
            cy = ops.cast(
                keras.random.uniform((), min_y, max_y, seed=new_seed),
                "int32",
            )
            mask = self._make_circle_mask(
                ops.stack([cx, cy])[None, :], h, w, (rx, ry), img2d.dtype
            )[0]
            img_aug = img2d * (1 - mask) + self.fill_value * mask
            center = ops.stack([cx, cy])
            return img_aug, center

        aug_imgs, centers = ops.vectorized_map(_draw_circle_2d, flat)
        aug_imgs = ops.reshape(aug_imgs, x.shape)
        aug_imgs = ops.transpose(aug_imgs, axes=self._inv_perm)
        centers_shape = [2] if flat_batch_size == 1 else [flat_batch_size, 2]
        centers = ops.reshape(centers, centers_shape)
        return (aug_imgs, centers)

    def _apply_circle_mask(self, flat, center, h, w):
        """Apply circle mask to flattened image data.

        Args:
            flat (Tensor): Flattened image data of shape (flat_batch, h, w).
            center (Tensor): Center coordinates, either (2,) or (flat_batch, 2).
            h (int): Height of images.
            w (int): Width of images.

        Returns:
            Tensor: Augmented images with circle applied.
        """
        rx, ry = self.radius

        # Ensure center has batch dimension for broadcasting
        if len(center.shape) == 1:
            # Single center (2,) -> broadcast to all slices
            center_batched = ops.tile(ops.reshape(center, [1, 2]), [flat.shape[0], 1])
        else:
            # Already batched (flat_batch, 2)
            center_batched = center

        # Create masks for all slices using vectorized_map or broadcasting
        masks = self._make_circle_mask(center_batched, h, w, (rx, ry), flat.dtype)

        # Apply masks
        aug_imgs = flat * (1 - masks) + self.fill_value * masks
        return aug_imgs

    def _call_with_fixed_center(self, x, fixed_center):
        """Apply augmentation using a pre-determined center.

        Args:
            x (Tensor): Input image tensor.
            fixed_center (Tensor): Pre-determined center coordinates, either (2,)
                for a single center or (flat_batch, 2) for per-slice centers.

        Returns:
            tuple: (augmented image, center coordinates).
        """
        x = self._permute_axes_to_circle_last(x)
        flat, flat_batch_size, h, w = self._flatten_batch_and_other_dims(x)

        # Apply circle mask with fixed center (handles both single and batched centers)
        aug_imgs = self._apply_circle_mask(flat, fixed_center, h, w)
        aug_imgs = ops.reshape(aug_imgs, x.shape)
        aug_imgs = ops.transpose(aug_imgs, axes=self._inv_perm)

        # Return centers matching the expected shape
        if len(fixed_center.shape) == 1:
            # Single center (2,) -> broadcast to match flat_batch_size
            if flat_batch_size == 1:
                centers = fixed_center
            else:
                centers = ops.tile(ops.reshape(fixed_center, [1, 2]), [flat_batch_size, 1])
        else:
            # Already batched centers (flat_batch, 2)
            centers = fixed_center

        return (aug_imgs, centers)

    def get_config(self):
        """
        Get layer configuration for serialization.

        Returns:
            dict: Dictionary of layer configuration.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "radius": self.radius,
                "fill_value": self.fill_value,
                "circle_axes": self.circle_axes,
                "return_centers": self.return_centers,
                "width_range": self.width_range,
                "height_range": self.height_range,
            }
        )
        return cfg

    def evaluate_recovered_circle_accuracy(
        self, images, centers, recovery_threshold, fill_value=None
    ):
        """
        Evaluate the percentage of the true circle that has been recovered in the images,
        and return a mask of the detected part of the circle.

        Args:
            images (Tensor): Tensor of images (any shape, with circle axes as specified).
            centers (Tensor): Tensor of circle centers (matching batch size).
            recovery_threshold (float): Threshold for considering a pixel as recovered.
            fill_value (float, optional): Optionally override fill_value for cases
                where image range has changed.

         Returns:
            Tuple[Tensor, Tensor]:
                - percent_recovered: [batch] - average recovery percentage per batch element,
                  averaged across all non-batch dimensions (e.g., frames, slices)
                - recovered_masks: [batch, flat_batch, h, w] or [batch, h, w] or [flat_batch, h, w]
                   depending on input shape - binary mask of detected circle regions
        """
        fill_value = fill_value or self.fill_value

        def _evaluate_recovered_circle_accuracy(image, center):
            image_perm = self._permute_axes_to_circle_last(image)
            h, w = image_perm.shape[-2], image_perm.shape[-1]
            flat_image, _, _, _ = self._flatten_batch_and_other_dims(image_perm)
            flat_center = ops.reshape(center, [-1, 2])
            mask = self._make_circle_mask(flat_center, h, w, self.radius, flat_image.dtype)
            diff = ops.abs(flat_image - fill_value)
            recovered = ops.cast(diff <= recovery_threshold, flat_image.dtype) * mask
            recovered_sum = ops.sum(recovered, axis=[1, 2])
            mask_sum = ops.sum(mask, axis=[1, 2])
            percent_recovered = recovered_sum / (mask_sum + 1e-8)
            # recovered_mask: binary mask of detected part of the circle
            recovered_mask = ops.cast(recovered > 0, flat_image.dtype)
            return percent_recovered, recovered_mask

        if self.with_batch_dim:
            results = ops.vectorized_map(
                lambda args: _evaluate_recovered_circle_accuracy(args[0], args[1]),
                (images, centers),
            )
            percent_recovered, recovered_masks = results
            # If there are multiple circles per batch element (e.g., multiple frames/slices),
            # take the mean across all non-batch dimensions to get one value per batch element
            if len(percent_recovered.shape) > 1:
                # Average over all axes except the batch dimension (axis 0)
                axes_to_reduce = tuple(range(1, len(percent_recovered.shape)))
                percent_recovered = ops.mean(percent_recovered, axis=axes_to_reduce)
            return percent_recovered, recovered_masks
        else:
            percent_recovered, recovered_mask = _evaluate_recovered_circle_accuracy(images, centers)
            return percent_recovered, recovered_mask
