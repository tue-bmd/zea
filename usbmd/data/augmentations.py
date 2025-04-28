"""Augmentation layers for ultrasound data."""

import numpy as np
import keras
from keras import layers, ops

# pylint: disable=arguments-differ, abstract-class-instantiated, pointless-string-statement


class RandomCircleInclusion(layers.Layer):
    """Randomly includes a filled circle at a random location in an image."""

    def __init__(
        self,
        radius: int,
        fill_value: float = 1.0,
        circle_axes: tuple[int, int] = (1, 2),
        with_batch_dim=True,
        seed=None,
        return_centers=False,
        recovery_threshold=0.1,
        randomize_location_across_batch=True,
        **kwargs,
    ):
        """
        Initialize RandomCircleInclusion.

        Args:
            radius: Radius of the circle.
            fill_value: Value to fill inside the circle.
            circle_axes: Axes along which to draw the circle.
            with_batch_dim: Whether input has a batch dimension.
            seed: Random seed.
            return_centers: Whether to return circle centers.
            recovery_threshold: Threshold for considering a pixel as recovered.
            randomize_location_across_batch: If True, randomize circle location per batch element.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.fill_value = fill_value
        self.circle_axes = circle_axes
        self.seed = seed
        self.with_batch_dim = with_batch_dim
        self.return_centers = return_centers
        self.recovery_threshold = recovery_threshold
        self.randomize_location_across_batch = randomize_location_across_batch
        self._axis1 = None
        self._axis2 = None
        self._perm = None
        self._inv_perm = None
        self._static_shape = None
        self._static_batch = None
        self._static_h = None
        self._static_w = None

    def build(self, input_shape):
        """
        Build the layer and compute static shape and permutation info.

        Args:
            input_shape: Shape of the input tensor.
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
            raise ValueError(
                f"circle_axes {self.circle_axes} out of range for rank {rank}"
            )
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
        else:
            self._static_flat_batch = 1
        self._static_h = int(permuted_shape[-2])
        self._static_w = int(permuted_shape[-1])
        self._static_shape = tuple(permuted_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        """
        Compute output shape for the layer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            The output shape (same as input).
        """
        return input_shape

    def _permute_axes_to_circle_last(self, x):
        """
        Permute axes so that circle axes are last.

        Args:
            x: Input tensor.

        Returns:
            Tensor with circle axes as the last two dimensions.
        """
        return ops.transpose(x, axes=self._perm)

    def _flatten_batch_and_other_dims(self, x):
        """
        Flatten all axes except the last two (circle axes).

        Args:
            x: Input tensor with circle axes last.

        Returns:
            Tuple of (reshaped tensor, flat batch size, height, width).
        """
        shape = x.shape
        flat_batch = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        h, w = shape[-2], shape[-1]
        return ops.reshape(x, [flat_batch, h, w]), flat_batch, h, w

    def _make_circle_mask(self, centers, h, w, radius, dtype):
        """
        Create a mask for each center (batch, h, w) using Keras ops.

        Args:
            centers: Tensor of shape (batch, 2) with circle centers.
            h: Height of the image.
            w: Width of the image.
            radius: Radius of the circle.
            dtype: Data type for the mask.

        Returns:
            Tensor mask of shape (batch, h, w).
        """
        Y = ops.arange(h)
        X = ops.arange(w)
        Y, X = ops.meshgrid(Y, X, indexing="ij")
        Y = ops.expand_dims(Y, 0)  # (1, h, w)
        X = ops.expand_dims(X, 0)  # (1, h, w)
        cx = ops.cast(centers[:, 0], "float32")[:, None, None]
        cy = ops.cast(centers[:, 1], "float32")[:, None, None]
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask = ops.cast(dist2 <= radius**2, dtype)
        return mask

    def call(self, x):
        """
        Apply the random circle inclusion augmentation.

        Args:
            x: Input tensor.

        Returns:
            Augmented images, and optionally the circle centers.
        """
        backend = keras.backend.backend()
        seed = self.seed

        if self.with_batch_dim:
            if backend == "jax":
                batch_size = x.shape[0]
                import jax

                if self.randomize_location_across_batch:
                    seeds = jax.random.split(seed, batch_size)
                    imgs, centers = ops.map(
                        lambda args: self._call(args[0], args[1]), (x, seeds)
                    )
                else:
                    imgs, centers = ops.map(lambda arg: self._call(arg, seed), x)
            else:
                imgs, centers = ops.map(lambda xi: self._call(xi, seed), x)
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
            x: Input image tensor with circle axes last.
            seed: Random seed for circle location.

        Returns:
            Tuple of (augmented image, center coordinates).
        """
        x = self._permute_axes_to_circle_last(x)
        flat, flat_batch_size, h, w = self._flatten_batch_and_other_dims(x)

        def _draw_circle_2d(img2d):
            cx = ops.cast(
                keras.random.uniform((), self.radius, w - self.radius, seed=seed),
                "int32",
            )
            cy = ops.cast(
                keras.random.uniform((), self.radius, h - self.radius, seed=seed),
                "int32",
            )
            mask = self._make_circle_mask(
                ops.stack([cx, cy])[None, :], h, w, self.radius, img2d.dtype
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

    def get_config(self):
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary of layer configuration.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "radius": self.radius,
                "fill_value": self.fill_value,
                "circle_axes": self.circle_axes,
                "seed": self.seed,
                "return_centers": self.return_centers,
            }
        )
        return cfg

    def evaluate_recovered_circle_accuracy(self, images, centers):
        """
        Evaluate the percentage of the true circle that has been recovered in the images.

        Args:
            images: Tensor of images (any shape, with circle axes as specified).
            centers: Tensor of circle centers (matching batch size).

        Returns:
            Tensor of percentage recovered for each circle (shape: [num_circles]).
        """
        images = ops.convert_to_tensor(images)
        centers = ops.convert_to_tensor(centers)
        # Permute axes so circle axes are last
        images_perm = self._permute_axes_to_circle_last(images)
        h, w = images_perm.shape[-2], images_perm.shape[-1]
        flat_images, flat_batch, _, _ = self._flatten_batch_and_other_dims(images_perm)
        flat_centers = ops.reshape(centers, [-1, 2])
        mask = self._make_circle_mask(
            flat_centers, h, w, self.radius, flat_images.dtype
        )
        diff = ops.abs(flat_images - self.fill_value)
        recovered = ops.cast(diff <= self.recovery_threshold, flat_images.dtype) * mask
        recovered_sum = ops.sum(recovered, axis=[1, 2])
        mask_sum = ops.sum(mask, axis=[1, 2])
        percent_recovered = recovered_sum / (mask_sum + 1e-8)
        return percent_recovered
