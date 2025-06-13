"""Augmentation layers for ultrasound data."""

import keras
import numpy as np
from keras import layers, ops

from zea.tensor_ops import is_jax_prng_key, split_seed


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
        radius: int,
        fill_value: float = 1.0,
        circle_axes: tuple[int, int] = (1, 2),
        with_batch_dim=True,
        return_centers=False,
        recovery_threshold=0.1,
        randomize_location_across_batch=True,
        seed=None,
        **kwargs,
    ):
        """
        Initialize RandomCircleInclusion.

        Args:
            radius (int): Radius of the circle to include.
            fill_value (float): Value to fill inside the circle.
            circle_axes (tuple[int, int]): Axes along which to draw the circle (height, width).
            with_batch_dim (bool): Whether input has a batch dimension.
            return_centers (bool): Whether to return circle centers along with images.
            recovery_threshold (float): Threshold for considering a pixel as recovered.
            randomize_location_across_batch (bool): If True, randomize circle location
                per batch element.
            seed (Any): Optional random seed for reproducibility.
            **kwargs: Additional keyword arguments for the parent Layer.
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.fill_value = fill_value
        self.circle_axes = circle_axes
        self.with_batch_dim = with_batch_dim
        self.return_centers = return_centers
        self.recovery_threshold = recovery_threshold
        self.randomize_location_across_batch = randomize_location_across_batch
        self.seed = seed
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
            radius (int): Radius of the circle.
            dtype (str or dtype): Data type for the mask.

        Returns:
            Tensor: Mask of shape (batch, h, w).
        """
        Y = ops.arange(h)
        X = ops.arange(w)
        Y, X = ops.meshgrid(Y, X, indexing="ij")
        Y = ops.expand_dims(Y, 0)  # (1, h, w)
        X = ops.expand_dims(X, 0)  # (1, h, w)
        # cx = ops.cast(centers[:, 0], "float32")[:, None, None]
        # cy = ops.cast(centers[:, 1], "float32")[:, None, None]
        cx = centers[:, 0][:, None, None]
        cy = centers[:, 1][:, None, None]
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2
        mask = ops.cast(dist2 <= radius**2, dtype)
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
            raise NotImplementedError(
                "jax.random.key() is not supported, please use jax.random.PRNGKey()"
            )
        seed = seed if seed is not None else self.seed

        if self.with_batch_dim:
            x_is_symbolic_tensor = not isinstance(ops.shape(x)[0], int)
            if x_is_symbolic_tensor:
                if self.randomize_location_across_batch:
                    imgs, centers = ops.map(lambda arg: self._call(arg, seed), x)
                else:
                    raise NotImplementedError(
                        "You cannot fix circle locations across while using"
                        + "RandomCircleInclusion as a dataset augmentation, "
                        + "since samples in a batch are handled independently."
                    )
            else:
                if self.randomize_location_across_batch:
                    batch_size = ops.shape(x)[0]
                    seeds = split_seed(seed, batch_size)
                    if all(seed is seeds[0] for seed in seeds):
                        imgs, centers = ops.map(lambda arg: self._call(arg, seeds[0]), x)
                    else:
                        imgs, centers = ops.map(
                            lambda args: self._call(args[0], args[1]), (x, seeds)
                        )
                else:
                    imgs, centers = ops.map(lambda arg: self._call(arg, seed), x)
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
            cx = ops.cast(
                keras.random.uniform((), self.radius, w - self.radius, seed=seed),
                "int32",
            )
            new_seed, _ = split_seed(seed, 2)  # ensure that cx and cy are independent
            cy = ops.cast(
                keras.random.uniform((), self.radius, h - self.radius, seed=new_seed),
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
            dict: Dictionary of layer configuration.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "radius": self.radius,
                "fill_value": self.fill_value,
                "circle_axes": self.circle_axes,
                "return_centers": self.return_centers,
            }
        )
        return cfg

    def evaluate_recovered_circle_accuracy(
        self, images, centers, recovery_threshold, fill_value=None
    ):
        """
        Evaluate the percentage of the true circle that has been recovered in the images.

        Args:
            images (Tensor): Tensor of images (any shape, with circle axes as specified).
            centers (Tensor): Tensor of circle centers (matching batch size).
            recovery_threshold (float): Threshold for considering a pixel as recovered.
            fill_value (float, optional): Optionally override fill_value for cases
                where image range has changed.

        Returns:
            Tensor: Percentage recovered for each circle (shape: [num_circles]).
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
            return percent_recovered

        if self.with_batch_dim:
            return ops.vectorized_map(
                lambda args: _evaluate_recovered_circle_accuracy(args[0], args[1]),
                (images, centers),
            )[..., 0]
        else:
            return _evaluate_recovered_circle_accuracy(images, centers)
