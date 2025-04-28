import keras
from keras import layers, ops


class RandomCircleInclusion(layers.Layer):
    def __init__(
        self,
        radius: int,
        fill_value: float = 1.0,
        circle_axes: tuple[int, int] = (0, 1),
        with_batch_dim=True,
        seed=None,
        return_centers=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.radius = radius
        self.fill_value = fill_value
        self.circle_axes = circle_axes
        self.seed = seed
        self.with_batch_dim = with_batch_dim
        self.return_centers = return_centers

    def build(self, input_shape):
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
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x):
        if self.with_batch_dim:
            imgs, centers = ops.map(self._call, x)
        else:
            imgs, centers = self._call(x)

        if self.return_centers:
            return imgs, centers
        else:
            return imgs

    def _call(self, x):
        x = ops.transpose(x, axes=self._perm)
        full_shape = ops.shape(x)
        squeeze_batch = full_shape[:-2] == ()
        batch_size = ops.prod(full_shape[:-2])
        h, w = full_shape[-2], full_shape[-1]
        flat = ops.reshape(x, [batch_size, h, w])

        def _draw_circle_2d(img2d):
            cx = ops.cast(
                keras.random.uniform((), self.radius, w - self.radius, seed=self.seed),
                "int32",
            )
            cy = ops.cast(
                keras.random.uniform((), self.radius, h - self.radius, seed=self.seed),
                "int32",
            )
            Y = ops.arange(h)
            X = ops.arange(w)
            Y, X = ops.meshgrid(Y, X, indexing="ij")
            dist = ops.sqrt(ops.cast((X - cx) ** 2 + (Y - cy) ** 2, "float32"))
            mask = ops.cast(dist <= self.radius, img2d.dtype)
            img_aug = img2d * (1 - mask) + self.fill_value * mask
            center = ops.stack([cx, cy])
            return img_aug, center

        aug_imgs, centers = ops.vectorized_map(_draw_circle_2d, flat)
        aug_imgs = ops.reshape(aug_imgs, full_shape)
        aug_imgs = ops.transpose(aug_imgs, axes=self._inv_perm)
        centers_shape = [2] if squeeze_batch else [batch_size, 2]
        centers = ops.reshape(centers, centers_shape)

        return (aug_imgs, centers)

    def get_config(self):
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

    def evaluate_recovered_circle_accuracy(self, image, center, threshold):
        """
        Evaluate what percentage of the true circle at `center` with `self.radius`
        has been recovered in `image`, with a pixel considered 'recovered' if it is
        within `threshold` of `self.fill_value`.

        Args:
            image: 2D numpy array or tensor (height, width).
            center: (cx, cy) tuple or array.
            threshold: float, absolute tolerance for fill_value.

        Returns:
            float: percentage of recovered pixels inside the true circle (0.0 - 1.0).
        """
        import numpy as np

        if hasattr(image, "numpy"):
            image = image.numpy()
        if hasattr(center, "numpy"):
            center = center.numpy()
        cx, cy = center
        h, w = image.shape[-2:]
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= self.radius**2
        recovered = np.abs(image[mask] - self.fill_value) <= threshold
        percent_recovered = np.sum(recovered) / np.sum(mask)
        return percent_recovered
