from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from zea import log


class Spec:
    SCHEMA = {}

    def __post_init__(self):
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            expected_dtype = field_info["dtype"]
            expected_shape = field_info["shape"]

            # Track dimension names and sizes for consistency checks
            for i, dim_name in enumerate(expected_shape):
                if isinstance(dim_name, str):
                    dim_to_fields[dim_name].add(field_name)
                    dim_to_sizes[dim_name].add(field_value.shape[i])

            # Check dtype
            try:
                expected_np_dtype = np.dtype(expected_dtype)
                is_numpy_dtype = True
            except TypeError:
                is_numpy_dtype = False

            if is_numpy_dtype:
                if not np.issubdtype(field_value.dtype, expected_np_dtype):
                    raise TypeError(
                        f"{field_name} must be of dtype compatible with {expected_np_dtype}, "
                        f"got {field_value.dtype}"
                    )
            else:
                if field_value.dtype != expected_dtype:
                    raise TypeError(
                        f"{field_name} must be of type {expected_dtype}, got {field_value.dtype}"
                    )

            # Check ndims
            if len(field_value.shape) != len(expected_shape):
                raise ValueError(
                    f"{field_name} must have {len(expected_shape)} dimensions, "
                    f"got {len(field_value.shape)} dimensions with shape {field_value.shape}"
                )

            # Check static shape dimensions
            for dim_size, expected_dim in zip(field_value.shape, expected_shape):
                if isinstance(expected_dim, str):
                    continue  # skip literal dimensions
                if dim_size != expected_dim:
                    raise ValueError(
                        f"{field_name} dimension size mismatch: expected {expected_dim}, "
                        f"got {dim_size} with shape {field_value.shape}"
                    )

        # Check that dimensions with the same name have consistent sizes across fields
        for dim_name, sizes in dim_to_sizes.items():
            if len(sizes) > 1:
                field_names = sorted(dim_to_fields[dim_name])
                raise ValueError(
                    f"Dimension '{dim_name}' has inconsistent sizes across "
                    f"fields {field_names}: {sorted(sizes)}"
                )


@dataclass
class Segmentation(Spec):
    """Segmentation data and spatial extent metadata.

    Args:
        pixels: The segmentation pixels of shape (n_frames, h, w, d) of type uint8.
        labels: The labels corresponding to the segmentation pixels, where each unique value
            in the pixels corresponds to a label in this list of shape (n_labels,) and type str.
        extent: The segmentation extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """

    pixels: np.ndarray
    labels: np.ndarray
    extent: np.ndarray

    SCHEMA = {
        "pixels": {"dtype": np.uint8, "shape": ("n_frames", "h", "w", "d")},
        "labels": {"dtype": np.str_, "shape": ("n_labels",)},
        "extent": {"dtype": np.float32, "shape": ("n_frames", 6)},
    }

    def __post_init__(self):
        if self.extent.ndim == 1:
            self.extent = np.broadcast_to(
                self.extent,
                (self.pixels.shape[0], self.extent.shape[0]),
            ).copy()

        super().__post_init__()

        # Check sensible values
        if np.any(self.extent[:, 0] >= self.extent[:, 1]):
            raise ValueError("Segmentation extent xlims must have xmin < xmax")
        if np.any(self.extent[:, 2] >= self.extent[:, 3]):
            raise ValueError("Segmentation extent ylims must have ymin < ymax")
        if np.any(self.extent[:, 4] >= self.extent[:, 5]):
            raise ValueError("Segmentation extent zlims must have zmax < zmin")
        if np.any(self.extent >= 1.0) or np.any(self.extent <= -1.0):
            log.warning(
                "Segmentation extent values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the extent values are correct and in meters."
            )

        # Check every pixel value corresponds to a label
        unique_pixel_values = np.unique(self.pixels)
        if not np.all(np.isin(unique_pixel_values, np.arange(len(self.labels)))):
            raise ValueError(
                "Segmentation pixels contain values that do not correspond to any label. "
                f"Unique pixel values: {unique_pixel_values}, number of labels: {len(self.labels)}"
            )


@dataclass
class Data:
    raw_data: np.ndarray
    segmentation: Segmentation


@dataclass
class Scan:
    t0_delays: np.ndarray


@dataclass
class DataSpec:
    data: Data
    scan: Scan


if __name__ == "__main__":
    import zea

    zea.init_device("cpu")
    # Example usage
    pixels = np.zeros((10, 256, 256, 1), dtype=np.uint8)
    labels = np.array(["background", "label1", "label2", "label3"], dtype=np.str_)
    extent = np.array([0.0, 1.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)

    segmentation = Segmentation(pixels=pixels, labels=labels, extent=extent)
