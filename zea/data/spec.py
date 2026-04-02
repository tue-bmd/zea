from collections import defaultdict
from dataclasses import dataclass
from typing import List

import numpy as np

from zea import log


def check_dtype(value: np.ndarray, expected_dtype: type) -> None:
    """Check if the dtype of a numpy array matches the expected dtype,
    allowing for compatible types."""
    try:
        expected_np_dtype = np.dtype(expected_dtype)
        is_numpy_dtype = True
    except TypeError:
        is_numpy_dtype = False

    if is_numpy_dtype:
        if not np.issubdtype(value.dtype, expected_np_dtype):
            raise TypeError(
                f"Expected dtype compatible with {expected_np_dtype}, got {value.dtype}"
            )
    else:
        if value.dtype != expected_dtype:
            raise TypeError(f"Expected type {expected_dtype}, got {value.dtype}")


def match_shape(value: np.ndarray, expected_shape: tuple) -> bool:
    """Check if the shape of a numpy array matches the expected shape specification."""
    if len(value.shape) != len(expected_shape):
        return False

    for dim_size, expected_dim in zip(value.shape, expected_shape):
        if isinstance(expected_dim, str):
            continue
        if dim_size != expected_dim:
            return False

    return True


def find_matched_shape(value: np.ndarray, expected_shapes: List[tuple]) -> tuple | None:
    """Find the first expected shape specification that matches the shape of the value."""
    for expected_shape in expected_shapes:
        if match_shape(value, expected_shape):
            return expected_shape
    return None


class Spec:
    """Base class for data specifications with schema validation.

    Subclasses should define a SCHEMA class variable that specifies the expected dtype and shape
    for each field. The __post_init__ method will validate that the actual fields match the schema,
    including checking that dimensions with the same name have consistent sizes across fields.
    """

    SCHEMA: dict

    def __post_init__(self):
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            expected_dtype = field_info["dtype"]
            shape_spec = field_info["shape"]

            if shape_spec and isinstance(shape_spec[0], tuple):
                expected_shapes = shape_spec
            else:
                expected_shapes = (shape_spec,)

            check_dtype(field_value, expected_dtype)

            matched_shape = find_matched_shape(field_value, expected_shapes)
            if matched_shape is None:
                allowed_shapes = ", ".join(str(shape) for shape in expected_shapes)
                raise ValueError(
                    f"{field_name} has shape {field_value.shape}, expected one of: {allowed_shapes}"
                )

            # Track dimension names and sizes for consistency checks
            for i, dim_name in enumerate(matched_shape):
                if isinstance(dim_name, str):
                    dim_to_fields[dim_name].add(field_name)
                    dim_to_sizes[dim_name].add(field_value.shape[i])

        # Check that dimensions with the same name have consistent sizes across fields
        for dim_name, sizes in dim_to_sizes.items():
            if len(sizes) > 1:
                field_names = sorted(dim_to_fields[dim_name])
                raise ValueError(
                    f"Dimension '{dim_name}' has inconsistent sizes across "
                    f"fields {field_names}: {sorted(sizes)}"
                )


@dataclass
class Map(Spec):
    """Map data and spatial extent metadata.

    Args:
        pixels: The map pixels of shape (n_frames, h, w, d) of type uint8.
        extent: The map extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """

    pixels: np.ndarray
    extent: np.ndarray

    SCHEMA = {
        "pixels": {"dtype": np.uint8, "shape": ("n_frames", "h", "w", "d")},
        "extent": {"dtype": np.float32, "shape": (("n_frames", 6), (6,))},
    }

    def __post_init__(self):
        super().__post_init__()

        # Check sensible values
        if np.any(self.extent[..., 0] >= self.extent[..., 1]):
            raise ValueError("Map extent xlims must have xmin < xmax")
        if np.any(self.extent[..., 2] >= self.extent[..., 3]):
            raise ValueError("Map extent ylims must have ymin < ymax")
        if np.any(self.extent[..., 4] >= self.extent[..., 5]):
            raise ValueError("Map extent zlims must have zmax < zmin")

        # Ultrasound specific warning: if extent values are unusually large, log a warning
        if np.any(self.extent >= 1.0) or np.any(self.extent <= -1.0):
            log.warning(
                "Map extent values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the extent values are correct and in meters."
            )


@dataclass
class Segmentation(Map):
    """Segmentation data and spatial extent metadata.

    Args:
        pixels: The segmentation pixels of shape (n_frames, h, w, d) of type uint8.
        labels: The labels corresponding to the segmentation pixels, where each unique value
            in the pixels corresponds to a label in this list of shape (n_labels,) and type str.
        extent: The segmentation extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """

    labels: np.ndarray

    SCHEMA = {
        **Map.SCHEMA,
        "labels": {"dtype": np.str_, "shape": ("n_labels",)},
    }

    def __post_init__(self):
        super().__post_init__()

        # Check every pixel value corresponds to a label
        unique_pixel_values = np.unique(self.pixels)
        if not np.all(np.isin(unique_pixel_values, np.arange(len(self.labels)))):
            raise ValueError(
                "Segmentation pixels contain values that do not correspond to any label. "
                f"Unique pixel values: {unique_pixel_values}, number of labels: {len(self.labels)}"
            )


@dataclass
class Data(Spec):
    raw_data: np.ndarray
    segmentation: Segmentation

    SCHEMA = {
        "raw_data": {"dtype": np.float32, "shape": ("n_frames", "n_tx", "n_el", "n_ax", "n_ch")},
    }


@dataclass
class Scan(Spec):
    t0_delays: np.ndarray

    SCHEMA = {
        "t0_delays": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
    }


@dataclass
class Metadata:
    pass


@dataclass
class Metrics:
    pass


# TODO: Neatly integrate this with zea.File
@dataclass
class Dataset:
    """A dataset containing all the data, scan parameters, metadata,
    and metrics for a single acquisition.

    This class can be used to create a new dataset, which is validated upon initialization.
    Afterwards, it can be saved to disk as hdf5 file.

    Args:
        data: The data for the acquisition.
        scan: The scan parameters.
        metadata: Additional metadata about the acquisition.
        metrics: Metrics computed from the acquisition.

    Example usage::

        dataset = Dataset(
            data={
                "raw_data": np.random.rand(100, 32, 64, 128, 8).astype(np.float32),
                "segmentation": {
                    "pixels": np.random.randint(0, 5, size=(100, 256, 256, 1)).astype(np.uint8),
                    "labels": np.array(["background", "tissue", "vessel", "bone", "artifact"]),
                    "extent": np.array([[-0.1, 0.1, -0.1, 0.1, -0.05, 0.05]], dtype=np.float32),
                },
            }
            scan={
                "t0_delays": np.random.rand(32, 64).astype(np.float32),
            }
        )
    """

    data: Data
    scan: Scan
    metadata: Metadata
    metrics: Metrics

    def __post_init__(self):
        if not isinstance(self.data, Data):
            self.data = Data(**self.data)
        if not isinstance(self.scan, Scan):
            self.scan = Scan(**self.scan)
        if not isinstance(self.metadata, Metadata):
            self.metadata = Metadata(**self.metadata)
        if not isinstance(self.metrics, Metrics):
            self.metrics = Metrics(**self.metrics)

    @classmethod
    def load(cls, path: str) -> "Dataset":
        """Load a dataset from the specified path."""
        pass

    def save(self, path: str) -> None:
        """Save the dataset to the specified path."""
        pass
