from collections import defaultdict
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, List

import h5py
import numpy as np

from zea import File, log


def check_dtype(value: Any, expected_dtype: type) -> None:
    """Check if the dtype of a value matches the expected dtype,
    allowing for compatible types.

    Works for numpy arrays and scalar values.
    """
    try:
        expected_np_dtype = np.dtype(expected_dtype)
        is_numpy_dtype = True
    except TypeError:
        is_numpy_dtype = False

    value_dtype = value.dtype if isinstance(value, np.ndarray) else np.asarray(value).dtype

    if is_numpy_dtype:
        if not np.issubdtype(value_dtype, expected_np_dtype):
            raise TypeError(
                f"Expected dtype compatible with {expected_np_dtype}, got {value_dtype}"
            )
    else:
        if value_dtype != expected_dtype:
            raise TypeError(f"Expected type {expected_dtype}, got {value_dtype}")


def value_shape(value: Any) -> tuple:
    """Return the shape tuple for numpy arrays and scalar values."""
    if isinstance(value, np.ndarray):
        return value.shape
    return ()


def match_shape(value: Any, expected_shape: tuple) -> bool:
    """Check if the shape of a value matches the expected shape specification."""
    shape = value_shape(value)
    if len(shape) != len(expected_shape):
        return False

    for dim_size, expected_dim in zip(shape, expected_shape):
        if isinstance(expected_dim, str):
            continue
        if dim_size != expected_dim:
            return False

    return True


def find_matched_shape(value: Any, expected_shapes: List[tuple]) -> tuple | None:
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
        dataclass_fields = {f.name: f for f in fields(self)}

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            field_def = dataclass_fields.get(field_name)
            is_optional = False
            if field_def is not None:
                is_optional = (
                    field_def.default is not MISSING or field_def.default_factory is not MISSING
                )

            if field_value is None:
                if not is_optional:
                    raise ValueError(f"Missing required field '{field_name}'")
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                if isinstance(field_value, dict):
                    field_value = nested_spec(**field_value)
                    setattr(self, field_name, field_value)
                if not isinstance(field_value, nested_spec):
                    raise TypeError(
                        f"Expected field '{field_name}' to be {nested_spec.__name__}, "
                        f"got {type(field_value).__name__}"
                    )
                continue

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
                    f"{field_name} has shape {value_shape(field_value)}, "
                    f"expected one of: {allowed_shapes}"
                )

            # Track dimension names and sizes for consistency checks
            for i, dim_name in enumerate(matched_shape):
                if isinstance(dim_name, str):
                    dim_to_fields[dim_name].add(field_name)
                    dim_to_sizes[dim_name].add(value_shape(field_value)[i])

        # Check that dimensions with the same name have consistent sizes across fields
        for dim_name, sizes in dim_to_sizes.items():
            if len(sizes) > 1:
                field_names = sorted(dim_to_fields[dim_name])
                raise ValueError(
                    f"Dimension '{dim_name}' has inconsistent sizes across "
                    f"fields {field_names}: {sorted(sizes)}"
                )

    @staticmethod
    def _is_string_value(value: Any) -> bool:
        """Return True for scalar/array values that should be stored as HDF5 strings."""
        if isinstance(value, (str, np.str_, bytes, np.bytes_)):
            return True

        if isinstance(value, np.ndarray):
            return value.dtype.kind in {"U", "S", "O"}

        return False

    @staticmethod
    def create_dataset(
        group: h5py.Group, field_name: str, value: Any, compression: str = "gzip"
    ) -> None:
        """Create a dataset in the given group for the specified field and value,
        handling string and scalar values appropriately."""
        dataset_is_scalar = np.isscalar(value) or value.ndim == 0
        compression = None if dataset_is_scalar else compression
        if Spec._is_string_value(value):
            string_dtype = h5py.string_dtype(encoding="utf-8")
            string_value = np.asarray(value, dtype=object)
            group.create_dataset(
                field_name,
                data=string_value,
                dtype=string_dtype,
                compression=compression,
            )
        else:
            group.create_dataset(field_name, data=value, compression=compression)

    def store_in_group(self, group: h5py.Group, compression: str = "gzip") -> None:
        """Store the data in the given group (e.g. hdf5 group)."""

        assert isinstance(group, h5py.Group), "group must be an h5py Group"

        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name)
            if value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                subgroup = group.create_group(field_name)
                value.store_in_group(subgroup)
            else:
                # TODO: store description and unit as h5 attrs (like zea does)
                self.create_dataset(group, field_name, value, compression=compression)


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
class FloatMap(Map):
    """Map data with float32 pixel values and spatial extent metadata.

    Args:
        pixels: The map pixels of shape (n_frames, h, w, d) and type float32.
        extent: The map extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """

    pixels: np.ndarray
    extent: np.ndarray

    SCHEMA = {
        "pixels": {"dtype": np.float32, "shape": ("n_frames", "h", "w", "d")},
        "extent": {"dtype": np.float32, "shape": (("n_frames", 6), (6,))},
    }


@dataclass
class Image(Map):
    """Reconstructed (log-compressed) image data and spatial extent metadata.

    Args:
        pixels: The image pixels of shape (n_frames, h, w, d) and type uint8.
        extent: The image extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """


@dataclass
class SosMap(FloatMap):
    """Speed-of-sound map data and spatial extent metadata.

    Args:
        pixels: The speed-of-sound map pixels in m/s of shape (n_frames, h, w, d)
            and type float32.
        extent: The speed-of-sound map extent in meters of shape (n_frames, 6) or (6,).
    """


@dataclass
class StrainMap(FloatMap):
    """Strain map data and spatial extent metadata.

    Args:
        pixels: The strain pixels in % of shape (n_frames, h, w, d) and type float32.
        extent: The strain extent in meters of shape (n_frames, 6) or (6,).
    """


@dataclass
class SweMap(FloatMap):
    """Shear-wave elastography data and spatial extent metadata.

    Args:
        pixels: The shear-wave elastography pixels in m/s of shape
            (n_frames, h, w, d) and type float32.
        extent: The SWE extent in meters of shape (n_frames, 6) or (6,).
    """


@dataclass
class TissueDopplerMap(FloatMap):
    """Tissue Doppler data and spatial extent metadata.

    Args:
        pixels: The tissue Doppler pixels in m/s of shape (n_frames, h, w, d)
            and type float32.
        extent: The tissue Doppler extent in meters of shape (n_frames, 6) or (6,).
    """


@dataclass
class Data(Spec):
    """Data group containing raw channels and optional derived data products.

    Args:
        raw_data: Raw channel data of shape (n_frames, n_tx, n_el, n_ax, n_ch)
            and type float32.
        image: Reconstructed image data and extent metadata.
        segmentation: Segmentation data and extent metadata.
        sos_map: Speed-of-sound map data and extent metadata.
        strain: Strain map data and extent metadata.
        swe: Shear-wave elastography data and extent metadata.
        tissue_doppler: Tissue Doppler data and extent metadata.
    """

    raw_data: np.ndarray | None = None
    image: Image | dict | None = None
    segmentation: Segmentation | dict | None = None
    sos_map: SosMap | dict | None = None
    strain: StrainMap | dict | None = None
    swe: SweMap | dict | None = None
    tissue_doppler: TissueDopplerMap | dict | None = None

    SCHEMA = {
        "raw_data": {
            "dtype": np.float32,
            "shape": ("n_frames", "n_tx", "n_el", "n_ax", "n_ch"),
        },
        "image": {"spec": Image},
        "segmentation": {"spec": Segmentation},
        "sos_map": {"spec": SosMap},
        "strain": {"spec": StrainMap},
        "swe": {"spec": SweMap},
        "tissue_doppler": {"spec": TissueDopplerMap},
    }


@dataclass
class Scan(Spec):
    """Scan group with acquisition and transmit metadata.

    All fields are aligned with the data format specification.
    """

    probe_geometry: np.ndarray
    sampling_frequency: np.ndarray | float
    center_frequency: np.ndarray | float
    demodulation_frequency: np.ndarray | float
    initial_times: np.ndarray
    t0_delays: np.ndarray
    tx_apodizations: np.ndarray
    focus_distances: np.ndarray
    transmit_origins: np.ndarray
    polar_angles: np.ndarray
    azimuth_angles: np.ndarray
    time_to_next_transmit: np.ndarray
    us_machine: np.ndarray | str | None = None
    probe_name: np.ndarray | str | None = None
    sound_speed: np.ndarray | float | None = None
    tgc_gain_curve: np.ndarray | None = None
    element_width: np.ndarray | float | None = None
    waveforms_one_way: np.ndarray | None = None
    waveforms_two_way: np.ndarray | None = None

    SCHEMA = {
        "probe_geometry": {"dtype": np.float32, "shape": ("n_el", 3)},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
        "center_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "demodulation_frequency": {"dtype": np.float32, "shape": ((), ("n_tx",))},
        "initial_times": {"dtype": np.float32, "shape": ("n_tx",)},
        "t0_delays": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "tx_apodizations": {"dtype": np.float32, "shape": ("n_tx", "n_el")},
        "focus_distances": {"dtype": np.float32, "shape": ("n_tx",)},
        "transmit_origins": {"dtype": np.float32, "shape": ("n_tx", 3)},
        "polar_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "azimuth_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "time_to_next_transmit": {"dtype": np.float32, "shape": ("n_frames", "n_tx")},
        "us_machine": {"dtype": np.str_, "shape": ()},
        "probe_name": {"dtype": np.str_, "shape": ()},
        "sound_speed": {"dtype": float, "shape": ()},
        "tgc_gain_curve": {"dtype": np.float32, "shape": ("n_ax",)},
        "element_width": {"dtype": np.float32, "shape": ()},
        "waveforms_one_way": {"dtype": np.float32, "shape": ("n_tx", 500)},
        "waveforms_two_way": {"dtype": np.float32, "shape": ("n_tx", 500)},
    }


@dataclass
class Subject(Spec):
    """Subject metadata associated with the study.

    Args:
        type: Subject type, e.g. human, phantom, animal.
        age: Subject age in years.
        sex: Subject sex.
        fat: Subject fat percentage.
    """

    type: np.ndarray | str | None = None
    age: np.ndarray | int | None = None
    sex: np.ndarray | str | None = None
    fat: np.ndarray | float | None = None

    SCHEMA = {
        "type": {"dtype": np.str_, "shape": ()},
        "age": {"dtype": np.uint8, "shape": ()},
        "sex": {"dtype": np.str_, "shape": ()},
        "fat": {"dtype": np.float32, "shape": ()},
    }


@dataclass
class ProbeOrientation(Spec):
    """Probe pose and timing metadata.

    Args:
        pose: Probe pose in meters of shape (T, 6), ordered as
            (x, y, z, az, el, roll).
        offset: Time offset in seconds relative to frame timing.
        sampling_frequency: Sampling frequency in Hz for probe orientation samples.
    """

    pose: np.ndarray
    offset: np.ndarray | float | None = None
    sampling_frequency: np.ndarray | float | None = None

    SCHEMA = {
        "pose": {"dtype": np.float32, "shape": ("T", 6)},
        "offset": {"dtype": np.float32, "shape": ()},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
    }


@dataclass
class TimedSignal(Spec):
    """One-dimensional sampled signal with timing metadata.

    Args:
        samples: Signal samples of shape (T, 1) and type uint8.
        offset: Time offset in seconds relative to frame timing.
        sampling_frequency: Sampling frequency in Hz for signal samples.
    """

    samples: np.ndarray
    offset: np.ndarray | float | None = None
    sampling_frequency: np.ndarray | float | None = None

    SCHEMA = {
        "samples": {"dtype": np.uint8, "shape": ("T", 1)},
        "offset": {"dtype": np.float32, "shape": ()},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
    }


@dataclass
class Annotations(Spec):
    """Frame-level annotations, either per frame or broadcast labels.

    Args:
        anatomy: Anatomy label.
        view: View label of shape (n_frames,).
        label: Pathology or classification label of shape (n_frames,).
        image_quality: Image quality label, e.g. low, mid, high.
    """

    anatomy: np.ndarray | str | None = None
    view: np.ndarray | None = None
    label: np.ndarray | None = None
    image_quality: np.ndarray | str | None = None

    SCHEMA = {
        "anatomy": {"dtype": np.str_, "shape": (("n_frames",), ())},
        "view": {"dtype": np.str_, "shape": ("n_frames",)},
        "label": {"dtype": np.str_, "shape": ("n_frames",)},
        "image_quality": {"dtype": np.str_, "shape": (("n_frames",), ())},
    }


@dataclass
class Metadata(Spec):
    """Metadata group with subject, acquisition context, and annotations."""

    subject: Subject | dict | None = None
    credit: np.ndarray | str | None = None
    probe_orientation: ProbeOrientation | dict | None = None
    voice_narration: TimedSignal | dict | None = None
    ecg: TimedSignal | dict | None = None
    text_report: np.ndarray | str | None = None
    annotations: Annotations | dict | None = None

    SCHEMA = {
        "subject": {"spec": Subject},
        "credit": {"dtype": np.str_, "shape": ()},
        "probe_orientation": {"spec": ProbeOrientation},
        "voice_narration": {"spec": TimedSignal},
        "ecg": {"spec": TimedSignal},
        "text_report": {"dtype": np.str_, "shape": ()},
        "annotations": {"spec": Annotations},
    }


@dataclass
class Metrics(Spec):
    """Metrics group for acquisition-level quality/performance metrics.

    Args:
        common_midpoint_phase_error: Common midpoint phase error in radians of
            shape (n_frames,) and type float32.
        coherence_factor: Coherence factor of shape (n_frames,) and type float32.
    """

    common_midpoint_phase_error: np.ndarray | None = None
    coherence_factor: np.ndarray | None = None

    SCHEMA = {
        "common_midpoint_phase_error": {
            "dtype": np.float32,
            "shape": ("n_frames",),
        },
        "coherence_factor": {"dtype": np.float32, "shape": ("n_frames",)},
    }


@dataclass
class DatasetBuilder(Spec):
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

    data: Data | dict
    scan: Scan | dict
    metadata: Metadata | dict = field(default_factory=Metadata)
    metrics: Metrics | dict = field(default_factory=Metrics)

    SCHEMA = {
        "data": {"spec": Data},
        "scan": {"spec": Scan},
        "metadata": {"spec": Metadata},
        "metrics": {"spec": Metrics},
    }

    def save(self, path: str) -> None:
        """Save the dataset to the specified path."""
        with File(path, "w") as f:
            for group_name in self.SCHEMA.keys():
                # Create group
                group = f.create_group(group_name)

                value: Spec = getattr(self, group_name)
                value.store_in_group(group)
