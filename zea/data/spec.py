from collections import defaultdict
from dataclasses import MISSING, dataclass, field, fields
from typing import Any, List

import h5py
import numpy as np

from zea import log

CONSISTENCY_DIMENSIONS = {"n_frames", "n_tx", "n_ax", "n_el", "n_ch"}


def check_dtype(value: Any, expected_dtype: List[type]) -> None:
    """Check if the dtype of a value matches the expected dtype,
    allowing for compatible types.

    Works for numpy arrays, numpy scalars, and Python native types.
    """
    for dt in expected_dtype:
        try:
            expected_np_dtype = np.dtype(dt)
            is_numpy_dtype = True
        except TypeError:
            is_numpy_dtype = False

        if is_numpy_dtype:
            if hasattr(value, "dtype"):
                if np.issubdtype(value.dtype, expected_np_dtype):
                    return
            elif np.issubdtype(expected_np_dtype, np.character) and isinstance(value, (str, bytes)):
                return
        else:
            if isinstance(value, dt):
                return

    actual_type = (
        f"dtype {value.dtype}" if hasattr(value, "dtype") else f"Python {type(value).__name__}"
    )
    expected_dtypes_str = ", ".join(str(dt) for dt in expected_dtype)
    raise TypeError(
        f"Expected dtype compatible with one of ({expected_dtypes_str}), got {actual_type}. "
        f"Hint: wrap the value with the appropriate numpy type, "
        f"e.g. np.float32(...), np.str_(...), np.uint8(...)."
    )


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

    @staticmethod
    def _is_optional_dataclass_field(field_def: Any) -> bool:
        if field_def is None:
            return False
        return field_def.default is not MISSING or field_def.default_factory is not MISSING

    @staticmethod
    def _expected_shapes(shape_spec: Any) -> tuple[tuple, ...]:
        if shape_spec and isinstance(shape_spec[0], tuple):
            return tuple(shape_spec)
        return (shape_spec,)

    @staticmethod
    def _merge_dimension_info(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
        nested_dim_to_fields: defaultdict[str, set[str]],
        nested_dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        for dim_name, nested_fields in nested_dim_to_fields.items():
            dim_to_fields[dim_name].update(nested_fields)
        for dim_name, nested_sizes in nested_dim_to_sizes.items():
            dim_to_sizes[dim_name].update(nested_sizes)

    @staticmethod
    def _track_named_dimensions(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
        field_path: str,
        matched_shape: tuple,
        shape: tuple,
    ) -> None:
        for i, dim_name in enumerate(matched_shape):
            if isinstance(dim_name, str) and dim_name in CONSISTENCY_DIMENSIONS:
                dim_to_fields[dim_name].add(field_path)
                dim_to_sizes[dim_name].add(shape[i])

    @staticmethod
    def _raise_if_shape_mismatch(
        field_name: str, value: Any, expected_shapes: tuple[tuple, ...]
    ) -> None:
        allowed_shapes = ", ".join(str(shape) for shape in expected_shapes)
        raise ValueError(
            f"{field_name} has shape {value_shape(value)}, expected one of: {allowed_shapes}"
        )

    def _validate_nested_field(
        self, field_name: str, nested_spec: "Spec", field_value: Any
    ) -> "Spec":
        """Validate a nested spec field, recursively validating its contents."""
        if isinstance(field_value, dict):
            field_value = nested_spec(**field_value)
            setattr(self, field_name, field_value)

        # Check that the nested spec field is now an instance of the expected Spec subclass
        # E.g. Segmentation if nested_spec is Map
        if not issubclass(type(field_value), nested_spec):
            raise TypeError(
                f"Expected field '{field_name}' to be {nested_spec}, got {type(field_value)}"
            )

        return field_value

    @staticmethod
    def _cast_native_to_numpy(value: Any, expected_dtype: list) -> Any:
        """Cast values to expected numpy dtypes when possible.

        For fields that expect a floating dtype, all floating-point inputs are
        accepted and normalized to the first floating dtype in ``expected_dtype``
        (typically ``np.float32``).
        """
        expected_np_dtypes = []
        for dt in expected_dtype:
            try:
                expected_np_dtypes.append(np.dtype(dt))
            except TypeError:
                continue

        expected_float_dtype = next(
            (dt for dt in expected_np_dtypes if np.issubdtype(dt, np.floating)),
            None,
        )

        # Keep native string/bytes values as-is instead of converting to numpy string scalars.
        if isinstance(value, (str, bytes)):
            return value

        if hasattr(value, "dtype"):
            value_dtype = np.dtype(value.dtype)

            if (
                expected_float_dtype is not None
                and np.issubdtype(value_dtype, np.floating)
                and value_dtype != expected_float_dtype
            ):
                return value.astype(expected_float_dtype, copy=False)

            return value

        for dt in expected_dtype:
            try:
                target_dtype = np.dtype(dt)
                return target_dtype.type(value)
            except (TypeError, ValueError, OverflowError):
                continue

        return value

    def _validate_and_track_primitive_field(
        self,
        field_name: str,
        field_info: dict,
        field_value: Any,
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        expected_dtype = field_info["dtype"]
        if not isinstance(expected_dtype, (list, tuple)):
            expected_dtype = [expected_dtype]
        expected_shapes = self._expected_shapes(field_info["shape"])

        # Auto-cast Python native types (str, int, float) to numpy equivalents
        field_value = self._cast_native_to_numpy(field_value, expected_dtype)
        setattr(self, field_name, field_value)

        try:
            check_dtype(field_value, expected_dtype)
        except TypeError as e:
            raise TypeError(f"Field '{field_name}' has invalid dtype: {e}")

        matched_shape = find_matched_shape(field_value, expected_shapes)
        if matched_shape is None:
            self._raise_if_shape_mismatch(field_name, field_value, expected_shapes)

        self._track_named_dimensions(
            dim_to_fields=dim_to_fields,
            dim_to_sizes=dim_to_sizes,
            field_path=field_name,
            matched_shape=matched_shape,
            shape=value_shape(field_value),
        )

    @staticmethod
    def _raise_if_inconsistent_dimensions(
        dim_to_fields: defaultdict[str, set[str]],
        dim_to_sizes: defaultdict[str, set[int]],
    ) -> None:
        for dim_name, sizes in dim_to_sizes.items():
            if len(sizes) > 1:
                field_names = sorted(dim_to_fields[dim_name])
                raise ValueError(
                    f"Dimension '{dim_name}' has inconsistent sizes across "
                    f"fields {field_names}: {sorted(sizes)}"
                )

    def _collect_dimension_info(
        self, prefix: str = ""
    ) -> tuple[defaultdict[str, set[str]], defaultdict[str, set[int]]]:
        """Collect named dimension usage and observed sizes for this spec subtree."""
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            if field_value is None:
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                nested_dim_to_fields, nested_dim_to_sizes = field_value._collect_dimension_info(
                    prefix=f"{prefix}{field_name}."
                )
                self._merge_dimension_info(
                    dim_to_fields,
                    dim_to_sizes,
                    nested_dim_to_fields,
                    nested_dim_to_sizes,
                )
                continue

            expected_shapes = self._expected_shapes(field_info["shape"])

            matched_shape = find_matched_shape(field_value, expected_shapes)
            if matched_shape is None:
                # Child specs are already validated; skip defensively if no shape can be matched.
                continue

            self._track_named_dimensions(
                dim_to_fields=dim_to_fields,
                dim_to_sizes=dim_to_sizes,
                field_path=f"{prefix}{field_name}",
                matched_shape=matched_shape,
                shape=value_shape(field_value),
            )

        return dim_to_fields, dim_to_sizes

    def __post_init__(self):
        dim_to_fields = defaultdict(set)
        dim_to_sizes = defaultdict(set)
        dataclass_fields = {f.name: f for f in fields(self)}

        for field_name, field_info in self.SCHEMA.items():
            field_value = getattr(self, field_name)
            field_def = dataclass_fields.get(field_name)
            is_optional = self._is_optional_dataclass_field(field_def)

            if field_value is None:
                if not is_optional:
                    raise ValueError(f"Missing required field '{field_name}'")
                continue

            nested_spec = field_info.get("spec")
            if nested_spec is not None:
                field_value = self._validate_nested_field(field_name, nested_spec, field_value)

                nested_dim_to_fields, nested_dim_to_sizes = field_value._collect_dimension_info(
                    prefix=f"{field_name}."
                )
                self._merge_dimension_info(
                    dim_to_fields,
                    dim_to_sizes,
                    nested_dim_to_fields,
                    nested_dim_to_sizes,
                )
                continue

            self._validate_and_track_primitive_field(
                field_name=field_name,
                field_info=field_info,
                field_value=field_value,
                dim_to_fields=dim_to_fields,
                dim_to_sizes=dim_to_sizes,
            )

        self._raise_if_inconsistent_dimensions(dim_to_fields, dim_to_sizes)

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

    def to_dict(self) -> dict[str, Any]:
        """Return this spec as a nested dictionary based on ``SCHEMA`` fields.

        Nested specs are converted recursively.
        """
        result = {}
        for field_name, field_info in self.SCHEMA.items():
            value = getattr(self, field_name)
            nested_spec = field_info.get("spec")

            if nested_spec is not None and value is not None:
                if isinstance(value, Spec):
                    result[field_name] = value.to_dict()
                elif isinstance(value, dict):
                    result[field_name] = {
                        k: v.to_dict() if isinstance(v, Spec) else v for k, v in value.items()
                    }
                else:
                    result[field_name] = value
            else:
                result[field_name] = value

        return result


@dataclass
class Map(Spec):
    """Map data and spatial extent metadata.

    Args:
        pixels: The map pixels of shape (n_frames, h, w, d) of type uint8 or float32
        extent: The map extent in meters of shape (n_frames, 6) or (6,).
            A shape of (6,) is broadcast to all frames. Values are ordered as
            (xmin, xmax, ymin, ymax, zmax, zmin) and stored as float32.
    """

    pixels: np.ndarray
    extent: np.ndarray

    SCHEMA = {
        "pixels": {"dtype": (np.uint8, np.float32), "shape": ("n_frames", "h", "w", "d")},
        "extent": {"dtype": np.float32, "shape": (("n_frames", 6), (6,))},
    }

    def __post_init__(self):
        super().__post_init__()

        # Check sensible values
        if np.any(self.extent[..., 0] > self.extent[..., 1]):
            raise ValueError("Map extent xlims must have xmin <= xmax")
        if np.any(self.extent[..., 2] > self.extent[..., 3]):
            raise ValueError("Map extent ylims must have ymin <= ymax")
        if np.any(self.extent[..., 4] > self.extent[..., 5]):
            raise ValueError("Map extent zlims must have zmax <= zmin")

        # Ultrasound specific warning: if extent values are unusually large, log a warning
        if np.any(self.extent >= 1.0) or np.any(self.extent <= -1.0):
            log.warning(
                "Map extent values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the extent values are correct and in meters."
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
class UnsignedIntMap(Map):
    """Map data with uint8 pixel values and spatial extent metadata.

    Args:
        pixels: The map pixels of shape (n_frames, h, w, d) and type uint8.
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


@dataclass
class Segmentation(UnsignedIntMap):
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
class Image(UnsignedIntMap):
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

    def __post_init__(self):
        super().__post_init__()

        # Check sensible values for speed of sound
        if np.any(self.pixels < 300):
            log.warning(
                "Speed-of-sound map contains values below 300 m/s, which is unusually low. "
                "Please verify that the speed-of-sound values are correct and in m/s."
            )


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
class ColorDopplerMap(FloatMap):
    """Color Doppler (velocity) data and spatial extent metadata.

    Args:
        pixels: The color Doppler velocity pixels in m/s of shape
            (n_frames, h, w, d) and type float32. Positive values
            indicate flow towards the transducer, negative values
            indicate flow away from the transducer.
        extent: The color Doppler extent in meters of shape (n_frames, 6) or (6,).
    """


@dataclass(init=False)
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
        color_doppler: Color Doppler velocity data and extent metadata.
        **kwargs: Any other spatially aligned map data and extent metadata.
    """

    raw_data: np.ndarray
    image: Image | dict | None = None
    segmentation: Segmentation | dict | None = None
    sos_map: SosMap | dict | None = None
    strain: StrainMap | dict | None = None
    swe: SweMap | dict | None = None
    tissue_doppler: TissueDopplerMap | dict | None = None
    color_doppler: ColorDopplerMap | dict | None = None

    SCHEMA = {
        "raw_data": {
            "dtype": (np.float32, np.int16),
            "shape": ("n_frames", "n_tx", "n_ax", "n_el", "n_ch"),
        },
        "image": {"spec": Image},
        "segmentation": {"spec": Segmentation},
        "sos_map": {"spec": SosMap},
        "strain": {"spec": StrainMap},
        "swe": {"spec": SweMap},
        "tissue_doppler": {"spec": TissueDopplerMap},
        "color_doppler": {"spec": ColorDopplerMap},
    }

    def __init__(
        self,
        raw_data: np.ndarray,
        image: Image | dict | None = None,
        segmentation: Segmentation | dict | None = None,
        sos_map: SosMap | dict | None = None,
        strain: StrainMap | dict | None = None,
        swe: SweMap | dict | None = None,
        tissue_doppler: TissueDopplerMap | dict | None = None,
        color_doppler: ColorDopplerMap | dict | None = None,
        **extra_maps,
    ):
        self.raw_data = raw_data
        self.image = image
        self.segmentation = segmentation
        self.sos_map = sos_map
        self.strain = strain
        self.swe = swe
        self.tissue_doppler = tissue_doppler
        self.color_doppler = color_doppler

        reserved_keys = set(self.SCHEMA) | set(self.__dataclass_fields__) | set(dir(Spec))
        for key, value in extra_maps.items():
            if key in reserved_keys:
                raise TypeError(f"Invalid custom data key '{key}': reserved name")
            setattr(self, key, value)

        self._extra_map_keys = tuple(extra_maps.keys())
        if getattr(self, "_extra_map_keys", ()):
            self.SCHEMA = {
                **self.SCHEMA,
                **{key: {"spec": Map} for key in self._extra_map_keys},
            }

        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()

        suggested_map_keys = ", ".join(
            sorted(
                key
                for key, value in type(self).SCHEMA.items()
                if "spec" in value and issubclass(value["spec"], Map)
            )
        )

        if getattr(self, "_extra_map_keys", ()):
            custom_keys = ", ".join(sorted(self._extra_map_keys))
            log.warning(
                "Custom keys were added to 'data' and validated as generic Map specs: "
                f"{custom_keys}. If these keys match standard categories, consider using: "
                f"{suggested_map_keys}"
            )


@dataclass
class Scan(Spec):
    """Scan group with acquisition and transmit metadata.

    All fields are aligned with the data format specification.

    Args:
        probe_geometry: The probe geometry in meters of shape (n_el, 3),
            represented as (x, y, z) coordinates.
        sampling_frequency: The sampling frequency in Hz.
        center_frequency: The center frequency in Hz of the transmit pulse.
            Single scalar if all transmits share the same center frequency;
            otherwise an array of shape (n_tx,) with one frequency per transmit.
        demodulation_frequency: The frequency in Hz at which the data should
            be demodulated. Usually the same as center_frequency, but different
            when doing harmonic imaging. Single scalar if all transmits share
            the same center frequency; otherwise an array of shape (n_tx,) with
            one frequency per transmit.
        initial_times: The times in seconds when the A/D converter starts sampling
            of shape (n_tx,). This is the time between the first element firing
            and the first recorded sample.
        t0_delays: The transmit delays in seconds for each element of shape
            (n_tx, n_el). This is the time at which each element fires, shifted
            such that the first element fires at t=0.
        tx_apodizations: The apodization values that were applied to each
            element during transmit of shape (n_tx, n_el). This is a value
            between -1 and 1 that indicates how much each element contributed
            to the transmit beam, with 0 meaning no contribution and 1 meaning
            full contribution. Negative values indicate that the element was
            fired with opposite polarity.
        focus_distances: The transmit focus distances in meters of shape (n_tx,).
            This is the distance from the origin point on the transducer to
            where the beam comes to focus. For planewaves this is set to
            infinity or zero.
        transmit_origins: The transmit origins of the transmit beams in meters of
            shape (n_tx, 3). This is the (x, y, z) position from which the beam
            is transmitted.
        polar_angles: The polar angles in radians of the transmit beams of shape (n_tx,).
        time_to_next_transmit: The time in s between subsequent transmit events
            of shape (n_frames, n_tx).
        azimuth_angles: The azimuthal angles in radians of the transmit beams of
            shape (n_tx,).
        sound_speed: The speed of sound in meters per second.
        tgc_gain_curve: The time-gain-compensation that was applied to every
            sample in the raw_data of shape (n_ax,). Divide by this curve to
            undo the TGC.
        element_width: The width of the elements in the probe in meters.
        waveforms_one_way: One-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth once.
        waveforms_two_way: Two-way waveforms of shape (n_tx, .) as simulated
            by the Verasonics system. This is the waveform after being filtered
            by the transducer bandwidth twice.
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
    time_to_next_transmit: np.ndarray = None
    azimuth_angles: np.ndarray = None
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
        "time_to_next_transmit": {"dtype": np.float32, "shape": ("n_frames", "n_tx")},
        "azimuth_angles": {"dtype": np.float32, "shape": ("n_tx",)},
        "sound_speed": {"dtype": np.float32, "shape": ()},
        "tgc_gain_curve": {"dtype": np.float32, "shape": ("n_ax",)},
        "element_width": {"dtype": np.float32, "shape": ()},
        "waveforms_one_way": {
            "dtype": np.float32,
            "shape": ("n_tx", "n_samples_one_way"),
        },
        "waveforms_two_way": {
            "dtype": np.float32,
            "shape": ("n_tx", "n_samples_two_way"),
        },
    }

    @property
    def n_tx(self) -> int:
        """Number of transmits."""
        return self.t0_delays.shape[0]

    @property
    def n_el(self) -> int:
        """Number of elements."""
        return self.t0_delays.shape[1]

    def __post_init__(self):
        super().__post_init__()

        if np.any(self.probe_geometry > 1.0) or np.any(self.probe_geometry < -1.0):
            log.warning(
                "Probe geometry values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the probe geometry values are correct and in meters."
            )
        if self.sampling_frequency <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_frequency}")
        if np.any(self.center_frequency < 0):
            raise ValueError(f"Center frequency cannot be negative, got {self.center_frequency}")
        if np.any(self.demodulation_frequency < 0):
            raise ValueError(
                f"Demodulation frequency cannot be negative, got {self.demodulation_frequency}"
            )
        if np.any(self.t0_delays < 0):
            raise ValueError(f"Transmit delays cannot be negative, got {self.t0_delays}")
        if np.any(np.logical_and(self.focus_distances >= 1, self.focus_distances != np.inf)):
            log.warning(
                "Focus distances greater than or equal to 1 meter may be unusually large. "
                "Maybe you have to convert to meters?"
            )
        if np.any(self.transmit_origins > 1.0) or np.any(self.transmit_origins < -1.0):
            log.warning(
                "Transmit origin values are unusually large, extending beyond +/- 1.0 meters. "
                "Please verify that the transmit origin values are correct and in meters."
            )
        if np.any(self.polar_angles < -np.pi) or np.any(self.polar_angles > np.pi):
            raise ValueError(
                f"Polar angles should be between -pi and pi radians, got values between "
                f"{np.min(self.polar_angles)} and {np.max(self.polar_angles)}"
            )
        if np.any(self.azimuth_angles < -np.pi) or np.any(self.azimuth_angles > np.pi):
            raise ValueError(
                f"Azimuth angles should be between -pi and pi radians, got values between "
                f"{np.min(self.azimuth_angles)} and {np.max(self.azimuth_angles)}"
            )
        if self.sound_speed is not None and self.sound_speed <= 0:
            raise ValueError(f"Sound speed must be positive, got {self.sound_speed}")
        if self.tgc_gain_curve is not None and np.any(self.tgc_gain_curve < 0):
            raise ValueError(
                f"TGC gain curve values must be non-negative, got values between "
                f"{np.min(self.tgc_gain_curve)} and {np.max(self.tgc_gain_curve)}"
            )
        if self.element_width is not None and self.element_width <= 0:
            raise ValueError(f"Element width must be positive, got {self.element_width}")

        # Try to simplify the data by squeezing out any singleton dimensions,
        # e.g. if center_frequency is an array with all the same value
        if isinstance(self.center_frequency, np.ndarray) and self.center_frequency.ndim == 1:
            if np.all(self.center_frequency == self.center_frequency[0]):
                self.center_frequency = self.center_frequency[0]
        if (
            isinstance(self.demodulation_frequency, np.ndarray)
            and self.demodulation_frequency.ndim == 1
        ):
            if np.all(self.demodulation_frequency == self.demodulation_frequency[0]):
                self.demodulation_frequency = self.demodulation_frequency[0]


@dataclass
class Subject(Spec):
    """Subject metadata associated with the study.

    Args:
        type: Subject type, e.g. human, phantom, animal.
        age: Subject age in years.
        sex: Subject sex.
        fat: Subject fat percentage.
    """

    type: str | None = None
    age: np.uint8 | None = None
    sex: str | None = None
    fat_percentage: np.float32 | None = None

    SCHEMA = {
        "type": {"dtype": str, "shape": ()},
        "age": {"dtype": np.uint8, "shape": ()},
        "sex": {"dtype": str, "shape": ()},
        "fat_percentage": {"dtype": np.float32, "shape": ()},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.fat_percentage is not None and (
            self.fat_percentage < 0 or self.fat_percentage > 100
        ):
            raise ValueError(
                f"Subject fat percentage must be between 0 and 100, got {self.fat_percentage}"
            )


@dataclass
class AdditionalSignal(Spec):
    """Additional signal related to the scan, such as voice narration or ECG.

    Args:
        offset: Time offset in seconds relative to frame timing.
        sampling_frequency: Sampling frequency in Hz for the additional signal samples.
    """

    offset: np.ndarray | float
    sampling_frequency: np.ndarray | float

    SCHEMA = {
        "offset": {"dtype": np.float32, "shape": ()},
        "sampling_frequency": {"dtype": np.float32, "shape": ()},
    }

    def __post_init__(self):
        super().__post_init__()

        if self.sampling_frequency <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.sampling_frequency}")


@dataclass
class ProbeOrientation(AdditionalSignal):
    """Probe pose and timing metadata.

    Args:
        pose: Probe pose in meters of shape (T, 6), ordered as (x, y, z, az, el, roll).
        offset: Time offset in seconds relative to frame timing.
        sampling_frequency: Sampling frequency in Hz for probe orientation samples.
    """

    pose: np.ndarray

    SCHEMA = {
        "pose": {"dtype": np.float32, "shape": ("T", 6)},
        **AdditionalSignal.SCHEMA,
    }


@dataclass
class TimedSignal(AdditionalSignal):
    """One-dimensional sampled signal with timing metadata.

    Args:
        samples: Signal samples of shape (T, 1) and type float32.
        offset: Time offset in seconds relative to frame timing.
        sampling_frequency: Sampling frequency in Hz for signal samples.
    """

    samples: np.ndarray

    SCHEMA = {
        "samples": {"dtype": np.uint8, "shape": ("T", 1)},
        **AdditionalSignal.SCHEMA,
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
    credit: str | None = None
    probe_orientation: ProbeOrientation | dict | None = None
    voice_narration: TimedSignal | dict | None = None
    ecg: TimedSignal | dict | None = None
    text_report: str | None = None
    annotations: Annotations | dict | None = None

    SCHEMA = {
        "subject": {"spec": Subject},
        "credit": {"dtype": str, "shape": ()},
        "probe_orientation": {"spec": ProbeOrientation},
        "voice_narration": {"spec": TimedSignal},
        "ecg": {"spec": TimedSignal},
        "text_report": {"dtype": str, "shape": ()},
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
class FileSpec(Spec):
    """A dataset containing all the data, scan parameters, metadata,
    and metrics for a single acquisition.

    This class can be used to create a new dataset, which is validated upon initialization.
    Afterwards, it can be saved to disk as hdf5 file.

    Args:
        data: The data for the acquisition.
        scan: The scan parameters.
        metadata: Additional metadata about the acquisition.
        metrics: Metrics computed from the acquisition.
        probe_name: The name of the probe used to acquire the data.
        us_machine: The ultrasound machine used to acquire the data.

    Example:
        .. doctest::
            >>> from zea.data.spec import FileSpec
            >>> import numpy as np

            >>> dataset = FileSpec(
            ...    data={
            ...        "raw_data": np.random.rand(100, 32, 64, 128, 1).astype(np.float32),
            ...        "segmentation": {
            ...            "pixels": np.random.randint(0, 4, size=(100, 64, 64, 1), dtype=np.uint8),
            ...            "labels": np.array(["background", "tissue", "vessel", "bone"]),
            ...            "extent": np.array([-0.1, 0.1, -0.1, 0.1, -0.1, 0.1], dtype=np.float32),
            ...        },
            ...    }
            ...    scan={
            ...        "probe_geometry": np.zeros((64, 3), dtype=np.float32),
            ...        "sampling_frequency": np.float32(30e6),
            ...        "center_frequency": np.linspace(5e6, 6e6, 32, dtype=np.float32),
            ...        "demodulation_frequency": np.linspace(5e6, 6e6, 32, dtype=np.float32),
            ...        "initial_times": np.linspace(0, 1e-6, 32, dtype=np.float32),
            ...        "t0_delays": np.random.rand(32, 64).astype(np.float32),
            ...        "tx_apodizations": np.random.rand(32, 64).astype(np.float32),
            ...        "focus_distances": np.linspace(0.01, 0.1, 32, dtype=np.float32),
            ...        "transmit_origins": np.zeros((32, 3), dtype=np.float32),
            ...        "polar_angles": np.linspace(-0.1, 0.1, 32, dtype=np.float32),
            ...    }
            ... )
    """

    data: Data | dict
    scan: Scan | dict
    metadata: Metadata | dict = field(default_factory=Metadata)
    metrics: Metrics | dict = field(default_factory=Metrics)
    probe_name: str | None = None
    us_machine: str | None = None
    description: str | None = None

    SCHEMA = {
        "data": {"spec": Data},
        "scan": {"spec": Scan},
        "metadata": {"spec": Metadata},
        "metrics": {"spec": Metrics},
        "probe_name": {"dtype": str, "shape": ()},
        "us_machine": {"dtype": str, "shape": ()},
        "description": {"dtype": str, "shape": ()},
    }

    def save(self, path: str, compression: str = "gzip") -> None:
        """Save the dataset to the specified path."""
        from zea import File

        with File(path, "w") as f:
            for group_name, schema in self.SCHEMA.items():
                if "spec" in schema:
                    group = f.create_group(group_name)
                    value: Spec = getattr(self, group_name)
                    value.store_in_group(group, compression=compression)
                else:
                    value = getattr(self, group_name)
                    if value is not None:
                        f.attrs[group_name] = value
        log.info(f"File saved to {log.yellow(path)}")
