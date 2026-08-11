"""Base classes for the toolbox"""

import enum
import hashlib
import json
import pickle

import keras
import numpy as np

CONVERT_TO_KERAS_TYPES = (np.ndarray, int, float, list, tuple, bool)
BASE_FLOAT_PRECISION = "float32"
BASE_INT_PRECISION = "int32"
DEFAULT_DYNAMIC_RANGE = (-60, 0)


class DataTypes(enum.Enum):
    """Enum class for zea data types.

    The following terminology is used in the code when referring to different
    data types.

    raw_data        --> The raw channel data, storing the time-samples from each
                        distinct ultrasound transducer.
    aligned_data    --> Time-of-flight (TOF) corrected data. This is the data
                        that is time aligned with respect to the array geometry.
    beamformed_data --> Beamformed or also known as beamsummed data. Aligned
                        data is coherently summed together along the elements.
                        The data has now been transformed from the aperture
                        domain to the spatial domain.
    envelope_data   --> The envelope of the signal is here detected and the
                        center frequency is removed from the signal.
    image           --> After log compression of the envelope data, the
                        image is formed.
    image_sc        --> (DEPRECATED, legacy read-only) The scan converted image is
                        transformed to cartesian (x, y) format to account for possible
                        curved arrays. This data type is retained only so that legacy
                        files containing it can still be read; new files should store
                        the (polar) ``image`` together with per-pixel coordinates and
                        rely on :func:`zea.display.scan_convert` for scan conversion.
    """

    RAW_DATA = "raw_data"
    ALIGNED_DATA = "aligned_data"
    BEAMFORMED_DATA = "beamformed_data"
    ENVELOPE_DATA = "envelope_data"
    IMAGE = "image"
    IMAGE_SC = "image_sc"


class ModTypes(enum.Enum):
    """Enum class for zea modulation types."""

    RF = "rf"
    IQ = "iq"
    NONE = None


class classproperty(property):
    """Define a class level property."""

    def __get__(self, _, owner_cls):
        # ``property.fget`` is typed as optional, but a ``classproperty`` is only
        # ever constructed as a decorator, so the getter is always present.
        assert self.fget is not None
        return self.fget(owner_cls)


def _skip_to_tensor(value):
    """Check if the value should be skipped for conversion to tensor."""
    # Skip str (because JIT does not support it)
    # Skip methods and functions
    # Skip byte strings
    return isinstance(value, str) or callable(value) or isinstance(value, bytes)


def dict_to_tensor(dictionary: dict, keep_as_is: list | None = None) -> dict:
    """Convert an object to a dictionary of tensors."""
    snapshot = {}

    for key in dictionary:
        # Skip dunder/hidden methods
        if key.startswith("_"):
            continue

        # Get the value from the dictionary
        value = dictionary[key]

        if hasattr(value, "to_tensor"):
            snapshot[key] = value.to_tensor(keep_as_is=keep_as_is)
            continue

        # Skip certain types
        if _skip_to_tensor(value):
            continue

        # Convert the value to a tensor
        snapshot[key] = _to_tensor(key, value, keep_as_is=keep_as_is)

    return snapshot


def _to_tensor(key: str, val, keep_as_is: list | None = None):
    if keep_as_is is None:
        keep_as_is = []

    if key in keep_as_is:
        return val

    if not isinstance(val, CONVERT_TO_KERAS_TYPES):
        return val

    if val is None:
        return None
    # Recursively handle dicts
    if isinstance(val, dict):
        return {k: _to_tensor(k, v, keep_as_is=keep_as_is) for k, v in val.items()}
    # Use float precision for all floats (including np.float32/64)
    if isinstance(val, float) or (isinstance(val, np.ndarray) and np.issubdtype(val.dtype, float)):
        dtype = BASE_FLOAT_PRECISION
    # Use int precision for all ints (including np.int32/64)
    elif isinstance(val, bool) or (isinstance(val, np.ndarray) and np.issubdtype(val.dtype, bool)):
        dtype = bool
    elif isinstance(val, int) or (isinstance(val, np.ndarray) and np.issubdtype(val.dtype, int)):
        dtype = BASE_INT_PRECISION
    else:
        dtype = None
    return keras.ops.convert_to_tensor(val, dtype=dtype)


class ZEAEncoderJSON(json.JSONEncoder):
    """
    A custom JSONEncoder that:
      - Converts NumPy arrays to native Python types.
      - Converts zea Enums to their values
    """

    def default(self, o):
        """Convert objects to JSON serializable types."""
        obj = o
        # Convert zea Enums to their values
        if isinstance(obj, enum.Enum):
            return obj.value

        # Convert NumPy types to native Python
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


class ZEADecoderJSON(json.JSONDecoder):
    """
    A custom JSONDecoder that:
      - Converts lists into NumPy arrays.
      - Restores zea enum fields to their respective enum members.
    """

    # Create maps for quick enum lookups based on their .value
    _DATA_TYPES_MAP = {dt.value: dt for dt in DataTypes}
    _MOD_TYPES_MAP = {mt.value: mt for mt in ModTypes if mt.value is not None}

    def __init__(self, *args, **kwargs):
        # We supply our custom object_hook
        super().__init__(object_hook=self._object_hook, *args, **kwargs)

    def _object_hook(self, obj):
        """
        Called once for every JSON object (dict). We iterate through each key/value
        to see if we need to convert it into an enum or a NumPy array.
        """
        for key, value in list(obj.items()):
            # Convert lists to NumPy arrays
            if isinstance(value, list):
                # If you want a more selective approach (e.g. only numeric lists -> arrays),
                # you could check if all elements are numeric before converting.
                obj[key] = np.array(value)

            # Convert strings to DataTypes enum if it matches
            elif isinstance(value, str) and value in self._DATA_TYPES_MAP:
                obj[key] = self._DATA_TYPES_MAP[value]

            # Convert string to ModTypes enum if it matches. Also, allow None for the 'modtype' key.
            elif (key == "modtype" and value is None) or (
                isinstance(value, str) and value in self._MOD_TYPES_MAP
            ):
                obj[key] = self._MOD_TYPES_MAP[value] if value is not None else None

        return obj


def serialize_elements(key_elements: list) -> str:
    """Serialize elements of a list to a string.

    Generally, uses the pickle representation of the elements.

    Args:
        key_elements (list): List of elements to serialize. Can be nested lists
            or tuples. In this case the elements are serialized recursively.

    Returns:
        str: A serialized string representation of the elements, joined by underscores.
    """

    def _serialize(element) -> str:
        return pickle.dumps(element).hex()

    def _serialize_element(element) -> str:
        if isinstance(element, (list, tuple)):
            # If element is a list or tuple, serialize its elements recursively
            element = serialize_elements(element)
        elif hasattr(element, "serialized"):
            # Objects opt in to content-based keys by exposing a `serialized` property
            # (see :class:`zea.internal.parameters.BaseParameters`). Their own pickle
            # representation would depend on attribute insertion order and on derived
            # state such as caches.
            element = str(element.serialized)
        elif isinstance(element, keras.random.SeedGenerator):
            # If element is a SeedGenerator, use the state
            element = keras.ops.convert_to_numpy(element.state.value)
            element = _serialize(element)
        elif isinstance(element, dict):
            # If element is a dictionary, sort its keys and serialize its values recursively.
            # This is needed to ensure the internal state and ordering of the dictionary does
            # not affect the serialization.
            keys = list(sorted(element.keys()))
            values = [element[k] for k in keys]
            keys = serialize_elements(keys)
            values = serialize_elements(values)
            element = f"k_{keys}_v_{values}"
        else:
            # Otherwise, serialize the element directly
            element = _serialize(element)

        return element

    serialized_elements = []
    for element in key_elements:
        serialized_elements.append(_serialize_element(element))

    return "_".join(serialized_elements)


def hash_elements(key_elements: list) -> str:
    """Generate an MD5 hash of the elements.

    Args:
        key_elements (list): List of elements to serialize and hash.

    Returns:
        str: An MD5 hash of the serialized elements.
    """
    serialized = serialize_elements(key_elements)
    return hashlib.md5(serialized.encode()).hexdigest()
