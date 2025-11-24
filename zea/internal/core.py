"""Base classes for the toolbox"""

import enum
import hashlib
import json
import pickle
from copy import deepcopy

import keras
import numpy as np

from zea.internal.utils import reduce_to_signature
from zea.utils import update_dictionary

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
    image_sc        --> The scan converted image is transformed to cartesian
                        (x, y) format to account for possible curved arrays.

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
        return self.fget(owner_cls)


class Object:
    """Base class for all data objects in the toolbox"""

    def __init__(self):
        self._serialized = None

    @property
    def serialized(self):
        """Compute the checksum of the object only if not already done"""
        if self._serialized is None:
            attributes = self.__dict__.copy()
            attributes.pop(
                "_serialized", None
            )  # Remove the cached serialized attribute to avoid recursion
            self._serialized = serialize_elements([attributes])
        return self._serialized

    def __setattr__(self, name: str, value):
        """Reset the serialized data if the object is modified"""
        if name != "_serialized":  # Avoid resetting when setting _serialized itself
            self._serialized = None
        super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.serialized == other.serialized

    def __hash__(self):
        return hash(self.serialized)

    def copy(self):
        """Return a copied version of the object"""
        return deepcopy(self)

    def update(self, **kwargs):
        """Update the attributes of the object if they exist"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    @classmethod
    def safe_initialize(cls, **kwargs):
        """Safely initialize a class by removing any invalid arguments."""
        reduced_params = reduce_to_signature(cls.__init__, kwargs)
        return cls(**reduced_params)

    @classmethod
    def merge(cls, obj1: dict, obj2: dict, safe: bool = False):
        """Merge multiple objects and safely initialize a new object.

        Optionally can safely initialize the object, which removes any invalid
        arguments.
        """
        # TODO: support actual zea.core.Objects, now we only support dictionaries
        params = update_dictionary(obj1, obj2)
        if not safe:
            return cls(**params)
        else:
            return cls.safe_initialize(**params)

    @classmethod
    def _tree_unflatten(cls, aux, children):
        if cls is not Object:
            raise NotImplementedError(f"{cls.__name__} must implement _tree_unflatten.")
        return cls(*children)

    def _tree_flatten(self):
        if not isinstance(self, Object):
            raise NotImplementedError(f"{type(self).__name__} must implement _tree_flatten.")
        return (), ()

    @classmethod
    def register_pytree_node(cls):
        """Register the object as a PyTree node for JAX.
        https://docs.jax.dev/en/latest/_autosummary/jax.tree_util.register_pytree_node.html
        """
        try:
            from jax import tree_util
        except ImportError as exc:
            raise ImportError(
                "JAX is not installed. Please install JAX to use `register_pytree_node`."
            ) from exc

        tree_util.register_pytree_node(
            cls,
            cls._tree_flatten,
            cls._tree_unflatten,
        )


def _skip_to_tensor(value):
    """Check if the value should be skipped for conversion to tensor."""
    # Skip str (because JIT does not support it)
    # Skip methods and functions
    # Skip byte strings
    return isinstance(value, str) or callable(value) or isinstance(value, bytes)


def dict_to_tensor(dictionary, keep_as_is=None):
    """Convert an object to a dictionary of tensors."""
    snapshot = {}

    for key in dictionary:
        # Skip dunder/hidden methods
        if key.startswith("_"):
            continue

        # Get the value from the dictionary
        value = dictionary[key]

        if isinstance(value, Object) and hasattr(value, "to_tensor"):
            snapshot[key] = value.to_tensor(keep_as_is=keep_as_is)
            continue

        # Skip certain types
        if _skip_to_tensor(value):
            continue

        # Convert the value to a tensor
        snapshot[key] = _to_tensor(key, value, keep_as_is=keep_as_is)

    return snapshot


def _to_tensor(key, val, keep_as_is: list = None):
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
        return {k: _to_tensor(k, v) for k, v in val.items()}
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
        elif isinstance(element, Object) and hasattr(element, "serialized"):
            # Use the serialized attribute if it exists
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
