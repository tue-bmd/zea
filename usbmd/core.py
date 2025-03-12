"""Base classes for the toolbox"""

import enum
import pickle
from copy import deepcopy

import keras
import numpy as np

CONVERT_TO_KERAS_TYPES = (np.ndarray, int, float, list, tuple, bool)
BASE_PRECISION = "float32"

# TODO: make static more neat
STATIC = ["f_number", "demodulation_frequency", "apply_lens_correction", "Nx", "Nz"]


class DataTypes(enum.Enum):
    """Enum class for USBMD data types."""

    RAW_DATA = "raw_data"
    ALIGNED_DATA = "aligned_data"
    BEAMFORMED_DATA = "beamformed_data"
    ENVELOPE_DATA = "envelope_data"
    IMAGE = "image"
    IMAGE_SC = "image_sc"


class ModTypes(enum.Enum):
    """Enum class for USBMD modulation types."""

    NONE = "none"
    RF = "rf"
    IQ = "iq"


class classproperty(property):
    """Define a class level property."""

    def __get__(self, _, owner_cls):
        return self.fget(owner_cls)


class Object:
    """Base class for all data objects in the toolbox"""

    def __init__(self):
        self._serialized = None
        self._except_tensors = []  # To be filled by child classes

    @property
    def serialized(self):
        """Compute the checksum of the object only if not already done"""
        if self._serialized is None:
            attributes = self.__dict__.copy()
            attributes.pop(
                "_serialized", None
            )  # Remove the cached serialized attribute to avoid recursion
            self._serialized = pickle.dumps(attributes)
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

    def to_tensor(self):
        """Convert the attributes in the object to keras tensors"""
        return object_to_tensor(self)


def object_to_tensor(obj: Object):
    """Convert an object to a tensor"""
    snapshot = {}
    if hasattr(obj, "_except_tensors"):
        except_tensors = obj._except_tensors
    else:
        except_tensors = []

    # Check if the object has static attributes, we will not convert them to tensors
    if hasattr(obj, "_static_attrs"):
        static_attrs = obj._static_attrs
    else:
        static_attrs = []

    for key in dir(obj):
        # Skip dunder/hidden methods and excepted tensors
        if key.startswith("_") or key in except_tensors:
            continue

        # Some objects have a _set_params dict that stores if parameters have
        # been (lazily) set. We don't want to convert these attributes to tensors
        # if hasattr(obj, "_set_params") and not obj._set_params.get(key, True):
        #     continue

        # Skip methods
        try:
            value = getattr(obj, key)
        except ValueError:
            continue

        if callable(value):
            continue

        # Skip byte strings
        if isinstance(value, bytes):
            continue

        if key in static_attrs or not isinstance(value, CONVERT_TO_KERAS_TYPES):
            snapshot[key] = value
            continue

        dtype = None
        # Convert double precision arrays to float32
        if isinstance(value, np.ndarray) and value.dtype == np.float64:
            dtype = BASE_PRECISION

        snapshot[key] = keras.ops.convert_to_tensor(value, dtype=dtype)
    return snapshot
