"""Base classes for the toolbox"""

import enum
import json
import pickle
from copy import deepcopy

import keras
import numpy as np

CONVERT_TO_KERAS_TYPES = (np.ndarray, int, float, list, tuple, bool)
BASE_PRECISION = "float32"

# TODO: make static more neat
# These are global static attributes for all ops. Ops specific
# static attributes should be defined in the respective ops class
STATIC = [
    "f_number",
    "apply_lens_correction",
    "apply_phase_rotation",
    "Nx",
    "Nz",
]


class DataTypes(enum.Enum):
    """Enum class for USBMD data types.

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
    """Enum class for USBMD modulation types."""

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

    def to_tensor(self, except_tensors=None):
        """Convert the attributes in the object to keras tensors"""
        return object_to_tensor(self, except_tensors)


def object_to_tensor(obj: Object, except_tensors=None):
    """Convert an object to a tensor"""
    snapshot = {}
    if except_tensors is None:
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


class USBMDEncoderJSON(json.JSONEncoder):
    """
    A custom JSONEncoder that:
      - Converts NumPy arrays to native Python types.
      - Converts USBMD Enums to their values
    """

    def default(self, o):
        """Convert objects to JSON serializable types."""
        obj = o
        # Convert USBMD Enums to their values
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


class USBMDDecoderJSON(json.JSONDecoder):
    """
    A custom JSONDecoder that:
      - Converts lists into NumPy arrays.
      - Restores USBMD enum fields to their respective enum members.
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
