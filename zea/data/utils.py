"""Utility functions for zea datasets."""

import json
from pathlib import Path

from keras import ops


class ZeaJSONEncoder(json.JSONEncoder):
    """Wrapper for json.dumps to encode range and slice objects.

    Example:
        >>> import json
        >>> from zea.data.utils import ZeaJSONEncoder
        >>> json.dumps(range(10), cls=ZeaJSONEncoder)
        '{"__type__": "range", "start": 0, "stop": 10, "step": 1}'

    Note:
        Probably you would use the `zea.data.dataloader.json_dumps()`
        function instead of using this class directly.
    """

    def default(self, o):
        if isinstance(o, range):
            return {
                "__type__": "range",
                "start": o.start,
                "stop": o.stop,
                "step": o.step,
            }
        if isinstance(o, slice):
            return {
                "__type__": "slice",
                "start": o.start,
                "stop": o.stop,
                "step": o.step,
            }
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def json_dumps(obj):
    """Used to serialize objects that contain range and slice objects.
    Args:
        obj: object to serialize (most likely a dictionary).
    Returns:
        str: serialized object (json string).
    """
    return json.dumps(obj, cls=ZeaJSONEncoder)


def json_loads(obj):
    """Used to deserialize objects that contain range and slice objects.
    Args:
        obj: object to deserialize (most likely a json string).
    Returns:
        object: deserialized object (dictionary).
    """
    return json.loads(obj, object_hook=_zea_datasets_json_decoder)


def decode_file_info(file_info):
    """Decode file info from a json string.
    A batch of H5Generator can return a list of file_info that are json strings.
    This function decodes the json strings and returns a list of dictionaries
    with the information, namely:
    - full_path: full path to the file
    - file_name: file name
    - indices: indices used to extract the image from the file
    """

    if file_info.ndim == 0:
        file_info = [file_info]

    decoded_info = []
    for info in file_info:
        info = ops.convert_to_numpy(info)[()].decode("utf-8")
        decoded_info.append(json_loads(info))
    return decoded_info


def _zea_datasets_json_decoder(dct):
    """Wrapper for json.loads to decode range and slice objects."""
    if "__type__" in dct:
        if dct["__type__"] == "range":
            return range(dct["start"], dct["stop"], dct["step"])
        if dct["__type__"] == "slice":
            return slice(dct["start"], dct["stop"], dct["step"])
    return dct
