"""Register classes
Author(s): Tristan Stevens
Date: 28/02/2023
"""
_DATASETS = {}
_DATASET_TO_PROBE_NAME = {}
_DATASET_TO_SCAN_CLASS = {}
_PROBES = {}

def register_probe(cls=None, *, name=None):
    """A decorator for registering dataset classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PROBES:
            raise ValueError(f'Already registered probe with name: {local_name}')
        _PROBES[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_dataset(cls=None, *, name=None, probe_name=None, scan_class=None):
    """A decorator for registering dataset classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _DATASETS:
            raise ValueError(f'Already registered dataset with name: {local_name}')

        # dict linking dataset names (as defined in config files) to DataSet subclasses.
        _DATASETS[local_name] = cls
        # dict linking DataSet classes to probe names.
        _DATASET_TO_PROBE_NAME[cls] = probe_name
        # dict linking DataSet classes to scan classes.
        _DATASET_TO_SCAN_CLASS[cls] = scan_class
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)
