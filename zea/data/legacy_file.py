import numpy as np
from keras.utils import pad_sequences

from zea import log
from zea.data.spec import DataSpec


def dict_to_sorted_list(dictionary: dict):
    """Convert a dictionary with sortable keys to a sorted list of values.

    .. note::

        This function operates on the top level of the dictionary only.
        If the dictionary contains nested dictionaries, those will not be sorted.

    Example:
        .. doctest::

            >>> from zea.data.legacy_file import dict_to_sorted_list
            >>> input_dict = {"number_000": 5, "number_001": 1, "number_002": 23}
            >>> dict_to_sorted_list(input_dict)
            [5, 1, 23]

    Args:
        dictionary (dict): The dictionary to convert. The keys must be sortable.

    Returns:
        list: The sorted list of values.
    """
    return [value for _, value in sorted(dictionary.items())]


def _waveforms_dict_to_array(waveforms_dict: dict):
    """Convert waveforms stored as a dictionary to a padded numpy array."""
    waveforms = dict_to_sorted_list(waveforms_dict)
    return pad_sequences(waveforms, dtype=np.float32, padding="post")


def _reformat_waveforms(scan_kwargs: dict) -> dict:
    """Reformat waveforms from dict to array if needed. This is for backwards compatibility and will
    be removed in a future version of zea.

    Args:
        scan_kwargs (dict): The scan parameters.

    Returns:
        scan_kwargs (dict): The scan parameters with the keys waveforms_one_way and
            waveforms_two_way reformatted to arrays if they were stored as dicts.
    """

    if "waveforms_one_way" in scan_kwargs and isinstance(scan_kwargs["waveforms_one_way"], dict):
        log.warning(
            "The waveforms_one_way parameter is stored as a dictionary in the file. "
            "Converting to array. This will be deprecated in future versions of zea. "
            "Please update your files to store waveforms as arrays of shape `(n_tx, n_samples)`."
        )
        scan_kwargs["waveforms_one_way"] = _waveforms_dict_to_array(
            scan_kwargs["waveforms_one_way"]
        )

    if "waveforms_two_way" in scan_kwargs and isinstance(scan_kwargs["waveforms_two_way"], dict):
        log.warning(
            "The waveforms_two_way parameter is stored as a dictionary in the file. "
            "Converting to array. This will be deprecated in future versions of zea. "
            "Please update your files to store waveforms as arrays of shape `(n_tx, n_samples)`."
        )
        scan_kwargs["waveforms_two_way"] = _waveforms_dict_to_array(
            scan_kwargs["waveforms_two_way"]
        )
    return scan_kwargs


def check_focus_distances(scan_parameters: dict) -> dict:
    """Warn and auto-convert focus distances stored in wavelengths to metres.

    Some older files store ``focus_distances`` in wavelengths rather than
    metres.  This helper detects the pattern (values ≥ 1 and ≠ ``inf``) and
    converts them using ``sound_speed / center_frequency``.

    Args:
        scan_parameters: Raw scan parameter dict loaded from HDF5.

    Returns:
        dict: The same dict, with ``focus_distances`` converted when needed.
    """
    if "focus_distances" in scan_parameters:
        focus_distances = scan_parameters["focus_distances"]
        if np.any(np.logical_and(focus_distances >= 1, focus_distances != np.inf)):
            log.warning(
                "We have detected that focus distances are (probably) stored in "
                "wavelengths. Please update your file! "
                "Converting to metres automatically for now, but this assumes that "
                "`center_frequency` is the probe centre frequency which is not always "
                "the case!"
            )
            if "sound_speed" not in scan_parameters:
                raise ValueError(
                    "Cannot convert focus distances from wavelengths to metres "
                    "because sound_speed is not defined in the scan parameters."
                )
            if "center_frequency" not in scan_parameters:
                raise ValueError(
                    "Cannot convert focus distances from wavelengths to metres "
                    "because center_frequency is not defined in the scan parameters."
                )
            wavelength = scan_parameters["sound_speed"] / scan_parameters["center_frequency"]
            scan_parameters["focus_distances"] = focus_distances * wavelength
    return scan_parameters


def _if_exists_cast_to_float(key, parameters):
    """Cast a value to float if it exists."""
    if key in parameters:
        parameters[key] = np.float32(parameters[key])


def infer_n_tx(scan_parameters: dict):
    """Infer n_tx from n_frames and n_ax."""
    if "n_tx" in scan_parameters:
        return scan_parameters["n_tx"]
    if "t0_delays" in scan_parameters:
        return scan_parameters["t0_delays"].shape[0]
    if "focus_distances" in scan_parameters:
        return scan_parameters["focus_distances"].shape[0]
    if "polar_angles" in scan_parameters:
        return scan_parameters["polar_angles"].shape[0]
    raise ValueError("Cannot infer 'n_tx' from scan parameters. ")


def legacy_scan(scan_parameters: dict):
    """Format scan parameters for legacy file."""
    if set(scan_parameters.keys()) == {"n_ax", "n_frames", "n_tx"}:
        return {}

    scan_parameters = check_focus_distances(scan_parameters)
    scan_parameters = _reformat_waveforms(scan_parameters)

    scan_parameters.pop("probe_geometry", None)
    scan_parameters.pop("n_ax", None)
    scan_parameters.pop("n_el", None)
    n_tx = scan_parameters.pop("n_tx", None)
    scan_parameters.pop("n_ch", None)
    scan_parameters.pop("n_frames", None)
    scan_parameters.pop("bandwidth_percent", None)

    if "demodulation_frequency" not in scan_parameters:
        if "center_frequency" in scan_parameters:
            scan_parameters["demodulation_frequency"] = scan_parameters["center_frequency"]
        else:
            raise ValueError("No demodulation or center frequency found in scan parameters.")

    if "transmit_origins" not in scan_parameters:
        n_tx = infer_n_tx(scan_parameters)
        scan_parameters["transmit_origins"] = np.zeros((int(n_tx), 3), dtype=np.float32)

    for key in ["sampling_frequency", "sound_speed", "center_frequency", "demodulation_frequency"]:
        if key in scan_parameters:
            scan_parameters[key] = np.squeeze(scan_parameters[key])

    for key in [
        "sampling_frequency",
        "demodulation_frequency",
        "center_frequency",
        "initial_times",
        "transmit_origins",
        "sound_speed",
    ]:
        _if_exists_cast_to_float(key, scan_parameters)

    return scan_parameters


def legacy_probe(scan_parameters: dict):
    """Format probe parameters for legacy file."""

    probe_parameters = {}
    if "probe_geometry" in scan_parameters:
        probe_parameters["probe_geometry"] = scan_parameters["probe_geometry"]

    return probe_parameters


def legacy_data(data: dict) -> dict:
    """Format a legacy ``data`` dict for :class:`~zea.data.spec.DataSpec`.

    In old files spatial maps (``image``, ``image_sc``, ``envelope_data``, …)
    were stored as plain arrays of shape ``(n_frames, z, x)`` rather than groups
    with ``values`` + ``coordinates``.  Wrap each such array as
    ``{"values": array}`` so :class:`~zea.data.spec.DataSpec` accepts it.  The
    plain-array fields ``raw_data`` and ``aligned_data`` are left untouched.
    """
    formatted = dict(data)
    for key, value in data.items():
        if not isinstance(value, np.ndarray):
            continue
        schema_entry = DataSpec.SCHEMA.get(key)
        # raw_data / aligned_data are valid plain-array fields — leave as-is.
        if schema_entry is not None and "spec" not in schema_entry:
            continue
        log.warning(
            f"Legacy flat dataset 'data/{key}' has no spatial coordinates. "
            "The array has been loaded as 'values'; coordinates were not stored "
            "in this file and will be None."
        )
        formatted[key] = {"values": value}
    return formatted
