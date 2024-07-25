"""
Functions to write and validate datasets in the USBMD format.
"""

import inspect
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from usbmd.data.read_h5 import recursively_load_dict_contents_from_group
from usbmd.probes import get_probe
from usbmd.scan import Scan, cast_scan_parameters
from usbmd.utils import first_not_none_item, log, update_dictionary
from usbmd.utils.checks import _DATA_TYPES, validate_dataset


@dataclass
class DatasetElement:
    """Class to store a dataset element with a name, data, description and unit. Used to
    supply additional dataset elements to the generate_usbmd_dataset function."""

    # The group name to store the dataset under. This can be a nested group, e.g.
    # "scan/waveforms"
    group_name: str
    # The name of the dataset. This will be the key in the group.
    dataset_name: str
    # The data to store in the dataset.
    data: np.ndarray
    description: str
    unit: str


def generate_example_dataset(path, add_optional_fields=False):
    """Generates an example dataset that contains all the necessary fields.
    Note: This dataset does not contain actual data, but is filled with random
    values.

    Args:
        path (str): The path to write the dataset to.
        add_optional_fields (bool, optional): Whether to add optional fields to
            the dataset. Defaults to False.

    Returns:
        (h5py.File): The example dataset.
    """

    n_ax = 2048
    n_el = 128
    n_tx = 11
    n_ch = 1
    n_frames = 2
    sound_speed = 1540
    center_frequency = 7e6
    sampling_frequency = 40e6

    # creating some fake raw and image data
    raw_data = np.ones((n_frames, n_tx, n_ax, n_el, n_ch))
    # image data is in dB
    image = np.ones((n_frames, 512, 512)) * -40

    # creating some fake scan parameters
    t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
    tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)
    initial_times = np.zeros((n_tx,), dtype=np.float32)

    if add_optional_fields:
        focus_distances = np.ones((n_tx,), dtype=np.float32) * np.inf
        tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)
        polar_angles = np.zeros((n_tx,), dtype=np.float32)
        azimuth_angles = np.zeros((n_tx,), dtype=np.float32)
    else:
        focus_distances = None
        tx_apodizations = None
        polar_angles = None
        azimuth_angles = None

    generate_usbmd_dataset(
        path,
        raw_data=raw_data,
        image=image,
        probe_geometry=probe_geometry,
        sampling_frequency=sampling_frequency,
        center_frequency=center_frequency,
        initial_times=initial_times,
        t0_delays=t0_delays,
        sound_speed=sound_speed,
        tx_apodizations=tx_apodizations,
        probe_name="generic",
        focus_distances=focus_distances,
        polar_angles=polar_angles,
        azimuth_angles=azimuth_angles,
    )


def validate_input_data(
    raw_data, aligned_data, envelope_data, beamformed_data, image, image_sc
):
    """
    Validates input data for generate_usbmd_dataset

    Args:
        raw_data (np.ndarray): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        aligned_data (np.ndarray): The aligned data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        envelope_data (np.ndarray): The envelope data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        beamformed_data (np.ndarray): The beamformed data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        image (np.ndarray): The ultrasound images to be saved of shape (n_frames, n_z, n_x).
        image_sc (np.ndarray): The scan converted ultrasound images to be saved
            of shape (n_frames, output_size_z, output_size_x).
    """
    assert (
        raw_data is not None
        or aligned_data is not None
        or envelope_data is not None
        or beamformed_data is not None
        or image is not None
        or image_sc is not None
    ), f"At least one of the data types {_DATA_TYPES} must be specified."

    # specific checks for each data type are done in validate_dataset


def _write_datasets(
    dataset,
    data_group_name="data",
    scan_group_name="scan",
    raw_data=None,
    aligned_data=None,
    envelope_data=None,
    beamformed_data=None,
    image=None,
    image_sc=None,
    n_ax=None,
    n_el=None,
    n_tx=None,
    n_ch=None,
    n_frames=None,
    sound_speed=None,
    probe_geometry=None,
    sampling_frequency=None,
    center_frequency=None,
    initial_times=None,
    t0_delays=None,
    tx_apodizations=None,
    focus_distances=None,
    polar_angles=None,
    azimuth_angles=None,
    bandwidth_percent=None,
    time_to_next_transmit=None,
    tgc_gain_curve=None,
    element_width=None,
    tx_waveform_indices=None,
    waveforms_one_way=None,
    waveforms_two_way=None,
    additional_elements=None,
    **kwargs,
):
    # weird pylint work around
    if kwargs:
        raise ValueError(f"Unknown arguments: {list(kwargs.keys())}")

    def _convert_datatype(x, astype=np.float32):
        return x.astype(astype) if x is not None else None

    def _first_not_none_shape(arr, axis):
        data = first_not_none_item(arr)
        return data.shape[axis] if data is not None else None

    def _add_dataset(
        group_name: str, name: str, data: np.ndarray, description: str, unit: str
    ):
        """Adds a dataset to the given group with a description and unit.
        If data is None, the dataset is not added."""
        if data is None:
            return

        # Create the group if it does not exist
        if group_name not in dataset:
            group = dataset.create_group(group_name)
        else:
            group = dataset[group_name]

        new_dataset = group.create_dataset(name, data=data)
        new_dataset.attrs["description"] = description
        new_dataset.attrs["unit"] = unit

    # Write data group
    data_group = dataset.create_group(data_group_name)
    data_group.attrs["description"] = "This group contains the data."

    if n_frames is None:
        n_frames = first_not_none_item(
            [raw_data, aligned_data, envelope_data, beamformed_data, image, image_sc]
        ).shape[0]
    if n_tx is None:
        n_tx = _first_not_none_shape([raw_data, aligned_data], axis=1)
    if n_ax is None:
        n_ax = _first_not_none_shape([raw_data, aligned_data, beamformed_data], axis=-3)
    if n_ax is None:
        n_ax = _first_not_none_shape([envelope_data, image, image_sc], axis=-2)
    if n_el is None:
        n_el = _first_not_none_shape([raw_data], axis=-2)
    if n_ch is None:
        n_ch = _first_not_none_shape([raw_data, aligned_data, beamformed_data], axis=-1)
    if n_tx is None:
        n_tx = _first_not_none_shape([t0_delays, focus_distances, polar_angles], axis=0)
    if n_el is None:
        n_el = _first_not_none_shape([t0_delays], axis=1)
    if n_el is None:
        n_el = _first_not_none_shape([probe_geometry], axis=0)
    if n_tx is None:
        n_tx = 1

    _add_dataset(
        group_name=data_group_name,
        name="raw_data",
        data=_convert_datatype(raw_data),
        description="The raw_data of shape (n_frames, n_tx, n_ax, n_el, n_ch).",
        unit="unitless",
    )

    _add_dataset(
        group_name=data_group_name,
        name="aligned_data",
        data=_convert_datatype(aligned_data),
        description="The aligned_data of shape (n_frames, n_tx, n_ax, n_el, n_ch).",
        unit="unitless",
    )

    _add_dataset(
        group_name=data_group_name,
        name="envelope_data",
        data=_convert_datatype(envelope_data),
        description="The envelope_data of shape (n_frames, n_z, n_x).",
        unit="unitless",
    )

    _add_dataset(
        group_name=data_group_name,
        name="beamformed_data",
        data=_convert_datatype(beamformed_data),
        description="The beamformed_data of shape (n_frames, n_z, n_x).",
        unit="unitless",
    )

    _add_dataset(
        group_name=data_group_name,
        name="image",
        data=_convert_datatype(image),
        unit="unitless",
        description="The images of shape [n_frames, n_z, n_x]",
    )

    _add_dataset(
        group_name=data_group_name,
        name="image_sc",
        data=_convert_datatype(image_sc),
        unit="unitless",
        description=(
            "The scan converted images of shape [n_frames, output_size_z,"
            " output_size_x]"
        ),
    )

    # Write scan group
    scan_group = dataset.create_group(scan_group_name)
    scan_group.attrs["description"] = "This group contains the scan parameters."

    _add_dataset(
        group_name=scan_group_name,
        name="n_ax",
        data=n_ax,
        description="The number of axial samples.",
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="n_el",
        data=n_el,
        description="The number of elements in the probe.",
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="n_tx",
        data=n_tx,
        description="The number of transmits per frame.",
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="n_ch",
        data=n_ch,
        description=(
            "The number of channels. For RF data this is 1. For IQ data " "this is 2."
        ),
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="n_frames",
        data=n_frames,
        description="The number of frames.",
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="sound_speed",
        data=sound_speed,
        description="The speed of sound in m/s",
        unit="m/s",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="probe_geometry",
        data=probe_geometry,
        description="The probe geometry of shape (n_el, 3).",
        unit="m",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="sampling_frequency",
        data=sampling_frequency,
        description="The sampling frequency in Hz.",
        unit="Hz",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="center_frequency",
        data=center_frequency,
        description="The center frequency in Hz.",
        unit="Hz",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="initial_times",
        data=initial_times,
        description=(
            "The times when the A/D converter starts sampling "
            "in seconds of shape (n_tx,). This is the time between the "
            "first element firing and the first recorded sample."
        ),
        unit="s",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="t0_delays",
        data=t0_delays,
        description="The t0_delays of shape (n_tx, n_el).",
        unit="s",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="tx_apodizations",
        data=tx_apodizations,
        description=(
            "The transmit delays for each element defining the"
            " wavefront in seconds of shape (n_tx, n_elem). This is"
            " the time at which each element fires shifted such that"
            " the first element fires at t=0."
        ),
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="focus_distances",
        data=focus_distances,
        description=(
            "The transmit focus distances in meters of "
            "shape (n_tx,). For planewaves this is set to Inf."
        ),
        unit="m",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="polar_angles",
        data=polar_angles,
        description=(
            "The polar angles of the transmit beams in radians of shape (n_tx,)."
        ),
        unit="rad",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="azimuth_angles",
        data=azimuth_angles,
        description=(
            "The azimuthal angles of the transmit beams in radians of shape (n_tx,)."
        ),
        unit="rad",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="bandwidth_percent",
        data=bandwidth_percent,
        description=(
            "The receive bandwidth of RF signal in percentage of center frequency."
        ),
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="time_to_next_transmit",
        data=time_to_next_transmit,
        description=(
            "The time between subsequent transmit events of shape " "(n_frames, n_tx)."
        ),
        unit="s",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="tgc_gain_curve",
        data=tgc_gain_curve,
        description=(
            "The time-gain-compensation that was applied to every sample in the "
            "raw_data of shape (n_ax,). Divide by this curve to undo the TGC."
        ),
        unit="unitless",
    )

    _add_dataset(
        group_name=scan_group_name,
        name="element_width",
        data=element_width,
        description="The width of the elements in the probe in meters.",
        unit="m",
    )

    if tx_waveform_indices is not None and (
        waveforms_one_way is not None or waveforms_two_way is not None
    ):
        _add_dataset(
            group_name=scan_group_name,
            name="tx_waveform_indices",
            data=tx_waveform_indices,
            description=(
                "Transmit indices for waveforms, indexing waveforms_one_way "
                "and waveforms_two_way. This indicates which transmit waveform was "
                "used for each transmit event."
            ),
            unit="-",
        )
        n_waveforms = len(waveforms_one_way)
        for n, waveform_1way, waveform_2way in zip(
            range(n_waveforms), waveforms_one_way, waveforms_two_way
        ):
            _add_dataset(
                group_name=scan_group_name + "/waveforms_one_way",
                name=f"waveform_{str(n).zfill(3)}",
                data=waveform_1way,
                description=(
                    "One-way waveform as simulated by the Verasonics system, "
                    "sampled at 250MHz. This is the waveform after being filtered "
                    "by the tranducer bandwidth once."
                ),
                unit="V",
            )
            _add_dataset(
                group_name=scan_group_name + "/waveforms_two_way",
                name=f"waveform_{str(n).zfill(3)}",
                data=waveform_2way,
                description=(
                    "Two-way waveform as simulated by the Verasonics system, "
                    "sampled at 250MHz. This is the waveform after being filtered "
                    "by the tranducer bandwidth twice."
                ),
                unit="V",
            )

    # Add additional elements
    if additional_elements is not None:
        for element in additional_elements:
            _add_dataset(
                group_name=element.group_name,
                name=element.dataset_name,
                data=element.data,
                description=element.description,
                unit=element.unit,
            )


def generate_usbmd_dataset(
    path,
    raw_data=None,
    aligned_data=None,
    envelope_data=None,
    beamformed_data=None,
    image=None,
    image_sc=None,
    probe_geometry=None,
    sampling_frequency=None,
    center_frequency=None,
    initial_times=None,
    t0_delays=None,
    sound_speed=None,
    probe_name=None,
    description="No description was supplied",
    focus_distances=None,
    polar_angles=None,
    azimuth_angles=None,
    tx_apodizations=None,
    bandwidth_percent=None,
    time_to_next_transmit=None,
    tgc_gain_curve=None,
    element_width=None,
    tx_waveform_indices=None,
    waveforms_one_way=None,
    waveforms_two_way=None,
    additional_elements=None,
    event_structure=False,
):
    """Generates a dataset in the USBMD format.

    Args:
        path (str): The path to write the dataset to.
        raw_data (np.ndarray): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        aligned_data (np.ndarray): The aligned data of the ultrasound measurement of
            shape (n_frames, n_tx, n_ax, n_el, n_ch).
        envelope_data (np.ndarray): The envelope data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        beamformed_data (np.ndarray): The beamformed data of the ultrasound measurement of
            shape (n_frames, n_z, n_x).
        image (np.ndarray): The ultrasound images to be saved of shape (n_frames, n_z, n_x).
        image_sc (np.ndarray): The scan converted ultrasound images to be saved
            of shape (n_frames, output_size_z, output_size_x).
        probe_geometry (np.ndarray): The probe geometry of shape (n_el, 3).
        sampling_frequency (float): The sampling frequency in Hz.
        center_frequency (float): The center frequency in Hz.
        initial_times (list): The times when the A/D converter starts sampling
            in seconds of shape (n_tx,). This is the time between the first element
            firing and the first recorded sample.
        t0_delays (np.ndarray): The t0_delays of shape (n_tx, n_el).
        sound_speed (float): The speed of sound in m/s.
        probe_name (str): The name of the probe.
        description (str): The description of the dataset.
        focus_distances (np.ndarray): The focus distances of shape (n_tx, n_el).
        polar_angles (np.ndarray): The polar angles (radians) of shape (n_el,).
        azimuth_angles (np.ndarray): The azimuth angles (radians) of shape (n_tx,).
        tx_apodizations (np.ndarray): The transmit delays for each element defining
            the wavefront in seconds of shape (n_tx, n_elem).
            This is the time between the first element firing and the last element firing.
        bandwidth_percent (float): The bandwidth of the transducer as a
            percentage of the center frequency.
        time_to_next_transmit (np.ndarray): The time between subsequent transmit events in s
            of shape (n_frames, n_tx).
        tgc_gain_curve (np.ndarray): The TGC gain that was applied to every sample in the
            raw_data of shape (n_ax).
        element_width (float): The width of the elements in the probe in meters of
            shape (n_tx,).
        tx_waveform_indices (np.ndarray): Transmit indices for waveforms, indexing
            waveforms_one_way and waveforms_two_way. This indicates which transmit
            waveform was used for each transmit event.
        waveforms_one_way (list): List of one-way waveforms as simulated by the Verasonics
            system, sampled at 250MHz. This is the waveform after being filtered by the
            tranducer bandwidth once. Every element in the list is a 1D numpy array.
        waveforms_two_way (list): List of two-way waveforms as simulated by the Verasonics
            system, sampled at 250MHz. This is the waveform after being filtered by the
            tranducer bandwidth twice. Every element in the list is a 1D numpy array.
        additional_elements (List[DatasetElement]): A list of additional dataset
            elements to be added to the dataset. Each element should be a DatasetElement
            object. The additional elements are added under the scan group.
        event_structure (bool): Whether to write the dataset with an event structure.
            In that case all data should be lists with the same length (number of events).
            The data will be stored under event_i/data and event_i/scan for each event i.
            Instead of just a single data and scan group.

    Returns:
        (h5py.File): The example dataset.
    """
    # check if all args are lists
    if isinstance(probe_name, list):
        # all names in probe_name list should be the same
        assert (
            len(set(probe_name)) == 1
        ), "Probe names for all events should be the same"

    data_and_parameters = {
        "raw_data": raw_data,
        "aligned_data": aligned_data,
        "envelope_data": envelope_data,
        "beamformed_data": beamformed_data,
        "image": image,
        "image_sc": image_sc,
        "probe_geometry": probe_geometry,
        "sampling_frequency": sampling_frequency,
        "center_frequency": center_frequency,
        "initial_times": initial_times,
        "t0_delays": t0_delays,
        "sound_speed": sound_speed,
        "probe_name": probe_name,
        "description": description,
        "focus_distances": focus_distances,
        "polar_angles": polar_angles,
        "azimuth_angles": azimuth_angles,
        "tx_apodizations": tx_apodizations,
        "bandwidth_percent": bandwidth_percent,
        "time_to_next_transmit": time_to_next_transmit,
        "tgc_gain_curve": tgc_gain_curve,
        "element_width": element_width,
        "tx_waveform_indices": tx_waveform_indices,
        "waveforms_one_way": waveforms_one_way,
        "waveforms_two_way": waveforms_two_way,
        "additional_elements": additional_elements,
    }

    # make sure input arguments of func is same length as data_and_parameters
    # except `path` and `event_structure` arguments and ofcourse `data_and_parameters` itself
    assert (
        len(data_and_parameters)
        == len(inspect.signature(generate_usbmd_dataset).parameters) - 2
    ), (
        "All arguments should be put in data_and_parameters except "
        "`path` and `event_structure` arguments."
    )

    if event_structure:
        for argument, argument_value in data_and_parameters.items():
            _num_events = None
            if argument_value is not None:
                assert isinstance(
                    argument_value, list
                ), f"{argument} should be a list when event_structure is set to True."
                num_events = len(argument_value)
                if _num_events is not None:
                    assert (
                        num_events == _num_events
                    ), "All arguments should have the same number of events."
                _num_events = num_events

        assert (
            len(set(probe_name)) == 1
        ), "Probe names for all events should be the same"
        log.info(
            f"Event structure is set to True. Writing dataset with event "
            f"structure (found {len(probe_name)} events)."
        )
        num_events = len(probe_name)
        probe_name = probe_name[0]
        description = description[0]

    assert isinstance(probe_name, str), "The probe name must be a string."
    assert isinstance(description, str), "The description must be a string."
    assert isinstance(event_structure, bool), "The event_structure must be a boolean."

    validate_input_data(
        raw_data=raw_data,
        aligned_data=aligned_data,
        envelope_data=envelope_data,
        beamformed_data=beamformed_data,
        image=image,
        image_sc=image_sc,
    )

    # Convert path to Path object
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"The file {path} already exists.")

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as dataset:
        dataset.attrs["probe"] = probe_name
        dataset.attrs["description"] = description
        dataset.attrs["event_structure"] = event_structure
        # remove probe and description from data_and_parameters
        data_and_parameters.pop("probe_name")
        data_and_parameters.pop("description")

        if event_structure:
            for i in range(num_events):
                _data_and_parameters = {
                    k: v[i] for k, v in data_and_parameters.items() if v is not None
                }
                _write_datasets(
                    dataset,
                    data_group_name=f"event_{i}/data",
                    scan_group_name=f"event_{i}/scan",
                    **_data_and_parameters,
                )

        else:
            _write_datasets(
                dataset,
                data_group_name="data",
                scan_group_name="scan",
                **data_and_parameters,
            )

    validate_dataset(path)
    log.info(f"USBMD dataset written to {log.yellow(path)}")


def load_usbmd_file(
    path, frames=None, transmits=None, data_type="raw_data", config=None
):
    """Loads a hdf5 file in the USBMD format and returns the data together with
    a scan object containing the parameters of the acquisition and a probe
    object containing the parameters of the probe.

    Args:
        path (str, pathlike): The path to the hdf5 file.
        frames (tuple, list, optional): The frames to load. Defaults to None in
            which case all frames are loaded.
        transmits (tuple, list, optional): The transmits to load. Defaults to
            None in which case all transmits are used.
        data_type (str, optional): The type of data to load. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        config (utils.config.Config, optional): A config object containing parameters.
            This function only uses parameters from config.scan.

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_ax, n_el, n_ch).
        (Scan): A scan object containing the parameters of the acquisition.
        (Probe): A probe object containing the parameters of the probe.
    """

    assert isinstance(
        path, (str, Path)
    ), "The path must be a string or a pathlib.Path object."

    assert isinstance(
        frames, (tuple, list, type(None))
    ), "The frames must be a tuple, list or None."

    if frames is not None:
        # Assert that all frames are integers
        assert all(
            isinstance(frame, int) for frame in frames
        ), "All frames must be integers."

    if transmits is not None:
        # Assert that all frames are integers
        assert all(
            isinstance(tx, int) for tx in transmits
        ), "All transmits must be integers."

    assert (
        data_type in _DATA_TYPES
    ), f"Data type {data_type} does not exist, should be in {_DATA_TYPES}"

    with h5py.File(path, "r") as hdf5_file:
        # Define the probe
        probe_name = hdf5_file.attrs["probe"]
        file_scan_parameters = recursively_load_dict_contents_from_group(
            hdf5_file, "scan"
        )
        file_scan_parameters = cast_scan_parameters(file_scan_parameters)
        sig = inspect.signature(get_probe("generic").__init__)
        file_probe_params = {
            key: file_scan_parameters[key]
            for key in sig.parameters
            if key in file_scan_parameters
        }

        if probe_name == "generic":
            probe = get_probe(probe_name, **file_probe_params)
        else:
            probe = get_probe(probe_name)

            probe_geometry = file_probe_params.get("probe_geometry", None)

            # Verify that the probe geometry matches the probe geometry in the
            # dataset
            if not np.allclose(probe_geometry, probe.probe_geometry):
                probe.probe_geometry = probe_geometry
                log.warning(
                    "The probe geometry in the data file does not "
                    "match the probe geometry of the probe. The probe "
                    "geometry has been updated to match the data file."
                )

        n_frames = file_scan_parameters.pop(
            "n_frames"
        )  # this is not part of Scan class
        remove_params = ["PRF", "origin"]
        for param in remove_params:
            if param in file_scan_parameters:
                file_scan_parameters.pop(param)

        n_tx = file_scan_parameters["n_tx"]

        if frames is None:
            frames = np.arange(n_frames, dtype=np.int32)

        if transmits is None:
            transmits = np.arange(n_tx, dtype=np.int32)

        n_tx = len(transmits)

        # Load the desired frames from the file
        data = hdf5_file["data"][data_type][frames]

        if data_type in ["raw_data", "aligned_data", "beamformed_data"]:
            if data.shape[-1] != 1 and data.shape[-1] != 2:
                raise ValueError(
                    f"The data has an unexpected shape: {data.shape}. Last "
                    "dimension must be 1 (RF) or 2 (IQ), when data_type is "
                    f"{data_type}."
                )
        # Define the additional keyword parameters from the config object or an emtpy
        # dict if no config object is provided.
        if config is None:
            config_scan_dict = {}
        else:
            config_scan_dict = config.scan

        # merge file scan parameters with config scan parameters
        scan_params = update_dictionary(file_scan_parameters, config_scan_dict)

        # Initialize the scan object
        scan = Scan(**scan_params)

        return data, scan, probe
