"""
Functions to write and validate datasets in the USBMD format.
"""

import logging
from pathlib import Path

import h5py
import numpy as np

from usbmd.probes import Probe, get_probe
from usbmd.scan import Scan, cast_scan_parameters
from usbmd.utils.checks import _DATA_TYPES, validate_dataset
from usbmd.utils.read_h5 import recursively_load_dict_contents_from_group
from usbmd.utils.utils import first_not_none_item, update_dictionary


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
    n_tx = 8
    n_ch = 1
    n_frames = 2

    raw_data = np.ones((n_frames, n_tx, n_ax, n_el, n_ch))

    t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
    tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)

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
        probe_geometry=probe_geometry,
        sampling_frequency=40e6,
        center_frequency=7e6,
        initial_times=np.zeros((n_tx,)),
        t0_delays=t0_delays,
        sound_speed=1540,
        tx_apodizations=tx_apodizations,
        probe_name="example_probe",
        focus_distances=focus_distances,
        polar_angles=polar_angles,
        azimuth_angles=azimuth_angles,
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
):
    """
    Generates a dataset in the USBMD format.

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
        polar_angles (np.ndarray): The polar angles of shape (n_el,).
        azimuth_angles (np.ndarray): The azimuth angles of shape (n_tx,).
        tx_apodizations (np.ndarray): The transmit delays for each element defining
            the wavefront in seconds of shape (n_tx, n_elem).
            This is the time between the first element firing and the last element firing.
        bandwidth_percent (float): The bandwidth of the transducer as a
            percentage of the center frequency.
        time_to_next_transmit (np.ndarray): The time between subsequent transmit events in s
            of shape (n_frames, n_tx).

    Returns:
        (h5py.File): The example dataset.
    """

    # Assertions
    assert (
        raw_data is not None
        or aligned_data is not None
        or envelope_data is not None
        or beamformed_data is not None
        or image is not None
        or image_sc is not None
    ), f"At least one of the data types {_DATA_TYPES} must be specified."

    assert isinstance(probe_name, str), "The probe name must be a string."
    assert isinstance(description, str), "The description must be a string."

    # Convert path to Path object
    path = Path(path)

    if path.exists():
        raise FileExistsError(f"The file {path} already exists.")

    # Create the directory if it does not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as dataset:
        dataset.attrs["probe"] = probe_name
        dataset.attrs["description"] = description

        assert (
            isinstance(raw_data, np.ndarray) and raw_data.ndim == 5
        ), "The raw_data must be a numpy array of shape (n_frames, n_tx, n_ax, n_el, n_ch)."

        def convert_datatype(x, astype=np.float32):
            return x.astype(astype) if x is not None else None

        def first_not_none_shape(arr, axis):
            data = first_not_none_item(arr)
            return data.shape[axis] if data is not None else None

        def add_dataset(group, name, data, description, unit):
            """Adds a dataset to the given group with a description and unit.
            If data is None, the dataset is not added."""
            if data is None:
                return
            dataset = group.create_dataset(name, data=data)
            dataset.attrs["description"] = description
            dataset.attrs["unit"] = unit

        n_frames = first_not_none_item(
            [raw_data, aligned_data, envelope_data, beamformed_data, image_sc, image]
        ).shape[0]
        n_tx = first_not_none_shape([raw_data, aligned_data], axis=1)
        n_el = first_not_none_shape([raw_data, aligned_data], axis=3)
        n_ax = first_not_none_shape([raw_data, aligned_data], axis=2)
        n_ch = first_not_none_shape([raw_data, aligned_data], axis=4)

        # Write data group
        data_group = dataset.create_group("data")
        data_group.attrs["description"] = "This group contains the data."

        add_dataset(
            group=data_group,
            name="raw_data",
            data=convert_datatype(raw_data),
            description="The raw_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="aligned_data",
            data=convert_datatype(aligned_data),
            description="The aligned_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="envelope_data",
            data=convert_datatype(envelope_data),
            description="The envelope_data of shape (n_frames, n_z, n_x).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="beamformed_data",
            data=convert_datatype(beamformed_data),
            description="The beamformed_data of shape (n_frames, n_z, n_x).",
            unit="unitless",
        )

        add_dataset(
            group=data_group,
            name="image",
            data=convert_datatype(image),
            unit="unitless",
            description="The images of shape [n_frames, n_z, n_x]",
        )

        add_dataset(
            group=data_group,
            name="image_sc",
            data=convert_datatype(image_sc),
            unit="unitless",
            description=(
                "The scan converted images of shape [n_frames, output_size_z,"
                " output_size_x]"
            ),
        )

        # Write scan group
        scan_group = dataset.create_group("scan")
        scan_group.attrs["description"] = "This group contains the scan parameters."

        add_dataset(
            group=scan_group,
            name="n_ax",
            data=n_ax,
            description="The number of axial samples.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_el",
            data=n_el,
            description="The number of elements in the probe.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_tx",
            data=n_tx,
            description="The number of transmits per frame.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_ch",
            data=n_ch,
            description=(
                "The number of channels. For RF data this is 1. For IQ data "
                "this is 2."
            ),
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="n_frames",
            data=n_frames,
            description="The number of frames.",
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="sound_speed",
            data=sound_speed,
            description="The speed of sound in m/s",
            unit="m/s",
        )

        add_dataset(
            group=scan_group,
            name="probe_geometry",
            data=probe_geometry,
            description="The probe geometry of shape (n_el, 3).",
            unit="m",
        )

        add_dataset(
            group=scan_group,
            name="sampling_frequency",
            data=sampling_frequency,
            description="The sampling frequency in Hz.",
            unit="Hz",
        )

        add_dataset(
            group=scan_group,
            name="center_frequency",
            data=center_frequency,
            description="The center frequency in Hz.",
            unit="Hz",
        )

        add_dataset(
            group=scan_group,
            name="initial_times",
            data=initial_times,
            description=(
                "The times when the A/D converter starts sampling "
                "in seconds of shape (n_tx,). This is the time between the "
                "first element firing and the first recorded sample."
            ),
            unit="s",
        )

        add_dataset(
            group=scan_group,
            name="t0_delays",
            data=t0_delays,
            description="The t0_delays of shape (n_tx, n_el).",
            unit="s",
        )

        add_dataset(
            group=scan_group,
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

        add_dataset(
            group=scan_group,
            name="focus_distances",
            data=focus_distances,
            description=(
                "The transmit focus distances in meters of "
                "shape (n_tx,). For planewaves this is set to Inf."
            ),
            unit="m",
        )

        add_dataset(
            group=scan_group,
            name="polar_angles",
            data=polar_angles,
            description=(
                "The polar angles of the transmit beams in radians of shape (n_tx,)."
            ),
            unit="rad",
        )

        add_dataset(
            group=scan_group,
            name="azimuth_angles",
            data=azimuth_angles,
            description=(
                "The azimuthal angles of the transmit beams in radians of shape (n_tx,)."
            ),
            unit="rad",
        )

        add_dataset(
            group=scan_group,
            name="bandwidth_percent",
            data=bandwidth_percent,
            description=(
                "The receive bandwidth of RF signal in percentage of center frequency."
            ),
            unit="unitless",
        )

        add_dataset(
            group=scan_group,
            name="time_to_next_transmit",
            data=time_to_next_transmit,
            description=(
                "The time between subsequent transmit events of shape "
                "(n_frames, n_tx)."
            ),
            unit="s",
        )

    validate_dataset(path)

    
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
        probe_geometry = hdf5_file["scan"]["probe_geometry"][:]

        # Try to load a known probe type. If this fails, use a generic probe
        # instead, but warn the user.
        try:
            probe = get_probe(probe_name)
        except NotImplementedError:
            logging.warning(
                "The probe %s is not implemented. Using a generic probe instead.",
                probe_name,
            )

            probe = Probe(probe_geometry=probe_geometry)

        # Verify that the probe geometry matches the probe geometry in the
        # dataset
        if not np.allclose(probe_geometry, probe.probe_geometry):
            probe.probe_geometry = probe_geometry
            logging.warning(
                "The probe geometry in the data file does not "
                "match the probe geometry of the probe. The probe "
                "geometry has been updated to match the data file."
            )

        file_scan_parameters = recursively_load_dict_contents_from_group(
            hdf5_file, "scan"
        )
        file_scan_parameters = cast_scan_parameters(file_scan_parameters)

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

        # Select only the desired transmits if
        if data_type in ["raw_data", "aligned_data"]:
            data = data[:, scan.selected_transmits]

        return data, scan, probe
