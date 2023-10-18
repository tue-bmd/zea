"""
Functions to write and validate datasets in the USBMD format.
"""
import logging
from pathlib import Path

import h5py
import numpy as np

from usbmd.probes import Probe, get_probe
from usbmd.processing import _DATA_TYPES
from usbmd.scan import Scan


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

    raw_data = np.ones((n_frames, n_tx, n_el, n_ax, n_ch))

    t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
    tx_apodizations = np.zeros((n_tx, n_el), dtype=np.float32)
    probe_geometry = np.zeros((n_el, 3), dtype=np.float32)
    probe_geometry[:, 0] = np.linspace(-0.02, 0.02, n_el)

    if add_optional_fields:
        focus_distances = np.ones((n_tx,), dtype=np.float32)*np.inf
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
    raw_data,
    probe_geometry,
    sampling_frequency,
    center_frequency,
    initial_times,
    t0_delays,
    sound_speed,
    probe_name,
    description="No description was supplied",
    focus_distances=None,
    polar_angles=None,
    azimuth_angles=None,
    tx_apodizations=None,
    bandwidth_percent=None,
):
    """Generates a dataset in the USBMD format.

    Args:
        path (str): The path to write the dataset to.
        raw_data (np.ndarray): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_el, n_ax, n_ch).
        add_optional_fields (bool, optional): Whether to add optional fields to
            the dataset. Defaults to False.

    Returns:
        (h5py.File): The example dataset.
    """

    dataset = h5py.File(path, "w")

    dataset.attrs["probe"] = probe_name
    dataset.attrs["description"] = description

    def add_dataset(group, name, data, description, unit):
        """Adds a dataset to the given group with a description and unit."""
        dataset = group.create_dataset(name, data=data)
        dataset.attrs["description"] = description
        dataset.attrs["unit"] = unit

    n_frames = raw_data.shape[0]
    n_tx = raw_data.shape[1]
    n_el = raw_data.shape[2]
    n_ax = raw_data.shape[3]
    n_ch = raw_data.shape[4]

    # Write data group
    data_group = dataset.create_group("data")
    data_group.attrs["description"] = "This group contains the data."
    data_shape = (n_frames, n_tx, n_el, n_ax, n_ch)
    assert raw_data.shape == data_shape, (
        f"The raw_data has the wrong shape. Expected {data_shape}, "
        f"got {raw_data.shape}."
    )

    add_dataset(
        group=data_group,
        name="raw_data",
        data=raw_data.astype(np.float32),
        description="The raw_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).",
        unit="unitless",
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

    add_dataset(group=scan_group,
                name='n_frames',
                data=n_frames,
                description='The number of frames.',
                unit='unitless')

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
        description="The times when the A/D converter starts sampling "
        "in seconds of shape (n_tx,). This is the time between the "
        "first element firing and the first recorded sample.",
        unit="s",
    )

    add_dataset(
        group=scan_group,
        name="t0_delays",
        data=t0_delays,
        description="The t0_delays of shape (n_tx, n_el).",
        unit="s",
    )

    if tx_apodizations is not None:
        add_dataset(
            group=scan_group,
            name="tx_apodizations",
            data=tx_apodizations,
            description="The transmit delays for each element defining the"
            " wavefront in seconds of shape (n_tx, n_elem). This is"
            " the time at which each element fires shifted such that"
            " the first element fires at t=0.",
            unit="unitless",
        )

    if focus_distances is not None:
        add_dataset(
            group=scan_group,
            name="focus_distances",
            data=focus_distances,
            description="The transmit focus distances in meters of "
            "shape (n_tx,). For planewaves this is set to Inf.",
            unit="m",
        )

    if polar_angles is not None:
        add_dataset(
            group=scan_group,
            name="polar_angles",
            data=polar_angles,
            description="The polar angles of the transmit beams in "
            "radians of shape (n_tx,).",
            unit="rad",
        )

    if azimuth_angles is not None:
        add_dataset(
            group=scan_group,
            name="azimuth_angles",
            data=azimuth_angles,
            description="The azimuthal angles of the transmit beams "
            "in radians of shape (n_tx,).",
            unit="rad",
        )
    
    if bandwidth_percent is not None:
        add_dataset(
            group=bandwidth_percent,
            name="bandwidth_percent",
            data=bandwidth_percent,
            description="The receive bandwidth of RF signal in "
            "percentage of center frequency.",
            unit="unitless",
        )

    dataset.close()
    validate_dataset(path)


def validate_dataset(path):
    """Reads the hdf5 dataset at the given path and validates its structure.

    Args:
        path (str, pathlike): The path to the hdf5 dataset.

    """
    dataset = h5py.File(path, "r")

    def check_key(dataset, key):
        assert key in dataset.keys(
        ), f"The dataset does not contain the key {key}."

    # Validate the root group
    check_key(dataset, "data")
    check_key(dataset, "scan")

    # validate the scan group
    check_key(dataset["scan"], "n_ax")
    check_key(dataset["scan"], "n_el")
    check_key(dataset["scan"], "n_tx")
    check_key(dataset["scan"], "probe_geometry")
    check_key(dataset["scan"], "sampling_frequency")
    check_key(dataset["scan"], "center_frequency")
    check_key(dataset["scan"], "t0_delays")

    # validate the data group
    allowed_data_keys = _DATA_TYPES

    for key in dataset["data"].keys():
        assert key in allowed_data_keys, "The data group contains an unexpected key."

        # Validate data shape
        data_shape = dataset["data"][key].shape
        if key == "raw_data":
            assert (
                len(data_shape) == 5
            ), "The raw_data group does not have a shape of length 5."
            assert (
                data_shape[1] == dataset["scan"]["n_tx"][()]
            ), "n_tx does not match the second dimension of raw_data."
            assert (
                data_shape[2] == dataset["scan"]["n_el"][()]
            ), "n_el does not match the third dimension of raw_data."
            assert (
                data_shape[3] == dataset["scan"]["n_ax"][()]
            ), "n_ax does not match the fourth dimension of raw_data."
            assert data_shape[4] in (
                1,
                2,
            ), "The fifth dimension of raw_data is not 1 or 2."

        elif key == "aligned_data":
            logging.warning("No validation has been defined for aligned data.")
        elif key == "beamformed_data":
            logging.warning(
                "No validation has been defined for beamformed data.")
        elif key == "envelope_data":
            logging.warning(
                "No validation has been defined for envelope data.")
        elif key == "image":
            logging.warning("No validation has been defined for image data.")
        elif key == "image_sc":
            logging.warning(
                "No validation has been defined for image_sc data.")

    required_scan_keys = [
        "n_ax",
        "n_el",
        "n_tx",
        "n_frames",
        "probe_geometry",
        "sampling_frequency",
        "center_frequency",
    ]

    # Ensure that all required keys are present
    for required_key in required_scan_keys:
        assert required_key in dataset["scan"].keys(), (
            "The scan group does not contain the required key " f"{required_key}."
        )

    # Ensure that all keys have the correct shape
    for key in dataset["scan"].keys():
        if key == "probe_geometry":
            correct_shape = (dataset["scan"]["n_el"][()], 3)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The probe_geometry does not have the correct shape."

        elif key == "t0_delays":
            correct_shape = (dataset["scan"]["n_tx"]
                             [()], dataset["scan"]["n_el"][()])
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The t0_delays does not have the correct shape."

        elif key == "tx_apodizations":
            correct_shape = (dataset["scan"]["n_tx"]
                             [()], dataset["scan"]["n_el"][()])
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The tx_apodizations does not have the correct shape."

        elif key == "focus_distances":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The focus_distances does not have the correct shape."

        elif key == "polar_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The polar_angles does not have the correct shape."

        elif key == "azimuth_angles":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The azimuthal_angles does not have the correct shape."

        elif key == "bandwidth_percent":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The percent bandwidth does not have the correct shape."

        elif key == "initial_times":
            correct_shape = (dataset["scan"]["n_tx"][()],)
            assert (
                dataset["scan"][key].shape == correct_shape
            ), "The initial_times does not have the correct shape."

        elif key in (
            "sampling_frequency",
            "center_frequency",
            "n_frames",
            "n_tx",
            "n_el",
            "n_ax",
            "sound_speed",
        ):
            assert (
                dataset["scan"][key].size == 1
            ), f"{key} does not have the correct shape."

        else:
            logging.warning("No validation has been defined for %s.", key)

    assert_unit_and_description_present(dataset)


def assert_unit_and_description_present(hdf5_file, _prefix=""):
    """Checks that all datasets have a unit and description attribute.

    Args:
        hdf5_file (h5py.File): The hdf5 file to check.

    Raises:
        AssertionError: If a dataset does not have a unit or description
            attribute.
    """
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            assert_unit_and_description_present(
                hdf5_file[key], _prefix=_prefix + key + "/"
            )
        else:
            assert (
                "unit" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a unit attribute."
            assert (
                "description" in hdf5_file[key].attrs.keys()
            ), f"The dataset {_prefix}/{key} does not have a description attribute."


def load_usbmd_file(path, frames=None, transmits=None, data_type="raw_data"):
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

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_el, n_ax, n_ch).
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
        assert all(isinstance(tx, int) for tx in transmits), \
            'All transmits must be integers.'

    assert (
        data_type in _DATA_TYPES
    ), f"Data type {data_type} does not exist, should be in {_DATA_TYPES}"

    with h5py.File(path, "r") as hdf5_file:
        # data = hdf5_file['data']['raw_data'][:]
        # scan = Scan(hdf5_file['scan'])

        # Define the probe
        probe_name = hdf5_file.attrs["probe"]
        ele_pos = hdf5_file["scan"]["probe_geometry"][:]

        # Try to load a known probe type. If this fails, use a generic probe
        # instead, but warn the user.
        try:
            probe = get_probe(probe_name)
        except NotImplementedError:
            logging.warning(
                "The probe %s is not implemented. Using a generic probe instead.",
                probe_name,
            )

            probe = Probe(ele_pos=ele_pos)

        # Verify that the probe geometry matches the probe geometry in the
        # dataset
        if not np.allclose(ele_pos, probe.ele_pos):
            probe.ele_pos = ele_pos
            logging.warning(
                "The probe geometry in the data file does not "
                "match the probe geometry of the probe. The probe "
                "geometry has been updated to match the data file."
            )

        # Define the scan
        n_frames = int(hdf5_file['scan']['n_frames'][()])
        n_ax = int(hdf5_file['scan']['n_ax'][()])
        n_tx = int(hdf5_file['scan']['n_tx'][()])
        c = float(hdf5_file['scan']['sound_speed'][()])
        fs = float(hdf5_file['scan']['sampling_frequency'][()])
        fc = float(hdf5_file['scan']['center_frequency'][()])
        bandwidth_percent = float(hdf5_file['scan']['bandwidth_percent'][()])

        if frames is None:
            frames = np.arange(n_frames, dtype=np.int32)

        if transmits is None:
            transmits = np.arange(n_tx, dtype=np.int32)

        # Compute the depth of the scan from the number of axial samples
        depth = n_ax / fs * c / 2

        # Set the scan limits to the limits of the probe and the depth of the
        # scan
        x0, x1 = ele_pos[0, 0], ele_pos[-1, 0]
        z0, z1 = 0, depth

        n_tx = len(transmits)

        initial_times = hdf5_file['scan']['initial_times'][transmits]
        tx_apodizations = hdf5_file['scan']['tx_apodizations'][transmits]
        t0_delays = hdf5_file['scan']['t0_delays'][transmits]
        polar_angles = hdf5_file['scan']['polar_angles'][transmits]
        azimuth_angles = hdf5_file['scan']['azimuth_angles'][transmits]
        focus_distances = hdf5_file['scan']['focus_distances'][transmits]

        # Initialize the scan object
        scan = Scan(
            n_tx=n_tx,
            t0_delays=t0_delays,
            initial_times=initial_times,
            tx_apodizations=tx_apodizations,
            xlims=(x0, x1),
            zlims=(z0, z1),
            fc=fc,
            fs=fs,
            bandwidth_percent=bandwidth_percent,
            n_ax=n_ax,
            c=c,
            polar_angles=polar_angles,
            azimuth_angles=azimuth_angles,
            focus_distances=focus_distances,
        )

        # Load the desired frames from the file
        data = hdf5_file["data"][data_type][frames]

        if data_type in ["raw_data", "aligned_data"]:
            data = data[:, transmits]

        return data, scan, probe
