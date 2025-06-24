"""Functionality to convert Verasonics matlab raw files to the zea format.

Example (MATLAB):

    .. code-block:: matlab

        >> setup_script;
        >> VSX;
        >> save_raw('C:/path/to/raw_data.mat');

Then in python:

    .. code-block:: python

        from zea.data_format.zea_from_matlab_raw import zea_from_matlab_raw

        zea_from_matlab_raw("C:/path/to/raw_data.mat", "C:/path/to/output.hdf5")

Or alternatively, use the script below to convert all .mat files in a directory:

    .. code-block:: bash

        python zea/data/convert/matlab.py "C:/path/to/directory"

or without the directory argument, the script will prompt you to select a directory
using a file dialog.

Event structure
---------------

By default the zea dataformat saves all the data to an hdf5 file with the following structure:

.. code-block:: text

    regular_zea_dataset.hdf5
    ├── data
    └── scan
          └── center_frequency: 1MHz

The data is stored in the ``data`` group and the scan parameters are stored in the ``scan``.
However, when we do an adaptive acquisition, some scanning parameters might change. These
blocks of data with consistent scanning parameters we call events. In the case we have multiple
events, we store the data in the following structure:

.. code-block:: text

    zea_dataset.hdf5
    ├── event_0
    │   ├── data
    │   └── scan
    │       └── center_frequency: 1MHz
    ├── event_1
    │   ├── data
    │   └── scan
    │       └── center_frequency: 2MHz
    ├── event_2
    │   ├── data
    │   └── scan
    └── event_3
        ├── data
        └── scan

This structure is supported by the zea toolbox. The way we can save the data in this structure
from the Verasonics, is by changing the setup script to keep track of the TX struct at each event.

The way this is done is still in development, an example of such an acquisition script that is
compatible with saving event structures is found here:
`setup_agent.m <https://github.com/tue-bmd/needle-tracking/blob/ius2024-demo-nc/verasonics/setup_agent.m>`_

Adding additional elements
--------------------------

You can add additional elements to the dataset by defining a function that reads the
data from the file and returns a ``DatasetElement``. Then pass the function to the
``zea_from_matlab_raw`` function as a list.

.. code-block:: python

    def read_max_high_voltage(file):
        lens_correction = file["Trans"]["lensCorrection"][0, 0].item()
        return lens_correction


    def read_high_voltage_func(file):
        return DatasetElement(
            group_name="scan",
            dataset_name="max_high_voltage",
            data=read_max_high_voltage(file),
            description="The maximum high voltage used by the Verasonics system.",
            unit="V",
        )


    zea_from_matlab_raw(
        "C:/path/to/raw_data.mat",
        "C:/path/to/output.hdf5",
        [read_high_voltage_func],
    )
"""  # noqa: E501

import argparse
import os
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog

import h5py
import numpy as np

from zea import log
from zea.data.data_format import DatasetElement, generate_zea_dataset
from zea.ops import LogCompress, Normalize
from zea.utils import strtobool

_VERASONICS_TO_ZEA_PROBE_NAMES = {
    "L11-4v": "verasonics_l11_4v",
    "L11-5v": "verasonics_l11_5v",
}


def dereference_index(file, dataset, index, event=None, subindex=None):
    """Get the element at the given index from the dataset, dereferencing it if
    necessary.

    MATLAB stores items in struct array differently depending on the size. If the size
    is 1, the item is stored as a regular dataset. If the size is larger, the item is
    stored as a dataset of references to the actual data.

    This function dereferences the dataset if it is a reference. Otherwise, it returns
    the dataset.

    Args:
        file (h5py.File): The file to read the dataset from.
        dataset (h5py.Dataset): The dataset to read the element from.
        index (int): The index of the element to read.
        event (int, optional): The event index. Usually we store each event in the
            second dimension of the dataset. Defaults to None in this case we assume
            that there is only a single event.
        subindex (slice, optional): The subindex of the element to read after
            referencing the actual data. Defaults to None. In this case, all the data
            is returned.
    """
    if isinstance(dataset.fillvalue, h5py.h5r.Reference):
        if event is not None:
            reference = dataset[index, event]
        else:
            reference = dataset[index, 0]
        if subindex is None:
            return file[reference][:]
        else:
            return file[reference][subindex]
    else:
        if index > 0:
            log.warning(
                f"index {index} is not a reference. You are probably "
                "incorrectly indexing a dataset."
            )
        return dataset


def get_reference_size(dataset):
    """Get the size of a reference dataset."""
    if isinstance(dataset.fillvalue, h5py.h5r.Reference):
        return len(dataset)
    else:
        return 1


def decode_string(dataset):
    """Decode a string dataset."""
    return "".join([chr(c) for c in dataset.squeeze()])


def read_probe_geometry(file):
    """
    Read the probe geometry from the file.

    Args:
        file (h5py.File): The file to read the probe geometry from. (The file should
            be opened in read mode.)

    Returns:
        probe_geometry (np.ndarray): The probe geometry of shape (n_el, 3).
    """
    # Read the probe geometry from the file
    probe_geometry = file["Trans"]["ElementPos"][:3, :]

    # Transpose the probe geometry to have the shape (n_el, 3)
    probe_geometry = probe_geometry.T

    # Read the unit
    unit = decode_string(file["Trans"]["units"][:])

    # Convert the probe geometry to meters
    if unit == "mm":
        probe_geometry = probe_geometry / 1000
    else:
        wavelength = read_wavelength(file)
        probe_geometry = probe_geometry * wavelength

    return probe_geometry


def read_wavelength(file):
    """Reads the wavelength from the file.

    Args:
        `file` (`h5py.File`): The file to read the wavelength from. (The file should be
            opened in read mode.)

    Returns:
        `wavelength` (`float`): The wavelength of the probe.
    """
    center_frequency = read_probe_center_frequency(file)
    sound_speed = read_sound_speed(file)
    wavelength = sound_speed / center_frequency
    return wavelength


def read_transmit_events(file, event=None, frames="all"):
    """Read the events from the file and finds the order in which transmits and receives
    appear in the events.

    Args:
        file (h5py.File): The file to read the events from.
            The file should be opened in read mode.
        event (int, optional): The event index. Defaults to None.
        frames (str or list, optional): The frames to read. Defaults to "all".

    Returns:
        tuple: (tx_order, rcv_order, time_to_next_acq)
            tx_order (list): The order in which the transmits appear in the events.
            rcv_order (list): The order in which the receives appear in the events.
            time_to_next_acq (np.ndarray): The time to next acquisition of shape (n_acq, n_tx).
    """

    num_events = file["Event"]["info"].shape[0]

    # In the Verasonics the transmits may not be in order in the TX structure and a
    # transmit might be reused. Therefore, we need to keep track of the order in which
    # the transmits appear in the Events.
    tx_order = []
    rcv_order = []
    time_to_next_acq = []

    frame_indices = get_frame_indices(file, frames)

    for i in range(num_events):
        # Get the tx
        event_tx = dereference_index(file, file["Event"]["tx"], i)
        event_tx = int(event_tx.item())

        # Get the rcv
        event_rcv = dereference_index(file, file["Event"]["rcv"], i)
        event_rcv = int(event_rcv.item())

        if not bool(event_tx) == bool(event_rcv):
            log.warning(
                "Events should have both a transmit and a receive or neither. "
                f"Event {i} has a transmit but no receive or vice versa."
            )

        if not event_tx:
            continue

        # Subtract one to make the indices 0-based
        event_tx -= 1
        event_rcv -= 1

        # Check in the Receive structure if this is still the first frame
        framenum_ref = file["Receive"]["framenum"][event_rcv, 0]
        framenum = file[framenum_ref][:].item()

        # Only add the event to the list if it is the first frame since we assume
        # that all frames have the same transmits and receives
        if framenum == 1:
            # Add the event to the list
            tx_order.append(event_tx)
            rcv_order.append(event_rcv)

        # Read the time_to_next_acq
        seq_control_indices = dereference_index(file, file["Event"]["seqControl"], i)

        for seq_control_index in seq_control_indices:
            seq_control_index = int(seq_control_index.item() - 1)
            seq_control = dereference_index(file, file["SeqControl"]["command"], seq_control_index)
            # Decode the seq_control int array into a string
            seq_control = decode_string(seq_control)
            if seq_control == "timeToNextAcq":
                value = dereference_index(
                    file, file["SeqControl"]["argument"], seq_control_index
                ).item()
                value = value * 1e-6
                time_to_next_acq.append(value)

    n_tx = len(tx_order)
    time_to_next_acq = np.array(time_to_next_acq)
    time_to_next_acq = np.reshape(time_to_next_acq, (-1, n_tx))

    if event is not None:
        time_to_next_acq = time_to_next_acq[event]
        time_to_next_acq = np.expand_dims(time_to_next_acq, axis=0)

    time_to_next_acq = time_to_next_acq[frame_indices]

    return tx_order, rcv_order, time_to_next_acq


def read_t0_delays_apod(file, tx_order, event=None):
    """
    Read the t0 delays and apodization from the file.

    Args:
        file (h5py.File): The file to read the t0 delays from. (The file should be
            opened in read mode.)

    Returns:
        t0_delays (np.ndarray): The t0 delays of shape (n_tx, n_el).
        apod (np.ndarray): The apodization of shape (n_el,).
    """

    t0_delays_list = []
    tx_apodizations_list = []

    wavelength = read_wavelength(file)
    sound_speed = read_sound_speed(file)

    for n in tx_order:
        # Get column vector of t0_delays
        if event is None:
            t0_delays = dereference_index(file, file["TX"]["Delay"], n)
        else:
            t0_delays = dereference_index(file, file["TX_Agent"]["Delay"], n, event)
        # Turn into 1d array
        t0_delays = t0_delays[:, 0]

        t0_delays_list.append(t0_delays)

        # Get column vector of apodizations
        if event is None:
            tx_apodizations = dereference_index(file, file["TX"]["Apod"], n)
        else:
            tx_apodizations = dereference_index(file, file["TX_Agent"]["Apod"], n, event)
        # Turn into 1d array
        tx_apodizations = tx_apodizations[:, 0]
        tx_apodizations_list.append(tx_apodizations)

    t0_delays = np.stack(t0_delays_list, axis=0)
    apodizations = np.stack(tx_apodizations_list, axis=0)

    # Convert the t0_delays to meters
    t0_delays = t0_delays * wavelength / sound_speed

    return t0_delays, apodizations


def read_sampling_frequency(file):
    """
    Read the sampling frequency from the file.

    Args:
        file (h5py.File): The file to read the sampling frequency from. (The file
            should be opened in read mode.)

    Returns:
        sampling_frequency (float): The sampling frequency.
    """
    # Read the sampling frequency from the file
    adc_rate = dereference_index(file, file["Receive"]["decimSampleRate"], 0)

    # The Vantage NXT has renamed this field to sampleSkip
    if "quadDecim" in file["Receive"]:
        quaddecim = dereference_index(file, file["Receive"]["quadDecim"], 0)
    else:
        quaddecim = 1.0

    sampling_frequency = adc_rate / quaddecim * 1e6

    return sampling_frequency[0, 0]


def read_waveforms(file, tx_order, event=None):
    """
    Read the waveforms from the file.

    Args:
        file (h5py.File): The file to read the waveforms from. (The file should be
            opened in read mode.)

    Returns:
        waveforms (np.ndarray): The waveforms of shape (n_tx, n_samples).
    """
    waveforms_one_way_list = []
    waveforms_two_way_list = []

    # Read all the waveforms from the file
    n_tw = get_reference_size(file["TW"]["Wvfm1Wy"])
    for n in range(n_tw):
        # Get the row vector of the 1-way waveform
        waveform_one_way = dereference_index(file, file["TW"]["Wvfm1Wy"], n)[:]
        # Turn into 1d array
        waveform_one_way = waveform_one_way[0, :]

        # Get the row vector of the 2-way waveform
        waveform_two_way = dereference_index(file, file["TW"]["Wvfm2Wy"], n)[:]
        # Turn into 1d array
        waveform_two_way = waveform_two_way[0, :]

        waveforms_one_way_list.append(waveform_one_way)
        waveforms_two_way_list.append(waveform_two_way)

    tx_waveform_indices = []

    for n in tx_order:
        # Read the waveform
        if event is None:
            waveform_index = dereference_index(file, file["TX"]["waveform"], n)[:]
        else:
            waveform_index = dereference_index(file, file["TX_Agent"]["waveform"], n, event)[:]
        # Subtract one to make the indices 0-based
        waveform_index -= 1
        # Turn into integer
        waveform_index = int(waveform_index.item())
        tx_waveform_indices.append(waveform_index)

    return tx_waveform_indices, waveforms_one_way_list, waveforms_two_way_list


def read_polar_angles(file, tx_order, event=None):
    """
    Read the polar angles from the file.

    Args:
        file (h5py.File): The file to read the polar angles from. (The file should
            be opened in read mode.)

    Returns:
        polar_angles (np.ndarray): The polar angles of shape (n_tx,).
    """
    polar_angles_list = []

    for n in tx_order:
        # Read the polar angle
        if event is None:
            polar_angle = dereference_index(file, file["TX"]["Steer"], n)[:]
        else:
            polar_angle = dereference_index(file, file["TX_Agent"]["Steer"], n, event)[:]
        # Turn into 1d array
        polar_angle = polar_angle[0, 0]

        polar_angles_list.append(polar_angle)

    polar_angles = np.stack(polar_angles_list, axis=0)

    return polar_angles


def read_azimuth_angles(file, tx_order, event=None):
    """
    Read the azimuth angles from the file.

    Args:
        file (h5py.File): The file to read the azimuth angles from. (The file should
            be opened in read mode.)

    Returns:
        azimuth_angles (np.ndarray): The azimuth angles of shape (n_tx,).
    """
    azimuth_angles_list = []

    for n in tx_order:
        # Read the azimuth angle
        if event is None:
            azimuth_angle = dereference_index(file, file["TX"]["Steer"], n)[:]
        else:
            azimuth_angle = dereference_index(file, file["TX_Agent"]["Steer"], n, event)[:]
        # Turn into 1d array
        azimuth_angle = azimuth_angle[1, 0]

        azimuth_angles_list.append(azimuth_angle)

    azimuth_angles = np.stack(azimuth_angles_list, axis=0)

    return azimuth_angles


def read_raw_data(file, event=None, frames="all"):
    """
    Read the raw data from the file.

    Args:
        file (h5py.File): The file to read the raw data from. (The file should be
            opened in read mode.)

    Returns:
        raw_data (np.ndarray): The raw data of shape (n_rcv, n_samples).
    """

    # Get the number of axial samples
    start_sample = dereference_index(file, file["Receive"]["startSample"], 0).item()
    end_sample = dereference_index(file, file["Receive"]["endSample"], 0).item()
    n_ax = int(end_sample - start_sample + 1)

    # Obtain the number of transmit events per frame
    tx_order, _, _ = read_transmit_events(file)
    n_tx = len(tx_order)

    # Read the raw data from the file
    if event is None:
        raw_data = dereference_index(file, file["RcvData"], 0)
    else:
        # for now we only index frames as events
        raw_data = dereference_index(file, file["RcvData"], 0, subindex=event)
        raw_data = np.expand_dims(raw_data, axis=0)

    frame_indices = get_frame_indices(file, frames)

    raw_data = raw_data[frame_indices]

    raw_data = raw_data[:, :, : n_ax * n_tx]

    raw_data = raw_data.reshape((raw_data.shape[0], raw_data.shape[1], n_tx, -1))

    raw_data = np.transpose(raw_data, (0, 2, 3, 1))

    # Add channel dimension
    raw_data = raw_data[..., None]

    return raw_data


def read_probe_center_frequency(file):
    """Reads the center frequency of the probe from the file.

    Args:
        file (h5py.File): The file to read the center frequency from. (The file
            should be opened in read mode.)

    Returns:
        center_frequency (float): The center frequency of the probe.
    """
    center_frequency = file["Trans"]["frequency"][0, 0] * 1e6
    return center_frequency


def read_sound_speed(file):
    """Reads the speed of sound from the file.

    Args:
        file (h5py.File): The file to read the speed of sound from. (The file
            should be opened in read mode.)

    Returns:
        sound_speed (float): The speed of sound.
    """

    sound_speed = file["Resource"]["Parameters"]["speedOfSound"][0, 0].item()
    return sound_speed


def read_initial_times(file, rcv_order, sound_speed):
    """Reads the initial times from the file.

    Args:
        file (h5py.File): The file to read the initial times from. (The file should
            be opened in read mode.)
        rcv_order (list): The order in which the receives appear in the events.
        wavelength (float): The wavelength of the probe.
        sound_speed (float): The speed of sound.

    Returns:
        initial_times (np.ndarray): The initial times of shape (n_rcv,).
    """
    wavelength = read_wavelength(file)
    initial_times = []
    for n in rcv_order:
        start_depth = dereference_index(file, file["Receive"]["startDepth"], n).item()

        initial_times.append(2 * start_depth * wavelength / sound_speed)

    return np.array(initial_times).astype(np.float32)


def read_probe_name(file):
    """Reads the name of the probe from the file.

    Args:
        file (h5py.File): The file to read the name of the probe from. (The file
            should be opened in read mode.)

    Returns:
        probe_name (str): The name of the probe.
    """
    probe_name = file["Trans"]["name"][:]
    probe_name = decode_string(probe_name)
    # Translates between verasonics probe names and zea probe names
    if probe_name in _VERASONICS_TO_ZEA_PROBE_NAMES:
        probe_name = _VERASONICS_TO_ZEA_PROBE_NAMES[probe_name]
    else:
        log.warning(
            f"Probe name {probe_name} is not in the list of known probes. "
            "Please add it to the _VERASONICS_TO_ZEA_PROBE_NAMES dictionary. "
            "Falling back to generic probe."
        )
        probe_name = "generic"

    return probe_name


def read_focus_distances(file, tx_order, event=None):
    """Reads the focus distances from the file.

    Args:
        file (h5py.File): The file to read the focus distances from. (The file
            should be opened in read mode.)
        tx_order (list): The order in which the transmits appear in the events.

    Returns:
        focus_distances (list): The focus distances.
    """
    focus_distances = []
    for n in tx_order:
        if event is None:
            focus_distance = dereference_index(file, file["TX"]["focus"], n)[0, 0]
        else:
            focus_distance = dereference_index(file, file["TX_Agent"]["focus"], n, event)[0, 0]
        focus_distances.append(focus_distance)
    return np.array(focus_distances)


def _probe_geometry_is_ordered_ula(probe_geometry):
    """Checks if the probe geometry is ordered as a uniform linear array (ULA)."""
    diff_vec = probe_geometry[1:] - probe_geometry[:-1]
    return np.isclose(diff_vec, diff_vec[0]).all()


def planewave_focal_distance_to_inf(focus_distances, t0_delays, tx_apodizations, probe_geometry):
    """Detects plane wave transmits and sets the focus distance to infinity.

    Args:
        focus_distances (np.ndarray): The focus distances of shape (n_tx,).
        t0_delays (np.ndarray): The t0 delays of shape (n_tx, n_el).
        tx_apodizations (np.ndarray): The apodization of shape (n_tx, n_el).

    Returns:
        focus_distances (np.ndarray): The focus distances of shape (n_tx,).

    Note:
        This function assumes that the probe_geometry is a 1d uniform linear array.
        If not it will warn and return.
    """
    if not _probe_geometry_is_ordered_ula(probe_geometry):
        log.warning(
            "The probe geometry is not ordered as a uniform linear array. "
            "Focal distances are not set to infinity for plane waves."
        )
        return focus_distances

    for tx in range(focus_distances.size):
        mask_active = np.abs(tx_apodizations[tx]) > 0
        if np.sum(mask_active) < 2:
            continue
        t0_delays_active = t0_delays[tx][mask_active]

        # If the t0_delays all have the same offset, we assume it is a plane wave
        if np.std(np.diff(t0_delays_active)) < 1e-16:
            focus_distances[tx] = np.inf

    return focus_distances


def read_bandwidth_percent(file):
    """Reads the bandwidth percent from the file.

    Args:
        file (h5py.File): The file to read the bandwidth percent from. (The file
            should be opened in read mode.)

    Returns:
        bandwidth_percent (int): The bandwidth percent.
    """
    bandwidth_percent = dereference_index(file, file["Receive"]["sampleMode"], 0)
    bandwidth_percent = decode_string(bandwidth_percent)
    bandwidth_percent = int(bandwidth_percent[2:-2])
    return bandwidth_percent


def read_lens_correction(file):
    """Reads the lens correction from the file.

    Args:
        `file` (`h5py.File`): The file to read the lens correction from. (The file
            should be opened in read mode.)

    Returns:
        `lens_correction` (`np.ndarray`): The lens correction.
    """
    lens_correction = file["Trans"]["lensCorrection"][0, 0].item()
    return lens_correction


def read_tgc_gain_curve(file):
    """Reads the TGC gain curve from the file.

    Parameters
    ----------
    file : h5py.File
        The file to read the TGC gain curve from. (The file should be opened in read
        mode.)

    Returns
    -------
    np.ndarray
        The TGC gain curve of shape `(n_ax,)`.
    """

    gain_curve = file["TGC"]["Waveform"][:][:, 0]

    # Normalize the gain_curve to [0, 40]dB
    gain_curve = gain_curve / 1023 * 40

    # The gain curve is sampled at 800ns (See Verasonics documentation for details.
    # Specifically the tutorial sequence programming)
    gain_curve_sampling_period = 800e-9

    # Define the time axis for the gain curve
    t_gain_curve = np.arange(gain_curve.size) * gain_curve_sampling_period

    # Read the number of axial samples
    start_sample = dereference_index(file, file["Receive"]["startSample"], 0).item()
    end_sample = dereference_index(file, file["Receive"]["endSample"], 0).item()
    n_ax = int(end_sample - start_sample + 1)

    # Read the sampling frequency
    sampling_frequency = read_sampling_frequency(file)

    # Define the time axis for the axial samples
    t_samples = np.arange(n_ax) / sampling_frequency

    # Interpolate the gain_curve to the number of axial samples
    gain_curve = np.interp(t_samples, t_gain_curve, gain_curve)

    # The gain_curve gains are in dB, so we need to convert them to linear scale
    gain_curve = 10 ** (gain_curve / 20)

    return gain_curve


def read_image_data_p(file, event=None, frames="all"):
    """Reads the image data from the file.

    Args:
        `file` (`h5py.File`): The file to read the image data from. (The file should be
            opened in read mode.)

    Returns:
        `image_data` (`np.ndarray`): The image data.
    """
    # Check if the file contains image data
    if "ImgDataP" not in file:
        return None

    frame_indices = get_frame_indices(file, frames)

    # Get the dataset reference
    image_data_ref = file["ImgDataP"][0, 0]
    # Dereference the dataset
    if event is None:
        image_data = file[image_data_ref][:]
    else:
        image_data = file[image_data_ref][event]
        image_data = np.expand_dims(image_data, axis=0)

    # Get the relevant dimensions
    image_data = image_data[:, 0, :, :]

    # Convert to [-60, 0] dB range based on min and max values
    normalize = Normalize(output_range=(0, 1), input_range=None)
    log_compress = LogCompress()

    image_data = normalize(data=image_data)["data"]
    image_data = log_compress(data=image_data, dynamic_range=(-np.inf, 0))["data"]

    # Reshape so that [n_frames, n_samples, n_lines]
    image_data = np.transpose(image_data, (0, 2, 1))

    image_data = image_data[frame_indices]

    return image_data


def read_probe_element_width(file):
    """Reads the element width from the file.

    Args:
        file (h5py.File): The file to read the element width from.
            The file should be opened in read mode.

    Returns:
        float: The element width.
    """
    element_width = file["Trans"]["elementWidth"][:][0, 0]

    # Read the unit
    unit = decode_string(file["Trans"]["units"][:])

    # Convert the probe element width to meters
    if unit == "mm":
        element_width = element_width / 1000
    else:
        wavelength = read_wavelength(file)
        element_width = element_width * wavelength

    return element_width


def read_verasonics_file(file, event=None, additional_functions=None, frames="all"):
    """Reads data from a .mat Verasonics output file.

    Args:
        file (h5py.File): The file to read the data from. (The file should be opened in
            read mode.)
        event (int, optional): The event index. Defaults to None in this case we assume
            the data file is stored without event structure.
        additional_functions (list, optional): A list of functions that read additional
            data from the file. Each function should take the file as input and return a
            `DatasetElement`. Defaults to None.
    """

    probe_geometry = read_probe_geometry(file)

    # same for all events
    tx_order, rcv_order, time_to_next_transmit = read_transmit_events(file, frames=frames)
    sampling_frequency = read_sampling_frequency(file)
    bandwidth_percent = read_bandwidth_percent(file)
    center_frequency = read_probe_center_frequency(file)
    sound_speed = read_sound_speed(file)
    initial_times = read_initial_times(file, rcv_order, sound_speed)
    probe_name = read_probe_name(file)
    tgc_gain_curve = read_tgc_gain_curve(file)
    element_width = read_probe_element_width(file)

    # these are capable of handling multiple events
    raw_data = read_raw_data(file, event, frames=frames)
    image = read_image_data_p(file, event, frames=frames)

    polar_angles = read_polar_angles(file, tx_order, event)
    azimuth_angles = read_azimuth_angles(file, tx_order, event)
    t0_delays, tx_apodizations = read_t0_delays_apod(file, tx_order, event)
    focus_distances = read_focus_distances(file, tx_order, event)

    tx_waveform_indices, waveforms_one_way_list, waveforms_two_way_list = read_waveforms(
        file, tx_order, event
    )
    focus_distances = planewave_focal_distance_to_inf(
        focus_distances, t0_delays, tx_apodizations, probe_geometry
    )

    # If the data is captured in BS100BW mode or BS50BW mode, the data is stored in
    # as complex IQ data and the sampling frequency is halved.
    if bandwidth_percent in (50, 100):
        raw_data = np.concatenate(
            (
                raw_data[:, :, 0::2, :, :],
                -raw_data[:, :, 1::2, :, :],
            ),
            axis=-1,
        )
        # Two sequential samples are interpreted as a single complex sample
        # Therefore, we need to halve the sampling frequency
        sampling_frequency = sampling_frequency / 2

        # We have halved the number of samples, so we need to halve the number
        # of samples in the gain curve as well
        tgc_gain_curve = tgc_gain_curve[0::2]

    lens_correction = read_lens_correction(file)
    if event is None:
        group_name = "scan"
    else:
        group_name = f"event_{event}/scan"

    el_lens_correction = DatasetElement(
        group_name=group_name,
        dataset_name="lens_correction",
        data=lens_correction,
        description=(
            "The lens correction value used by Verasonics. This value is the "
            "additional path length in wavelength that the lens introduces. "
            "(This disregards refraction.)"
        ),
        unit="wavelengths",
    )

    additional_elements = []
    if additional_functions is not None:
        for additional_function in additional_functions:
            additional_elements.append(additional_function(file))

    data = {
        "probe_geometry": probe_geometry,
        "time_to_next_transmit": time_to_next_transmit,
        "t0_delays": t0_delays,
        "tx_apodizations": tx_apodizations,
        "sampling_frequency": sampling_frequency,
        "polar_angles": polar_angles,
        "azimuth_angles": azimuth_angles,
        "bandwidth_percent": bandwidth_percent,
        "raw_data": raw_data,
        "image": image,
        "center_frequency": center_frequency,
        "sound_speed": sound_speed,
        "initial_times": initial_times,
        "probe_name": probe_name,
        "focus_distances": focus_distances,
        "tx_waveform_indices": tx_waveform_indices,
        "waveforms_one_way": waveforms_one_way_list,
        "waveforms_two_way": waveforms_two_way_list,
        "tgc_gain_curve": tgc_gain_curve,
        "element_width": element_width,
        "additional_elements": [el_lens_correction, *additional_elements],
    }

    return data


def get_frame_indices(file, frames):
    """Creates a numpy array of frame indices from the file and the frames argument.

    Args:
        file (h5py.File): The file to read the frame indices from.
        frames (str): The frames argument. This can be "all" or a list of frame indices.

    Returns:
        frame_indices (np.ndarray): The frame indices.
    """
    # Read the number of frames from the file
    n_frames = int(file["Resource"]["RcvBuffer"]["numFrames"][0][0])

    if isinstance(frames, str) and frames == "all":
        # Create an array of all frame-indices
        frame_indices = np.arange(n_frames)
    else:
        frame_indices = np.array(frames)
        frame_indices.sort()

    if np.any(frame_indices >= n_frames):
        log.error(
            f"Frame indices {frame_indices} are out of bounds. "
            f"The file contains {n_frames} frames. "
            f"Using only the indices that are within bounds."
        )
        # Remove out of bounds indices
        frame_indices = frame_indices[frame_indices < n_frames]

    return frame_indices


def zea_from_matlab_raw(input_path, output_path, additional_functions=None, frames="all"):
    """Converts a Verasonics matlab raw file to the zea format. The MATLAB file
    should be created using the `save_raw` function and be stored in "v7.3" format.

    Args:
        input_path (str): The path to the input file (.mat file).
        output_path (str): The path to the output file (.hdf5 file).
        additional_functions (list, optional): A list of functions that read additional
            data from the file. Each function should take the file as input and return a
            `DatasetElement`. Defaults to None.
        frames (str or list of int, optional): The frames to add to the file. This can be
            a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
            'all'.
    """
    # Create the output directory if it does not exist
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert input_path.is_file(), log.error(f"Input file {log.yellow(input_path)} does not exist.")

    # Load the data
    with h5py.File(input_path, "r") as file:
        if "TX_Agent" in file:
            active_keys = file["TX_Agent"].keys()
            log.info(
                f"Found active imaging data with {len(active_keys)} events. "
                "Will convert and save all parameters for each event separately."
            )
            num_events = set(file["TX_Agent"][key].shape[-1] for key in active_keys)
            assert len(num_events) == 1, (
                "All TX_Agent entries should have the same number of events."
            )
            num_events = num_events.pop()

            # loop over TX_Agent entries and overwrite TX each time
            data = {}
            for event in range(num_events):
                data[event] = read_verasonics_file(
                    file,
                    event=event,
                    additional_functions=additional_functions,
                )

            # convert dict of events to dict of lists
            data = {key: [data[event][key] for event in data] for key in data[0]}
            description = ["Verasonics data with multiple events"] * num_events
            # Generate the zea dataset
            generate_zea_dataset(
                path=output_path,
                **data,
                event_structure=True,
                description=description,
            )

        else:
            # Here we call al the functions to read the data from the file
            data = read_verasonics_file(
                file, additional_functions=additional_functions, frames=frames
            )

            # Generate the zea dataset
            generate_zea_dataset(path=output_path, **data, description="Verasonics data")

    log.success(f"Converted {log.yellow(input_path)} to {log.yellow(output_path)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Verasonics matlab raw files to the zea format."
        "Example usage: python zea/data/convert/matlab.py raw_file.mat output.hdf5 --frames 1-5 7"
    )
    parser.add_argument(
        "input_path",
        default=None,
        type=str,
        nargs="?",
        help="The path to a file or directory containing raw Verasonics data.",
    )

    parser.add_argument(
        "output_path",
        default=None,
        type=str,
        nargs="?",
        help="The path to the output file or directory.",
    )

    parser.add_argument(
        "--frames",
        default=["all"],
        type=str,
        nargs="+",
        help="The frames to add to the file. This can be a list of integers, a range "
        "of integers (e.g. 4-8), or 'all'.",
    )

    return parser.parse_args()


def get_answer(prompt, additional_options=None):
    """Get a yes or no answer from the user. There is also the option to provide
    additional options. In case yes or no is selected, the function returns a boolean.
    In case an additional option is selected, the function returns the selected option
    as a string.

    Args:
        prompt (str): The prompt to show the user.
        additional_options (list, optional): Additional options to show the user.
            Defaults to None.

    Returns:
        str: The user's answer.
    """
    while True:
        answer = input(prompt)
        try:
            bool_answer = strtobool(answer)
            return bool_answer
        except ValueError:
            if additional_options is not None and answer in additional_options:
                return answer
        log.warning("Invalid input.")


if __name__ == "__main__":
    args = parse_args()

    # Variable to indicate what to do with existing files.
    # Is set by the user in case these are found.
    existing_file_policy = None

    if args.input_path is None:
        log.info("Select a directory containing Verasonics matlab raw files.")
        # Create a Tkinter root window
        try:
            root = tk.Tk()
            root.withdraw()
            # Prompt the user to select a file or directory
            selected_path = filedialog.askdirectory()
        except Exception as e:
            raise ValueError(
                log.error(
                    "Failed to open a file dialog (possibly in headless state). "
                    "Please provide a path as an argument. "
                )
            ) from e
    else:
        selected_path = args.input_path

    # Exit when no path is selected
    if not selected_path:
        log.error("No path selected.")
        sys.exit()
    else:
        selected_path = Path(selected_path)

    selected_path_is_directory = os.path.isdir(selected_path)

    # Set the output path to be next to the input directory with _zea appended
    # to the name
    if args.output_path is None:
        if selected_path_is_directory:
            output_path = selected_path.parent / (Path(selected_path).name + "_zea")
        else:
            output_path = str(selected_path.with_suffix("")) + "_zea.hdf5"
            output_path = Path(output_path)
    else:
        output_path = Path(args.output_path)
        if selected_path.is_file() and output_path.suffix not in (".hdf5", ".h5"):
            log.error(
                "When converting a single file, the output path should have the .hdf5 "
                "or .h5 extension."
            )
            sys.exit()
        elif selected_path.is_dir() and output_path.is_file():
            log.error("When converting a directory, the output path should be a directory.")
            sys.exit()
        #
        if output_path.is_dir() and not selected_path_is_directory:
            output_path = output_path / (selected_path.name + "_zea.hdf5")

    log.info(f"Selected path: {log.yellow(selected_path)}")

    # Parse frames
    frames = args.frames
    if frames[0] == "all":
        frames = "all"
    else:
        indices = set()
        for frame in frames:
            if "-" in frame:
                start, end = frame.split("-")
                indices.update(range(int(start), int(end) + 1))
            else:
                indices.add(int(frame))
        frames = list(indices)
        frames.sort()
    # Do the conversion of a single file
    if not selected_path_is_directory:
        if output_path.is_file():
            answer = get_answer(
                f"File {log.yellow(output_path)} exists. Overwrite?"
                "\n\ty\t - Overwrite"
                "\n\tn\t - Skip"
                "\nAnswer: "
            )
            if answer is True:
                log.warning(f"{selected_path} exists. Deleting...")
                output_path.unlink(missing_ok=False)
            else:
                log.info("Aborting...")
                sys.exit()
        zea_from_matlab_raw(selected_path, output_path, frames=frames)
    else:
        # Continue with the rest of your code...
        for root, dirs, files in os.walk(selected_path):
            for mat_file in files:
                # Skip non-mat files
                if not mat_file.endswith(".mat"):
                    continue

                log.info(f"Found raw data file {log.yellow(mat_file)}")

                # Convert the file to a Path object
                mat_file = Path(mat_file)

                # Construct the output path
                relative_path = (Path(root) / Path(mat_file)).relative_to(selected_path)
                file_output_path = output_path / (relative_path.with_suffix(".hdf5"))

                full_path = selected_path / relative_path

                # Handle existing files
                if file_output_path.is_file():
                    if existing_file_policy is None:
                        answer = get_answer(
                            f"File {log.yellow(file_output_path)} exists. Overwrite?"
                            "\n\ty\t - Overwrite"
                            "\n\tn\t - Skip"
                            "\n\tya\t - Overwrite all existing files"
                            "\n\tna\t - Skip all existing files"
                            "\nAnswer: ",
                            additional_options=("ya", "na"),
                        )
                        if answer == "ya":
                            existing_file_policy = "overwrite"
                        elif answer == "na":
                            existing_file_policy = "skip"
                            continue

                    if existing_file_policy == "skip" or answer is False:
                        log.info("Skipping...")
                        continue

                    if existing_file_policy == "overwrite" or answer is True:
                        log.warning(f"{log.yellow(full_path)} exists. Deleting...")
                        file_output_path.unlink(missing_ok=False)

                try:
                    zea_from_matlab_raw(full_path, file_output_path, frames=frames)
                except Exception:
                    # Print error message without raising it
                    log.error(f"Failed to convert {mat_file}")
                    # Print stacktrace
                    traceback.print_exc()

                    continue
