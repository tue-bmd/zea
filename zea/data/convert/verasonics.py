"""Functionality to convert Verasonics MATLAB workspace to the zea format.

Example of saving the entire workspace to a .mat file (MATLAB):

    .. code-block:: matlab

        >> setup_script;
        >> VSX;
        >> save_raw('C:/path/to/raw_data.mat');

Then convert the saved `raw_data.mat` file to zea format using the following code (Python):

    .. code-block:: python

        from zea.data.convert.verasonics import VerasonicsFile

        VerasonicsFile("C:/path/to/raw_data.mat").to_zea("C:/path/to/output.hdf5")

Or alternatively, use the script below to convert all .mat files in a directory:

    .. code-block:: bash

        python zea/data/convert/verasonics.py "C:/path/to/directory"

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
``to_zea`` method as a list.

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


    VerasonicsFile("C:/path/to/raw_data.mat").to_zea(
        "C:/path/to/output.hdf5",
        [read_high_voltage_func],
    )
"""  # noqa: E501

import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from keras import ops

from zea import log
from zea.data.data_format import DatasetElement, generate_zea_dataset
from zea.func import log_compress, normalize
from zea.internal.device import init_device
from zea.utils import strtobool

_VERASONICS_TO_ZEA_PROBE_NAMES = {
    "L11-4v": "verasonics_l11_4v",
    "L11-5v": "verasonics_l11_5v",
}


class VerasonicsFile(h5py.File):
    """HDF5 File class for Verasonics MATLAB workspace files.

    This class extends the h5py.File class to handle Verasonics-specific
    data structures and conventions.
    """

    def dereference_index(self, dataset, index, event=None, subindex=None):
        """Get the element at the given index from the dataset, dereferencing it if
        necessary.

        MATLAB stores items in struct array differently depending on the size. If the size
        is 1, the item is stored as a regular dataset. If the size is larger, the item is
        stored as a dataset of references to the actual data.

        This function dereferences the dataset if it is a reference. Otherwise, it returns
        the dataset.

        Args:
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
                return self[reference][:]
            else:
                return self[reference][subindex]
        else:
            return dataset

    @staticmethod
    def get_reference_size(dataset):
        """Get the size of a reference dataset."""
        if isinstance(dataset.fillvalue, h5py.h5r.Reference):
            return len(dataset)
        else:
            return 1

    @staticmethod
    def decode_string(dataset):
        """Decode a string dataset."""
        return "".join([chr(c) for c in dataset.squeeze()])

    @property
    def probe_unit(self):
        """The unit the probe dimensions are defined in."""
        _ALLOWED_UNITS = {"wavelengths", "mm"}
        unit = self.decode_string(self["Trans"]["units"][:])
        assert unit in {"wavelengths", "mm"}, (
            f"Unexpected unit '{unit}' in file, must be one of {_ALLOWED_UNITS}"
        )
        return unit

    @property
    def probe_geometry(self):
        """The probe geometry of shape (n_el, 3)."""
        # Read the probe geometry from the file
        probe_geometry = self["Trans"]["ElementPos"][:3, :]

        # Transpose the probe geometry to have the shape (n_el, 3)
        probe_geometry = probe_geometry.T

        # Convert the probe geometry to meters
        if self.probe_unit == "mm":
            probe_geometry = probe_geometry / 1000
        else:
            probe_geometry = probe_geometry * self.wavelength

        return probe_geometry

    @property
    def wavelength(self):
        """Wavelength of the probe from the file in meters."""

        return self.sound_speed / self.center_frequency

    def read_transmit_events(self, event=None, frames="all", allow_accumulate=False):
        """Read the events from the file and finds the order in which transmits and receives
        appear in the events.

        Args:
            event (int, optional): The event index. Defaults to None.
            frames (str or list, optional): The frames to read. Defaults to "all".
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.

        Returns:
            tuple: (tx_order, rcv_order, time_to_next_acq)
                tx_order (list): The order in which the transmits appear in the events.
                rcv_order (list): The order in which the receives appear in the events.
                time_to_next_acq (np.ndarray): The time to next acquisition of shape (n_acq, n_tx).
        """

        num_events = self["Event"]["info"].shape[0]

        # In the Verasonics the transmits may not be in order in the TX structure and a
        # transmit might be reused. Therefore, we need to keep track of the order in which
        # the transmits appear in the Events.
        tx_order = []
        rcv_order = []
        time_to_next_acq = []
        modes = []

        frame_indices = self.get_frame_indices(frames)

        for i in range(num_events):
            # Get the tx
            event_tx = self.dereference_index(self["Event"]["tx"], i)
            event_tx = int(event_tx.item())

            # Get the rcv
            event_rcv = self.dereference_index(self["Event"]["rcv"], i)
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

            # Read mode
            mode = self.dereference_index(self["Receive"]["mode"], event_rcv)
            mode = int(mode.item())

            # Check in the Receive structure if this is still the first frame
            framenum_ref = self["Receive"]["framenum"][event_rcv, 0]
            framenum = self[framenum_ref][:].item()

            # Only add the event to the list if it is the first frame since we assume
            # that all frames have the same transmits and receives
            if framenum == 1:
                # Add the event to the list
                tx_order.append(event_tx)
                rcv_order.append(event_rcv)
                modes.append(mode)

            # Read the time_to_next_acq
            seq_control_indices = self.dereference_index(self["Event"]["seqControl"], i)

            for seq_control_index in seq_control_indices:
                seq_control_index = int(seq_control_index.item() - 1)
                seq_control = self.dereference_index(
                    self["SeqControl"]["command"], seq_control_index
                )
                # Decode the seq_control int array into a string
                seq_control = self.decode_string(seq_control)
                if seq_control == "timeToNextAcq":
                    value = self.dereference_index(
                        self["SeqControl"]["argument"], seq_control_index
                    ).item()
                    value = value * 1e-6
                    time_to_next_acq.append(value)

        modes = np.array(modes)
        tx_order = np.array(tx_order)
        rcv_order = np.array(rcv_order)
        time_to_next_acq = np.array(time_to_next_acq)
        time_to_next_acq = np.reshape(time_to_next_acq, (-1, tx_order.size))

        if np.any(modes == 1) and not allow_accumulate:
            raise ValueError(
                "Some receive events are in accumulate mode (mode=1). "
                "This indicates that the data is already accumulated on the Verasonics system. "
                "Set allow_accumulate=True to allow this."
            )
        elif np.any(modes == 1) and allow_accumulate:
            # We only keep the transmits that are in mode 0 (normal acquisition)
            log.info(
                "Data contains both receives in accumulate mode and replace mode.\n"
                "Discarding transmits in accumulate mode (mode=1). "
                "Keeping transmits in replace mode (mode=0)."
            )
            tx_order = tx_order[modes == 0]
            rcv_order = rcv_order[modes == 0]
            time_to_next_acq = time_to_next_acq[:, modes == 0]

        if event is not None:
            time_to_next_acq = time_to_next_acq[event]
            time_to_next_acq = np.expand_dims(time_to_next_acq, axis=0)

        time_to_next_acq = time_to_next_acq[frame_indices]

        return tx_order, rcv_order, time_to_next_acq

    def read_t0_delays_apod(self, tx_order, event=None):
        """
        Read the t0 delays and apodization from the file.

        Returns:
            t0_delays (np.ndarray): The t0 delays of shape (n_tx, n_el).
            apod (np.ndarray): The apodization of shape (n_el,).
        """

        t0_delays_list = []
        tx_apodizations_list = []

        for n in tx_order:
            # Get column vector of t0_delays
            if event is None:
                t0_delays = self.dereference_index(self["TX"]["Delay"], n)
            else:
                t0_delays = self.dereference_index(self["TX_Agent"]["Delay"], n, event)
            # Turn into 1d array
            t0_delays = t0_delays[:, 0]

            t0_delays_list.append(t0_delays)

            # Get column vector of apodizations
            if event is None:
                tx_apodizations = self.dereference_index(self["TX"]["Apod"], n)
            else:
                tx_apodizations = self.dereference_index(self["TX_Agent"]["Apod"], n, event)
            # Turn into 1d array
            tx_apodizations = tx_apodizations[:, 0]
            tx_apodizations_list.append(tx_apodizations)

        t0_delays = np.stack(t0_delays_list, axis=0)
        apodizations = np.stack(tx_apodizations_list, axis=0)

        # Convert the t0_delays to meters
        t0_delays = t0_delays * self.wavelength / self.sound_speed

        return t0_delays, apodizations

    @property
    def sampling_frequency(self):
        """The sampling frequency in Hz from the file."""
        # Read the sampling frequency from the file
        adc_rate = self.dereference_index(self["Receive"]["decimSampleRate"], 0)

        if "quadDecim" in self["Receive"]:
            quaddecim = self.dereference_index(self["Receive"]["quadDecim"], 0)
        else:
            # TODO: Verify if this is correct.
            # On the Vantage NXT the quadDecim field is missing. It seems that it should be
            # set to 1.0 (that decimSampleRate is the actual sampling frequency).
            quaddecim = 1.0

        sampling_frequency = adc_rate / quaddecim * 1e6
        sampling_frequency = sampling_frequency.item()

        if self.is_baseband_mode:
            # Two sequential samples are interpreted as a single complex sample
            # Therefore, we need to halve the sampling frequency
            sampling_frequency = sampling_frequency / 2

        return sampling_frequency

    def read_waveforms(self, tx_order, event=None):
        """
        Read the waveforms from the file.

        Returns:
            waveforms (np.ndarray): The waveforms of shape (n_tx, n_samples).
        """
        waveforms_one_way_list = []
        waveforms_two_way_list = []

        # Read all the waveforms from the file
        n_waveforms = self.get_reference_size(self["TW"]["Wvfm1Wy"])
        for n in range(n_waveforms):
            # Get the row vector of the 1-way waveform
            waveform_one_way = self.dereference_index(self["TW"]["Wvfm1Wy"], n)[:]
            # Turn into 1d array
            waveform_one_way = waveform_one_way[0, :]

            # Get the row vector of the 2-way waveform
            waveform_two_way = self.dereference_index(self["TW"]["Wvfm2Wy"], n)[:]
            # Turn into 1d array
            waveform_two_way = waveform_two_way[0, :]

            waveforms_one_way_list.append(waveform_one_way)
            waveforms_two_way_list.append(waveform_two_way)

        tx_waveform_indices = []

        for n in tx_order:
            # Read the waveform
            if event is None:
                waveform_index = self.dereference_index(self["TX"]["waveform"], n)[:]
            else:
                waveform_index = self.dereference_index(self["TX_Agent"]["waveform"], n, event)[:]
            # Subtract one to make the indices 0-based
            waveform_index -= 1
            # Turn into integer
            waveform_index = int(waveform_index.item())
            tx_waveform_indices.append(waveform_index)

        return tx_waveform_indices, waveforms_one_way_list, waveforms_two_way_list

    def read_beamsteering_angles(self, tx_order, event=None):
        """Beam steering angles in radians (theta, alpha) for each transmit.

        Returns:
            angles (np.ndarray): The beam steering angles of shape (n_tx, 2).
        """
        angles_list = []

        for n in tx_order:
            # Read the polar angle
            if event is None:
                angle = self.dereference_index(self["TX"]["Steer"], n)[:]
            else:
                angle = self.dereference_index(self["TX_Agent"]["Steer"], n, event)[:]

            angles_list.append(angle)
        angles = np.stack(angles_list, axis=0)
        angles = np.squeeze(angles, axis=-1)

        assert angles.shape == (len(tx_order), 2), (
            f"Expected angles shape to be {(len(tx_order), 2)}, but got {angles.shape}"
        )
        return angles

    def read_polar_angles(self, tx_order, event=None):
        """Read the polar angles  of shape (n_tx,) from the file."""
        return self.read_beamsteering_angles(tx_order, event)[:, 0]

    def read_azimuth_angles(self, tx_order, event=None):
        """Read the azimuth angles of shape (n_tx,) from the file."""
        return self.read_beamsteering_angles(tx_order, event)[:, 1]

    @property
    def end_samples(self):
        """The index of the last sample for each receive event."""
        length = self["Receive"]["endSample"].shape[0]
        return np.concatenate(
            [self.dereference_index(self["Receive"]["endSample"], i) for i in range(length)]
        )

    @property
    def start_samples(self):
        """The index of the first sample for each receive event."""
        length = self["Receive"]["startSample"].shape[0]
        return np.concatenate(
            [self.dereference_index(self["Receive"]["startSample"], i) for i in range(length)]
        )

    @property
    def n_ax(self):
        """Number of axial samples."""
        n_ax = (self.end_samples - self.start_samples + 1).astype(np.int32)
        n_ax = np.unique(n_ax)
        if n_ax.size != 1:
            raise ValueError(
                "The number of axial samples is not the same for all receive events."
                "We do not support this case yet."
            )
        return n_ax.item()

    @property
    def probe_connector(self):
        """Probe connector indices."""
        probe_connector = self["Trans"]["ConnectorES"][:]
        probe_connector = np.squeeze(probe_connector, axis=0)
        probe_connector = probe_connector.astype(np.int32)
        probe_connector = probe_connector - 1  # make 0-based
        return probe_connector

    @property
    def is_new_save_raw_format(self):
        return "save_raw_version" in self.keys()

    def get_image_raw_data_order(self, raw_data: np.ndarray):
        """The order of frames in the RcvBuffer buffer.

        Because of the circular buffer used in Verasonics, the frames in the RcvBuffer
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.
        """
        n_frames = raw_data.shape[0]
        try:
            last_frame = int(
                self.dereference_index(self["Resource"]["RcvBuffer"]["lastFrame"], 0)[()].item() - 1
            )
        except KeyError:
            log.warning(
                "Could not find 'lastFrame' in 'Resource/RcvBuffer'. "
                "Assuming data is already in correct order."
            )
            return np.arange(n_frames)
        first_frame = (last_frame + 1) % n_frames
        indices = np.arange(first_frame, first_frame + n_frames) % n_frames
        return indices

    def read_raw_data(self, event=None, frames="all"):
        """
        Read the raw data from the file.

        Returns:
            raw_data (np.ndarray): The raw data of shape (n_rcv, n_samples).
        """

        # Read the raw data from the file
        if event is None:
            raw_data = self.dereference_index(self["RcvData"], 0)
        else:
            # for now we only index frames as events
            raw_data = self.dereference_index(self["RcvData"], 0, subindex=event)
            raw_data = np.expand_dims(raw_data, axis=0)

        # Convert the raw data to a numpy array to allow out-of-order indexing later
        raw_data = np.array(raw_data, dtype=np.int16)

        # Reorder and select channels based on probe elements
        if self.is_new_save_raw_format:
            raw_data = raw_data[:, self.probe_connector, :]
        else:
            log.warning(
                "Data was not saved using the updated `save_raw` function (version >= 1.0). "
                "In that case, we assume that the channel order in the data matches the "
                "probe element order. Please verify that this is correct!"
            )

        # Re-order frames such that sequence is correct
        indices = self.get_image_raw_data_order(raw_data)
        raw_data = raw_data[indices]

        # Select only the requested frames
        frame_indices = self.get_frame_indices(frames)
        raw_data = raw_data[frame_indices]

        # Trim the raw data to the final sample in the buffer
        final_sample_in_buffer = int(self.end_samples.max())
        raw_data = raw_data[:, :, :final_sample_in_buffer]

        # Determine n_tx based on the final sample in buffer and n_ax
        # For some sequences, transmits are already aggregated in the raw data
        # (e.g. harmonic imaging through pulse inversion)
        n_tx = final_sample_in_buffer // self.n_ax

        # Reshape the raw data to (n_frames, n_el, n_tx, n_ax)
        raw_data = raw_data.reshape((raw_data.shape[0], raw_data.shape[1], n_tx, self.n_ax))

        # Transpose the raw data to (n_frames, n_tx, n_ax, n_el)
        raw_data = np.transpose(raw_data, (0, 2, 3, 1))

        # Add channel dimension
        raw_data = raw_data[..., None]

        # If the data is captured in BS100BW mode or BS50BW mode, the data is stored in
        # as complex IQ data.
        if self.is_baseband_mode:
            raw_data = np.concatenate(
                (
                    raw_data[:, :, 0::2, :, :],
                    -raw_data[:, :, 1::2, :, :],
                ),
                axis=-1,
            )

        return raw_data

    @property
    def center_frequency(self):
        """Center frequency of the probe from the file in Hz."""

        return self["Trans"]["frequency"][0, 0] * 1e6

    @property
    def sound_speed(self):
        """Speed of sound in the medium in m/s."""

        return self["Resource"]["Parameters"]["speedOfSound"][0, 0].item()

    def read_initial_times(self, rcv_order):
        """Reads the initial times from the file.

        Args:
            rcv_order (list): The order in which the receives appear in the events.
            wavelength (float): The wavelength of the probe.

        Returns:
            initial_times (np.ndarray): The initial times of shape (n_rcv,).
        """
        initial_times = []
        for n in rcv_order:
            start_depth = self.dereference_index(self["Receive"]["startDepth"], n).item()

            initial_times.append(2 * start_depth * self.wavelength / self.sound_speed)

        return np.array(initial_times).astype(np.float32)

    @property
    def probe_name(self):
        """The name of the probe from the file."""
        probe_name = self["Trans"]["name"][:]
        probe_name = self.decode_string(probe_name)
        # Translates between verasonics probe names and zea probe names
        if probe_name in _VERASONICS_TO_ZEA_PROBE_NAMES:
            probe_name = _VERASONICS_TO_ZEA_PROBE_NAMES[probe_name]
        else:
            log.warning(
                f"Probe name '{probe_name}' is not in the list of known probes. "
                "Please add it to the _VERASONICS_TO_ZEA_PROBE_NAMES dictionary. "
                "Falling back to generic probe."
            )
            probe_name = "generic"

        return probe_name

    def read_focus_distances(self, tx_order, event=None):
        """Reads the focus distances from the file.

        Args:
            tx_order (list): The order in which the transmits appear in the events.

        Returns:
            focus_distances (list): The focus distances of shape (n_tx,) in meters.
        """
        focus_distances = []
        for n in tx_order:
            if event is None:
                focus_distance = self.dereference_index(self["TX"]["focus"], n)[0, 0]
            else:
                focus_distance = self.dereference_index(self["TX_Agent"]["focus"], n, event)[0, 0]
            focus_distances.append(focus_distance)

        # Convert focus distances from wavelengths to meters
        focus_distances = np.array(focus_distances) * self.wavelength

        return focus_distances

    @property
    def _probe_geometry_is_ordered_ula(self):
        """Checks if the probe geometry is ordered as a uniform linear array (ULA)."""
        diff_vec = self.probe_geometry[1:] - self.probe_geometry[:-1]
        return np.isclose(diff_vec, diff_vec[0]).all()

    def planewave_focal_distance_to_inf(self, focus_distances, t0_delays, tx_apodizations):
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
        if not self._probe_geometry_is_ordered_ula:
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

    @property
    def bandwidth_percent(self):
        """Receive bandwidth as a percentage of center frequency."""
        bandwidth_percent = self.dereference_index(self["Receive"]["sampleMode"], 0)
        bandwidth_percent = self.decode_string(bandwidth_percent)
        bandwidth_percent = int(bandwidth_percent[2:-2])
        return bandwidth_percent

    @property
    def is_baseband_mode(self):
        """If the data is captured in 'BS100BW' mode or 'BS50BW' mode.

        - The data is stored as complex IQ data.
        - The sampling frequency is halved.
        - Two sequential samples are interpreted as a single complex sample.
          Therefore, we need to halve the sampling frequency.
        """
        return self.bandwidth_percent in (50, 100)

    @property
    def lens_correction(self):
        """The lens correction: 1 way delay in wavelengths thru lens"""
        lens_correction = self["Trans"]["lensCorrection"][0, 0].item()
        return lens_correction

    @property
    def tgc_gain_curve(self):
        """The TGC gain curve from the file interpolated to the number of axial samples (n_ax,)."""

        gain_curve = self["TGC"]["Waveform"][:][:, 0]

        # Normalize the gain_curve to [0, 40]dB
        gain_curve = gain_curve / 1023 * 40

        # The gain curve is sampled at 800ns (See Verasonics documentation for details.
        # Specifically the tutorial sequence programming)
        gain_curve_sampling_period = 800e-9

        # Define the time axis for the gain curve
        t_gain_curve = np.arange(gain_curve.size) * gain_curve_sampling_period

        # Define the time axis for the axial samples
        t_samples = np.arange(self.n_ax) / self.sampling_frequency

        # Interpolate the gain_curve to the number of axial samples
        gain_curve = np.interp(t_samples, t_gain_curve, gain_curve)

        # The gain_curve gains are in dB, so we need to convert them to linear scale
        gain_curve = 10 ** (gain_curve / 20)

        return gain_curve

    def get_image_data_p_frame_order(self, image_data: np.ndarray):
        """The order of frames in the ImgDataP buffer.

        Because of the circular buffer used in Verasonics, the frames in the ImgDataP
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.

        Uses the first buffer, so does not support multiple buffers.
        """
        FIRST_BUFFER = 0
        n_frames = image_data.shape[0]
        try:
            first_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["firstFrame"], FIRST_BUFFER
            )[()].item()
            last_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["lastFrame"], FIRST_BUFFER
            )[()].item()
            first_frame -= 1  # make 0-based
            last_frame -= 1  # make 0-based
            indices = np.arange(first_frame, first_frame + n_frames) % n_frames
            assert indices[-1] == last_frame, (
                "The last frame index does not match the expected last frame index."
            )
            return indices
        except KeyError:
            log.warning(
                "Could not find 'firstFrame' or 'lastFrame' in 'Resource/ImageBuffer'. "
                "Assuming data is already in correct order."
            )
            return np.arange(n_frames)

    def read_image_data_p(self, event=None, frames="all"):
        """Reads the image data from the file.

        Uses the ``ImgDataP`` buffer, which is used for spatial filtering
        and persistence processing. Generally, this buffer does not contain the same frames
        as the raw data buffer. This happens because the Verasonics often does not reconstruct
        every acquired frame. This means that the images in this buffer often skip frames, and
        span a longer time period than the raw data buffer.

        Returns:
            `image_data` (`np.ndarray`): The image data.
        """
        # Check if the file contains image data
        if "ImgDataP" not in self:
            return None

        # Get the dataset reference
        image_data_ref = self["ImgDataP"][0, 0]
        # Dereference the dataset
        if event is None:
            image_data = self[image_data_ref][:]
        else:
            image_data = self[image_data_ref][event]
            image_data = np.expand_dims(image_data, axis=0)

        # Re-order images such that sequence is correct
        indices = self.get_image_data_p_frame_order(image_data)
        image_data = image_data[indices, :, :]

        # Normalize and log-compress the image data
        image_data = normalize(image_data, output_range=(0, 1), input_range=(0, None))
        image_data = log_compress(image_data)
        image_data = ops.convert_to_numpy(image_data)

        # Select only the requested frames
        frame_indices = self.get_frame_indices(frames)
        image_data = image_data[frame_indices]

        return image_data

    @property
    def element_width(self):
        """The element width in meters from the file."""
        element_width = self["Trans"]["elementWidth"][:][0, 0]

        # Convert the probe element width to meters
        if self.probe_unit == "mm":
            element_width = element_width / 1000
        else:
            element_width = element_width * self.wavelength

        return element_width

    def read_verasonics_file(
        self, event=None, additional_functions=None, frames="all", allow_accumulate=False
    ):
        """Reads data from a .mat Verasonics output file.

        Args:
            event (int, optional): The event index. Defaults to None in this case we assume
                the data file is stored without event structure.
            additional_functions (list, optional): A list of functions that read additional
                data from the file. Each function should take the file as input and return a
                `DatasetElement`. Defaults to None.
        """

        if additional_functions is None:
            additional_functions = []

        tx_order, rcv_order, time_to_next_transmit = self.read_transmit_events(
            frames=frames, allow_accumulate=allow_accumulate
        )
        initial_times = self.read_initial_times(rcv_order)

        # these are capable of handling multiple events
        raw_data = self.read_raw_data(event, frames=frames)

        verasonics_image_buffer = self.read_image_data_p(event, frames=frames)

        polar_angles = self.read_polar_angles(tx_order, event)
        azimuth_angles = self.read_azimuth_angles(tx_order, event)
        t0_delays, tx_apodizations = self.read_t0_delays_apod(tx_order, event)
        focus_distances = self.read_focus_distances(tx_order, event)

        tx_waveform_indices, waveforms_one_way_list, waveforms_two_way_list = self.read_waveforms(
            tx_order, event
        )
        focus_distances = self.planewave_focal_distance_to_inf(
            focus_distances, t0_delays, tx_apodizations
        )

        if event is None:
            group_name = "scan"
        else:
            group_name = f"event_{event}/scan"

        el_lens_correction = DatasetElement(
            group_name=group_name,
            dataset_name="lens_correction",
            data=self.lens_correction,
            description=(
                "The lens correction value used by Verasonics. This value is the "
                "additional path length in wavelength that the lens introduces. "
                "(This disregards refraction.)"
            ),
            unit="wavelengths",
        )

        verasonics_image_buffer = DatasetElement(
            dataset_name="verasonics_image_buffer",
            data=verasonics_image_buffer,
            description=(
                "The Verasonics ImgDataP buffer. "
                "WARNING: This buffer may skip frames compared to the raw data! "
                "Use only for reference."
            ),
            unit="unitless",
        )

        additional_elements = []
        for additional_function in additional_functions:
            additional_elements.append(additional_function(self))

        data = {
            "probe_geometry": self.probe_geometry,
            "time_to_next_transmit": time_to_next_transmit,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "sampling_frequency": self.sampling_frequency,
            "polar_angles": polar_angles,
            "azimuth_angles": azimuth_angles,
            "bandwidth_percent": self.bandwidth_percent,
            "raw_data": raw_data,
            "center_frequency": self.center_frequency,
            "sound_speed": self.sound_speed,
            "initial_times": initial_times,
            "probe_name": self.probe_name,
            "focus_distances": focus_distances,
            "tx_waveform_indices": tx_waveform_indices,
            "waveforms_one_way": waveforms_one_way_list,
            "waveforms_two_way": waveforms_two_way_list,
            "tgc_gain_curve": self.tgc_gain_curve,
            "element_width": self.element_width,
            "additional_elements": [
                el_lens_correction,
                verasonics_image_buffer,
                *additional_elements,
            ],
        }

        return data

    def get_frame_indices(self, frames):
        """Creates a numpy array of frame indices from the file and the frames argument.

        Args:
            frames (str): The frames argument. This can be "all" or a list of frame indices.

        Returns:
            frame_indices (np.ndarray): The frame indices.
        """
        # Read the number of frames from the file
        n_frames = int(self.dereference_index(self["Resource"]["RcvBuffer"]["numFrames"], 0)[0][0])

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

    def to_zea(
        self,
        output_path,
        additional_functions=None,
        frames="all",
        allow_accumulate=False,
        enable_compression=True,
    ):
        """Converts the Verasonics file to the zea format.

        Args:
            output_path (str): The path to the output file (.hdf5 file).
            additional_functions (list, optional): A list of functions that read additional
                data from the file. Each function should take the file as input and return a
                `DatasetElement`. Defaults to None.
            frames (str or list of int, optional): The frames to add to the file. This can be
                a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
                'all'.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
                Defaults to False.
            enable_compression (bool, optional): Whether to enable compression when saving
                the zea file. Defaults to True.
        """
        if "TX_Agent" in self:
            active_keys = self["TX_Agent"].keys()
            log.info(
                f"Found active imaging data with {len(active_keys)} events. "
                "Will convert and save all parameters for each event separately."
            )
            num_events = set(self["TX_Agent"][key].shape[-1] for key in active_keys)
            assert len(num_events) == 1, (
                "All TX_Agent entries should have the same number of events."
            )
            num_events = num_events.pop()

            # loop over TX_Agent entries and overwrite TX each time
            data = {}
            for event in range(num_events):
                data[event] = self.read_verasonics_file(
                    event=event,
                    additional_functions=additional_functions,
                    allow_accumulate=allow_accumulate,
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
                enable_compression=enable_compression,
            )

        else:
            # Here we call al the functions to read the data from the file
            log.info("Reading Verasonics file...")
            data = self.read_verasonics_file(
                additional_functions=additional_functions,
                frames=frames,
                allow_accumulate=allow_accumulate,
            )

            # Generate the zea dataset
            log.info("Generating zea dataset...")
            generate_zea_dataset(
                path=output_path,
                **data,
                description="Verasonics data",
                enable_compression=enable_compression,
            )


def _zea_from_verasonics_workspace(input_path, output_path, **kwargs):
    """Helper function around ``VerasonicsFile.to_zea``"""

    # Create the output directory if it does not exist
    input_path = Path(input_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    assert input_path.is_file(), log.error(f"Input file {log.yellow(input_path)} does not exist.")

    # Load the data
    with VerasonicsFile(input_path, "r") as file:
        file.to_zea(output_path, **kwargs)

    log.success(f"Converted {log.yellow(input_path)} to {log.yellow(output_path)}")


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


def convert_verasonics(args):
    """
    Converts a Verasonics MATLAB workspace file (.mat) or a directory containing multiple
    such files to the zea format.

    Args:
        args (argparse.Namespace): An object with attributes:

            - src (str): Source folder path.
            - dst (str): Destination folder path.
            - frames (list[str]): MATLAB frames spec (e.g., ["all"], integers, or ranges like "4-8")
            - allow_accumulate (bool): Whether to allow accumulate mode.
            - device (str): Device to use for processing.
    """

    init_device(args.device)

    # Variable to indicate what to do with existing files.
    # Is set by the user in case these are found.
    existing_file_policy = None

    if args.src is None:
        log.info("Select a directory containing Verasonics MATLAB workspace files.")
        # Create a Tkinter root window
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            # Prompt the user to select a file or directory
            selected_path = filedialog.askdirectory()
        except ImportError as e:
            raise ImportError(
                log.error(
                    "tkinter is not installed. Please install it with 'apt install python3-tk'."
                )
            ) from e
        except Exception as e:
            raise ValueError(
                log.error(
                    "Failed to open a file dialog (possibly in headless state). "
                    "Please provide a path as an argument. "
                )
            ) from e
    else:
        selected_path = args.src

    # Exit when no path is selected
    if not selected_path:
        log.error("No path selected.")
        sys.exit()
    else:
        selected_path = Path(selected_path)

    selected_path_is_directory = os.path.isdir(selected_path)

    # Set the output path to be next to the input directory with _zea appended
    # to the name
    if args.dst is None:
        if selected_path_is_directory:
            output_path = selected_path.parent / (Path(selected_path).name + "_zea")
        else:
            output_path = str(selected_path.with_suffix("")) + "_zea.hdf5"
            output_path = Path(output_path)
    else:
        output_path = Path(args.dst)
        if selected_path.is_file() and output_path.suffix not in (".hdf5", ".h5"):
            log.error(
                "When converting a single file, the output path should have the .hdf5 "
                "or .h5 extension."
            )
            sys.exit()
        elif selected_path.is_dir() and output_path.is_file():
            log.error("When converting a directory, the output path should be a directory.")
            sys.exit()

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
        _zea_from_verasonics_workspace(
            selected_path,
            output_path,
            frames=frames,
            allow_accumulate=args.allow_accumulate,
            enable_compression=not args.no_compression,
        )
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
                        log.warning(f"{log.yellow(file_output_path)} exists. Deleting...")
                        file_output_path.unlink(missing_ok=False)

                try:
                    _zea_from_verasonics_workspace(
                        full_path,
                        file_output_path,
                        frames=frames,
                        allow_accumulate=args.allow_accumulate,
                        enable_compression=not args.no_compression,
                    )
                except Exception:
                    # Print error message without raising it
                    log.error(f"Failed to convert {mat_file}")
                    # Print stacktrace
                    traceback.print_exc()

                    continue
