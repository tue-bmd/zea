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

---------------

By default the zea dataformat saves all the data to an hdf5 file with the following structure:

.. code-block:: text

    regular_zea_dataset.hdf5
    ├── data
    └── scan
          └── center_frequency: 1MHz

The data is stored in the ``data`` group and the scan parameters are stored in the ``scan``.
"""  # noqa: E501

import os
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import yaml
from keras import ops
from schema import And, Optional, Or, Regex, Schema

from zea import log
from zea.data.convert.utils import (
    require_output_dir_ownership,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.data_format import DatasetElement
from zea.data.file import File
from zea.data.spec import DEFAULT_COMPRESSION
from zea.func import log_compress, normalize
from zea.internal.device import init_device
from zea.utils import strtobool

_VERASONICS_TO_ZEA_PROBE_NAMES = {
    "L11-4v": "verasonics_l11_4v",
    "L11-5v": "verasonics_l11_5v",
}


_CONVERT_YAML_SCHEMA = Schema(
    {
        "files": [
            {
                "name": str,
                Optional("first_frame"): And(int, lambda x: x >= 0),
                Optional("frames"): Or(
                    "all",
                    And(str, Regex(r"^\d+(-\d+)?$")),  # Matches "30-99" or single number like "5"
                    [And(int, lambda x: x >= 0)],  # List of non-negative integers
                ),
                Optional("transmits"): Or("all", [And(int, lambda x: x >= 0)]),
            }
        ]
    }
)


class VerasonicsFile(h5py.File):
    """HDF5 File class for Verasonics MATLAB workspace files.

    This class extends the h5py.File class to handle Verasonics-specific
    data structures and conventions.
    """

    def dereference_index(self, dataset, index):
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
        """
        if isinstance(dataset.fillvalue, h5py.h5r.Reference):
            reference = dataset[index].item()
            return self[reference][:]
        else:
            return dataset

    def dereference_all(self, dataset, func=None):
        """Dereference all elements in a dataset.

        Args:
            dataset (h5py.Dataset): The dataset to dereference.
            func (callable, optional): A function to apply to each dereferenced element.

        Returns:
            list: The dereferenced data.
        """
        size = self.get_reference_size(dataset)
        dereferenced_data = []
        for i in range(size):
            element = self.dereference_index(dataset, i)
            element = func(element) if func is not None else element
            dereferenced_data.append(element)
        return dereferenced_data

    @staticmethod
    def get_reference_size(dataset):
        """Get the size of a reference dataset."""
        if isinstance(dataset.fillvalue, h5py.h5r.Reference):
            return len(dataset)
        else:
            return 1

    @staticmethod
    def decode_string(dataset: np.ndarray) -> str:
        """Decode a string dataset."""
        return "".join([chr(c) for c in dataset.squeeze()])

    @staticmethod
    def cast_to_integer(dataset):
        """Cast a h5py dataset to an integer."""
        return int(dataset[:].item())

    @property
    def wavelength(self):
        """Wavelength of the probe from the file in meters."""

        return self.sound_speed / self.probe.center_frequency

    def read_transmit_events(self, frames="all", allow_accumulate=False, buffer_index=0):
        """Read the events from the file and finds the order in which transmits and receives
        appear in the events.

        Args:
            frames (str or list, optional): The frames to read. Defaults to "all".
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
            buffer_index (int, optional): The buffer index to read from. Defaults to 0.

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

        frame_indices = self.get_frame_indices(frames, buffer_index)

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
            framenum = self.dereference_index(self["Receive"]["framenum"], event_rcv)
            framenum = self.cast_to_integer(framenum)

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

        modes = np.stack(modes)
        tx_order = np.stack(tx_order)
        rcv_order = np.stack(rcv_order)
        time_to_next_acq = np.stack(time_to_next_acq)
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

            log.info("Dropping time to next acquisition for accumulate mode transmits.")
            time_to_next_acq = None

        if time_to_next_acq is not None:
            time_to_next_acq = time_to_next_acq[frame_indices]

        return tx_order, rcv_order, time_to_next_acq

    def read_t0_delays_apod(self, tx_order):
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
            t0_delays = self.dereference_index(self["TX"]["Delay"], n)
            # Turn into 1d array
            t0_delays = t0_delays[:, 0]

            t0_delays_list.append(t0_delays)

            # Get column vector of apodizations
            tx_apodizations = self.dereference_index(self["TX"]["Apod"], n)
            # Turn into 1d array
            tx_apodizations = tx_apodizations[:, 0]
            tx_apodizations_list.append(tx_apodizations)

        t0_delays = np.stack(t0_delays_list, axis=0)
        apodizations = np.stack(tx_apodizations_list, axis=0)

        # Convert the t0_delays to seconds
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

    def read_tx_waveform_indices(self, tx_order):
        tx_waveform_indices = []
        for n in tx_order:
            # Read the waveform
            waveform_index = self.dereference_index(self["TX"]["waveform"], n)[:]
            # Subtract one to make the indices 0-based
            waveform_index -= 1
            # Turn into integer
            waveform_index = int(waveform_index.item())
            tx_waveform_indices.append(waveform_index)
        return tx_waveform_indices

    def read_waveforms(self):
        """Read the waveforms from the file."""
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

        return waveforms_one_way_list, waveforms_two_way_list

    def read_beamsteering_angles(self, tx_order):
        """Beam steering angles in radians (theta, alpha) for each transmit.

        Returns:
            angles (np.ndarray): The beam steering angles of shape (n_tx, 2).
        """
        angles_list = []

        for n in tx_order:
            # Read the polar angle
            angle = self.dereference_index(self["TX"]["Steer"], n)[:]

            angles_list.append(angle)
        angles = np.stack(angles_list, axis=0)
        angles = np.squeeze(angles, axis=-1)

        assert angles.shape == (len(tx_order), 2), (
            f"Expected angles shape to be {(len(tx_order), 2)}, but got {angles.shape}"
        )
        return angles

    def read_polar_angles(self, tx_order):
        """Read the polar angles  of shape (n_tx,) from the file."""
        return self.read_beamsteering_angles(tx_order)[:, 0]

    def read_azimuth_angles(self, tx_order):
        """Read the azimuth angles of shape (n_tx,) from the file."""
        return self.read_beamsteering_angles(tx_order)[:, 1]

    @property
    def end_samples(self):
        """The index of the last sample for each receive event."""
        return np.concatenate(self.dereference_all(self["Receive"]["endSample"])).squeeze()

    @property
    def start_samples(self):
        """The index of the first sample for each receive event."""
        return np.concatenate(self.dereference_all(self["Receive"]["startSample"])).squeeze()

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
    def is_new_save_raw_format(self):
        return "save_raw_version" in self.keys()

    def load_convert_config(self):
        """
        Can load additional conversion configuration from a `convert.yaml` file.

        The `convert.yaml` file should be in the same directory as the .mat file and have
        the following structure:

        .. code-block:: yaml

            files:
            - name: raw_data.mat
              first_frame: 26  # 0-based indexing
              frames: 30-99  # 0-based indexing

        If ``first_frame`` is provided, it will reorder the frames first and use
        the ``frames`` key to subsample afterwards.

        In the example ``frames: 30-99`` means frames 30 to 99 inclusive.

        Returns:
            dict: The configuration for the current file, or an empty dict if no
                configuration is found.
        """
        path = Path(self.filename)
        config_file = path.parent / "convert.yaml"
        if config_file.exists():
            log.info(f"Found convert config file: {log.yellow(config_file)}")
            with open(config_file, "r", encoding="utf-8") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)

            # Validate the YAML structure
            validated_data = _CONVERT_YAML_SCHEMA.validate(data)

            files = validated_data["files"]
            filenames = [file["name"] for file in files]
            if path.name in filenames:
                return files[filenames.index(path.name)]
            elif path.stem in filenames:
                return files[filenames.index(path.stem)]
        return {}

    def get_frame_count(self, buffer_index=0):
        """Get the total number of frames in the RcvBuffer buffer."""
        n_frames = self.dereference_index(self["Resource"]["RcvBuffer"]["numFrames"], buffer_index)
        n_frames = self.cast_to_integer(n_frames)
        return n_frames

    def get_indices_to_reorder(self, first_frame: int, n_frames: int):
        return (np.arange(n_frames) + first_frame) % n_frames

    def get_raw_data_order(self, buffer_index=0):
        """The order of frames in the RcvBuffer buffer.

        Because of the circular buffer used in Verasonics, the frames in the RcvBuffer
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.
        """
        n_frames = self.get_frame_count(buffer_index)
        try:
            last_frame = self.dereference_index(
                self["Resource"]["RcvBuffer"]["lastFrame"], buffer_index
            )
            last_frame = self.cast_to_integer(last_frame) - 1
            first_frame = (last_frame + 1) % n_frames
        except KeyError:
            log.warning(
                "Could not find 'lastFrame' in 'Resource/RcvBuffer'. "
                "Assuming data is already in correct order."
            )
            return np.arange(n_frames)

        return self.get_indices_to_reorder(first_frame, n_frames)

    def read_raw_data(self, frames="all", buffer_index=0, first_frame_idx=None):
        """
        Read the raw data from the file.

        Returns:
            raw_data (np.ndarray): The raw data of shape (n_rcv, n_samples).
        """

        # Read the raw data from the file
        raw_data = self.dereference_index(self["RcvData"], buffer_index)

        # Convert the raw data to a numpy array to allow out-of-order indexing later
        raw_data = np.asarray(raw_data, dtype=np.int16)

        # Add n_frames dimension if it is missing
        if raw_data.ndim == 2:
            raw_data = np.expand_dims(raw_data, axis=0)

        assert raw_data.ndim == 3, (
            "Expected raw data to have 3 dimensions at this point "
            f"(n_frames, n_channels, n_samples), but got {raw_data.shape}."
        )

        # Reorder and select channels based on probe elements
        if self.is_new_save_raw_format:
            raw_data = raw_data[:, self.probe.connector, :]
        else:
            log.warning(
                "Data was not saved using the updated `save_raw` function (version >= 1.0). "
                "In that case, we assume that the channel order in the data matches the "
                "probe element order. Please verify that this is correct!"
            )

        # Re-order frames such that sequence is correct
        if first_frame_idx is not None:
            n_frames = self.get_frame_count(buffer_index)
            indices = self.get_indices_to_reorder(first_frame_idx, n_frames)
        else:
            indices = self.get_raw_data_order(buffer_index)
        raw_data = raw_data[indices]

        # Select only the requested frames
        frame_indices = self.get_frame_indices(frames, buffer_index)
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

    def read_center_frequency(self, waveform_index):
        """Center frequency of the transmit from the file in Hz."""

        tw_type = self.dereference_index(self["TW"]["type"], waveform_index)[:]
        tw_type = self.decode_string(tw_type)
        if tw_type == "parametric":
            center_freq, _, _, _ = self.dereference_index(
                self["TW"]["Parameters"],
                waveform_index,
            )[:].squeeze()
        else:
            raise ValueError(
                f"Unsupported waveform type '{tw_type}' for center frequency extraction."
            )

        return center_freq.item() * 1e6  # Convert MHz to Hz

    def read_center_frequencies(self, tx_waveform_indices):
        """Center frequencies of the transmits from the file in Hz."""

        center_frequencies = []
        for waveform_index in tx_waveform_indices:
            center_frequency = self.read_center_frequency(waveform_index)
            center_frequencies.append(center_frequency)

        return np.stack(center_frequencies)

    @property
    def demodulation_frequency(self):
        """Demodulation frequency of the probe from the file in Hz."""

        demod_freq = self.dereference_all(self["Receive"]["demodFrequency"])
        demod_freq = np.unique(demod_freq)
        assert demod_freq.size == 1, (
            f"Multiple demodulation frequencies found in file: {demod_freq}. "
            "We do not support this case."
        )

        return demod_freq.item() * 1e6

    @property
    def sound_speed(self):
        """Speed of sound in the medium in m/s."""

        return self["Resource"]["Parameters"]["speedOfSound"][:].item()

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

        return np.stack(initial_times).astype(np.float32)

    @property
    def probe(self) -> "VerasonicsProbe":
        """The probe object from the file."""
        return VerasonicsProbe(self)

    def read_focus_distances(self, tx_order):
        """Reads the focus distances from the file.

        Args:
            tx_order (list): The order in which the transmits appear in the events.

        Returns:
            focus_distances (list): The focus distances of shape (n_tx,) in meters.
        """
        focus_distances = []
        for n in tx_order:
            focus_distance = self.dereference_index(self["TX"]["focus"], n)[:].item()
            focus_distances.append(focus_distance)

        # Convert focus distances from wavelengths to meters
        focus_distances = np.stack(focus_distances) * self.wavelength

        return focus_distances

    def read_transmit_origins(self, tx_order):
        """Reads the transmit origins from the file.

        Args:
            tx_order (list): The order in which the transmits appear in the events.

        Returns:
            origins (np.ndarray): The transmit origins of shape (n_tx, 3) in meters.
        """
        origins = []
        for n in tx_order:
            origin = self.dereference_index(self["TX"]["Origin"], n)
            origins.append(origin.squeeze())

        # Convert origins from wavelengths to meters
        origins = np.stack(origins) * self.wavelength

        return origins

    def planewave_focal_distance_to_inf(self, focus_distances, t0_delays, tx_apodizations):
        """Detects plane wave transmits and sets the focus distance to infinity.

        Args:
            focus_distances (np.ndarray): The focus distances of shape (n_tx,).
            t0_delays (np.ndarray): The t0 delays of shape (n_tx, n_el).
            tx_apodizations (np.ndarray): The apodization of shape (n_tx, n_el).

        Returns:
            focus_distances (np.ndarray): The focus distances of shape (n_tx,).

        Note:
            This function assumes that the probe geometry is a 1d uniform linear array.
            If not it will warn and return.
        """
        if not self.probe._probe_geometry_is_ordered_ula:
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
    def sample_mode(self):
        """Receive bandwidth as a percentage of center frequency."""
        SUPPORTED_SAMPLE_MODES = ["NS200BW", "BS100BW", "BS67BW", "BS50BW"]

        # For all unique sample modes
        sample_mode = self.dereference_all(self["Receive"]["sampleMode"], func=self.decode_string)
        sample_mode = set(sample_mode)

        # Ensure only a single sample mode is used
        assert len(sample_mode) == 1, (
            f"Multiple sample modes found in file: {sample_mode}. We do not support this case."
        )
        sample_mode = sample_mode.pop()

        # Check if the sample mode is supported, and extract the percentage
        assert sample_mode in SUPPORTED_SAMPLE_MODES, (
            f"Unexpected sample mode '{sample_mode}' in file."
            f"Expected one of {SUPPORTED_SAMPLE_MODES}"
        )
        return int(sample_mode[2:-2])

    @property
    def is_baseband_mode(self):
        """If the data is captured in 'BS100BW' mode or 'BS50BW' mode.

        - The data is stored as complex IQ data.
        - The sampling frequency is halved.
        - Two sequential samples are interpreted as a single complex sample.
          Therefore, we need to halve the sampling frequency.
        """
        return self.sample_mode in (50, 100)

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

        # For baseband mode two consecutive samples are combined into a single complex sample
        n_ax = self.n_ax if not self.is_baseband_mode else self.n_ax // 2

        # Define the time axis for the axial samples
        t_samples = np.arange(n_ax) / self.sampling_frequency

        # Interpolate the gain_curve to the number of axial samples
        gain_curve = np.interp(t_samples, t_gain_curve, gain_curve)

        # The gain_curve gains are in dB, so we need to convert them to linear scale
        gain_curve = 10 ** (gain_curve / 20)

        return gain_curve

    def get_image_data_p_frame_order(self, buffer_index=0):
        """The order of frames in the ImgDataP buffer.

        Because of the circular buffer used in Verasonics, the frames in the ImgDataP
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.
        """
        n_frames = self.dereference_index(
            self["Resource"]["ImageBuffer"]["numFrames"], buffer_index
        )
        n_frames = self.cast_to_integer(n_frames)
        try:
            first_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["firstFrame"], buffer_index
            )
            last_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["lastFrame"], buffer_index
            )
            first_frame = self.cast_to_integer(first_frame) - 1  # make 0-based
            last_frame = self.cast_to_integer(last_frame) - 1  # make 0-based
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

    def read_image_data_p(self, frames="all", buffer_index=0):
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
        image_data_ref = self["ImgDataP"][:].squeeze()[buffer_index]
        # Dereference the dataset
        image_data = self[image_data_ref][:]

        # Re-order images such that sequence is correct
        indices = self.get_image_data_p_frame_order(buffer_index)
        image_data = image_data[indices, :, :]

        # Normalize and log-compress the image data
        image_data = normalize(image_data, output_range=(0, 1), input_range=(0, None))
        image_data = log_compress(image_data)
        image_data = ops.convert_to_numpy(image_data)

        # Select only the requested frames
        frame_indices = self.get_frame_indices(frames, buffer_index)
        image_data = image_data[frame_indices]

        return image_data

    def read_scan(self, frames=None, allow_accumulate=False, buffer_index=0) -> dict:
        """Reads all scan parameters from the file and returns them in a dictionary.

        Args:
            frames (str or list of int, optional): The frames to add to the file. This can be
                a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
                None, which means all frames, unless specified in a `convert.yaml` file.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
            buffer_index (int, optional): The buffer index to read from. Defaults to 0.
        """

        convert_config = self.load_convert_config()

        if frames is None:
            frames = convert_config.get("frames", "all")

        tx_order, rcv_order, time_to_next_transmit = self.read_transmit_events(
            frames=frames, allow_accumulate=allow_accumulate, buffer_index=buffer_index
        )
        initial_times = self.read_initial_times(rcv_order)

        polar_angles = self.read_polar_angles(tx_order)
        azimuth_angles = self.read_azimuth_angles(tx_order)
        t0_delays, tx_apodizations = self.read_t0_delays_apod(tx_order)
        focus_distances = self.read_focus_distances(tx_order)
        transmit_origins = self.read_transmit_origins(tx_order)

        waveforms_one_way_list, waveforms_two_way_list = self.read_waveforms()
        tx_waveform_indices = self.read_tx_waveform_indices(tx_order)

        # stack waveforms to (n_tx, n_samples) using the tx_waveform_indices
        waveforms_one_way = np.stack([waveforms_one_way_list[i] for i in tx_waveform_indices])
        waveforms_two_way = np.stack([waveforms_two_way_list[i] for i in tx_waveform_indices])

        center_frequency = self.read_center_frequencies(tx_waveform_indices)
        focus_distances = self.planewave_focal_distance_to_inf(
            focus_distances, t0_delays, tx_apodizations
        )

        return {
            "time_to_next_transmit": time_to_next_transmit,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "sampling_frequency": self.sampling_frequency,
            "polar_angles": polar_angles,
            "azimuth_angles": azimuth_angles,
            "center_frequency": center_frequency,
            "demodulation_frequency": self.demodulation_frequency,
            "sound_speed": self.sound_speed,
            "initial_times": initial_times,
            "focus_distances": focus_distances,
            "transmit_origins": transmit_origins,
            "waveforms_one_way": waveforms_one_way,
            "waveforms_two_way": waveforms_two_way,
            "tgc_gain_curve": self.tgc_gain_curve,
        }

    def read_verasonics_file(
        self,
        additional_functions=None,
        frames=None,
        allow_accumulate=False,
        buffer_index=0,
    ):
        """Reads data from a .mat Verasonics output file.

        Args:
            additional_functions (list, optional): A list of functions that read additional
                data from the file. Each function should take the file as input and return a
                `DatasetElement`. Defaults to None.
            frames (str or list of int, optional): The frames to add to the file. This can be
                a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
                None, which means all frames, unless specified in a `convert.yaml` file.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
            buffer_index (int, optional): The buffer index to read from. Defaults to 0.
        """

        if additional_functions is None:
            additional_functions = []

        convert_config = self.load_convert_config()

        if frames is None:
            frames = convert_config.get("frames", "all")

        scan_dict = self.read_scan(
            frames=frames,
            allow_accumulate=allow_accumulate,
            buffer_index=buffer_index,
        )

        raw_data = self.read_raw_data(
            frames=frames,
            buffer_index=buffer_index,
            first_frame_idx=convert_config.get("first_frame", None),
        )

        additional_elements = []

        if self.probe.lens_correction is not None:
            el_lens_correction = DatasetElement(
                dataset_name="lens_correction",
                data=self.probe.lens_correction,
                description=(
                    "The lens correction value used by Verasonics. This value is the "
                    "additional path length in wavelength that the lens introduces. "
                    "(This disregards refraction.)"
                ),
                unit="wavelengths",
            )
            additional_elements.append(el_lens_correction)

        # Add additional elements from user-defined functions
        for additional_function in additional_functions:
            additional_elements.append(additional_function(self))

        # Add Verasonics ImgDataP buffer to additional elements
        try:
            verasonics_image_buffer = self.read_image_data_p(frames=frames)
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
            additional_elements.append(verasonics_image_buffer)
        except Exception as e:
            log.error(f"Could not read Verasonics ImgDataP buffer: {e}, skipping.")

        return {"raw_data": raw_data}, scan_dict, additional_elements

    def _parse_frames_argument(self, frames, n_frames):
        value_error = ValueError(
            f"Invalid frames argument: {frames}. "
            "Expected 'all', a range (e.g. '4-8'), or a list of integers."
        )

        if isinstance(frames, str):
            if frames == "all":
                return list(range(n_frames))
            elif "-" in frames:
                start, end = frames.split("-")
                return list(range(int(start), int(end) + 1))
            else:
                # Try to convert to integer
                try:
                    frame_index = int(frames)
                    return [frame_index]
                except ValueError:
                    raise value_error
        elif isinstance(frames, (list, tuple)):
            # Recursively parse each element
            frame_indices = []
            for frame in frames:
                frame_indices.extend(self._parse_frames_argument(frame, n_frames))
            return frame_indices
        elif isinstance(frames, int):
            return [frames]
        else:
            raise value_error

    def get_frame_indices(self, frames, buffer_index=0):
        """Creates a numpy array of frame indices from the file and the frames argument.

        Args:
            frames (str): The frames argument. This can be "all", a range of integers
                (e.g. "4-8"), or a list of frame indices.

        Returns:
            frame_indices (np.ndarray): The frame indices.
        """
        # Read the number of frames from the file
        n_frames = self.get_frame_count(buffer_index)

        frame_indices = self._parse_frames_argument(frames, n_frames)
        frame_indices = np.asarray(frame_indices)
        frame_indices = np.unique(frame_indices)  # Remove duplicates
        frame_indices.sort()  # Sort the indices

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
        frames=None,
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
                None, which means all frames are used, unless specified otherwise in a
                `convert.yaml` file.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
                Defaults to False.
            enable_compression (bool, optional): Whether to enable compression when saving
                the zea file. Defaults to True.
        """
        # Here we call all the functions to read the data from the file
        log.info("Reading Verasonics file...")
        data_dict, scan_dict, additional_elements = self.read_verasonics_file(
            additional_functions=additional_functions,
            frames=frames,
            allow_accumulate=allow_accumulate,
        )

        # Generate the zea dataset
        log.info("Generating zea dataset...")
        compression = DEFAULT_COMPRESSION if enable_compression else None
        File.create(
            path=output_path,
            data=data_dict,
            scan=scan_dict,
            probe=self.probe.to_probe_spec(),
            description="Verasonics data",
            compression=compression,
        )

        if additional_elements:
            _write_user_additional_elements_to_file(output_path, additional_elements)


class VerasonicsProbe:
    def __init__(self, file: VerasonicsFile):
        self._file = file
        self.trans_obj = file["Trans"]

    @property
    def wavelength(self):
        """The wavelength of the probe in meters."""
        return self._file.wavelength

    @property
    def name(self):
        """The name of the probe from the file."""
        name = self.trans_obj["name"][:]
        name = self._file.decode_string(name)
        # Translates between verasonics probe names and zea probe names
        if name in _VERASONICS_TO_ZEA_PROBE_NAMES:
            name = _VERASONICS_TO_ZEA_PROBE_NAMES[name]
        else:
            log.warning(
                f"Probe name '{name}' is not in the list of known probes. "
                "Please add it to the _VERASONICS_TO_ZEA_PROBE_NAMES dictionary. "
                "Falling back to generic probe."
            )
            name = "generic"

        return name

    @property
    def unit(self):
        """The unit some probe dimensions are defined in.

        This concerns ElementPos, elementWidth and lensCorrection.
        """
        _ALLOWED_UNITS = {"wavelengths", "mm"}
        unit = self._file.decode_string(self.trans_obj["units"][:])
        assert unit in {"wavelengths", "mm"}, (
            f"Unexpected unit '{unit}' in file, must be one of {_ALLOWED_UNITS}"
        )
        return unit

    @property
    def center_frequency(self):
        """Center frequency of the probe from the file in Hz."""

        return self.trans_obj["frequency"][:].item() * 1e6

    @property
    def geometry(self):
        """The probe geometry of shape (n_el, 3)."""
        # Read the probe geometry from the file
        geometry = self.trans_obj["ElementPos"][:3, :]

        # Transpose the probe geometry to have the shape (n_el, 3)
        geometry = geometry.T

        # Convert the probe geometry to meters
        if self.unit == "mm":
            geometry = geometry / 1000
        else:
            geometry = geometry * self.wavelength

        return geometry

    @property
    def bandwidth(self):
        """Bandwidth of the probe: -6dB lower and upper cutoff pts in Hz."""
        if "Bandwidth" in self.trans_obj.keys():
            return self.trans_obj["Bandwidth"][:].squeeze() * 1e6

    @property
    def bandwidth_percent(self):
        """Bandwidth of the probe as a percentage of the center frequency."""
        if self.bandwidth is not None:
            assert self.bandwidth[1] > self.bandwidth[0], "Bandwidth must be positive"
            diff = self.bandwidth[1] - self.bandwidth[0]
            return 100 * (diff / self.center_frequency)

    @property
    def type(self):
        """The type of the probe from the file."""
        if "type" in self.trans_obj.keys():
            _id_to_str = {
                0: "linear",
                1: "curved",
                2: "2D-array",
                3: "annular",
                4: "row-column",
            }
            probe_type_id = int(self.trans_obj["type"][:].item())
            return _id_to_str.get(probe_type_id)

    @property
    def element_width(self):
        """The element width in meters from the file."""
        element_width = self.trans_obj["elementWidth"][:].item()

        # Convert the probe element width to meters
        if self.unit == "mm":
            element_width = element_width / 1000  # mm -> m
        else:
            element_width = element_width * self.wavelength  # wavelengths -> m

        return element_width

    @property
    def element_length(self):
        """Element length for row-column probes in meters."""
        if "ElementLength" in self.trans_obj.keys():
            return self.trans_obj["ElementLength"][:].item() / 1000  # mm -> m

    @property
    def connector(self):
        """Probe connector indices."""
        connector = self.trans_obj["ConnectorES"][:]
        connector = np.squeeze(connector, axis=0)
        connector = connector.astype(np.int32)
        connector = connector - 1  # make 0-based
        return connector

    @property
    def lens_correction(self):
        """The lens correction: 1 way delay in wavelengths thru lens"""

        if "lensCorrection" in self.trans_obj.keys():
            return self.trans_obj["lensCorrection"][:].item()

    @property
    def _probe_geometry_is_ordered_ula(self):
        """Checks if the probe geometry is ordered as a uniform linear array (ULA)."""
        diff_vec = self.geometry[1:] - self.geometry[:-1]
        return np.isclose(diff_vec, diff_vec[0]).all()

    def to_probe_spec(self):
        """Convert the probe to a dict compatible with :class:`~zea.data.spec.ProbeSpec`."""
        return {
            "name": self.name,
            "type": self.type,
            "probe_center_frequency": self.center_frequency,
            "probe_bandwidth_percent": self.bandwidth_percent,
            "probe_geometry": self.geometry,
            "element_width": self.element_width,
        }


def _write_user_additional_elements(h5file, additional_elements, prefix=""):
    """Write user-provided DatasetElement objects into an HDF5 file.

    Args:
        h5file: Open h5py.File in write/append mode.
        additional_elements: List of DatasetElement objects from user-defined functions.
        prefix: Path prefix for element group names.
    """
    if not additional_elements:
        return

    nse_path = f"{prefix}/non_standard_elements" if prefix else "non_standard_elements"
    if nse_path not in h5file:
        nse = h5file.create_group(nse_path)
        nse.attrs["description"] = (
            "This group contains non-standard elements that can be added by the user."
        )

    for element in additional_elements:
        group_path = nse_path
        if element.group_name:
            group_path = f"{nse_path}/{element.group_name}"
            if group_path not in h5file:
                h5file.create_group(group_path)

        data = np.asarray(element.data)
        is_scalar = np.isscalar(data) or data.ndim == 0
        compression = DEFAULT_COMPRESSION if not is_scalar else None
        ds = h5file[group_path].create_dataset(
            element.dataset_name, data=data, compression=compression
        )
        ds.attrs["description"] = element.description
        ds.attrs["unit"] = element.unit


def _write_user_additional_elements_to_file(path, additional_elements):
    """Write user-provided additional elements into an existing HDF5 file."""
    with h5py.File(str(path), "a") as f:
        _write_user_additional_elements(f, additional_elements)


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


def make_dataset_card(repo_id):
    return f"""\
---
zea_repo_id: {repo_id}
---

# Verasonics ultrasound data (zea format)

This dataset contains raw ultrasound data acquired with Verasonics systems,
converted to the [zea](https://github.com/tue-bmd/zea) HDF5 format.
"""


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

    if getattr(args, "upload", False):
        assert args.hf_repo_id, "hf_repo_id must be provided when --upload is True."
        assert args.revision, "revision must be provided when --upload is True."

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

    # Build the list of (input, output) file pairs to convert
    if selected_path_is_directory:
        file_pairs = []
        for root, _dirs, files in os.walk(selected_path):
            for mat_file in files:
                # Skip non-mat files
                if not mat_file.endswith(".mat"):
                    continue

                log.info(f"Found raw data file {log.yellow(mat_file)}")

                relative_path = (Path(root) / mat_file).relative_to(selected_path)
                file_pairs.append(
                    (
                        selected_path / relative_path,
                        output_path / relative_path.with_suffix(".hdf5"),
                    )
                )
    else:
        file_pairs = [(selected_path, output_path)]

    num_converted = 0
    for full_path, file_output_path in file_pairs:
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
                frames=args.frames,
                allow_accumulate=args.allow_accumulate,
                enable_compression=not args.no_compression,
            )
            num_converted += 1
        except Exception:
            # Print error message without raising it
            log.error(f"Failed to convert {full_path.name}")
            # Print stacktrace
            traceback.print_exc()

            continue

    # Do not write a dataset card or upload anything when nothing was converted
    # (e.g. all files failed or were skipped).
    if getattr(args, "upload", False) and num_converted == 0:
        log.error("No files were converted successfully; skipping upload.")
        return

    # Write the dataset card next to the converted output: in the output
    # directory for a directory conversion, or alongside the file for a single
    # file. The card is required for the upload ownership check below.
    if selected_path_is_directory:
        write_dataset_card(output_path, make_dataset_card(args.hf_repo_id))
    elif getattr(args, "upload", False):
        write_dataset_card(output_path.parent, make_dataset_card(args.hf_repo_id))

    if getattr(args, "upload", False):
        assert args.hf_repo_id, "hf_repo_id must be provided when --upload is True."
        assert args.revision, "revision must be provided when --upload is True."
        upload_verasonics(
            output_path,
            revision=args.revision,
            repo_id=args.hf_repo_id,
        )


def upload_verasonics(
    output_path: str | Path, revision: str, repo_id: str
) -> None:  # pragma: no cover
    """Upload a converted Verasonics dataset to a HuggingFace Hub revision branch.

    Accepts either a directory of converted HDF5 files or a single converted
    HDF5 file.  For a single file only that file (and its dataset card) is
    uploaded, leaving any sibling files in the same directory untouched.

    Only for zea maintainers with push access to the repository.  Upload to
    ``main`` is blocked; merge the revision branch into ``main`` manually after
    verifying the upload.

    Args:
        output_path: Directory containing the converted HDF5 files, or a single
            converted HDF5 file.
        revision: Target branch name on the Hub (must not be ``"main"``).
        repo_id: Target HuggingFace repository ID.
    """
    output_path = Path(output_path)
    if output_path.is_dir():
        folder = output_path
        file_glob = "*.hdf5"
        allow_patterns = None
    else:
        # Single file: upload only this file plus its dataset card, scoped via
        # allow_patterns so sibling files in the directory are not uploaded.
        folder = output_path.parent
        file_glob = output_path.name
        allow_patterns = [output_path.name, "README.md"]

    require_output_dir_ownership(folder, repo_id)
    upload_dataset_to_hf(
        folder=folder,
        repo_id=repo_id,
        revision=revision,
        file_glob=file_glob,
        allow_patterns=allow_patterns,
        commit_message=f"Upload Verasonics dataset (zea format) to {revision}",
    )
