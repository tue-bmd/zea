"""Functionality to convert Verasonics MATLAB workspace to the zea format.

Example of saving the entire workspace to a .mat file (MATLAB):

    .. code-block:: matlab

        >> setup_script;
        >> VSX;
        >> save('C:/path/to/raw_data.mat', '-v7.3');

.. important::

    The ``.mat`` file **must** be saved in HDF5 format (MATLAB v7.3 or later).
    Older ``.mat`` files are not HDF5-compatible and cannot be opened by this converter.
    To save in the correct format from MATLAB, use the ``-v7.3`` flag:

.. note::

    We also have a `save_raw` function (not available in zea at the moment)
    which saves all relevant variables from the workspace only. This results in a smaller file size and faster conversion.

Then convert the saved `raw_data.mat` file to zea format using the following code (Python):

    .. code-block:: python

        from zea.data.convert.verasonics import VerasonicsFile

        VerasonicsFile("C:/path/to/raw_data.mat").to_zea("C:/path/to/output.hdf5")

Or alternatively, use the CLI to convert all .mat files in a directory:

    .. code-block:: bash

        zea convert verasonics "C:/path/to/source" "C:/path/to/output"

A ``.mat`` with several RF buffers (``RcvData{1}``, ``RcvData{2}``, ...) is written
to one zea file with one track per buffer. Read them with ``File(path).tracks`` or
``File(path).get_track("rf_buffer_1")``.

Like the rest of zea, the converter counts buffers and frames from 0, while MATLAB counts
from 1. So RF buffer 0 is ``RcvData{1}`` and becomes the track ``rf_buffer_0``, and image
buffer 0 is ``ImgDataP{1}`` and becomes the custom element ``verasonics_image_buffer_0``.
Messages name both, e.g. "RF buffer 0 (RcvData{1})". The external ``RF_data_k.bin``
files keep the MATLAB number: ``RF_data_1.bin`` holds RF buffer 0.

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
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
import scipy.fft
import yaml
from keras import ops

from zea import log
from zea.data.convert import _verasonics_bin
from zea.data.convert._verasonics_bin import rf_buffer_name
from zea.data.convert.utils import (
    require_output_dir_ownership,
    upload_dataset_to_hf,
    write_dataset_card,
)
from zea.data.file import CustomElement, File
from zea.func import log_compress, normalize
from zea.utils import strtobool

_VERASONICS_TO_ZEA_PROBE_NAMES = {
    "L11-4v": "verasonics_l11_4v",
    "L11-5v": "verasonics_l11_5v",
    "C5-2v": "verasonics_c5_2v",
    "C5-2V": "verasonics_c5_2v",
    "P4-2v": "verasonics_p4_2v",
    "P4-2V": "verasonics_p4_2v",
}


def classify_ordered_geometry(geometry, rtol=1e-3):
    """Classify a probe geometry as an ordered linear array, ordered arc, or neither.

    Used when interpreting transmit delays during conversion. A **uniform linear
    array** (linear/phased probes) has identical step vectors between consecutive
    elements. A **convex/curved** array has steps of equal length that turn by a
    constant angle (a uniform arc). Anything else — a matrix array, or an
    unordered element list — is ``"unknown"``.

    This lets the converter pick the right check per probe instead of demanding a
    strict linear array for everything (which needlessly flags valid curved
    probes).

    Args:
        geometry (np.ndarray): Element positions of shape ``(n_el, 3)``.
        rtol (float): Relative tolerance (scaled by the element spacing) for the
            uniformity checks.

    Returns:
        tuple[bool, str]: ``(is_ordered, kind)`` with ``kind`` in
        ``{"linear", "arc", "unknown"}``.
    """
    n_el = geometry.shape[0]
    if n_el < 2:
        return False, "unknown"

    steps = geometry[1:] - geometry[:-1]
    lengths = np.linalg.norm(steps, axis=1)
    scale = lengths[0]
    if scale == 0:
        return False, "unknown"
    tol = rtol * scale

    # Linear: every step vector equals the first (same direction and spacing).
    if np.max(np.linalg.norm(steps - steps[0], axis=1)) < tol:
        return True, "linear"

    if n_el < 3:
        return False, "unknown"

    # Arc: equal step lengths and a constant turning angle between steps.
    if np.max(np.abs(lengths - scale)) < tol:
        units = steps / lengths[:, None]
        cos_turn = np.clip(np.sum(units[1:] * units[:-1], axis=1), -1.0, 1.0)
        if np.max(np.abs(cos_turn - cos_turn[0])) < rtol:
            return True, "arc"

    return False, "unknown"


_FRAMES_RANGE_RE = re.compile(r"^\d+(-\d+)?$")


def estimate_lens_probe_params(
    lens_correction: float | None,
    center_frequency: float,
    lens_sound_speed: float = 1000.0,
) -> dict:
    """Return ProbeSpec lens fields derived from the Verasonics lens correction.

    Converts the Verasonics one-way scalar delay (wavelengths) to a physical
    ``(lens_thickness, lens_sound_speed)`` pair::

        t_one_way = lens_correction / f_c  =  lens_thickness / lens_sound_speed
        → lens_thickness = lens_correction × lens_sound_speed / f_c

    The raw Verasonics value (``Trans.lensCorrection``) is a scalar one-way
    delay offset in wavelengths applied uniformly across all elements.  It is
    not directly compatible with zea's Fermat-based model, which solves the
    refracted path per element–pixel pair via Newton-Raphson.

    Args:
        lens_correction (float or None): One-way delay through the lens in
            wavelengths (``Trans.lensCorrection``).  Returns an empty dict
            when ``None``.
        center_frequency (float): Center frequency in Hz.
        lens_sound_speed (float, optional): Speed of sound in the lens
            material in m/s. Defaults to 1000.0.

    Returns:
        dict: ``{"lens_sound_speed": ..., "lens_thickness": ...}`` ready to
        merge into a ProbeSpec dict, or ``{}`` when ``lens_correction`` is
        ``None``.
    """
    if lens_correction is None:
        return {}
    if not (np.isfinite(lens_sound_speed) and lens_sound_speed > 0):
        raise ValueError(f"lens_sound_speed must be finite and positive, got {lens_sound_speed!r}")
    if not (np.isfinite(center_frequency) and center_frequency > 0):
        raise ValueError(f"center_frequency must be finite and positive, got {center_frequency!r}")
    if not (np.isfinite(lens_correction) and lens_correction >= 0):
        raise ValueError(
            f"lens_correction must be finite and non-negative, got {lens_correction!r}"
        )
    lens_thickness = np.float32(float(lens_correction) * lens_sound_speed / center_frequency)
    log.info(
        f"Lens: {float(lens_correction):.3f} wl → {float(lens_thickness) * 1e3:.3f} mm "
        f"(c_lens = {lens_sound_speed:.0f} m/s)"
    )
    return {
        "lens_sound_speed": np.float32(lens_sound_speed),
        "lens_thickness": lens_thickness,
    }


def baseband_mode_to_iq(data: np.ndarray, carrier_periods_per_pair: int = 1) -> np.ndarray:
    """Convert BS100BW/BS50BW interleaved data to IQ format.

    Both Verasonics baseband modes (BS100BW and BS50BW) store pairs of samples taken a
    quarter carrier period apart. One pair is taken every carrier period (BS100BW) or
    every two carrier periods (BS50BW).

    With ``x(t) = I(t) cos(2π f_c t) - Q(t) sin(2π f_c t)``, the even samples fall where
    ``cos = 1`` and hold ``I``, and the odd samples fall where ``sin = 1`` and hold
    ``-Q``. The odd samples are therefore negated to recover ``Q``. Because Q is sampled
    a quarter carrier period after I, Q is then shifted back to the I sample times with
    a band-limited fractional delay.

    Args:
        data: Input array of shape ``(n_frames, n_tx, n_ax_raw, n_el, 1)``.
        carrier_periods_per_pair: Number of carrier periods between consecutive sample
            pairs: 1 for BS100BW, 2 for BS50BW. Defaults to 1.

    Returns:
        IQ array of shape ``(n_frames, n_tx, n_ax, n_el, 2)``.
    """
    if data.ndim != 5 or data.shape[-1] != 1:
        raise ValueError(f"Expected shape (n_frames, n_tx, n_ax_raw, n_el, 1), got {data.shape}")
    if data.shape[2] % 2 != 0:
        raise ValueError(f"Axial dimension must be even for IQ deinterleaving, got {data.shape[2]}")
    if carrier_periods_per_pair not in (1, 2):
        raise ValueError(
            f"carrier_periods_per_pair must be 1 (BS100BW) or 2 (BS50BW), "
            f"got {carrier_periods_per_pair}"
        )
    d = data.astype(np.float32)
    # Even samples hold I, odd samples hold -Q.
    iq_data = np.stack([d[:, :, 0::2, :, 0], -d[:, :, 1::2, :, 0]], axis=-1)
    del d
    # Q lags I by a quarter carrier period, expressed here in IQ samples.
    q_delay = 0.25 / carrier_periods_per_pair
    return _align_baseband_quadrature(iq_data, axis=2, q_delay=q_delay)


def _align_baseband_quadrature(iq: np.ndarray, axis: int = 2, q_delay: float = 0.25) -> np.ndarray:
    """Shift Q back to the I sample times in IQ data built from BS100BW/BS50BW pairs.

    The baseband sample modes capture I at time ``t`` and Q a quarter carrier period
    later. With ``Ts`` the IQ sample period, that is ``Q(t + Ts/4)`` for BS100BW and
    ``Q(t + Ts/8)`` for BS50BW. This function applies a band-limited fractional delay
    of ``q_delay`` samples to Q so that it is evaluated at the I sample times.

    Q is overwritten in place to limit peak memory on full recordings.

    Args:
        iq: Float32 IQ data with I and Q on the last axis. Modified in place.
        axis: The axial (fast-time) axis.
        q_delay: How far Q lags I, in IQ samples. 0.25 for BS100BW, 0.125 for BS50BW.
            Defaults to 0.25.

    Returns:
        ``iq``, with Q evaluated at the I sample times.
    """
    q_data = np.moveaxis(iq[..., 1], axis, -1)
    n_ax = q_data.shape[-1]
    # Mirror the record so the circular FFT shift sees no jump between its end and start.
    n_fft = 2 * n_ax
    q_data_extended = np.concatenate([q_data, q_data[..., ::-1]], axis=-1)
    spectrum = scipy.fft.rfft(q_data_extended, axis=-1, workers=-1)
    del q_data_extended
    frequency = scipy.fft.rfftfreq(n_fft)
    # Q[n] holds Im b at sample n + q_delay; evaluate it at sample n.
    spectrum *= np.exp(-2j * np.pi * frequency * q_delay).astype(spectrum.dtype)
    q_aligned = scipy.fft.irfft(spectrum, n=n_fft, axis=-1, workers=-1)
    del spectrum

    iq[..., 1] = np.moveaxis(q_aligned[..., :n_ax], -1, axis)
    return iq


def _validate_convert_config(data):
    """Validate the structure of a convert.yaml config dict.

    Expected shape::

        files:
          - name: <str>
            first_frame: <int >= 0>          # optional
            frames: all | "N" | "N-M" | [N, ...] # optional
            transmits: all | [N, ...]         # optional
            buffers: [N, ...]                 # optional
            connector_applied: <bool>         # optional
    """
    if not isinstance(data, dict) or "files" not in data:
        raise ValueError("convert.yaml must have a top-level 'files' key")
    if not isinstance(data["files"], list):
        raise ValueError("'files' must be a list")
    for entry in data["files"]:
        if not isinstance(entry, dict):
            raise ValueError(f"each entry in 'files' must be a dict, got {type(entry).__name__}")
        if not isinstance(entry.get("name"), str):
            raise ValueError(f"each file entry must have a string 'name', got {entry!r}")
        if "first_frame" in entry:
            ff = entry["first_frame"]
            if not isinstance(ff, int) or isinstance(ff, bool) or ff < 0:
                raise ValueError(f"'first_frame' must be a non-negative int, got {ff!r}")
        if "frames" in entry:
            fr = entry["frames"]
            if isinstance(fr, str) and _FRAMES_RANGE_RE.fullmatch(fr) and "-" in fr:
                start, end = map(int, fr.split("-"))
                if start > end:
                    raise ValueError(f"'frames' range must be ascending (start <= end), got {fr!r}")
            if not (
                fr == "all"
                or (isinstance(fr, str) and _FRAMES_RANGE_RE.fullmatch(fr))
                or (
                    isinstance(fr, list)
                    and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in fr)
                )
            ):
                raise ValueError(
                    f"'frames' must be 'all', a range string like '30-99', or a list of "
                    f"non-negative ints, got {fr!r}"
                )
        if "transmits" in entry:
            tr = entry["transmits"]
            if not (
                tr == "all"
                or (
                    isinstance(tr, list)
                    and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in tr)
                )
            ):
                raise ValueError(
                    f"'transmits' must be 'all' or a list of non-negative ints, got {tr!r}"
                )
        if "buffers" in entry:
            bu = entry["buffers"]
            if not (
                isinstance(bu, list)
                and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in bu)
            ):
                raise ValueError(f"'buffers' must be a list of non-negative ints, got {bu!r}")
        if "connector_applied" in entry and not isinstance(entry["connector_applied"], bool):
            raise ValueError(
                f"'connector_applied' must be a bool, got {entry['connector_applied']!r}"
            )
    return data


def _is_all_frames(frames) -> bool:
    # True for "all", or a list/tuple of only "all" (the CLI passes frames as a list).
    if isinstance(frames, str):
        return frames == "all"
    if isinstance(frames, (list, tuple)):
        return len(frames) > 0 and all(isinstance(f, str) and f == "all" for f in frames)
    return False


def _is_reference_dataset(dataset) -> bool:
    """Whether an HDF5 dataset stores object references.

    MATLAB v7.3 stores cell arrays (e.g. ``RcvData``) and struct-array fields
    (e.g. ``Receive.bufnum``) as datasets of HDF5 object references.

    Args:
        dataset (h5py.Dataset): The dataset to inspect.

    Returns:
        bool: True if the dataset's elements are HDF5 object references.
    """
    # Check the dtype, not ``dataset.fillvalue``: only some writers (e.g. MATLAB) set
    # a reference fill value.
    return h5py.check_dtype(ref=dataset.dtype) is not None


class VerasonicsFile(h5py.File):
    """HDF5 File class for Verasonics MATLAB workspace files.

    This class extends the h5py.File class to handle Verasonics-specific
    data structures and conventions.

    .. note::

        The ``.mat`` file must be saved in HDF5 format (MATLAB v7.3).
        Use ``save('file.mat', '-v7.3')`` in MATLAB before converting.
    """

    def __init__(self, name, mode="r", **kwargs):
        # Checked up front: h5py reports a missing file as the OSError caught below.
        if mode == "r" and isinstance(name, (str, os.PathLike)) and not Path(name).is_file():
            raise FileNotFoundError(f"Verasonics file '{name}' does not exist.")
        try:
            super().__init__(name, mode, **kwargs)
        except OSError as e:
            raise OSError(
                f"Cannot open '{name}' as an HDF5 file.\n\n"
                "This usually means the .mat file was not saved in HDF5 format.\n"
                "MATLAB saves in HDF5 format only when you use the '-v7.3' flag:\n\n"
                "    save('C:/path/to/raw_data.mat', '-v7.3')\n\n"
                "Re-save the workspace in MATLAB with this flag and try again."
            ) from e

    def dereference_index(self, dataset, index):
        """Get the element at the given index from the dataset, dereferencing it if
        necessary.

        MATLAB stores items in struct array differently depending on the size. If the size
        is 1, the item is stored as a regular dataset. If the size is larger, the item is
        stored as a dataset of references to the actual data.

        This function dereferences the dataset if it is a reference. Otherwise, it returns
        the dataset.

        If the struct array has a single element, it is stored directly rather than
        as references, so only ``index == 0`` is valid.

        Args:
            dataset (h5py.Dataset): The dataset to read the element from.
            index (int): The index of the element to read.
        """
        if _is_reference_dataset(dataset):
            if index < 0 or index >= dataset.size:
                raise IndexError(
                    f"index {index} is out of range for a reference dataset with "
                    f"{dataset.size} element(s)."
                )
            # Read only this reference, not the whole dataset: callers loop over every
            # transmit or receive. unravel_index handles both (N, 1) and (1, N) layouts.
            reference = dataset[np.unravel_index(int(index), dataset.shape)]
            return self[reference][:]
        else:
            # A non-reference dataset is a single stored element.
            if index != 0:
                raise IndexError(
                    f"index {index} is out of range: this dataset is stored as a "
                    "single (non-reference) element, so only index 0 is valid. "
                    "For RcvData this means the file contains a single RF buffer."
                )
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

    def _read_scalar_field(self, group_name, field_name):
        """Read every entry of a scalar struct-array field (e.g. ``Receive.mode``) as
        one numpy array. The result is cached per field.

        Returns:
            np.ndarray: 1-D array with one value per struct-array entry.
        """
        # Cached because each entry is a separate HDF5 read (tens of thousands for
        # large Event/Receive structs) and several callers read the same fields.
        cache = self.__dict__.setdefault("_scalar_field_cache", {})
        key = (group_name, field_name)
        if key in cache:
            return cache[key]

        dataset = self[group_name][field_name]
        if _is_reference_dataset(dataset):
            references = np.asarray(dataset[:]).reshape(-1)
            values = np.empty(references.size, dtype=np.float64)
            file = self  # local ref for speed
            for i, ref in enumerate(references):
                values[i] = np.asarray(file[ref]).reshape(-1)[0]
        else:
            values = np.asarray(dataset[:]).reshape(-1).astype(np.float64)

        cache[key] = values
        return values

    def _read_constant_receive_field(self, field_name, buffer_index, decode, description):
        """Read a ``Receive`` field that is constant within a buffer (e.g. ``sampleMode``).

        Every entry is read and checked to be equal. The result is cached per field and
        buffer.

        Args:
            field_name (str): The ``Receive`` field to read.
            buffer_index (int or None): The 0-based RF buffer, or None for all receives.
            decode (callable): Turns one dereferenced entry into a hashable value.
            description (str): Plural name of the values, for the error message.
        """
        cache = self.__dict__.setdefault("_constant_field_cache", {})
        key = (field_name, buffer_index)
        if key in cache:
            return cache[key]

        dataset = self["Receive"][field_name]
        if buffer_index is not None:
            indices = self.receive_indices_for_buffer(buffer_index)
        else:
            indices = np.arange(self.get_reference_size(dataset))

        # Each entry is a separate HDF5 object (about 0.1 ms per read), so this is the
        # slow part for large sequences, but a partial check could miss a mixed buffer.
        values = {decode(self.dereference_index(dataset, int(i))) for i in indices}
        assert len(values) == 1, (
            f"Multiple {description} found in file: {values}. We do not support this case."
        )
        cache[key] = values.pop()
        return cache[key]

    def _clear_caches(self):
        """Clear every memoized value (e.g. after reopening)."""
        for name in ("_scalar_field_cache", "_constant_field_cache", "_probe"):
            self.__dict__.pop(name, None)

    @staticmethod
    def get_reference_size(dataset):
        """Get the number of references in a reference dataset.

        A non-reference dataset is a single stored element, so its size is 1.
        """
        if _is_reference_dataset(dataset):
            # Not len(): cell arrays such as RcvData are stored as (1, N) row vectors.
            return int(dataset.size)
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

    def read_receive_bufnums(self):
        """Read the (0-based) RF buffer index for every ``Receive`` entry.

        This is ``Receive.bufnum - 1``, i.e. the receive is stored in
        ``RcvData{bufnum}``.

        Returns:
            np.ndarray: Integer array of shape ``(n_receive,)``. All zeros if the file
            has no ``bufnum`` field (older single-buffer files).
        """
        if "bufnum" not in self["Receive"]:
            # Older single-buffer files may not store bufnum; assume buffer 0.
            # endSample is not used to count receives, since it can be missing.
            for field in ("startDepth", "endDepth", "mode", "framenum"):
                if field in self["Receive"]:
                    n_receive = self.get_reference_size(self["Receive"][field])
                    return np.zeros(n_receive, dtype=np.int64)
            raise KeyError(
                "Could not determine the number of Receive entries: none of the "
                "expected Receive fields (bufnum, startDepth, endDepth, mode, "
                "framenum) are present."
            )

        bufnums = self._read_scalar_field("Receive", "bufnum")
        return np.round(bufnums).astype(np.int64) - 1

    def receive_indices_for_buffer(self, buffer_index=0):
        """Indices into the ``Receive`` array that belong to ``buffer_index``.

        Args:
            buffer_index (int, optional): The 0-based RF buffer to select.
                Defaults to 0.

        Returns:
            np.ndarray: Sorted integer array of ``Receive`` indices whose
            ``bufnum`` matches ``buffer_index``.
        """
        bufnums = self.read_receive_bufnums()
        indices = np.flatnonzero(bufnums == buffer_index)
        if indices.size == 0:
            raise ValueError(
                f"No Receive entries found for buffer_index={buffer_index}. "
                f"Available buffers (0-based): {sorted(np.unique(bufnums).tolist())}."
            )
        return indices

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

        frame_indices = self.get_frame_indices(frames, buffer_index)

        receive_bufnums = self.read_receive_bufnums()

        event_tx_all = np.round(self._read_scalar_field("Event", "tx")).astype(np.int64)
        event_rcv_all = np.round(self._read_scalar_field("Event", "rcv")).astype(np.int64)
        receive_mode_all = np.round(self._read_scalar_field("Receive", "mode")).astype(np.int64)
        receive_framenum_all = np.round(self._read_scalar_field("Receive", "framenum")).astype(
            np.int64
        )

        # An event contributes only if it has BOTH a transmit and a receive.
        has_tx = event_tx_all != 0
        has_rcv = event_rcv_all != 0
        if np.any(has_tx != has_rcv):
            log.warning(
                "Some events have a transmit but no receive or vice versa; these are skipped."
            )
        valid = has_tx & has_rcv

        # 0-based tx/rcv indices for the valid events (order preserved).
        event_indices = np.flatnonzero(valid)
        tx_idx = event_tx_all[event_indices] - 1
        rcv_idx = event_rcv_all[event_indices] - 1

        # Keep only events whose receive belongs to the selected buffer.
        in_buffer = receive_bufnums[rcv_idx] == buffer_index
        event_indices = event_indices[in_buffer]
        tx_idx = tx_idx[in_buffer]
        rcv_idx = rcv_idx[in_buffer]

        # The tx/rcv order is taken from the first frame, assuming all frames match.
        framenum = receive_framenum_all[rcv_idx]
        first_frame = framenum == 1
        tx_order = tx_idx[first_frame]
        rcv_order = rcv_idx[first_frame]
        modes = receive_mode_all[rcv_idx[first_frame]]

        time_to_next_acq = self._collect_time_to_next_acq(event_indices)

        tx_order = np.asarray(tx_order)
        rcv_order = np.asarray(rcv_order)
        modes = np.asarray(modes)
        time_to_next_acq = np.asarray(time_to_next_acq)

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

            log.info(
                "Dropping time to next acquisition (time_to_next_transmit will be "
                "None). In accumulate sequences (e.g. pulse inversion), each stored "
                "acquisition is formed from several transmits that each carry their "
                "own timeToNextAcq, so the per-transmit timing no longer maps "
                "one-to-one onto the stored transmits and cannot be represented "
                "correctly. It is left unset rather than filled with a misleading value."
            )
            time_to_next_acq = None

        # Reshape to (n_frames, n_tx); dropped if the transmit structure is not
        # uniform across frames.
        if time_to_next_acq is not None:
            n_tx = tx_order.size
            if n_tx > 0 and time_to_next_acq.size % n_tx == 0:
                time_to_next_acq = np.reshape(time_to_next_acq, (-1, n_tx))
                time_to_next_acq = time_to_next_acq[frame_indices]
            else:
                log.warning(
                    "Could not reshape time_to_next_acq "
                    f"({time_to_next_acq.size} values) into frames of {n_tx} "
                    "transmits; the transmit structure is not uniform across "
                    "frames for this buffer. Dropping time-to-next-acquisition."
                )
                time_to_next_acq = None

        return tx_order, rcv_order, time_to_next_acq

    def _collect_time_to_next_acq(self, event_indices):
        """Collect the ``timeToNextAcq`` values (in seconds) of the given events, in order."""
        seq_control_dataset = self["Event"]["seqControl"]

        # The SeqControl table is small, so decode it once up front.
        n_seq = self.get_reference_size(self["SeqControl"]["command"])
        is_ttna = np.zeros(n_seq, dtype=bool)
        ttna_value = np.zeros(n_seq, dtype=np.float64)
        for j in range(n_seq):
            command = self.decode_string(self.dereference_index(self["SeqControl"]["command"], j))
            if command == "timeToNextAcq":
                is_ttna[j] = True
                ttna_value[j] = (
                    np.asarray(self.dereference_index(self["SeqControl"]["argument"], j)).reshape(
                        -1
                    )[0]
                    * 1e-6
                )

        # Each event's seqControl is a scalar index or a short vector.
        if _is_reference_dataset(seq_control_dataset):
            seq_refs = np.asarray(seq_control_dataset[:]).reshape(-1)
            resolve = self
        else:
            seq_refs = None

        time_to_next_acq = []
        for i in event_indices:
            if seq_refs is not None:
                seq_control_indices = np.asarray(resolve[seq_refs[int(i)]]).reshape(-1)
            else:
                seq_control_indices = np.asarray(
                    self.dereference_index(seq_control_dataset, int(i))
                ).reshape(-1)
            for raw in seq_control_indices:
                seq_control_index = int(raw) - 1  # 0-based; 0 -> -1 means "none"
                if seq_control_index < 0:
                    continue
                if is_ttna[seq_control_index]:
                    time_to_next_acq.append(ttna_value[seq_control_index])

        return time_to_next_acq

    def read_t0_delays_apod(self, tx_order):
        """
        Read the t0 delays and apodization from the file.

        Returns:
            tuple: ``(t0_delays, apodizations)`` — t0 delays of shape ``(n_tx, n_el)``
            and transmit apodizations of shape ``(n_tx, n_el)``.
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

    def get_sampling_frequency(self, buffer_index=None):
        """The sampling frequency in Hz from the file.

        Args:
            buffer_index (int, optional): If given, only consider receive events
                belonging to this (0-based) RF buffer. If ``None`` (default), all
                receive events are considered.
        """

        def scalar(value):
            return float(np.asarray(value).reshape(-1)[0])

        adc_rate = self._read_constant_receive_field(
            "decimSampleRate", buffer_index, scalar, "decimation sample rates"
        )

        if "quadDecim" in self["Receive"]:
            quaddecim = self._read_constant_receive_field(
                "quadDecim", buffer_index, scalar, "quadrature decimation factors"
            )
        else:
            # TODO: Verify if this is correct.
            # On the Vantage NXT the quadDecim field is missing. It seems that it should be
            # set to 1.0 (that decimSampleRate is the actual sampling frequency).
            quaddecim = 1.0

        sampling_frequency = adc_rate / quaddecim * 1e6

        if self.get_is_baseband_mode(buffer_index):
            # Two sequential samples are interpreted as a single complex sample
            # Therefore, we need to halve the sampling frequency
            sampling_frequency = sampling_frequency / 2

        return sampling_frequency

    @property
    def sampling_frequency(self):
        """The sampling frequency in Hz from the file (all buffers)."""
        return self.get_sampling_frequency()

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

    def get_end_samples(self, buffer_index=None):
        """The index of the last sample for each receive event.

        Args:
            buffer_index (int, optional): If given, only return the end samples
                for receive events belonging to this (0-based) RF buffer. If
                ``None`` (default), return the values for all receive events.
        """
        end_samples = self._read_scalar_field("Receive", "endSample")
        if buffer_index is not None:
            end_samples = end_samples[self.receive_indices_for_buffer(buffer_index)]
        return end_samples

    def get_start_samples(self, buffer_index=None):
        """The index of the first sample for each receive event.

        Args:
            buffer_index (int, optional): If given, only return the start samples
                for receive events belonging to this (0-based) RF buffer. If
                ``None`` (default), return the values for all receive events.
        """
        start_samples = self._read_scalar_field("Receive", "startSample")
        if buffer_index is not None:
            start_samples = start_samples[self.receive_indices_for_buffer(buffer_index)]
        return start_samples

    @property
    def end_samples(self):
        """The index of the last sample for each receive event (all buffers)."""
        return self.get_end_samples()

    @property
    def start_samples(self):
        """The index of the first sample for each receive event (all buffers)."""
        return self.get_start_samples()

    def get_n_ax(self, buffer_index=None):
        """Number of axial samples for the given RF buffer.

        Args:
            buffer_index (int, optional): The 0-based RF buffer to compute the
                number of axial samples for. If ``None`` (default), all receive
                events are considered.
        """
        n_ax = (
            self.get_end_samples(buffer_index) - self.get_start_samples(buffer_index) + 1
        ).astype(np.int32)
        n_ax = np.unique(n_ax)
        if n_ax.size != 1:
            raise ValueError(
                "The number of axial samples is not the same for all receive events."
                "We do not support this case yet."
            )
        return n_ax.item()

    @property
    def n_ax(self):
        """Number of axial samples (all buffers)."""
        return self.get_n_ax()

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
              buffers: [0]  # 0-based indexing, so RcvData{1}
              connector_applied: true  # see VerasonicsFile.read_raw_data

        If ``first_frame`` is provided, it will reorder the frames first and use
        the ``frames`` key to subsample afterwards.

        In the example ``frames: 30-99`` means frames 30 to 99 inclusive.

        ``buffers`` selects the RF buffers to convert. ``frames`` and ``first_frame``
        need a single buffer when the file has several.

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
            validated_data = _validate_convert_config(data)

            files = validated_data["files"]
            filenames = [file["name"] for file in files]
            if path.name in filenames:
                return files[filenames.index(path.name)]
            elif path.stem in filenames:
                return files[filenames.index(path.stem)]
        return {}

    def get_frame_count(self, buffer_index=0):
        """Get the total number of frames stored for an RF buffer.

        For a binary workspace this is ``RF_frames``, the number of frames saved in
        ``RF_data_k.bin``, which can be fewer than ``Resource.RcvBuffer.numFrames``.
        """
        if _verasonics_bin.is_binary_workspace(self):
            return _verasonics_bin.frame_count(self, buffer_index)
        return self._rcvbuffer_frame_count(buffer_index)

    def _rcvbuffer_frame_count(self, buffer_index):
        n_frames = self.dereference_index(self["Resource"]["RcvBuffer"]["numFrames"], buffer_index)
        return self.cast_to_integer(n_frames)

    def get_indices_to_reorder(self, first_frame: int, n_frames: int):
        return (np.arange(n_frames) + first_frame) % n_frames

    def get_raw_data_order(self, buffer_index=0):
        """The order of frames in the RcvBuffer buffer.

        Because of the circular buffer used in Verasonics, the frames in the RcvBuffer
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.
        """
        n_frames = self.get_frame_count(buffer_index)
        if _verasonics_bin.is_binary_workspace(self) and not _verasonics_bin.holds_full_rcvbuffer(
            self, buffer_index
        ):
            return np.arange(n_frames)
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

    def available_buffers(self):
        """0-based indices of the RF buffers in this file.

        Every ``RcvData`` cell (or saved ``RF_data_k.bin``) that at least one
        ``Receive`` entry stores data in is a buffer. Unused cells are left out with a
        warning. Buffer ``k`` is the MATLAB cell ``RcvData{k + 1}``.
        """
        if _verasonics_bin.is_binary_workspace(self):
            stored = _verasonics_bin.available_buffers(self)
        else:
            stored = list(range(self.get_reference_size(self._rcvdata())))
        if "Receive" not in self:
            return stored

        used = set(np.unique(self.read_receive_bufnums()).tolist())
        for k in stored:
            if k not in used:
                log.warning(f"Skipping {rf_buffer_name(k)}: no Receive entry stores data in it.")
        return [k for k in stored if k in used]

    def _select_buffers(self, buffers):
        # Validate the requested 0-based buffers (None = all available) and return them as a list.
        available = self.available_buffers()
        buffers = available if buffers is None else list(buffers)
        repeated = sorted({k for k in buffers if buffers.count(k) > 1})
        if repeated:
            # Each buffer becomes the track rf_buffer_{k}, so a repeat would duplicate a label.
            raise ValueError(f"RF buffers {repeated} (0-based) are selected more than once.")
        missing = sorted(set(buffers) - set(available))
        if missing:
            names = ", ".join(rf_buffer_name(k) for k in available) or "none"
            raise ValueError(
                f"RF buffers {missing} (0-based) are not in this file or have no Receive "
                f"entries. Available: {names}."
            )
        if not buffers:
            raise ValueError("No RF buffers to convert.")
        return buffers

    def _rcvdata(self):
        if "RcvData" not in self:
            raise KeyError(
                f"'RcvData' not found in {self.filename}. Save the full workspace "
                "after the acquisition, e.g. save('raw_data.mat', '-v7.3')."
            )
        return self["RcvData"]

    def read_raw_buffer_array(self, buffer_index):
        """Return ``RcvData{buffer_index + 1}`` as ``(n_frames, n_channels, n_samples)`` int16."""
        if _verasonics_bin.is_binary_workspace(self):
            return _verasonics_bin.read_buffer(self, buffer_index)
        raw_data = self.dereference_index(self._rcvdata(), buffer_index)
        return np.asarray(raw_data, dtype=np.int16)

    def read_raw_data(
        self, frames="all", buffer_index=0, first_frame_idx=None, connector_applied=None
    ):
        """
        Read the raw data from the file.

        Args:
            connector_applied (bool, optional): Whether the raw data was already reordered
                from hardware channels to probe elements with ``Trans.ConnectorES`` before
                saving. Only used when the probe has a connector. Defaults to None, which
                applies the connector for files saved with ``save_raw_version`` or with more
                channels than elements, and raises a ``ValueError`` otherwise, because such
                files can be either.

        Returns:
            raw_data (np.ndarray): The raw data of shape (n_rcv, n_samples).
        """

        raw_data = self.read_raw_buffer_array(buffer_index)

        # Convert the raw data to a numpy array to allow out-of-order indexing later
        raw_data = np.asarray(raw_data, dtype=np.int16)

        # Add n_frames dimension if it is missing
        if raw_data.ndim == 2:
            raw_data = np.expand_dims(raw_data, axis=0)

        assert raw_data.ndim == 3, (
            "Expected raw data to have 3 dimensions at this point "
            f"(n_frames, n_channels, n_samples), but got {raw_data.shape}."
        )

        # Map hardware channels to probe elements (see VerasonicsProbe.connector).
        n_channels = raw_data.shape[1]
        connector = self.probe.connector
        n_elements = self.probe.geometry.shape[0]

        if connector is not None and connector_applied is None:
            if self.is_new_save_raw_format or n_channels != n_elements:
                connector_applied = False
            else:
                raise ValueError(
                    "Cannot tell whether the raw data was already reordered with "
                    "Trans.ConnectorES before saving: the file has as many channels as "
                    f"probe elements ({n_elements}) and no 'save_raw_version'. Set "
                    "connector_applied (convert.yaml, or --connector-applied True/False) "
                    "to True if it was reordered, or False to apply the connector."
                )

        if connector is not None and connector_applied:
            if n_channels != n_elements:
                raise ValueError(
                    "connector_applied=True, but the raw data has "
                    f"{n_channels} channels and the probe has {n_elements} elements."
                )
        elif connector is not None:
            if connector.size != n_elements:
                raise ValueError(
                    f"Trans.ConnectorES has {connector.size} entries but the probe has "
                    f"{n_elements} elements."
                )
            if connector.min() < 0 or connector.max() >= n_channels:
                raise ValueError(
                    "Cannot map acquired channels to probe elements: connector "
                    f"indices span [{int(connector.min())}, {int(connector.max())}] "
                    f"but the raw data has only {n_channels} channels."
                )
            if n_channels != n_elements:
                log.info(
                    f"Raw data has {n_channels} channels but the probe has "
                    f"{n_elements} elements; selecting element channels via "
                    "Trans.ConnectorES."
                )
            raw_data = raw_data[:, connector, :]
        elif n_channels != n_elements:
            log.warning(
                f"Raw data has {n_channels} channels but the probe has "
                f"{n_elements} elements, and no usable connector map "
                "(Trans.ConnectorES) was found. Assuming the channel order matches "
                "the probe element order. Please verify that this is correct!"
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
        n_ax = self.get_n_ax(buffer_index)
        final_sample_in_buffer = int(self.get_end_samples(buffer_index).max())
        raw_data = raw_data[:, :, :final_sample_in_buffer]

        # Determine n_tx based on the final sample in buffer and n_ax
        # For some sequences, transmits are already aggregated in the raw data
        # (e.g. harmonic imaging through pulse inversion)
        n_tx = final_sample_in_buffer // n_ax

        # Reshape the raw data to (n_frames, n_el, n_tx, n_ax)
        raw_data = raw_data.reshape((raw_data.shape[0], raw_data.shape[1], n_tx, n_ax))

        # Transpose the raw data to (n_frames, n_tx, n_ax, n_el)
        raw_data = np.transpose(raw_data, (0, 2, 3, 1))

        # Add channel dimension
        raw_data = raw_data[..., None]

        if self.get_is_baseband_mode(buffer_index):
            # BS100BW takes a sample pair every carrier period, BS50BW every two.
            raw_data = baseband_mode_to_iq(
                raw_data, carrier_periods_per_pair=100 // self.get_sample_mode(buffer_index)
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

    def get_demodulation_frequency(self, buffer_index=None):
        """Demodulation frequency of the probe from the file in Hz.

        Args:
            buffer_index (int, optional): If given, only consider receive events
                belonging to this (0-based) RF buffer. If ``None`` (default), all
                receive events are considered.
        """
        demod_freq = self._read_constant_receive_field(
            "demodFrequency",
            buffer_index,
            lambda value: float(np.asarray(value).reshape(-1)[0]),
            "demodulation frequencies",
        )
        return demod_freq * 1e6

    @property
    def demodulation_frequency(self):
        """Demodulation frequency of the probe from the file in Hz (all buffers)."""
        return self.get_demodulation_frequency()

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
            np.ndarray: The initial times of shape ``(n_rcv,)``.
        """
        depth_to_time = self.wavelength / self.sound_speed
        initial_times = []
        for n in rcv_order:
            start_depth = self.dereference_index(self["Receive"]["startDepth"], n).item()

            initial_times.append(2 * start_depth * depth_to_time)

        return np.stack(initial_times).astype(np.float32)

    @property
    def probe(self) -> "VerasonicsProbe":
        """The probe object from the file."""
        # Cached: it is accessed per transmit, and rebuilding it re-reads Trans and
        # repeats the unknown-probe warning.
        cache = self.__dict__
        if "_probe" not in cache:
            cache["_probe"] = VerasonicsProbe(self)
        return cache["_probe"]

    def read_focus_distances(self, tx_order):
        """Reads the focus distances from the file.

        Args:
            tx_order (list): The order in which the transmits appear in the events.

        Returns:
            np.ndarray: The focus distances of shape ``(n_tx,)`` in meters.
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

        .. note::
            For a 1d uniform linear array, transmits whose t0 delays form a linear ramp
            across the active elements are treated as plane waves and their focus
            distance is set to infinity.

        .. note::
            For a curved/arc array, a "flat-fire" transmit is a diverging wave rather
            than a plane wave, so this normalization does not apply; an informational
            log is emitted and the focus distances are returned unchanged.

        .. note::
            For any other (unrecognized) probe geometry, a warning is logged and the
            focus distances are returned unchanged.
        """
        is_ordered, kind = self.probe._probe_geometry_ordering
        if kind != "linear":
            if kind == "arc":
                # Valid curved/convex array: the linear-ramp plane-wave test below
                # does not apply (a "flat-fire" transmit on a curved array is a
                # diverging wave, not a plane wave). Keep the focus distances as
                # read from the file — for a diverging transmit that is a finite
                # (often 0 or negative) value, not infinity.
                log.info(
                    "Curved/arc probe geometry detected; keeping transmit focus "
                    "distances as read (plane-wave-to-infinity normalization only "
                    "applies to linear arrays)."
                )
            else:
                log.warning(
                    "The probe geometry is not a recognized ordered array (linear or "
                    "arc). Focal distances are not set to infinity for plane waves."
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

    def get_sample_mode(self, buffer_index=None):
        """Receive bandwidth as a percentage of center frequency.

        Args:
            buffer_index (int, optional): If given, only consider receive events
                belonging to this (0-based) RF buffer. If ``None`` (default), all
                receive events are considered.
        """
        SUPPORTED_SAMPLE_MODES = ["NS200BW", "BS100BW", "BS67BW", "BS50BW"]

        sample_mode = self._read_constant_receive_field(
            "sampleMode", buffer_index, self.decode_string, "sample modes"
        )

        # Check if the sample mode is supported, and extract the percentage
        assert sample_mode in SUPPORTED_SAMPLE_MODES, (
            f"Unexpected sample mode '{sample_mode}' in file."
            f"Expected one of {SUPPORTED_SAMPLE_MODES}"
        )
        return int(sample_mode[2:-2])

    @property
    def sample_mode(self):
        """Receive bandwidth as a percentage of center frequency (all buffers)."""
        return self.get_sample_mode()

    def get_is_baseband_mode(self, buffer_index=None):
        """If the data is captured in 'BS100BW' mode or 'BS50BW' mode.

        - The data is stored as complex IQ data.
        - The sampling frequency is halved.
        - Two sequential samples are interpreted as a single complex sample.
          Therefore, we need to halve the sampling frequency.

        Args:
            buffer_index (int, optional): If given, only consider receive events
                belonging to this (0-based) RF buffer. If ``None`` (default), all
                receive events are considered.
        """
        return self.get_sample_mode(buffer_index) in (50, 100)

    @property
    def is_baseband_mode(self):
        """If the data is captured in 'BS100BW' or 'BS50BW' mode (all buffers)."""
        return self.get_is_baseband_mode()

    def _read_tgc_waveform(self, buffer_index=None):
        """Read the raw ``TGC.Waveform`` samples used by the selected buffer's receives.

        Returns:
            np.ndarray: 1-D array of the TGC control samples.
        """
        waveform_ds = self["TGC"]["Waveform"]
        if not _is_reference_dataset(waveform_ds):
            # Single TGC struct: plain (n_samples, 1) matrix.
            return np.asarray(waveform_ds[:])[:, 0]

        # Multiple TGC structs: pick the one this buffer's receives reference.
        n_tgc = self.get_reference_size(waveform_ds)
        tgc_index = self._receive_tgc_index(buffer_index)
        if not 0 <= tgc_index < n_tgc:
            log.warning(
                f"TGC index {tgc_index} is out of range for {n_tgc} TGC waveform(s); "
                "using the first."
            )
            tgc_index = 0
        return np.asarray(self.dereference_index(waveform_ds, tgc_index)).reshape(-1)

    def _receive_tgc_index(self, buffer_index=None):
        """0-based index of the TGC waveform referenced by the selected buffer's receives."""
        if "TGC" not in self["Receive"]:
            return 0
        if buffer_index is None:
            recv_idx = np.arange(self.get_reference_size(self["Receive"]["TGC"]))
        else:
            recv_idx = self.receive_indices_for_buffer(buffer_index)
        tgc_values = np.unique(
            np.round(self._read_scalar_field("Receive", "TGC")[recv_idx]).astype(int)
        )
        if tgc_values.size != 1 and buffer_index is not None:
            # A track stores a single tgc_gain_curve, so a mixed buffer cannot be stored.
            raise ValueError(
                f"The receives of {rf_buffer_name(buffer_index)} reference several TGC "
                f"waveforms {tgc_values.tolist()}. We do not support this case yet."
            )
        if tgc_values.size != 1:
            log.warning(
                f"Receives for buffer_index={buffer_index} reference multiple TGC "
                f"waveforms {tgc_values.tolist()}; using the first."
            )
        return int(tgc_values[0]) - 1  # Verasonics TGC numbers are 1-based

    def get_tgc_gain_curve(self, buffer_index=None):
        """The TGC gain curve interpolated to the number of axial samples (n_ax,).

        Args:
            buffer_index (int, optional): The 0-based RF buffer. If ``None``
                (default), all receive events are considered.
        """

        gain_curve = self._read_tgc_waveform(buffer_index)

        # Normalize the gain_curve to [0, 40]dB
        gain_curve = gain_curve / 1023 * 40

        # The gain curve is sampled at 800ns (See Verasonics documentation for details.
        # Specifically the tutorial sequence programming)
        gain_curve_sampling_period = 800e-9

        # Define the time axis for the gain curve
        t_gain_curve = np.arange(gain_curve.size) * gain_curve_sampling_period

        # For baseband mode two consecutive samples are combined into a single complex sample
        base_n_ax = self.get_n_ax(buffer_index)
        n_ax = base_n_ax if not self.get_is_baseband_mode(buffer_index) else base_n_ax // 2

        # Define the time axis for the axial samples
        t_samples = np.arange(n_ax) / self.get_sampling_frequency(buffer_index)

        # Interpolate the gain_curve to the number of axial samples
        gain_curve = np.interp(t_samples, t_gain_curve, gain_curve)

        # The gain_curve gains are in dB, so we need to convert them to linear scale
        gain_curve = 10 ** (gain_curve / 20)

        return gain_curve

    @property
    def tgc_gain_curve(self):
        """The TGC gain curve interpolated to the number of axial samples (all buffers)."""
        return self.get_tgc_gain_curve()

    def get_image_data_p_frame_order(self, image_buffer_index=0):
        """The order of frames in the ImgDataP buffer.

        Because of the circular buffer used in Verasonics, the frames in the ImgDataP
        buffer are not necessarily in the correct order. This function computes the
        correct order of frames.

        Args:
            image_buffer_index (int, optional): The 0-based image buffer
                (``Resource.ImageBuffer``) to compute the frame order for.
                Defaults to 0.
        """
        n_frames = self.dereference_index(
            self["Resource"]["ImageBuffer"]["numFrames"], image_buffer_index
        )
        n_frames = self.cast_to_integer(n_frames)
        try:
            first_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["firstFrame"], image_buffer_index
            )
            last_frame = self.dereference_index(
                self["Resource"]["ImageBuffer"]["lastFrame"], image_buffer_index
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

    def read_image_data_p(self, frames="all", image_buffer_index=0):
        """Reads the image data from the file.

        Uses the ``ImgDataP`` buffer, which is used for spatial filtering
        and persistence processing. Generally, this buffer does not contain the same frames
        as the raw data buffer. This happens because the Verasonics often does not reconstruct
        every acquired frame. This means that the images in this buffer often skip frames, and
        span a longer time period than the raw data buffer.

        Args:
            frames (str or list, optional): The frames to read. Defaults to "all".
            image_buffer_index (int, optional): The 0-based image buffer
                (``Resource.ImageBuffer``) to read from. Image buffers are indexed
                independently from the RF buffers. Defaults to 0.

        Returns:
            `image_data` (`np.ndarray`): The image data.
        """
        # Check if the file contains image data
        if "ImgDataP" not in self:
            return None

        n_image_buffers = self.get_reference_size(self["ImgDataP"])
        if image_buffer_index < 0 or image_buffer_index >= n_image_buffers:
            raise ValueError(
                f"image_buffer_index={image_buffer_index} is out of range; the file "
                f"contains {n_image_buffers} image buffer(s) (valid indices "
                f"0..{n_image_buffers - 1})."
            )

        # Dereference the dataset
        image_data = self.dereference_index(self["ImgDataP"], image_buffer_index)
        image_data = np.asarray(image_data)

        # Re-order images such that sequence is correct.
        indices = self.get_image_data_p_frame_order(image_buffer_index)
        image_data = image_data[indices, :, :]

        # Normalize and log-compress the image data
        image_data = normalize(image_data, output_range=(0, 1), input_range=(0, None))
        image_data = log_compress(image_data)
        image_data = ops.convert_to_numpy(image_data)

        # Select only the requested frames
        n_image_frames = image_data.shape[0]
        frame_indices = self.get_frame_indices(frames, n_frames=n_image_frames)
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

        if frames is None:
            frames = self.load_convert_config().get("frames", "all")

        tx_order, rcv_order, time_to_next_transmit = self.read_transmit_events(
            frames=frames, allow_accumulate=allow_accumulate, buffer_index=buffer_index
        )
        initial_times = self.read_initial_times(rcv_order)

        # Read both angles at once to loop over TX.Steer only once.
        beamsteering_angles = self.read_beamsteering_angles(tx_order)
        polar_angles = beamsteering_angles[:, 0]
        azimuth_angles = beamsteering_angles[:, 1]
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

        sampling_frequency = self.get_sampling_frequency(buffer_index)
        demodulation_frequency = self.get_demodulation_frequency(buffer_index)
        sound_speed = self.sound_speed
        tgc_gain_curve = self.get_tgc_gain_curve(buffer_index)

        return {
            "time_to_next_transmit": time_to_next_transmit,
            "t0_delays": t0_delays,
            "tx_apodizations": tx_apodizations,
            "sampling_frequency": sampling_frequency,
            "polar_angles": polar_angles,
            "azimuth_angles": azimuth_angles,
            "center_frequency": center_frequency,
            "demodulation_frequency": demodulation_frequency,
            "sound_speed": sound_speed,
            "initial_times": initial_times,
            "focus_distances": focus_distances,
            "transmit_origins": transmit_origins,
            "waveforms_one_way": waveforms_one_way,
            "waveforms_two_way": waveforms_two_way,
            "tgc_gain_curve": tgc_gain_curve,
        }

    def read_verasonics_file(
        self,
        frames=None,
        allow_accumulate=False,
        buffers=None,
        additional_functions=None,
        lens_sound_speed: float = 1000.0,
        connector_applied=None,
    ):
        """Reads data from a .mat Verasonics output file.

        Args:
            frames (str or list of int, optional): The frames to add to the file. This can be
                a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
                None, which means all frames, unless specified in a `convert.yaml` file.
                Selecting frames is only supported for a single RF buffer.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
            buffers (list[int], optional): 0-based RF buffers to read, so 0 is
                ``RcvData{1}``. Buffer ``k`` becomes the track ``rf_buffer_{k}``. Defaults
                to the ``buffers`` in a `convert.yaml` file, or else every buffer in the
                file (see :meth:`available_buffers`).
            additional_functions (list, optional): A list of functions that read additional
                data from the file. Each function should take the `VerasonicsFile` as input
                and return a `CustomElement`. Defaults to None.
            lens_sound_speed (float, optional): Speed of sound in the lens material in m/s.
                Used to convert the Verasonics scalar lens correction (one-way delay in
                wavelengths) into ``lens_thickness`` and ``lens_sound_speed`` fields on the
                probe dict.  Only applied when the file contains a ``lensCorrection`` field.
                Defaults to 1000.0.
            connector_applied (bool, optional): See :meth:`read_raw_data`. Defaults to the
                ``connector_applied`` in a `convert.yaml` file, or else None.

        Returns:
            tuple: ``(tracks, probe_dict, custom_elements)`` with one track dict
            (``data``, ``scan``, ``label``) per RF buffer.
        """

        if additional_functions is None:
            additional_functions = []

        convert_config = self.load_convert_config()
        first_frame = convert_config.get("first_frame", None)
        if frames is None:
            frames = convert_config.get("frames", "all")
        if buffers is None:
            buffers = convert_config.get("buffers", None)
        if connector_applied is None:
            connector_applied = convert_config.get("connector_applied", None)

        buffers = self._select_buffers(buffers)
        if len(buffers) > 1 and (not _is_all_frames(frames) or first_frame is not None):
            raise NotImplementedError(
                "Selecting frames ('frames' / 'first_frame') is not supported for files "
                f"with several RF buffers ({len(buffers)} buffers): the buffers generally "
                "hold different numbers of frames. Select one 0-based buffer with "
                "'buffers' in convert.yaml or with --buffers, e.g. 0 for RcvData{1}."
            )

        tracks = []
        for buffer_index in buffers:
            log.info(f"Reading {rf_buffer_name(buffer_index)}...")
            try:
                scan_dict = self.read_scan(
                    frames=frames,
                    allow_accumulate=allow_accumulate,
                    buffer_index=buffer_index,
                )
                raw_data = self.read_raw_data(
                    frames=frames,
                    buffer_index=buffer_index,
                    first_frame_idx=first_frame,
                    connector_applied=connector_applied,
                )
            except Exception as e:
                e.add_note(f"While reading {rf_buffer_name(buffer_index)} of {self.filename}.")
                raise
            tracks.append(
                {
                    "data": {"raw_data": raw_data},
                    "scan": scan_dict,
                    "label": f"rf_buffer_{buffer_index}",
                }
            )

        custom_elements = []

        if self.probe.lens_correction is not None:
            el_lens_correction = CustomElement(
                name="lens_correction",
                data=self.probe.lens_correction,
                description=(
                    "The lens correction value used by Verasonics. This value is a "
                    "scalar one-way delay offset in wavelengths applied uniformly across "
                    "all elements (disregards refraction). "
                    "This is not directly compatible with zea's lens correction, which "
                    "uses Fermat's principle (Newton-Raphson) to find the shortest "
                    "refracted path per element-pixel pair."
                ),
                unit="wavelengths",
            )
            custom_elements.append(el_lens_correction)

        # Add additional elements from user-defined functions
        for additional_function in additional_functions:
            custom_elements.append(additional_function(self))

        # ImgDataP buffers are independent of the RF buffers, so store all of them.
        n_image_buffers = self.get_reference_size(self["ImgDataP"]) if "ImgDataP" in self else 0
        if n_image_buffers == 0:
            log.info("No ImgDataP image buffer found in file; skipping.")
        for image_buffer_index in range(n_image_buffers):
            custom_elements.append(
                CustomElement(
                    name=f"verasonics_image_buffer_{image_buffer_index}",
                    data=self.read_image_data_p(
                        frames=frames, image_buffer_index=image_buffer_index
                    ),
                    description=(
                        f"The Verasonics ImgDataP{{{image_buffer_index + 1}}} buffer. "
                        "WARNING: This buffer may skip frames compared to the raw data! "
                        "Use only for reference."
                    ),
                    unit="unitless",
                )
            )

        probe_dict = self.probe.to_probe_spec()
        f_c = self.probe.center_frequency
        probe_dict.update(
            estimate_lens_probe_params(self.probe.lens_correction, f_c, lens_sound_speed)
        )

        return tracks, probe_dict, custom_elements

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

    def get_frame_indices(self, frames, buffer_index=0, n_frames=None):
        """Creates a numpy array of frame indices from the file and the frames argument.

        Args:
            frames (str): The frames argument. This can be "all", a range of integers
                (e.g. "4-8"), or a list of frame indices.
            buffer_index (int, optional): The 0-based RF buffer whose frame count
                is used. Ignored when ``n_frames`` is given. Defaults to 0.
            n_frames (int, optional): Number of available frames, e.g. for the
                ``ImgDataP`` image buffer. Defaults to None.

        Returns:
            frame_indices (np.ndarray): The frame indices.
        """
        if n_frames is None:
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
        frames=None,
        allow_accumulate=False,
        additional_functions=None,
        lens_sound_speed: float = 1000.0,
        buffers=None,
        connector_applied=None,
    ):
        """Converts the Verasonics file to the zea format.

        Every RF buffer becomes one track in a single zea file: 0-based buffer ``k``
        (``RcvData{k + 1}``) is labelled ``rf_buffer_{k}``. A file with one RF buffer
        gives a single-track file.

        Args:
            output_path (str): The path to the output file (.hdf5 file).
            frames (str or list of int, optional): The frames to add to the file. This can be
                a list of integers, a range of integers (e.g. 4-8), or 'all'. Defaults to
                None, which means all frames are used, unless specified otherwise in a
                `convert.yaml` file. Only supported for a single RF buffer.
            allow_accumulate (bool, optional): Sometimes, some transmits are already accumulated
                on the Verasonics system (e.g. harmonic imaging through pulse inversion).
                In this case, the mode in the Receive structure is set to 1 (accumulate).
                If this flag is set to False, an error is raised when such a mode is detected.
                Defaults to False.
            additional_functions (list, optional): A list of functions that read additional
                data from the file. Each function should take the `VerasonicsFile` as input
                and return a `CustomElement`. Defaults to None.
            lens_sound_speed (float, optional): Speed of sound in the lens material in m/s.
                Used to convert ``Trans.lensCorrection`` (wavelengths) into
                ``lens_thickness`` and ``lens_sound_speed`` fields on the probe. Defaults
                to 1000.0.
            buffers (list[int], optional): 0-based RF buffers to convert, so 0 is
                ``RcvData{1}``. Defaults to the
                ``buffers`` in a `convert.yaml` file, or else every buffer in the file
                (see :meth:`available_buffers`).
            connector_applied (bool, optional): See :meth:`read_raw_data`. Defaults to the
                ``connector_applied`` in a `convert.yaml` file, or else None.
        """
        tracks, probe_dict, custom_elements = self.read_verasonics_file(
            frames=frames,
            allow_accumulate=allow_accumulate,
            buffers=buffers,
            additional_functions=additional_functions,
            lens_sound_speed=lens_sound_speed,
            connector_applied=connector_applied,
        )

        log.info("Generating zea dataset...")
        File.create(
            path=output_path,
            tracks=tracks,
            probe=probe_dict,
            description="Verasonics data",
            custom=custom_elements,
        )


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
        bandwidth = self.bandwidth
        if bandwidth is not None:
            assert bandwidth[1] > bandwidth[0], "Bandwidth must be positive"
            diff = bandwidth[1] - bandwidth[0]
            return 100 * (diff / self.center_frequency)

    @property
    def type(self):
        """The type of the probe from the file.

        ``None`` if the type is missing or not recognized.
        """
        _id_to_str = {
            0: "linear",
            1: "curved",
            2: "2D-array",
            3: "annular",
            4: "row-column",
        }
        if "type" not in self.trans_obj.keys():
            return None

        probe_type_id = int(self.trans_obj["type"][:].item())
        probe_type = _id_to_str.get(probe_type_id)
        if probe_type is None:
            log.warning(
                f"Unrecognized Trans.type id {probe_type_id}; expected one of "
                f"{sorted(_id_to_str)}. Leaving the probe type unset."
            )
        return probe_type

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
        """Probe connector indices (0-based) mapping elements to hardware channels.

        ``None`` if the transducer has no ``ConnectorES`` field, in which case the
        channel order is taken to match the element order. Whether the connector is
        applied to the raw data is decided by ``connector_applied`` in
        :meth:`VerasonicsFile.read_raw_data`.
        """
        if "ConnectorES" not in self.trans_obj.keys():
            return None
        connector = self.trans_obj["ConnectorES"][:]
        connector = np.squeeze(connector, axis=0)
        connector = connector.astype(np.int32)
        connector = connector - 1  # make 0-based
        return connector

    @property
    def lens_correction(self):
        """The lens correction: 1 way delay in wavelengths through lens.

        This is the Verasonics scalar offset added uniformly to all element delays.
        It is not equivalent to zea's lens correction, which uses Fermat's principle
        (solved via Newton-Raphson) to find the shortest refracted path through the
        lens geometry per element-pixel pair.
        """

        if "lensCorrection" in self.trans_obj.keys():
            return self.trans_obj["lensCorrection"][:].item()

    @property
    def _probe_geometry_ordering(self):
        """``(is_ordered, kind)`` for the geometry; see :func:`classify_ordered_geometry`."""
        return classify_ordered_geometry(self.geometry)

    @property
    def _probe_geometry_is_ordered_ula(self):
        """Checks if the probe geometry is ordered as a uniform linear array (ULA)."""
        is_ordered, kind = self._probe_geometry_ordering
        return is_ordered and kind == "linear"

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


def _zea_from_verasonics_workspace(input_path, output_path, **kwargs):
    """Helper function around ``VerasonicsFile.to_zea``"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
            - buffers (list[int] | None): 0-based RF buffers to convert (all if None).
            - connector_applied (bool | None): See :meth:`VerasonicsFile.read_raw_data`."""

    if getattr(args, "upload", False):
        assert args.hf_repo_id, "hf_repo_id must be provided when --upload is True."
        assert args.revision, "revision must be provided when --upload is True."

    # Variable to indicate what to do with existing files.
    # Is set by the user in case these are found.
    existing_file_policy = None

    selected_path = Path(args.src)
    selected_path_is_directory = os.path.isdir(selected_path)

    output_path = Path(args.dst)
    if selected_path.is_file() and output_path.suffix not in (".hdf5", ".h5"):
        log.error(
            "When converting a single file, the output path should have the .hdf5 or .h5 extension."
        )
        sys.exit(2)
    elif selected_path.is_dir() and output_path.is_file():
        log.error("When converting a directory, the output path should be a directory.")
        sys.exit(2)

    if output_path.is_dir() and not selected_path_is_directory:
        output_path = output_path / (selected_path.name + "_zea.hdf5")

    log.info(f"Selected path: {log.yellow(selected_path)}")

    num_converted = 0

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

    for full_path, file_output_path in file_pairs:
        if file_output_path.is_file():
            answer = None
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

            if existing_file_policy == "skip" or answer is False:
                log.info("Skipping...")
                continue

            log.warning(f"{log.yellow(file_output_path)} exists. Deleting...")
            file_output_path.unlink()

        try:
            _zea_from_verasonics_workspace(
                full_path,
                file_output_path,
                frames=args.frames,
                allow_accumulate=args.allow_accumulate,
                buffers=getattr(args, "buffers", None),
                connector_applied=getattr(args, "connector_applied", None),
            )
            num_converted += 1
        except Exception:
            log.error(f"Failed to convert {full_path.name}")
            traceback.print_exc()

    # Do not write a dataset card or upload anything when nothing was converted
    # (e.g. all files failed or were skipped).
    if getattr(args, "upload", False) and num_converted == 0:
        log.error("No files were converted successfully; skipping upload.")
        return

    # Write the dataset card next to the converted output: in the output
    # directory for a directory conversion, or alongside the file for a single
    # file. The card is required for the upload ownership check below.
    if selected_path_is_directory and args.hf_repo_id:
        write_dataset_card(output_path, make_dataset_card(args.hf_repo_id))
    elif getattr(args, "upload", False) and args.hf_repo_id:
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
