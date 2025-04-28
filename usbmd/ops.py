"""Deprecated ops module"""

import importlib
from abc import ABC, abstractmethod

import keras
import numpy as np
from keras import ops
from scipy import ndimage

import usbmd.beamformer as bmf
from usbmd import display
from usbmd.config import Config
from usbmd.ops_v2 import (
    channels_to_complex,
    complex_to_channels,
    demodulate_not_jitable,
    get_band_pass_filter,
    get_low_pass_iq_filter,
    hilbert,
    upmix,
)
from usbmd.probes import Probe
from usbmd.registry import ops_registry
from usbmd.scan import Scan
from usbmd.tensor_ops import patched_map
from usbmd.utils import deprecated, lens_correction, log, pfield
from usbmd.utils.checks import get_check

# make sure to reload all modules that import keras
# to be able to set backend properly
importlib.reload(bmf)
importlib.reload(pfield)
importlib.reload(lens_correction)
importlib.reload(display)

# clear registry upon import
ops_registry.clear()

log.warning("The ops module is deprecated. Please use usbmd.ops_v2 instead.")


def get_ops(ops_name):
    """Get the operation from the registry."""
    return ops_registry[ops_name]


class Operation(ABC):
    """Basic operation class as building block for processing pipeline and standalone operations."""

    @deprecated(replacement="usbmd.ops_v2")
    def __init__(
        self,
        input_data_type=None,
        output_data_type=None,
        with_batch_dim=True,
    ):
        """Initialize the operation.

        Args:
            input_data_type (type): The expected data type of the input data.
            output_data_type (type): The expected data type of the output data.
            with_batch_dim (bool): Whether the input data has a batch dimension.
        """
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.with_batch_dim = with_batch_dim

        self.config = None
        self.scan = None
        self.probe = None

        # this means that the operation doesn't change the data type
        if self.input_data_type is not None and self.output_data_type is None:
            self.output_data_type = self.input_data_type

    @property
    def name(self):
        """Return the name of the registered operation."""
        names = ops_registry.registry.keys()
        classes = ops_registry.registry.values()
        return list(names)[list(classes).index(self.__class__)]

    @abstractmethod
    def process(self, data):
        """Process the input data through the operation.

        Args:
            data: The input data to be processed.

        Returns:
            The processed data.

        """
        return data

    def __call__(self, data, *args, **kwargs):
        """Call the operation on the input data.

        Args:
            data: The input data to be processed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The processed data.

        """
        if self.input_data_type:
            check = get_check(self.input_data_type)
            check(data, with_batch_dim=self.with_batch_dim)
        return self.process(data, *args, **kwargs)

    @property
    def _ready(self):
        """Check if the operation is ready to be used.

        Returns:
            bool: True if the operation is ready, False otherwise.

        """
        return True

    def initialize(self):
        """Initialize the operation."""
        if not self._ready:
            raise ValueError(
                f"Operation {self.__class__.__name__} is not ready to be used, "
                "please set parameters: "
                "either using `op.set_params(config, scan, probe)` or "
                "manually setting the parameters during initialization."
            )

    def set_params(self, config: Config, scan: Scan, probe: Probe, override=False):
        """Set the parameters for the operation.

        Parameters are assigned to the operation from the config, scan, and probe
        and in that order of priority (i.e. config > scan > probe will be assigned
        for shared parameters between the three).

        Args:
            config (Config): Configuration parameters for the operation.
            scan (Scan): Scan parameters for the operation.
            probe (Probe): Probe parameters for the operation.
            override (bool): Whether to override parameters if they are already set.
                Defaults to False, meaning that only unset parameters will be set
                from the config, scan, and probe.
        """
        # combine all params with priority config > scan > probe
        # combine the dicts
        params = {}
        if config is not None:
            params.update(self._assign_config_params(config))
        if scan is not None:
            params.update(self._assign_scan_params(scan))
        if probe is not None:
            params.update(self._assign_probe_params(probe))

        for attr, value in params.items():
            if value is None:  # don't override with None values
                continue
            # only override if override is True or the attribute is not set yet
            if override or getattr(self, attr) is None:
                setattr(self, attr, value)

    def propagate_params(self, scan: Scan):
        """Update the parameters for the operation.

        Args:
            scan (Scan): Scan class with parameters passed from
                the previous operation in the pipeline.

        """
        updated_params = self._assign_update_params(scan)

        for attr, value in updated_params.items():
            if value is None:
                continue
            if not hasattr(scan, attr):
                log.warning(
                    f"Parameter {attr} is not part of the scan "
                    "class and cannot be updated. Please check "
                    f"{self.__class__.__name__}._assign_update_params for "
                    "faulty scan parameters."
                )
                continue
            setattr(scan, attr, value)
        return scan

    # pylint: disable=unused-argument
    def _assign_config_params(self, config: Config):
        """Return the config parameters for the operation.

        Args:
            config (Config): Configuration parameters for the operation.

        Returns:
            dict: The config parameters for the operation.

        """
        return {}

    # pylint: disable=unused-argument
    def _assign_scan_params(self, scan: Scan):
        """Return the scan parameters for the operation.

        Args:
            scan (Scan): Scan parameters for the operation.

        Returns:
            dict: The scan parameters for the operation.

        """
        return {}

    # pylint: disable=unused-argument
    def _assign_probe_params(self, probe: Probe):
        """Return the probe parameters for the operation.

        Args:
            probe (Probe): Probe parameters for the operation.

        Returns:
            dict: The probe parameters for the operation.

        """
        return {}

    # pylint: disable=unused-argument
    def _assign_update_params(self, scan: Scan):
        """Update the parameters for remaining operations in the pipeline.

        Args:
            scan (Scan): Scan class with parameters passed from
                the previous operation in the pipeline.

        """
        return {}

    def prepare_tensor(self, x, dtype=None):
        """Convert input array to appropriate tensor type for the operations package.

        Args:
            x: The input array to be converted.
            dtype: The desired data type of the converted tensor.
            device: The desired device for the converted tensor.

        Returns:
            The converted tensor.

        Raises:
            ValueError: If the operations package is not supported.

        """
        return ops.convert_to_tensor(x, dtype=dtype)

    def to_numpy(self, x):
        """Convert tensor to numpy array.

        Args:
            x: The input tensor to be converted.

        Returns:
            The converted numpy array.

        """
        return ops.convert_to_numpy(x)


class Pipeline:
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(self, operations, with_batch_dim=True, device=None):
        """Initialize a pipeline

        Args:
            operations (list): A list of Operation instances representing the operations
                to be performed.
            ops (module, str, optional): The type of operations to use. Defaults to "numpy".
            with_batch_dim (bool, optional): Whether to include batch dimension in the operations.
                Defaults to True.
            device (str, optional): The device to use for the operations. Defaults to None.
                Can be `cpu` or `cuda`, `cuda:0`, etc.
        """

        self.operations = operations

        self.device = self._check_device(device)

        for operation in self.operations:
            # operation.ops = ops
            operation.with_batch_dim = with_batch_dim

        # check if the operations are compatible
        for i in range(len(self.operations) - 1):
            if self.operations[i].output_data_type is None:
                continue
            if self.operations[i + 1].input_data_type is None:
                continue
            if (
                self.operations[i].output_data_type
                != self.operations[i + 1].input_data_type
            ):
                raise ValueError(
                    f"Operation {self.operations[i].name} output data type is not compatible "
                    f"with the input data type of operation {self.operations[i + 1].name}"
                )

        self._jitted_process = None

    def __str__(self):
        """String representation of the pipeline.

        Will print on two parallel pipeline lines if it detects a splitting operations
        (such as multi_bandpass_filter)
        Will merge the pipeline lines if it detects a stacking operation (such as stack)
        """
        split_operations = ["MultiBandPassFilter"]
        merge_operations = ["Stack"]

        operations = [operation.__class__.__name__ for operation in self.operations]
        string = " -> ".join(operations)

        if any(operation in split_operations for operation in operations):
            # a second line is needed with same length as the first line
            split_line = " " * len(string)
            # find the splitting operation and index and print \-> instead of -> after
            split_detected = False
            merge_detected = False
            split_operation = None
            for operation in operations:
                if operation in split_operations:
                    index = string.index(operation)
                    index = index + len(operation)
                    split_line = (
                        split_line[:index] + "\\->" + split_line[index + len("\\->") :]
                    )
                    split_detected = True
                    merge_detected = False
                    split_operation = operation
                    continue

                if operation in merge_operations:
                    index = string.index(operation)
                    index = index - 4
                    split_line = split_line[:index] + "/" + split_line[index + 1 :]
                    split_detected = False
                    merge_detected = True
                    continue

                if split_detected:
                    # print all operations in the second line
                    index = string.index(operation)
                    split_line = (
                        split_line[:index]
                        + operation
                        + " -> "
                        + split_line[index + len(operation) + len(" -> ") :]
                    )
            assert merge_detected is True, log.error(
                "Pipeline was never merged back together (with Stack operation), even "
                f"though it was split with {split_operation}. "
                "Please properly define your operation chain."
            )
            return f"\n{string}\n{split_line}\n"

        return string

    def __repr__(self):
        """String representation of the pipeline."""
        operations = [operation.__class__.__name__ for operation in self.operations]
        return ",".join(operations)

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.operations[0].with_batch_dim

    def on_device(self, func, data, device=None, return_numpy=False):
        """On device function for running pipeline on specific device."""
        backend = keras.backend.backend()
        if backend == "numpy":
            return func(data)
        elif backend == "tensorflow":
            on_device_tf = importlib.import_module(
                "usbmd.backend.tensorflow"
            ).on_device_tf
            return on_device_tf(func, data, device=device, return_numpy=return_numpy)
        elif backend == "torch":
            on_device_torch = importlib.import_module(
                "usbmd.backend.torch"
            ).on_device_torch
            return on_device_torch(func, data, device=device, return_numpy=return_numpy)
        elif backend == "jax":
            on_device_jax = importlib.import_module("usbmd.backend.jax").on_device_jax
            return on_device_jax(func, data, device=device, return_numpy=return_numpy)
        else:
            raise ValueError(f"Unsupported operations package {backend}.")

    def set_params(self, config: Config, scan: Scan, probe: Probe, override=False):
        """Set the parameters for the pipeline. See Operation.set_params for more info."""
        scan_objects = [scan]
        for operation in self.operations:
            # set parameters for each operation using initial scan, config, probe
            operation.set_params(config, scan, probe, override=override)
            # also propagate running list of updated parameters to the next operation
            if scan is not None:
                scan = operation.propagate_params(scan.copy())
            else:
                log.warning(
                    "Did not provide a scan object to the pipeline, and therefore "
                    "cannot propagate parameters through the pipeline."
                )
            scan_objects.append(scan)

        return scan_objects

    def process(self, data, return_numpy=False):
        """Process input data through the pipeline."""
        data = ops.convert_to_tensor(data)
        if not all(operation._ready for operation in self.operations):
            operations_not_ready = [
                operation.name for operation in self.operations if not operation._ready
            ]
            raise ValueError(
                log.error(
                    f"Operations {operations_not_ready} are not ready to be used, "
                    "please set parameters using `op.set_params(config, scan, probe)` "
                    "and initialize them using `op.initialize()`."
                )
            )
        if self._jitted_process is None:
            processing_func = self._process
        else:
            processing_func = self._jitted_process

        if self.device:
            return self.on_device(
                processing_func, data, device=self.device, return_numpy=return_numpy
            )
        data_out = processing_func(data)
        if return_numpy:
            return ops.convert_to_numpy(data_out)
        return data_out

    def _process(self, data):
        for operation in self.operations:
            if isinstance(data, list) and operation.__class__.__name__ != "Stack":
                data = [operation(_data) for _data in data]
            else:
                data = operation(data)
        return data

    def initialize(self):
        """Initialize all operations in the pipeline."""
        for operation in self.operations:
            operation.initialize()

    def compile(self, jit=True):
        """Compile the pipeline using jit."""
        backend = keras.backend.backend()
        if not jit:
            return
        log.info(f"Compiling pipeline, with backend {backend}.")
        if backend == "numpy":
            return
        elif backend == "tensorflow":
            tf_function = importlib.import_module("tensorflow").function

            self._jitted_process = tf_function(
                self._process, jit_compile=jit
            )  # tf.function
            return
        elif backend == "torch":
            log.warning("JIT compmilation is not yet supported for torch.")
            return
        elif backend == "jax":
            jax_jit = importlib.import_module("jax").jit
            self._jitted_process = jax_jit(self._process)  # jax.jit
            return

    def prepare_tensor(self, x, dtype=None, device=None):
        """Convert input array to appropriate tensor type for the operations package."""
        if len(self.operations) == 0:
            return x
        return self.operations[0].prepare_tensor(x, dtype=dtype, device=device)

    def _check_device(self, device):
        if device is None:
            return None

        if device == "cpu":
            return "cpu"

        backend = keras.backend.backend()

        if backend == "numpy":
            if device not in [None, "cpu"]:
                log.warning(
                    f"Device {device} is not supported for numpy operations, using cpu."
                )
            return "cpu"

        else:
            # assert device to be cpu, cuda, cuda:{int} or int or None
            assert isinstance(
                device, (str, int)
            ), f"device should be a string or int, got {device}"
            if isinstance(device, str):
                if backend == "tensorflow":
                    assert device.startswith(
                        "gpu"
                    ), f"device should be 'cpu' or 'gpu:*', got {device}"
                elif backend == "torch":
                    assert device.startswith(
                        "cuda"
                    ), f"device should be 'cpu' or 'cuda:*', got {device}"
                elif backend == "jax":
                    assert device.startswith(
                        ("gpu", "cuda")
                    ), f"device should be 'cpu', 'gpu:*', or 'cuda:*', got {device}"
                else:
                    raise ValueError(f"Unsupported backend {backend}.")
            return device


@ops_registry("delay_and_sum_multi")
class DelayAndSumMulti(Operation):
    """
    Sums time-delayed signals along channels and transmits for a list of receive apodizations.
    Each receive apodization in the list will generate a separate output in a list
    """

    def __init__(self, rx_apo=None, tx_apo=None, patches=1, **kwargs):
        """
        Args:
            rx_apo (list, optional):  Receive apodization windows. Defaults to None.
            tx_apo (array, optional): Transmit apodization window. Defaults to None.
            patches (int, optional): Number of patches to split the data into. Defaults to 1.
        """
        super().__init__(
            input_data_type=None,
            output_data_type="beamformed_data",
            **kwargs,
        )
        self.rx_apo = rx_apo
        self.tx_apo = tx_apo
        self.patches = patches
        self.rx_apo_ind = 0

    def initialize(self):
        if self.rx_apo is None:
            self.rx_apo = [
                1.0,
            ]  # single branch - standard das

        if self.tx_apo is None:
            self.tx_apo = 1.0

    def process_patch(self, patch):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_pix, n_tx, n_el, n_ch)`

        Returns:
            ops.Tensor: The beamformed data of shape `(n_pix, n_ch)`
        """
        # Sum over the channels, i.e. DAS
        data = ops.sum(self.rx_apo[self.rx_apo_ind] * patch, -2)

        # Sum over transmits, i.e. Compounding
        data = self.tx_apo * data
        data = ops.sum(data, 1)

        return data

    def process_item(self, data):
        """Performs DAS beamforming on tof-corrected input. Optionally splits the data into patches.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_tx, n_z, n_x, n_el, n_ch)`

        Returns:
            ops.Tensor: The beamformed data of shape `(n_z, n_x, n_ch)`
        """
        n_tx, n_z, n_x, n_el, n_ch = data.shape

        # Flatten grid and move n_pix=(n_z * n_x) to the front
        flat_data = ops.reshape(data, (n_tx, -1, n_el, n_ch))
        flat_data = ops.moveaxis(flat_data, 1, 0)

        data = []
        for i in range(0, len(self.rx_apo)):
            self.rx_apo_ind = i
            temp = patched_map(self.process_patch, flat_data, self.patches)

            # Reshape data back to original shape
            data.append(ops.reshape(temp, (n_z, n_x, n_ch)))

        return data

    def process(self, data):
        """Performs DAS beamforming on tof-corrected input.

        Args:
            data (ops.Tensor): The TOF corrected input of shape
                `(n_tx, n_z, n_x, n_el, n_ch)` with optional batch dimension.

        Returns:
            ops.Tensor: The beamformed data of shape `(n_z, n_x, n_ch)`
                with optional batch dimension.
        """

        if not self.with_batch_dim:
            return self.process_item(data)
        else:
            # TODO: could be ops.vectorized_map if enough memory
            return ops.map(self.process_item, data)


@ops_registry("interpolate")
class Interpolate(Operation):
    """Interpolate data along a specific axis using the downsample factor."""

    def __init__(
        self, factor: int = None, axis: int = -1, method: str = "bilinear", **kwargs
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.factor = factor
        self.axis = axis
        self.method = method

    def process(self, data):
        if self.factor is None or self.factor <= 1:
            return data  # No interpolation needed if factor is None or <= 1

        data_out = self.resize_along_axis(data, self.factor, self.axis, self.method)
        return data_out

    @staticmethod
    def resize_along_axis(data, factor, axis, method):
        """Resize data along a specific axis using the downsample factor."""
        shape = ops.shape(data)
        data_flat = ops.reshape(data, [-1, shape[axis]])
        # fill to four dimensions for `ops.image.resize` function
        data_flat = data_flat[..., None, None]
        data_out = ops.image.resize(
            data_flat, [shape[axis] * factor, 1], interpolation=method
        )
        new_shape = list(shape)
        new_shape[axis] = shape[axis] * factor
        data_out = ops.reshape(data_out, new_shape)
        return data_out


@ops_registry("bandpass_filter")
class BandPassFilter(Operation):
    """Band pass filter data."""

    def __init__(
        self,
        num_taps=None,
        sampling_frequency=None,
        center_frequency=None,
        f1=None,
        f2=None,
        axis=-3,
        **kwargs,
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.num_taps = num_taps
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.f1 = f1
        self.f2 = f2
        self.axis = axis

        if self._ready:
            self.initialize()

    def initialize(self):
        super().initialize()

        self.filter = get_band_pass_filter(
            self.num_taps, self.sampling_frequency, self.f1, self.f2
        )

    @property
    def _ready(self):
        return (
            self.axis is not None
            and self.num_taps is not None
            and self.sampling_frequency is not None
            and self.f1 is not None
            and self.f2 is not None
        )

    def process(self, data):
        axis = data.ndim + self.axis if self.axis < 0 else self.axis

        if data.shape[-1] == 2:
            data = channels_to_complex(data)

        data = ops.convert_to_numpy(data)
        data = ndimage.convolve1d(data, self.filter, mode="wrap", axis=axis)
        data = ops.convert_to_tensor(data)

        if data.dtype in ["complex64", "complex128"]:
            data = complex_to_channels(data, axis=-1)

        return data

    def _assign_scan_params(self, scan):
        return {
            "sampling_frequency": scan.sampling_frequency,
            "center_frequency": scan.center_frequency,
        }

    def _assign_config_params(self, config):
        return {
            "sampling_frequency": config.scan.sampling_frequency,
            "center_frequency": config.scan.center_frequency,
        }


@ops_registry("multi_bandpass_filter")
class MultiBandPassFilter(Operation):
    """Applies multiply band pass filters on beamformed data.

    Takes average in image domain of differend band passed filtered data if `to_image` set to true.
    Data is filtered in the RF / IQ domain. This function also can convert to image domain, since
    the compounding of filtered beamformed data takes place there (incoherent compounding).

    Args:
        beamformed_data (ndarray): input data, RF / IQ with shape [..., n_ax, n_el, n_ch].
            filtering is always applied over the n_ax axis.
        params (dict): dict with parameters for filter.
            Should include `num_taps`, `sampling_frequency`, `center_frequency` and two lists:
            `freqs` and `bandwidths` which define the filter characteristics. Lengths of those lists
            should be the same and is equal to the number of filters applied. Optionally the `units`
            can be specified, which is for instance `Hz` or `MHz`. Defaults to `Hz`.

    Returns:
        beamformed_data (list): list of filtered data, each element is filtered data
            with shape [..., n_ax, n_el, n_ch] for each filter applied.

    Example:
        >>> params = {
        >>>     'num_taps': 128,
        >>>     'sampling_frequency': 50e6,
        >>>     'center_frequency': 5e6,
        >>>     'freqs': [-2.5, 0, 2.5],
        >>>     'bandwidths': [1, 1, 1],
        >>>     'units': 'MHz'
        >>> }
        >>> mbpf = usbmd.ops.MultiBandPassFilter(
        >>>     params=params, modtype='iq', sampling_frequency=50e6, center_frequency=5e6, axis=-3)
        >>> filtered_data = mbpf(beamformed_data)
    """

    def __init__(
        self,
        params=None,
        modtype=None,
        sampling_frequency=None,
        center_frequency=None,
        axis=-3,
        **kwargs,
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.params = params
        self.modtype = modtype
        self.axis = axis
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency

        assert self.axis != -1, (
            "Axis of multibandpass filter cannot be the last axis "
            "as it is used for channels (RF / IQ)."
        )
        if self._ready:
            self.initialize()

    def initialize(self):
        super().initialize()

        if "units" in self.params:
            units = ["Hz", "kHz", "MHz", "GHz"]
            factors = [1, 1e3, 1e6, 1e9]
            unit_factor = factors[units.index(self.params["units"])]
        else:
            unit_factor = 1

        offsets = self.params["freqs"] * unit_factor
        bandwidths = self.params["bandwidths"] * unit_factor
        num_taps = self.params["num_taps"]
        # make sure sampling_frequency is correct for IQ (downsampled)
        sampling_frequency = self.sampling_frequency * unit_factor
        center_frequency = (
            self.center_frequency * unit_factor
        )  # center_frequency is only used when RF

        if self.modtype == "iq":
            center_frequency = 0  # center_frequency is automatically set to zero if IQ
            self.filter_params = [
                {
                    "num_taps": num_taps,
                    "sampling_frequency": sampling_frequency,
                    "f": center_frequency - offset,
                    "bw": bw,
                }
                for offset, bw in zip(offsets, bandwidths)
            ]
        elif self.modtype == "rf":
            self.filter_params = [
                {
                    "num_taps": num_taps,
                    "sampling_frequency": sampling_frequency,
                    "f1": center_frequency - offset - bw / 2,
                    "f2": center_frequency - offset + bw / 2,
                }
                for offset, bw in zip(offsets, bandwidths)
            ]
        self.filters = []
        for param in self.filter_params:
            if self.modtype == "iq":
                filter_weights = get_low_pass_iq_filter(**param)
            elif self.modtype == "rf":
                filter_weights = get_band_pass_filter(**param)
            else:
                raise ValueError(
                    f"Modulation type {self.modtype} is not supported for multibandpass filter."
                    "Supported types are 'iq' and 'rf'."
                )
            self.filters.append(filter_weights)

    @property
    def _ready(self):
        return (
            self.axis is not None
            and self.modtype is not None
            and self.params is not None
            and self.sampling_frequency is not None
            and self.center_frequency is not None
        )

    def process(self, data):
        axis = data.ndim + self.axis if self.axis < 0 else self.axis

        if self.modtype == "iq":
            assert data.shape[-1] == 2, "IQ data should have 2 channels."
            data = channels_to_complex(data)

        data_list = []
        for _filter in self.filters:
            data = ops.convert_to_numpy(data)
            _data = ndimage.convolve1d(data, _filter, mode="wrap", axis=axis)
            _data = ops.convert_to_tensor(_data)
            if self.modtype == "iq":
                _data = complex_to_channels(_data, axis=-1)
            data_list.append(_data)

        return data_list

    def _assign_scan_params(self, scan):
        return {
            "sampling_frequency": scan.sampling_frequency,
            "center_frequency": scan.center_frequency,
        }

    def _assign_config_params(self, config):
        return {
            "sampling_frequency": config.scan.sampling_frequency,
            "center_frequency": config.scan.center_frequency,
        }


@ops_registry("doppler")
class Doppler(Operation):
    """Compute the Doppler velocities from the I/ time series using a slow-time autocorrelator."""

    def __init__(
        self,
        PRF: float = None,
        sampling_frequency: float = None,
        center_frequency: float = None,
        c: float = None,
        M: int = None,
        lag: int = 1,
        nargout: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            center_frequency (float): Center frequency in Hz.
            c (float): Longitudinal velocity in m/s.
            PRF (float): Pulse repetition frequency in Hz.
            M (int, optional): Size of the hamming filter for spatial weighted average.
                Default is 1.
            The output Doppler velocity is estimated from M-by-M or M(1)-by-M(2)
                neighborhood around the corresponding pixel.
            lag (int, optional): LAG used in the autocorrelator. Default is 1.

        Note:
            This function is currently limited to use with beamformed data, but it
            can be modified to receive input_data_type = "raw_data".
        """
        super().__init__(
            input_data_type="beamformed_data",
            output_data_type=None,
            **kwargs,
        )
        self.PRF = PRF
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.c = c
        self.M = M
        self.lag = lag
        self.nargout = nargout
        self.warning_produced = False

        assert (
            self.with_batch_dim is True
        ), "Doppler requires multiple frames to compute"

    def process(self, data):

        assert data.ndim == 4, "Doppler requires multiple frames to compute"

        if data.shape[-1] == 2:
            data = channels_to_complex(data)

        # frames as last dimension for iq2doppler func
        data = ops.transpose(data, (1, 2, 0))

        doppler_velocities = self.iq2doppler(data)
        return doppler_velocities

    def iq2doppler(self, data):
        """Compute Doppler from packet of I/Q Data.

        Args:
            data (ndarray): I/Q complex data of shape (n_el, n_ax, n_frames).
                n_frames corresponds to the ensemble length used to compute
                the Doppler signal.
        Returns:
            doppler_velocities (ndarray): Doppler velocity map of shape (n_el, n_ax).

        """
        assert data.ndim == 3, "Data must be a 3-D array"

        if self.M is None:
            self.M = np.array([1, 1])
        elif np.isscalar(self.M):
            self.M = np.array([self.M, self.M])
        assert self.M.all() > 0 and np.all(
            np.equal(self.M, np.round(self.M))
        ), "M must contain integers > 0"

        assert (
            isinstance(self.lag, int) and self.lag >= 0
        ), "Lag must be a positive integer"

        if self.center_frequency is None:
            raise ValueError("A center frequency (center_frequency) must be specified")
        if self.PRF is None:
            raise ValueError("A pulse repetition frequency or period must be specified")

        # Auto-correlation method
        IQ1 = data[:, :, : data.shape[-1] - self.lag]
        IQ2 = data[:, :, self.lag :]
        AC = ops.sum(IQ1 * ops.conj(IQ2), axis=2)  # Ensemble auto-correlation

        # TODO: add spatial weighted average

        # Doppler velocity
        nyquist_velocities = self.c * self.PRF / (4 * self.center_frequency * self.lag)
        doppler_velocities = -nyquist_velocities * ops.imag(ops.log(AC)) / np.pi

        return doppler_velocities

    def _assign_scan_params(self, scan):
        return {
            "sampling_frequency": scan.sampling_frequency,
            "center_frequency": scan.center_frequency,
            "c": scan.sound_speed,
            "PRF": 1 / sum(scan.time_to_next_transmit[0]),
        }

    def _assign_config_params(self, config):
        return {
            "sampling_frequency": config.scan.sampling_frequency,
            "center_frequency": config.scan.center_frequency,
        }
