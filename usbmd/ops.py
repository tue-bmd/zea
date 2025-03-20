"""Ops module for processing ultrasound data.

- **Author(s)**     : Tristan Stevens
- **Date**          : 12/04/2024

Each step in the processing pipeline is defined as an operation. The operations
are then combined into a pipeline which can be used to process the data.

The pipeline as a whole has some additional functionality such as setting parameters
to all operations at once, initializing all operations at once, and running the
pipeline on a specific device and using a specific package.

## Stand-alone manual usage
Operations can also be run individually / standalone.
Examples:
```python
data = np.random.randn(2000, 128, 1)
envelope_detect = EnvelopeDetect(axis=-1)
envelope_data = envelope_detect(data)
```

## Using a pipeline
We can leave the arguments to the operation empty and set them later using the
`set_params` method. This is useful when using the operations in a pipeline.
```python
operations = [
    Beamform(),
    Sum(),
    Demodulate(),
    EnvelopeDetect(),
    Downsample(),
    Normalize(),
    LogCompress(),
    ScanConvert(),
]

config = ...
scan = ...
probe = ...

pipeline = Pipeline(operations, ops="numpy", device="cpu")
pipeline.set_params(config, scan, probe)
pipeline.initialize()

raw_data = np.random.randn(11, 2000, 128, 1)
image = pipeline.process(data)

```
If you do not have a config, scan, or probe, you can set the parameters to the
operations individually during initialization.

## The process class
Finally, there is an even higher level abstraction called the `Process` class.
This class defines a pipeline with a set of operations for you. You can use
this class to process data directly without having to define the operations explicitly.

```python
process = Process(config, scan, probe)
process.set_pipeline(
    operation_chain=[
        {"name": "tof_correction"},
        {"name": "delay_and_sum"},
        {"name": "demodulate", "params": {"sampling_frequency": 50e6, "center_frequency": 5e6}},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        {"name": "log_compress"},
        {"name": "scan_convert"},
    ],
)

raw_data = np.random.randn(11, 2000, 128, 1)
image = pipeline.process(data)
```

## Go crazy with parallel pipelines
You can also use blocks that output multiple data arrays as a list,
process them in parallel, and then stack them back together later in the pipeline.

```python
process = Process(config, scan, probe)
process.set_pipeline(
    operations_chain = [
        {
            "name": "multi_bandpass_filter",
            "params": {
                "params": {
                    "freqs": [-0.2e6, 0.0e6, 0.2e6],
                    "bandwidths": [1.2e6, 1.4e6, 1.0e6],
                    "num_taps": 81,
                },
                "modtype": "iq",
            },
        },  # this bandpass filters the data three times and returns a list
        {"name": "demodulate"},
        {"name": "envelope_detect"},
        {"name": "downsample"},
        {"name": "normalize"},
        {"name": "log_compress"},
        {
            "name": "stack",
            "params": {"axis": 0},
        },  # stack the data back together
        {
            "name": "mean",
            "params": {"axis": 0},
        },  # take the mean of the stack
    ]
)
```
which will give the following pipeline:
```bash
MBPF -> Demodulate -> EnvelopeDetect -> Downsample -> Normalize -> LogCompress -> Stack -> Mean
    \\-> Demodulate -> EnvelopeDetect -> Downsample -> Normalize -> LogCompress/->
```

TODO:
- Compilation of the pipeline using jit (currently some ops break the jit compatibility)

"""

import importlib
from abc import ABC, abstractmethod

import keras
import numpy as np
import scipy
from keras import ops
from scipy import ndimage, signal
from scipy.ndimage import gaussian_filter

import usbmd.beamformer as bmf
from usbmd import display
from usbmd.config import Config
from usbmd.probes import Probe
from usbmd.registry import ops_registry
from usbmd.scan import Scan
from usbmd.tensor_ops import patched_map
from usbmd.utils import lens_correction, log, pfield, translate
from usbmd.utils.checks import get_check

# make sure to reload all modules that import keras
# to be able to set backend properly
importlib.reload(bmf)
importlib.reload(pfield)
importlib.reload(lens_correction)
importlib.reload(display)

# clear registry upon import
ops_registry.clear()


def get_ops(ops_name):
    """Get the operation from the registry."""
    return ops_registry[ops_name]


class Operation(ABC):
    """Basic operation class as building block for processing pipeline and standalone operations."""

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


@ops_registry("identity")
class Identity(Operation):
    """Identity operation."""

    def process(self, data):
        return data


@ops_registry("delay_and_sum")
class DelayAndSum(Operation):
    """Sums time-delayed signals along channels and transmits."""

    def __init__(self, rx_apo=None, tx_apo=None, patches=1, **kwargs):
        """
        Args:
            rx_apo (array, optional): Receive apodization window. Defaults to None.
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

    def initialize(self):
        if self.rx_apo is None:
            self.rx_apo = 1.0

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
        data = ops.sum(self.rx_apo * patch, -2)

        # Sum over transmits, i.e. Compounding
        data = self.tx_apo * data
        data = ops.sum(data, 1)

        return data

    def process_item(self, data):
        """Performs DAS beamforming on tof-corrected input. Optionally splits the data into patches.

        Args:
            data (ops.Tensor): The TOF corrected input of shape `(n_tx, n_z, n_x, n_el, n_ch)`

        Returns:
            list[ops.Tensor]: The beamformed data of shape `(n_z, n_x, n_ch)`
        """
        n_tx, n_z, n_x, n_el, n_ch = data.shape

        # Flatten grid and move n_pix=(n_z * n_x) to the front
        flat_data = ops.reshape(data, (n_tx, -1, n_el, n_ch))
        flat_data = ops.moveaxis(flat_data, 1, 0)

        flat_data = patched_map(self.process_patch, flat_data, self.patches)

        # Reshape data back to original shape
        data = ops.reshape(flat_data, (n_z, n_x, n_ch))

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


@ops_registry("tof_correction")
class TOFCorrection(Operation):
    """Time-of-flight correction operation for ultrasound data."""

    def __init__(
        self,
        grid=None,
        sound_speed=None,
        polar_angles=None,
        focus_distances=None,
        sampling_frequency=None,
        f_number=None,
        n_el=None,
        n_tx=None,
        n_ax=None,
        demodulation_frequency=None,
        t0_delays=None,
        tx_apodizations=None,
        initial_times=None,
        probe_geometry=None,
        apply_lens_correction=None,
        lens_thickness=None,
        lens_sound_speed=None,
        patches=1,
    ):
        super().__init__(
            input_data_type="raw_data",
            output_data_type=None,
        )
        self.grid = grid
        self.sound_speed = sound_speed
        self.polar_angles = polar_angles
        self.focus_distances = focus_distances
        self.sampling_frequency = sampling_frequency
        self.f_number = f_number
        self.n_el = n_el
        self.n_tx = n_tx
        self.n_ax = n_ax
        self.demodulation_frequency = demodulation_frequency
        self.t0_delays = t0_delays
        self.tx_apodizations = tx_apodizations
        self.initial_times = initial_times
        self.probe_geometry = probe_geometry
        self.apply_lens_correction = apply_lens_correction
        self.lens_thickness = lens_thickness
        self.lens_sound_speed = lens_sound_speed
        self.patches = patches

    def initialize(self):
        self.grid = ops.convert_to_tensor(self.grid, dtype="float32")
        self.focus_distances = ops.convert_to_tensor(
            self.focus_distances, dtype="float32"
        )
        self.polar_angles = ops.convert_to_tensor(self.polar_angles, dtype="float32")
        self.t0_delays = ops.convert_to_tensor(self.t0_delays, dtype="float32")
        self.tx_apodizations = ops.convert_to_tensor(
            self.tx_apodizations, dtype="float32"
        )
        self.initial_times = ops.convert_to_tensor(self.initial_times, dtype="float32")
        self.probe_geometry = ops.convert_to_tensor(
            self.probe_geometry, dtype="float32"
        )

        super().initialize()

    def process_item(self, data):
        """Perform time-of-flight correction on a single item in the batch."""
        return bmf.tof_correction(
            data,
            grid=self.grid,
            t0_delays=self.t0_delays,
            tx_apodizations=self.tx_apodizations,
            sound_speed=self.sound_speed,
            probe_geometry=self.probe_geometry,
            initial_times=self.initial_times,
            sampling_frequency=self.sampling_frequency,
            demodulation_frequency=self.demodulation_frequency,
            fnum=self.f_number,
            angles=self.polar_angles,
            vfocus=self.focus_distances,
            apply_phase_rotation=bool(self.demodulation_frequency),
            apply_lens_correction=bool(self.apply_lens_correction),
            lens_thickness=self.lens_thickness,
            lens_sound_speed=self.lens_sound_speed,
            patches=self.patches,
        )

    def process(self, data):
        """Perform time-of-flight correction on a batch of data."""
        if not self.with_batch_dim:
            return self.process_item(data)
        else:
            return ops.map(self.process_item, data)

    def _assign_scan_params(self, scan: Scan):
        return {
            "grid": scan.grid,
            "focus_distances": scan.focus_distances,
            "t0_delays": scan.t0_delays,
            "tx_apodizations": scan.tx_apodizations,
            "initial_times": scan.initial_times,
            "probe_geometry": scan.probe_geometry,
            "sound_speed": scan.sound_speed,
            "polar_angles": scan.polar_angles,
            "sampling_frequency": scan.sampling_frequency,
            "f_number": scan.f_number,
            "demodulation_frequency": scan.demodulation_frequency,
            "apply_lens_correction": scan.apply_lens_correction,
            "lens_thickness": scan.lens_thickness,
            "lens_sound_speed": scan.lens_sound_speed,
        }

    @property
    def _ready(self):
        return all(
            [
                self.grid is not None,
                self.focus_distances is not None,
                self.t0_delays is not None,
                self.tx_apodizations is not None,
                self.initial_times is not None,
                self.probe_geometry is not None,
                self.sound_speed is not None,
                self.polar_angles is not None,
                self.sampling_frequency is not None,
                self.f_number is not None,
                self.demodulation_frequency is not None,
                self.apply_lens_correction is not None,
                self.lens_thickness is not None or self.apply_lens_correction is False,
                self.lens_sound_speed is not None
                or self.apply_lens_correction is False,
            ]
        )


@ops_registry("pfield_weighting")
class PfieldWeighting(Operation):
    """Weighting aligned data with the pressure field."""

    def __init__(self, pfield=None, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )

        self.pfield = pfield

    def _assign_scan_params(self, scan: Scan):
        return {
            "pfield": scan.pfield,
        }

    @property
    def _ready(self):
        return self.pfield is not None

    def process(self, data):
        # Perform element-wise multiplication with the pressure weight mask
        # Also add the required dimensions for broadcasting
        if self.with_batch_dim:
            pfield = ops.expand_dims(self.pfield, axis=0)
        else:
            pfield = self.pfield

        pfield = pfield[..., None, None]

        data_weighted = data * pfield
        return data_weighted


@ops_registry("stack")
class Stack(Operation):
    """Stack multiple data arrays along a new axis.
    Useful to merge data from parallel pipelines.
    """

    def __init__(self, axis=0, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.axis = axis

    def process(self, data):
        return ops.stack(data, axis=self.axis)


@ops_registry("mean")
class Mean(Operation):
    """Take the mean of the input data along a specific axis."""

    def __init__(self, axis=0, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.axis = axis

    def process(self, data):
        return ops.mean(data, axis=self.axis)


@ops_registry("sum")
class Sum(Operation):
    """Sum the input data along a specific axis."""

    def __init__(self, axis=0, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.axis = axis

    def process(self, data):
        return ops.sum(data, axis=self.axis)


@ops_registry("normalize")
class Normalize(Operation):
    """Normalize data to a given range."""

    def __init__(self, output_range=None, input_range=None, **kwargs):
        """Initialize the Normalize operation.

        Args:
            output_range (Tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (Tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.
        """
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.output_range = output_range
        self.input_range = input_range

    def process(self, data):
        if self.output_range is None:
            self.output_range = (0, 1)

        if self.input_range is None:
            minimum = ops.min(data)
            maximum = ops.max(data)
            self.input_range = (minimum, maximum)
        else:
            a_min, a_max = self.input_range
            data = ops.clip(data, a_min, a_max)
        return translate(data, self.input_range, self.output_range)

    def _assign_config_params(self, config):
        return {
            "input_range": config.data.input_range,
            "output_range": None,
        }


@ops_registry("log_compress")
class LogCompress(Operation):
    """Logarithmic compression of data."""

    def __init__(self, dynamic_range=None, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.dynamic_range = dynamic_range

    def process(self, data):
        if self.dynamic_range is None:
            self.dynamic_range = (-60, 0)
        small_number = ops.convert_to_tensor(1e-16, dtype=data.dtype)
        data = ops.where(data == 0, small_number, data)
        compressed_data = 20 * ops.log10(data)
        compressed_data = ops.clip(compressed_data, *self.dynamic_range)
        return compressed_data

    def _assign_config_params(self, config):
        return {
            "dynamic_range": config.data.dynamic_range,
        }


@ops_registry("downsample")
class Downsample(Operation):
    """Downsample data along a specific axis."""

    def __init__(self, factor: int = None, phase: int = None, axis: int = -1, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def process(self, data):
        if self.factor is None:
            return data
        length = ops.shape(data)[self.axis]
        if self.phase is None:
            self.phase = 0
        sample_idx = ops.arange(self.phase, length, self.factor)

        return ops.take(data, sample_idx, axis=self.axis)

    def _assign_config_params(self, config):
        return {
            "factor": config.scan.downsample,
        }


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


@ops_registry("companding")
class Companding(Operation):
    """Companding according to the A- or μ-law algorithm.
    Tensorflow versions of companding.

    Invertible compressing operation. Used to compress
    dynamic range of input data (and subsequently expand).

    μ-law companding:
    https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    A-law companding:
    https://en.wikipedia.org/wiki/A-law_algorithm

    The μ-law algorithm provides a slightly larger dynamic range
    than the A-law at the cost of worse proportional distortion
    for small signals.

    Args:
    array (ndarray): input array. expected to be in range [-1, 1].
        expand (bool, optional): If set to False (default),
            data is compressed, else expanded.
        comp_type (str): either `a` or `mu`.
        mu (float, optional): compression parameter. Defaults to 255.
        A (float, optional): compression parameter. Defaults to 255.

    Returns:
        ndarray: companded array. has values in range [-1, 1].
    """

    def __init__(self, expand=False, comp_type=None, mu=255, A=87.6, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.expand = expand
        self.comp_type = comp_type
        self.mu = mu
        self.A = A
        self.one = None

    def _assign_config_params(self, config):
        self.expand = config.expand
        self.comp_type = config.comp_type
        self.mu = config.mu
        self.A = config.A

    def process(self, data):
        self.one = ops.convert_to_tensor(1.0, dtype=data.dtype)
        self.A = ops.convert_to_tensor(self.A, dtype=data.dtype)
        self.mu = ops.convert_to_tensor(self.mu, dtype=data.dtype)

        data = ops.clip(data, -1, 1)

        if self.comp_type is None:
            self.comp_type = "mu"
        assert self.comp_type.lower() in ["a", "mu"]

        def mu_law_compress(x):
            y = (
                ops.sign(x)
                * ops.log(self.one + self.mu * ops.abs(x))
                / ops.log(self.one + self.mu)
            )
            return y

        def mu_law_expand(y):
            x = (
                ops.sign(y)
                * ((self.one + self.mu) ** (ops.abs(y)) - self.one)
                / self.mu
            )
            return x

        def a_law_compress(x):
            x_sign = ops.sign(x)
            x_abs = ops.abs(x)
            A_log = ops.log(self.A)

            val1 = x_sign * self.A * x_abs / (self.one + A_log)
            val2 = x_sign * (self.one + ops.log(self.A * x_abs)) / (self.one + A_log)

            y = ops.where((x_abs >= 0) & (x_abs < (self.one / self.A)), val1, val2)
            return y

        def a_law_expand(y):
            y_sign = ops.sign(y)
            y_abs = ops.abs(y)
            A_log = ops.log(self.A)

            val1 = y_sign * y_abs * (self.one + A_log) / self.A
            val2 = y_sign * ops.exp(y_abs * (self.one + A_log) - self.one) / self.A

            x = ops.where(
                (y_abs >= 0) & (y_abs < (self.one / (self.one + A_log))), val1, val2
            )
            return x

        if self.comp_type.lower() == "mu":
            if self.expand:
                data_out = mu_law_expand(data)
            else:
                data_out = mu_law_compress(data)
        elif self.comp_type.lower() == "a":
            if self.expand:
                data_out = a_law_expand(data)
            else:
                data_out = a_law_compress(data)
        else:
            raise ValueError(f"Invalid companding type {self.comp_type}.")

        return data_out


@ops_registry("envelope_detect")
class EnvelopeDetect(Operation):
    """Envelope detection of RF signals."""

    def __init__(self, axis=-3, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.axis = axis

    def process(self, data):
        if data.shape[-1] == 2:
            data = channels_to_complex(data)
        else:
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            # data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = hilbert(data, N=M, axis=self.axis)
            indices = ops.arange(n_ax)

            data = ops.take(data, indices, axis=self.axis)
            data = ops.squeeze(data, axis=-1)

        # data = ops.abs(data)
        real = ops.real(data)
        imag = ops.imag(data)
        data = ops.sqrt(real**2 + imag**2)
        data = ops.cast(data, "float32")
        return data


@ops_registry("demodulate")
class Demodulate(Operation):
    """Demodulate RF signals to IQ data (complex baseband)."""

    def __init__(
        self,
        sampling_frequency=None,
        center_frequency=None,
        bandwidth=None,
        filter_coeff=None,
        **kwargs,
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.bandwidth = bandwidth
        self.filter_coeff = filter_coeff
        self.warning_produced = False

    def process(self, data):

        if data.shape[-1] == 2:
            if not self.warning_produced:
                log.warning("Demodulation is not applicable to IQ data.")
                self.warning_produced = True
            return data
        elif data.shape[-1] == 1:
            data = ops.squeeze(data, axis=-1)

        data = demodulate_not_jitable(
            data,
            self.sampling_frequency,
            self.center_frequency,
            self.bandwidth,
            self.filter_coeff,
        )
        data = ops.convert_to_tensor(data)
        return complex_to_channels(data, axis=-1)

    def _assign_scan_params(self, scan):
        return {
            "sampling_frequency": scan.sampling_frequency,
            "center_frequency": scan.center_frequency,
            "bandwidth": scan.bandwidth_percent,
        }

    def _assign_config_params(self, config):
        return {
            "sampling_frequency": config.scan.sampling_frequency,
            "center_frequency": config.scan.center_frequency,
        }

    # pylint: disable=unused-argument
    def _assign_update_params(self, scan):
        return {
            "demodulation_frequency": self.center_frequency,
            "n_ch": 2,
        }


@ops_registry("upmix")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def __init__(
        self,
        sampling_frequency=None,
        center_frequency=None,
        upsampling_rate=6,
        **kwargs,
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.sampling_frequency = sampling_frequency
        self.center_frequency = center_frequency
        self.upsampling_rate = upsampling_rate

    def process(self, data):
        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data)
        data = upmix(
            data, self.sampling_frequency, self.center_frequency, self.upsampling_rate
        )
        data = ops.expand_dims(data, axis=-1)
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


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    def __init__(
        self,
        rho_range=None,
        theta_range=None,
        phi_range=None,
        resolution=None,
        fill_value=None,
        order=1,
        **kwargs,
    ):
        """Initialize the ScanConvert operation.

        Args:
            rho_range (Tuple): Range of the rho axis in the polar coordinate system.
                Defined in meters.
            theta_range (Tuple): Range of the theta axis in the polar coordinate system.
                Defined in radians.
            phi_range (Tuple): Range of the phi axis in the polar coordinate system.
                Defined in radians.
            resolution (float): Resolution of the output image in meters per pixel.
                if None, the resolution is computed based on the input data.
            fill_value (float): Value to fill the image with outside the defined region.
        Returns:
            image_sc (ndarray): Output image (converted to cartesian coordinates).

        """
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.rho_range = rho_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.resolution = resolution
        self.fill_value = fill_value
        self.order = order

    @property
    def _ready(self):
        return self.rho_range is not None and self.theta_range is not None

    def process(self, data):
        if self.phi_range is not None:
            data_out = display.scan_convert_3d(
                data,
                self.rho_range,
                self.theta_range,
                self.phi_range,
                self.resolution,
                self.fill_value,
                order=self.order,
            )
        else:
            data_out = display.scan_convert_2d(
                data,
                self.rho_range,
                self.theta_range,
                self.resolution,
                self.fill_value,
                order=self.order,
            )
        return data_out

    def _assign_scan_params(self, scan):
        return {
            "rho_range": (ops.min(scan.z_axis), ops.max(scan.z_axis)),
        }

    def _assign_config_params(self, config):
        return {
            "resolution": config.data.resolution,
            "fill_value": config.data.dynamic_range[0],
        }

    def _assign_probe_params(self, probe):
        # TODO: probably want to read coordinates from
        # usbmd file in the future (i.e. stored in scan class)
        if hasattr(probe, "angle_deg_axis"):
            angles = np.deg2rad(probe.angle_deg_axis)
            theta_range = (
                ops.min(angles),
                ops.max(angles),
            )
        else:
            # Probe does not have `angle_deg_axis` defined, using default
            # values (-45, 45 degree cone) for ScanConvert.
            theta_range = tuple(np.deg2rad([-45, 45]))

        return {
            "theta_range": theta_range,
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


@ops_registry("gaussian_blur")
class GaussianBlur(Operation):
    """
    GaussianBlur is an operation that applies a Gaussian blur to an input image.
    Uses scipy.ndimage.gaussian_filter to create a kernel.

    Src: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """

    def __init__(
        self,
        sigma: float,
        kernel_size: int | None = None,
        pad_mode="symmetric",
        truncate=4.0,
        **kwargs,
    ):
        """
        Args:
            sigma (float): Standard deviation for Gaussian kernel. The standard deviations of the
                Gaussian filter are given for each axis as a sequence, or as a single number,
                in which case it is equal for all axes.
            kernel_size (int, optional): The size of the kernel to be used. If None, the kernel
                size is calculated based on the sigma and truncate. Default is None.
            pad_mode (str): Padding mode for the input image. Default is 'symmetric'.
                See [keras docs](https://www.tensorflow.org/api_docs/python/tf/keras/ops/pad) for
                all options and [tensoflow docs](https://www.tensorflow.org/api_docs/python/tf/pad)
                for some examples. Note that the naming differs from scipy.ndimage.gaussian_filter!
        """
        super().__init__(**kwargs)
        if kernel_size is None:
            radius = round(truncate * sigma)
            self.kernel_size = 2 * radius + 1
        else:
            self.kernel_size = kernel_size
        self.sigma = sigma
        self.pad_mode = pad_mode
        self.radius = self.kernel_size // 2
        self.kernel = self.get_kernel()

    def get_kernel(self):
        """
        Create a gaussian kernel for blurring.

        Returns:
            kernel (Tensor): A gaussian kernel for blurring.
                Shape is (kernel_size, kernel_size, 1, 1).
        """
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.radius, self.radius] = 1
        kernel = gaussian_filter(
            n, sigma=self.sigma, radius=self.radius, mode="constant"
        ).astype(np.float32)
        kernel = kernel[:, :, None, None]
        return ops.convert_to_tensor(kernel)

    def process(self, data):
        """Blur the input image with a gaussian kernel."""

        # Add batch dimension if not present
        if not self.with_batch_dim:
            data = data[None]

        # Add channel dimension to kernel
        kernel = ops.tile(self.kernel, (1, 1, data.shape[-1], data.shape[-1]))

        # Pad the input image according to the padding mode
        padded = ops.pad(
            data,
            [[0, 0], [self.radius, self.radius], [self.radius, self.radius], [0, 0]],
            mode=self.pad_mode,
        )

        # Apply the gaussian kernel to the padded image
        out = ops.conv(padded, kernel, padding="valid", data_format="channels_last")

        # Remove padding
        out = keras.layers.CenterCrop(
            data.shape[-2], data.shape[-3], data_format="channels_last"
        )(out)

        # Remove batch dimension if it was not present before
        if not self.with_batch_dim:
            out = ops.squeeze(out, axis=0)

        return out


class LeeFilter(Operation):
    """
    The Lee filter is a speckle reduction filter commonly used insynthetic aperture radar (SAR)
    image processing. It smooths the image while preserving edges and details. This implementation
    uses Gaussian filter for local statistics and treats channels independently.
    Based on: https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    """

    def __init__(self, sigma=3, kernel_size=None, pad_mode="symmetric", **kwargs):
        """
        Args:
            sigma (float, optional): Standard deviation for Gaussian kernel. Default is 3.
            kernel_size (int or tuple, optional): Size of the Gaussian kernel. If None,
                it will be calculated based on sigma. See `GaussianBlur` for more details.
                Default is None.
            pad_mode (str, optional): Padding mode to be used before Gaussian blur.
                Default is "symmetric".
        """

        super().__init__(**kwargs)
        self.sigma = sigma
        self.kernel_size = kernel_size

        self.blur = GaussianBlur(
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            with_batch_dim=self.with_batch_dim,
            pad_mode=pad_mode,
        )

    def process(self, data):
        """
        Apply Lee filter to the input data.

        Args:
            data (Tensor): Input data to be filtered.

        Returns:
            Tensor: Filtered data.
        """
        img_mean = self.blur(data)
        img_sqr_mean = self.blur(data**2)
        img_variance = img_sqr_mean - img_mean**2

        # treating channels independently!
        overall_variance = ops.var(data, axis=(-2, -3), keepdims=True)

        img_weights = img_variance / (img_variance + overall_variance)
        img_output = img_mean + img_weights * (data - img_mean)
        return img_output


def demodulate_not_jitable(
    rf_data,
    sampling_frequency=None,
    center_frequency=None,
    bandwidth=None,
    filter_coeff=None,
):
    """Demodulates an RF signal to complex base-band (IQ).

    Demodulates the radiofrequency (RF) bandpass signals and returns the
    Inphase/Quadrature (I/Q) components. IQ is a complex whose real (imaginary)
    part contains the in-phase (quadrature) component.

    This function operates (i.e. demodulates) on the RF signal over the
    (fast-) time axis which is assumed to be the last axis.

    Args:
        rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
            second to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the RF signals (in Hz).
            Only not necessary when filter_coeff is provided.
        center_frequency (float, optional): represents the center frequency (in Hz).
            Defaults to None.
        bandwidth (float, optional): Bandwidth of RF signal in % of center
            frequency. Defaults to None.
            The bandwidth in % is defined by:
            B = Bandwidth_in_% = Bandwidth_in_Hz*(100/center_frequency).
            The cutoff frequency:
            Wn = Bandwidth_in_Hz/sampling_frequency, i.e:
            Wn = B*(center_frequency/100)/sampling_frequency.
        filter_coeff (list, optional): (b, a), numerator and denominator coefficients
            of FIR filter for quadratic band pass filter. All other parameters are ignored
            if filter_coeff are provided. Instead the given filter_coeff is directly used.
            If not provided, a filter is derived from the other params (sampling_frequency,
            center_frequency, bandwidth).
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

    Returns:
        iq_data (ndarray): complex valued base-band signal.

    """
    rf_data = ops.convert_to_numpy(rf_data)
    assert np.isreal(
        rf_data
    ).all(), f"RF must contain real RF signals, got {rf_data.dtype}"

    input_shape = rf_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, n_el = input_shape
    else:
        n_ax, n_el = input_shape

    if filter_coeff is None:
        assert (
            sampling_frequency is not None
        ), "provide sampling_frequency when no filter is given."
        # Time vector
        t = np.arange(n_ax) / sampling_frequency
        t0 = 0
        t = t + t0

        # Estimate center frequency
        if center_frequency is None:
            # Keep a maximum of 100 randomly selected scanlines
            idx = np.arange(n_el)
            if n_el > 100:
                idx = np.random.permutation(idx)[:100]
            # Power Spectrum
            P = np.sum(
                np.abs(np.fft.fft(np.take(rf_data, idx, axis=-1), axis=-2)) ** 2,
                axis=-1,
            )
            P = P[: n_ax // 2]
            # Carrier frequency
            idx = np.sum(np.arange(n_ax // 2) * P) / np.sum(P)
            center_frequency = idx * sampling_frequency / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * center_frequency / sampling_frequency, 0.5)
            bandwidth = center_frequency * Wn
        else:
            assert np.isscalar(
                bandwidth
            ), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (
                bandwidth <= 200
            ), "The signal bandwidth (in %) must be within the interval of ]0,200]."
            # bandwidth in Hz
            bandwidth = center_frequency * bandwidth / 100
            Wn = bandwidth / sampling_frequency
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * center_frequency * t)
        # add the singleton dimensions
        carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))
        iq_data = rf_data * carrier

        # Low-pass filter
        N = 5
        b, a = scipy.signal.butter(N, Wn, "low")

        # factor 2: to preserve the envelope amplitude
        iq_data = scipy.signal.filtfilt(b, a, iq_data, axis=-2) * 2

        # Display a warning message if harmful aliasing is suspected
        # the RF signal is undersampled
        if sampling_frequency < (2 * center_frequency + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = center_frequency - bandwidth / 2
            fH = center_frequency + bandwidth / 2
            n = fH // (fH - fL)
            harmless_aliasing = any(
                (2 * fH / np.arange(1, n) <= sampling_frequency)
                & (sampling_frequency <= 2 * fL / np.arange(1, n))
            )
            if not harmless_aliasing:
                log.warning(
                    "rf2iq:harmful_aliasing Harmful aliasing is present: the aliases"
                    " are not mutually exclusive!"
                )
    else:
        b, a = filter_coeff
        iq_data = scipy.signal.lfilter(b, a, rf_data, axis=-2) * 2

    return iq_data


def upmix(iq_data, sampling_frequency, center_frequency, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        sampling_frequency (float): the sampling frequency of the input IQ signal (in Hz).
            resulting sampling_frequency of RF data is upsampling_rate times higher.
        center_frequency (float, optional): represents the center frequency (in Hz).

    Returns:
        rf_data (ndarray): output real valued rf data.

    """
    assert iq_data.dtype in [
        "complex64",
        "complex128",
    ], "IQ must contain all complex signals."

    input_shape = iq_data.shape
    n_dim = len(input_shape)
    if n_dim > 2:
        *_, n_ax, _ = input_shape
    else:
        n_ax, _ = input_shape

    # Time vector
    n_ax *= upsampling_rate
    sampling_frequency *= upsampling_rate

    t = np.arange(n_ax) / sampling_frequency
    t0 = 0
    t = t + t0

    # interpolation
    iq_data_upsampled = signal.resample(iq_data, num=n_ax, axis=-2)

    # Up-mixing of the IQ signals
    carrier = np.exp(1j * 2 * np.pi * center_frequency * t)
    # add the singleton dimensions
    carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = np.real(rf_data) * np.sqrt(2)

    return rf_data.astype(np.float32)


def get_band_pass_filter(num_taps, sampling_frequency, f1, f2):
    """Band pass filter

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.

    Returns:
        ndarray: band pass filter
    """
    bpf = signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=sampling_frequency)
    return bpf


def get_low_pass_iq_filter(num_taps, sampling_frequency, f, bw):
    """Design low pass filter.

    LPF with num_taps points and cutoff at bw / 2

    Args:
        num_taps (int): number of taps in filter.
        sampling_frequency (float): sample frequency.
        f (float): center frequency.
        bw (float): bandwidth in Hz.
    Raises:
        AssertionError: if cutoff frequency (bw / 2) is not within (0, sampling_frequency / 2)

    Returns:
        ndarray: fx LP filter
    """
    assert (bw / 2 > 0) & (bw / 2 < sampling_frequency / 2), log.error(
        "Cutoff frequency must be within (0, sampling_frequency / 2), "
        f"got {bw / 2} Hz, must be within (0, {sampling_frequency / 2}) Hz"
    )
    t_qbp = np.arange(num_taps) / sampling_frequency
    lpf = signal.firwin(
        num_taps, bw / 2, pass_zero=True, fs=sampling_frequency
    ) * np.exp(1j * 2 * np.pi * f * t_qbp)
    return lpf


def complex_to_channels(complex_data, axis=-1):
    """Unroll complex data to separate channels.

    Args:
        complex_data (complex ndarray): complex input data.
        axis (int, optional): on which axis to extend. Defaults to -1.

    Returns:
        ndarray: real array with real and imaginary components
            unrolled over two channels at axis.
    """
    # assert ops.iscomplex(complex_data).any()
    q_data = ops.imag(complex_data)
    i_data = ops.real(complex_data)

    i_data = ops.expand_dims(i_data, axis=axis)
    q_data = ops.expand_dims(q_data, axis=axis)

    iq_data = ops.concatenate((i_data, q_data), axis=axis)
    return iq_data


def channels_to_complex(data):
    """Convert array with real and imaginary components at
    different channels to complex data array.

    Args:
        data (ndarray): input data, with at 0 index of axis
            real component and 1 index of axis the imaginary.

    Returns:
        ndarray: complex array with real and imaginary components.
    """
    assert data.shape[-1] == 2, "Data must have two channels."
    data = ops.cast(data, "complex64")
    return data[..., 0] + 1j * data[..., 1]


def hilbert(x, N: int = None, axis=-1):
    """Manual implementation of the Hilbert transform function. Tje function
    returns the analytical signal.

    Operated in the Fourier domain.

    Note:
        THIS IS NOT THE MATHEMATICAL THE HILBERT TRANSFORM as you will find it on
        wikipedia, but computes the analytical signal. The implementation reproduces
        the behavior of the `scipy.signal.hilbert` function.

    Args:
        x (ndarray): input data of any shape.
        N (int, optional): number of points in the FFT. Defaults to None.
        axis (int, optional): axis to operate on. Defaults to -1.
    Returns:
        x (ndarray): complex iq data of any shape.k

    """
    input_shape = x.shape
    n_dim = len(input_shape)

    n_ax = input_shape[axis]

    if axis < 0:
        axis = n_dim + axis

    if N is not None:
        if N < n_ax:
            raise ValueError("N must be greater or equal to n_ax.")
        # only pad along the axis, use manual padding
        pad = N - n_ax
        zeros = ops.zeros(
            input_shape[:axis] + (pad,) + input_shape[axis + 1 :],
        )

        x = ops.concatenate((x, zeros), axis=axis)
    else:
        N = n_ax

    # Create filter to zero out negative frequencies
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    idx = list(range(n_dim))
    # make sure axis gets to the end for fft (operates on last axis)
    idx.remove(axis)
    idx.append(axis)
    x = ops.transpose(x, idx)

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[-1] = slice(None)
        h = h[tuple(ind)]

    h = ops.convert_to_tensor(h)
    h = ops.cast(h, "complex64")
    h = h + 1j * ops.zeros_like(h)

    Xf_r, Xf_i = ops.fft((x, ops.zeros_like(x)))

    Xf_r = ops.cast(Xf_r, "complex64")
    Xf_i = ops.cast(Xf_i, "complex64")

    Xf = Xf_r + 1j * Xf_i
    Xf = Xf * h

    # x = np.fft.ifft(Xf)
    # do manual ifft using fft
    Xf_r = ops.real(Xf)
    Xf_i = ops.imag(Xf)
    Xf_r_inv, Xf_i_inv = ops.fft((Xf_r, -Xf_i))

    Xf_i_inv = ops.cast(Xf_i_inv, "complex64")
    Xf_r_inv = ops.cast(Xf_r_inv, "complex64")

    x = Xf_r_inv / N
    x = x + 1j * (-Xf_i_inv / N)

    # switch back to original shape
    idx = list(range(n_dim))
    idx.insert(axis, idx.pop(-1))
    x = ops.transpose(x, idx)
    return x


def demodulate(data, center_frequency, sampling_frequency, axis=-3):
    """Demodulates the input data to baseband. The function computes the analytical
    signal (the signal with negative frequencies removed) and then shifts the spectrum
    of the signal to baseband by multiplying with a complex exponential. Where the
    spectrum was centered around `center_frequency` before, it is now centered around
    0 Hz. The baseband IQ data are complex-valued. The real and imaginary parts
    are stored in two real-valued channels.

    Args:
        data (ops.Tensor): The input data to demodulate of shape `(..., axis, ..., 1)`.
        center_frequency (float): The center frequency of the signal.
        sampling_frequency (float): The sampling frequency of the signal.
        axis (int, optional): The axis along which to demodulate. Defaults to -3.

    Returns:
        ops.Tensor: The demodulated IQ data of shape `(..., axis, ..., 2)`.
    """
    # Compute the analytical signal
    analytical_signal = hilbert(data, axis=axis)

    # Define frequency indices
    frequency_indices = ops.arange(analytical_signal.shape[axis])

    # Expand the frequency indices to match the shape of the RF data
    indexing = [None] * data.ndim
    indexing[axis] = slice(None)
    indexing = tuple(indexing)
    frequency_indices_shaped_like_rf = frequency_indices[indexing]

    # Cast to complex64
    center_frequency = ops.cast(center_frequency, dtype="complex64")
    sampling_frequency = ops.cast(sampling_frequency, dtype="complex64")
    frequency_indices_shaped_like_rf = ops.cast(
        frequency_indices_shaped_like_rf, dtype="complex64"
    )

    # Shift to baseband
    phasor_exponent = (
        -1j
        * 2
        * np.pi
        * center_frequency
        * frequency_indices_shaped_like_rf
        / sampling_frequency
    )
    iq_data_signal_complex = analytical_signal * ops.exp(phasor_exponent)

    # Split the complex signal into two channels
    iq_data_two_channel = complex_to_channels(iq_data_signal_complex[..., 0])

    return iq_data_two_channel
