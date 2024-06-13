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
    operations_chain=[
        {name: "beamform"},
        {name: "demodulate": params={"fs": 50e6, "fc": 5e6}},
        {name: "envelope_detect"},
        {name: "downsample"},
        {name: "normalize"},
        {name: "log_compress"},
        {name: "scan_convert"},
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
- Test operations for jax (currently only np / torch / tensorflow tested)
- Compilation of the pipeline using jit (currently some ops break the jit compatibility)

"""

import importlib
from abc import ABC, abstractmethod

import numpy as np
import scipy
from scipy import ndimage, signal

from usbmd.display import scan_convert
from usbmd.probes import Probe
from usbmd.pytorch_ultrasound import on_device_torch
from usbmd.registry import (
    ops_registry,
    tf_beamformer_registry,
    torch_beamformer_registry,
)
from usbmd.scan import Scan
from usbmd.tensorflow_ultrasound import on_device_tf
from usbmd.utils import log
from usbmd.utils.checks import _ML_LIBRARIES, get_check
from usbmd.utils.config import Config
from usbmd.utils.utils import translate


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
        ops="numpy",
    ):
        """Initialize the operation.

        Args:
            input_data_type (type): The expected data type of the input data.
            output_data_type (type): The expected data type of the output data.
            with_batch_dim (bool): Whether the input data has a batch dimension.
            ops (module, str): The operations package used in the operation.
                Can be a module or a string to import the module. Defaults to "numpy".

        """
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.with_batch_dim = with_batch_dim
        self.ops = ops

        self.config = None
        self.scan = None
        self.probe = None

    @property
    def ops(self):
        """Get the operations package used in the operation."""
        assert self._ops is not None, "ops package is not set"
        return self._ops

    @ops.setter
    def ops(self, ops):
        """Set the package for the operation."""
        if isinstance(ops, str):
            ops = importlib.import_module(ops)
            importlib.import_module("usbmd.backend_aliases")
        assert ops.__name__ in _ML_LIBRARIES, f"Unsupported operations package {ops}"
        self._ops = ops

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

    def set_params(self, config: Config, scan: Scan, probe: Probe):
        """Set the parameters for the operation.

        Parameters are assigned to the operation from the config, scan, and probe
        and in that order of priority (i.e. config > scan > probe will be assigned
        for shared parameters between the three).

        Args:
            config (Config): Configuration parameters for the operation.
            scan (Scan): Scan parameters for the operation.
            probe (Probe): Probe parameters for the operation.

        """
        if config is not None:
            self._assign_config_params(config)
        if scan is not None:
            self._assign_scan_params(scan)
        if probe is not None:
            self._assign_probe_params(probe)

    # pylint: disable=unused-argument
    def _assign_config_params(self, config: Config):
        """Assign the config parameters to the operation.

        Args:
            config (Config): Configuration parameters for the operation.

        """
        # Assign the config parameters to the operation
        return

    # pylint: disable=unused-argument
    def _assign_scan_params(self, scan: Scan):
        """Assign the scan parameters to the operation.

        Args:
            scan (Scan): Scan parameters for the operation.

        """
        return

    # pylint: disable=unused-argument
    def _assign_probe_params(self, probe: Probe):
        """Assign the probe parameters to the operation.

        Args:
            probe (Probe): Probe parameters for the operation.

        """
        return

    def prepare_tensor(self, x, dtype=None, device=None):
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
        if self.ops.__name__ == "numpy":
            x = np.array(x)
            if dtype is not None:
                x = x.astype(dtype)
            return x
        elif self.ops.__name__ == "tensorflow":
            x = self.ops.convert_to_tensor(x)
            if dtype is not None:
                x = self.ops.cast(x, dtype)
            return x
        elif self.ops.__name__ == "torch":
            x = self.ops.tensor(x)
            if dtype is not None:
                x = x.type(dtype)
            if device is not None:
                x = x.to(device)
            return x
        else:
            raise ValueError("Unsupported operations package.")


class Pipeline:
    """Pipeline class for processing ultrasound data through a series of operations."""

    def __init__(self, operations, ops="numpy", with_batch_dim=True, device=None):
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

        self.device = self._check_device(device, ops)

        for operation in self.operations:
            operation.ops = ops
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

    @property
    def ops(self):
        """Get the operations package used in the pipeline."""
        assert all(
            operation.ops == self.operations[0].ops for operation in self.operations
        ), (
            "Operations in pipeline are not compatible, "
            "please use the same operations package for all operations."
        )
        return self.operations[0].ops

    @property
    def with_batch_dim(self):
        """Get the with_batch_dim property of the pipeline."""
        return self.operations[0].with_batch_dim

    def on_device(self, func, data, device=None, return_numpy=False):
        """On device function for running pipeline on specific device."""
        if self.ops.__name__ == "numpy":
            return func(data)
        elif self.ops.__name__ == "tensorflow":
            return on_device_tf(func, data, device=device, return_numpy=return_numpy)
        elif self.ops.__name__ == "torch":
            return on_device_torch(func, data, device=device, return_numpy=return_numpy)
        else:
            raise ValueError("Unsupported operations package.")

    def set_params(self, config: Config, scan: Scan, probe: Probe):
        """Set the parameters for the pipeline. See Operation.set_params for more info."""
        for operation in self.operations:
            operation.set_params(config, scan, probe)

    def process(self, data, return_numpy=False):
        """Process input data through the pipeline."""
        data = self.prepare_tensor(data)
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
        return processing_func(data)

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
        if not jit:
            return
        log.info(f"Compiling pipeline, with ops library: {self.ops.__name__}")
        if self.ops.__name__ == "numpy":
            return
        elif self.ops.__name__ == "tensorflow":
            self._jitted_process = self.ops.function(
                self._process, jit_compile=jit
            )  # tf.function
            return
        elif self.ops.__name__ == "torch":
            log.warning("JIT compmilation is not yet supported for torch.")
            return
        elif self.ops.__name__ == "jax":
            self._jitted_process = self.ops.jit(self._process)  # jax.jit
            return

    def prepare_tensor(self, x, dtype=None, device=None):
        """Convert input array to appropriate tensor type for the operations package."""
        if len(self.operations) == 0:
            return x
        return self.operations[0].prepare_tensor(x, dtype=dtype, device=device)

    def _check_device(self, device, ops):
        if device is None:
            return None

        if device == "cpu":
            return "cpu"

        if not isinstance(ops, str):
            ops = ops.__name__

        if ops == "numpy":
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
                if ops in ["tensorflow", "jax"]:
                    assert device.startswith(
                        "gpu"
                    ), f"device should be 'cpu' or 'gpu:*', got {device}"
                elif ops == "torch":
                    assert device.startswith(
                        "cuda"
                    ), f"device should be 'cpu' or 'cuda:*', got {device}"
            return device


@ops_registry("beamform")
class Beamform(Operation):
    """Beamforming operation for ultrasound data."""

    def __init__(self, beamformer=None, **kwargs):
        super().__init__(
            input_data_type="raw_data",
            output_data_type="beamformed_data",
            **kwargs,
        )

        self.beamformer = beamformer

    def initialize(self):
        super().initialize()

        if self.beamformer is not None:
            return

        beamformer_type = self.config.model.beamformer.type
        # pylint: disable=import-outside-toplevel
        if self.ops.__name__ == "torch":
            from usbmd.pytorch_ultrasound.layers.beamformers import get_beamformer

            _BEAMFORMER_TYPES = torch_beamformer_registry.registered_names()
        elif self.ops.__name__ == "tensorflow":
            from usbmd.tensorflow_ultrasound.layers.beamformers import get_beamformer

            _BEAMFORMER_TYPES = tf_beamformer_registry.registered_names()
        else:
            log.warning(
                f"Beamformer is not supported for the operations package: {self.ops.__name__} "
                f"Please use on of the supported beamformer packages: {['torch', 'tensorflow']}"
            )
            get_beamformer = None
            _BEAMFORMER_TYPES = []

        assert beamformer_type in _BEAMFORMER_TYPES, (
            f"Beamformer type {beamformer_type} is not supported, "
            f"should be in {_BEAMFORMER_TYPES}"
        )

        self.beamformer = get_beamformer(self.probe, self.scan, self.config)

    def _assign_config_params(self, config: Config):
        self.config = config

    def _assign_scan_params(self, scan: Scan):
        self.scan = scan

    def _assign_probe_params(self, probe: Probe):
        self.probe = probe

    def process(self, data):
        if self.with_batch_dim is False:
            data = self.ops.expand_dims(data, axis=0)
        if self.ops.__name__ == "torch":
            self.beamformer.to(data.device)
        data = self.beamformer(data, probe=self.probe, scan=self.scan)
        if self.with_batch_dim is False:
            data = self.ops.squeeze(data, axis=0)
        return data


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
        return self.ops.stack(data, axis=self.axis)


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
        return self.ops.mean(data, axis=self.axis)


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

    def _assign_config_params(self, config):
        self.input_range = config.data.input_range
        self.output_range = None

    def process(self, data):
        if self.output_range is None:
            self.output_range = (0, 1)

        if self.input_range is None:
            minimum = self.ops.min(data)
            maximum = self.ops.max(data)
            self.input_range = (minimum, maximum)
        else:
            a_min, a_max = self.input_range
            data = self.ops.clip(data, a_min, a_max)
        return translate(data, self.input_range, self.output_range)


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

    def _assign_config_params(self, config):
        self.dynamic_range = config.data.dynamic_range

    def process(self, data):
        if self.dynamic_range is None:
            self.dynamic_range = (-60, 0)

        device = None
        if self.ops.__name__ == "torch":
            device = data.device

        small_number = self.prepare_tensor(1e-16, dtype=data.dtype, device=device)
        data = self.ops.where(data == 0, small_number, data)
        compressed_data = 20 * self.ops.log10(data)
        compressed_data = self.ops.clip(compressed_data, *self.dynamic_range)
        return compressed_data


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

    def _assign_config_params(self, config):
        self.factor = config.scan.downsample

    def process(self, data):
        if self.factor is None:
            return data
        length = self.ops.shape(data)[self.axis]
        if self.phase is None:
            self.phase = 0
        sample_idx = self.ops.arange(self.phase, length, self.factor)
        if self.ops.__name__ == "torch":
            sample_idx = sample_idx.to(data.device)
        return take(data, sample_idx, axis=self.axis, ops=self.ops)


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
        self.one = self.prepare_tensor(1.0, dtype=data.dtype)
        self.A = self.prepare_tensor(self.A, dtype=data.dtype)
        self.mu = self.prepare_tensor(self.mu, dtype=data.dtype)

        data = self.ops.clip(data, -1, 1)

        if self.comp_type is None:
            self.comp_type = "mu"
        assert self.comp_type.lower() in ["a", "mu"]

        def mu_law_compress(x):
            y = (
                self.ops.sign(x)
                * self.ops.log(self.one + self.mu * self.ops.abs(x))
                / self.ops.log(self.one + self.mu)
            )
            return y

        def mu_law_expand(y):
            x = (
                self.ops.sign(y)
                * ((self.one + self.mu) ** (self.ops.abs(y)) - self.one)
                / self.mu
            )
            return x

        def a_law_compress(x):
            x_sign = self.ops.sign(x)
            x_abs = self.ops.abs(x)
            A_log = self.ops.log(self.A)

            val1 = x_sign * self.A * x_abs / (self.one + A_log)
            val2 = (
                x_sign * (self.one + self.ops.log(self.A * x_abs)) / (self.one + A_log)
            )

            y = self.ops.where((x_abs >= 0) & (x_abs < (self.one / self.A)), val1, val2)
            return y

        def a_law_expand(y):
            y_sign = self.ops.sign(y)
            y_abs = self.ops.abs(y)
            A_log = self.ops.log(self.A)

            val1 = y_sign * y_abs * (self.one + A_log) / self.A
            val2 = y_sign * self.ops.exp(y_abs * (self.one + A_log) - self.one) / self.A

            x = self.ops.where(
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
            data = channels_to_complex(data, ops=self.ops)
        else:
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            # data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = hilbert(data, N=M, axis=self.axis, ops=self.ops)
            data = take(data, self.ops.arange(n_ax), axis=self.axis, ops=self.ops)
            data = self.ops.squeeze(data, axis=-1)

        data = self.ops.abs(data)
        data = self.ops.cast(data, self.ops.float32)
        return data


@ops_registry("demodulate")
class Demodulate(Operation):
    """Demodulate RF signals to IQ data (complex baseband)."""

    def __init__(self, fs=None, fc=None, bandwidth=None, filter_coeff=None, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.filter_coeff = filter_coeff
        self.warning_produced = False

    def process(self, data):

        device = None
        if self.ops.__name__ == "torch":
            device = data.device

        if data.shape[-1] == 2:
            if not self.warning_produced:
                log.warning("Demodulation is not applicable to IQ data.")
                self.warning_produced = True
            return data
        elif data.shape[-1] == 1:
            data = self.ops.squeeze(data, axis=-1)

        # currently demodulate converts to numpy so we have to do some trickery
        if self.ops.__name__ == "torch":
            data = data.cpu().numpy()
        data = demodulate(data, self.fs, self.fc, self.bandwidth, self.filter_coeff)
        data = self.prepare_tensor(data, device=device)
        return complex_to_channels(data, axis=-1, ops=self.ops)

    def _assign_scan_params(self, scan):
        self.fs = scan.fs
        self.fc = scan.fc
        self.bandwidth = scan.bandwidth_percent

    def _assign_config_params(self, config):
        if config.scan.sampling_frequency is not None:
            self.fs = config.scan.sampling_frequency
        if config.scan.center_frequency is not None:
            self.fc = config.scan.center_frequency


@ops_registry("upmix")
class UpMix(Operation):
    """Upmix IQ data to RF data."""

    def __init__(self, fs=None, fc=None, upsampling_rate=6, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.fs = fs
        self.fc = fc
        self.upsampling_rate = upsampling_rate

    def process(self, data):
        if data.shape[-1] == 1:
            log.warning("Upmixing is not applicable to RF data.")
            return data
        elif data.shape[-1] == 2:
            data = channels_to_complex(data, ops=self.ops)
        data = upmix(data, self.fs, self.fc, self.upsampling_rate)
        data = self.ops.expand_dims(data, axis=-1)
        return data

    def _assign_scan_params(self, scan):
        self.fs = scan.fs
        self.fc = scan.fc

    def _assign_config_params(self, config):
        if config.scan.sampling_frequency is not None:
            self.fs = config.scan.sampling_frequency
        if config.scan.center_frequency is not None:
            self.fc = config.scan.center_frequency


@ops_registry("bandpass_filter")
class BandPassFilter(Operation):
    """Band pass filter data."""

    def __init__(
        self, num_taps=None, fs=None, fc=None, f1=None, f2=None, axis=-3, **kwargs
    ):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.num_taps = num_taps
        self.fs = fs
        self.fc = fc
        self.f1 = f1
        self.f2 = f2
        self.axis = axis

        if self._ready:
            self.initialize()

    def initialize(self):
        super().initialize()

        self.filter = get_band_pass_filter(self.num_taps, self.fs, self.f1, self.f2)

    @property
    def _ready(self):
        return (
            self.axis is not None
            and self.num_taps is not None
            and self.fs is not None
            and self.f1 is not None
            and self.f2 is not None
        )

    def _assign_scan_params(self, scan):
        self.fs = scan.fs
        self.fc = scan.fc

    def _assign_config_params(self, config):
        if config.scan.sampling_frequency is not None:
            self.fs = config.scan.sampling_frequency
        if config.scan.center_frequency is not None:
            self.fc = config.scan.center_frequency

    def process(self, data):
        axis = data.ndim + self.axis if self.axis < 0 else self.axis

        if data.shape[-1] == 2:
            data = channels_to_complex(data, ops=self.ops)

        if self.ops.__name__ == "torch":
            data = data.cpu().numpy()

        data = ndimage.convolve1d(data, self.filter, mode="wrap", axis=axis)
        data = self.prepare_tensor(data)

        if data.dtype in [self.ops.complex64, self.ops.complex128]:
            data = complex_to_channels(data, axis=-1, ops=self.ops)

        return data


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
            Should include `num_taps`, `fs`, `fc` and two lists: `freqs` and `bandwidths`
            which define the filter characteristics. Lengths of those lists should
            be the same and is equal to the number of filters applied. Optionally the
            `units` can be specified, which is for instance `Hz` or `MHz`. Defaults to `Hz`.

    Returns:
        beamformed_data (list): list of filtered data, each element is filtered data
            with shape [..., n_ax, n_el, n_ch] for each filter applied.

    Example:
        >>> params = {
        >>>     'num_taps': 128,
        >>>     'fs': 50e6,
        >>>     'fc': 5e6,
        >>>     'freqs': [-2.5, 0, 2.5],
        >>>     'bandwidths': [1, 1, 1],
        >>>     'units': 'MHz'
        >>> }
        >>> mbpf = usbmd.ops.MultiBandPassFilter(
        >>>     params=params, modtype='iq', fs=50e6, fc=5e6, axis=-3)
        >>> filtered_data = mbpf(beamformed_data)
    """

    def __init__(self, params=None, modtype=None, fs=None, fc=None, axis=-3, **kwargs):
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.params = params
        self.modtype = modtype
        self.axis = axis
        self.fs = fs
        self.fc = fc

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
        # make sure fs is correct for IQ (downsampled)
        fs = self.fs * unit_factor
        fc = self.fc * unit_factor  # fc is only used when RF

        if self.modtype == "iq":
            fc = 0  # fc is automatically set to zero if IQ
            self.filter_params = [
                {"num_taps": num_taps, "fs": fs, "f": fc - offset, "bw": bw}
                for offset, bw in zip(offsets, bandwidths)
            ]
        elif self.modtype == "rf":
            self.filter_params = [
                {
                    "num_taps": num_taps,
                    "fs": fs,
                    "f1": fc - offset - bw / 2,
                    "f2": fc - offset + bw / 2,
                }
                for offset, bw in zip(offsets, bandwidths)
            ]
        self.filters = []
        for param in self.filter_params:
            if self.modtype == "iq":
                filter_weights = get_low_pass_iq_filter(**param)
            elif self.modtype == "rf":
                filter_weights = get_band_pass_filter(**param)
            self.filters.append(filter_weights)

    @property
    def _ready(self):
        return (
            self.axis is not None
            and self.modtype is not None
            and self.params is not None
            and self.fs is not None
            and self.fc is not None
        )

    def _assign_scan_params(self, scan):
        self.fs = scan.fs
        self.fc = scan.fc

    def _assign_config_params(self, config):
        if config.scan.sampling_frequency is not None:
            self.fs = config.scan.sampling_frequency
        if config.scan.center_frequency is not None:
            self.fc = config.scan.center_frequency

    def process(self, data):
        axis = data.ndim + self.axis if self.axis < 0 else self.axis

        if self.modtype == "iq":
            assert data.shape[-1] == 2, "IQ data should have 2 channels."
            data = channels_to_complex(data, ops=self.ops)

        if self.ops.__name__ == "torch":
            data = data.cpu().numpy()

        data_list = []
        for _filter in self.filters:
            _data = ndimage.convolve1d(data, _filter, mode="wrap", axis=axis)
            _data = self.prepare_tensor(_data)
            if self.modtype == "iq":
                _data = complex_to_channels(_data, axis=-1, ops=self.ops)
            data_list.append(_data)

        return data_list


@ops_registry("scan_convert")
class ScanConvert(Operation):
    """Scan convert images to cartesian coordinates."""

    def __init__(
        self,
        x_axis=None,
        z_axis=None,
        spline_order=1,
        fill_value=None,
        n_pixels=None,
        **kwargs,
    ):
        """Initialize the ScanConvert operation.

        Args:
            x_axis (ndarray, optional): linspace of the angles
            z_axis (ndarray, optional): linspace of the depth
            spline_order (int, optional): Order of spline interpolation.
                Defaults to 1.
            fill_value (float, optional): Value of the points that cannot be
                mapped from sample_points to grid. Defaults to None, which sets
                value to minimun of dynamic range.
            n_pixels (int, optional): Number of pixels (widht) in output image.
                height is derived by division of sqrt(2) of width. Defaults to None.
                in this case n_pixels is set using default arg of scan_convert func.

        Returns:
            image_sc (ndarray): Output image (converted to cartesian coordinates).

        """
        super().__init__(
            input_data_type=None,
            output_data_type=None,
            **kwargs,
        )
        self.x_axis = x_axis
        self.z_axis = z_axis
        self.spline_order = spline_order
        self.fill_value = fill_value
        self.n_pixels = n_pixels
        self.probe_type = None

    @property
    def _ready(self):
        if self.probe_type != "phased":
            return True
        return (
            self.x_axis is not None
            and self.z_axis is not None
            and self.spline_order is not None
            and self.fill_value is not None
            and self.n_pixels is not None
        )

    def process(self, data):
        if self.probe_type != "phased":
            return data

        # TODO: not ready for torch yet
        if self.ops.__name__ == "torch":
            data = data.cpu().numpy()

        return scan_convert(
            data,
            self.x_axis,
            self.z_axis,
            n_pixels=self.n_pixels,
            spline_order=self.spline_order,
            fill_value=self.fill_value,
        )

    def _assign_config_params(self, config):
        self.n_pixels = config.data.output_size
        self.fill_value = config.data.dynamic_range[0]

    def _assign_probe_params(self, probe):
        self.probe_type = probe.probe_type
        if self.probe_type == "phased":
            self.x_axis = probe.angle_deg_axis

    def _assign_scan_params(self, scan):
        self.z_axis = scan.z_axis


def demodulate(rf_data, fs=None, fc=None, bandwidth=None, filter_coeff=None):
    """Demodulates an RF signal to complex base-band (IQ).

    Demodulates the radiofrequency (RF) bandpass signals and returns the
    Inphase/Quadrature (I/Q) components. IQ is a complex whose real (imaginary)
    part contains the in-phase (quadrature) component.

    This function operates (i.e. demodulates) on the RF signal over the
    (fast-) time axis which is assumed to be the last axis.

    Args:
        rf_data (ndarray): real valued input array of size [..., n_ax, n_el].
            second to last axis is fast-time axis.
        fs (float): the sampling frequency of the RF signals (in Hz).
            Only not necessary when filter_coeff is provided.
        fc (float, optional): represents the center frequency (in Hz).
            Defaults to None.
        bandwidth (float, optional): Bandwidth of RF signal in % of center
            frequency. Defaults to None.
            The bandwidth in % is defined by:
            B = Bandwidth_in_% = Bandwidth_in_Hz*(100/fc).
            The cutoff frequency:
            Wn = Bandwidth_in_Hz/Fs, i.e:
            Wn = B*(Fc/100)/Fs.
        filter_coeff (list, optional): (b, a), numerator and denominator coefficients
            of FIR filter for quadratic band pass filter. All other parameters are ignored
            if filter_coeff are provided. Instead the given filter_coeff is directly used.
            If not provided, a filter is derived from the other params (fs, fc, bandwidth).
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html

    Returns:
        iq_data (ndarray): complex valued base-band signal.

    """
    rf_data = np.array(rf_data)
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
        assert fs is not None, "provide fs when no filter is given."
        # Time vector
        t = np.arange(n_ax) / fs
        t0 = 0
        t = t + t0

        # Estimate center frequency
        if fc is None:
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
            fc = idx * fs / n_ax

        # Normalized cut-off frequency
        if bandwidth is None:
            Wn = min(2 * fc / fs, 0.5)
            bandwidth = fc * Wn
        else:
            assert np.isscalar(
                bandwidth
            ), "The signal bandwidth (in %) must be a scalar."
            assert (bandwidth > 0) & (
                bandwidth <= 200
            ), "The signal bandwidth (in %) must be within the interval of ]0,200]."
            # bandwidth in Hz
            bandwidth = fc * bandwidth / 100
            Wn = bandwidth / fs
        assert (Wn > 0) & (Wn <= 1), (
            "The normalized cutoff frequency is not within the interval of (0,1). "
            "Check the input parameters!"
        )

        # Down-mixing of the RF signals
        carrier = np.exp(-1j * 2 * np.pi * fc * t)
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
        if fs < (2 * fc + bandwidth):
            # lower and higher frequencies of the bandpass signal
            fL = fc - bandwidth / 2
            fH = fc + bandwidth / 2
            n = fH // (fH - fL)
            harmless_aliasing = any(
                (2 * fH / np.arange(1, n) <= fs) & (fs <= 2 * fL / np.arange(1, n))
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


def upmix(iq_data, fs, fc, upsampling_rate=6):
    """Upsamples and upmixes complex base-band signals (IQ) to RF.

    Args:
        iq_data (ndarray): complex valued input array of size [..., n_ax, n_el]. second
            to last axis is fast-time axis.
        fs (float): the sampling frequency of the input IQ signal (in Hz).
            resulting fs of RF data is upsampling_rate times higher.
        fc (float, optional): represents the center frequency (in Hz).

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
    fs *= upsampling_rate

    t = np.arange(n_ax) / fs
    t0 = 0
    t = t + t0

    # interpolation
    iq_data_upsampled = signal.resample(iq_data, num=n_ax, axis=-2)

    # Up-mixing of the IQ signals
    carrier = np.exp(1j * 2 * np.pi * fc * t)
    # add the singleton dimensions
    carrier = np.reshape(carrier, (*[1] * (n_dim - 2), n_ax, 1))

    rf_data = iq_data_upsampled * carrier
    rf_data = np.real(rf_data) * np.sqrt(2)

    return rf_data.astype(np.float32)


def get_band_pass_filter(num_taps, fs, f1, f2):
    """Band pass filter

    Args:
        num_taps (int): number of taps in filter.
        fs (float): sample frequency in Hz.
        f1 (float): cutoff frequency in Hz of left band edge.
        f2 (float): cutoff frequency in Hz of right band edge.

    Returns:
        ndarray: band pass filter
    """
    bpf = signal.firwin(num_taps, [f1, f2], pass_zero=False, fs=fs)
    return bpf


def get_low_pass_iq_filter(num_taps, fs, f, bw):
    """Design low pass filter.

    LPF with num_taps points and cutoff at bw / 2

    Args:
        num_taps (int): number of taps in filter.
        fs (float): sample frequency.
        f (float): center frequency.
        bw (float): bandwidth in Hz.
    Raises:
        AssertionError: if cutoff frequency (bw / 2) is not within (0, fs / 2)

    Returns:
        ndarray: complex LP filter
    """
    assert (bw / 2 > 0) & (bw / 2 < fs / 2), log.error(
        "Cutoff frequency must be within (0, fs / 2), "
        f"got {bw / 2} Hz, must be within (0, {fs / 2}) Hz"
    )
    t_qbp = np.arange(num_taps) / fs
    lpf = signal.firwin(num_taps, bw / 2, pass_zero=True, fs=fs) * np.exp(
        1j * 2 * np.pi * f * t_qbp
    )
    return lpf


def complex_to_channels(complex_data, axis=-1, ops=np):
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


def channels_to_complex(data, ops=np):
    """Convert array with real and imaginary components at
    different channels to complex data array.

    Args:
        data (ndarray): input data, with at 0 index of axis
            real component and 1 index of axis the imaginary.

    Returns:
        ndarray: complex array with real and imaginary components.
    """
    assert data.shape[-1] == 2, "Data must have two channels."
    assert data.dtype in [ops.float32, ops.float64], "Data must be float type."
    return ops.complex(data[..., 0], data[..., 1])


def take(data, indices, axis=-1, ops=np):
    """Take values from data along axis.

    Args:
        data (ndarray): input data.
        indices (ndarray): indices to take from data.
        axis (int, optional): axis to take from. Defaults to -1.
    """

    # make indices broadcastable by adding singleton dimensions around axis
    if axis < 0:
        axis = data.ndim + axis
    indices = ops.reshape(indices, [1] * axis + [-1] + [1] * (data.ndim - axis - 1))
    return ops.take_along_axis(data, indices, axis=axis)


def hilbert(x, N: int = None, axis=-1, ops=np):
    """Manual implementation of Hilbert transform.

    Operated in the Fourier domain.

    Args:
        x (ndarray): input data of any shape.
        N (int, optional): number of points in the FFT. Defaults to None.
        axis (int, optional): axis to operate on. Defaults to -1.
        ops (module, optional): operations module. Defaults to np (numpy).
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
        zeros = ops.zeros(input_shape[:axis] + (pad,) + input_shape[axis + 1 :])
        x = ops.concatenate((x, zeros), axis=axis)
        n_ax = N

    h = np.zeros(n_ax)

    # Create mask to remove the negative frequencies and double the frequencies above 0.
    if n_ax % 2 == 0:
        h[0] = h[n_ax // 2] = 1
        h[1 : n_ax // 2] = 2
    else:
        h[0] = 1
        h[1 : (n_ax + 1) // 2] = 2

    h = ops.convert_to_tensor(h)
    h = ops.expand_dims(ops.cast(ops.complex(h, h), ops.complex64), axis=0)

    # switch n_ax and n_el elements (based on ndim)
    idx = list(range(n_dim))
    # make sure axis gets to the end for fft (operates on last axis)
    idx.remove(axis)
    idx.append(axis)

    x = ops.permute(x, idx)

    x = ops.cast(ops.complex(x, x), ops.complex64)

    Xf = ops.fft.fft(x)
    x = ops.fft.ifft(Xf * h)

    # switch back to original shape
    idx = list(range(n_dim))
    idx.insert(axis, idx.pop(-1))
    x = ops.permute(x, idx)
    return x
