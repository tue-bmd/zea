from abc import ABC, abstractmethod

import numpy as np

from usbmd.utils.checks import get_check
from usbmd.utils.utils import translate

# import tensorflow as tf


class Operation(ABC):
    def __init__(
        self,
        name=None,
        input_data_type=None,
        output_data_type=None,
        batch_dim=True,
    ):
        self.name = name
        self.input_data_type = input_data_type
        self.output_data_type = output_data_type
        self.batch_dim = batch_dim

        self._ops = None

    @property
    def ops(self):
        assert self._ops is not None, "ops package is not set"
        return self._ops

    @ops.setter
    def ops(self, ops):
        self._ops = ops

    @abstractmethod
    def process(self, *args, **kwargs):
        # Process the input data
        pass

    def set_ops_pkg(self, ops):
        self.ops = ops

    def __call__(self, data, *args, **kwargs):
        if self.input_data_type:
            check = get_check(self.input_data_type)
            check(data, self.batch_dim)
        return self.process(data, *args, **kwargs)

    def set_params(self, config):
        # Set the arguments using the config dictionary
        pass


class Pipeline:
    def __init__(self, operations, ops=np, batch_dim=True):
        self.operations = operations
        for operation in self.operations:
            operation.ops = ops
            operation.batch_dim = batch_dim

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

    @property
    def ops(self):
        return self.operations[0].ops

    def set_params(self, config):
        for operation in self.operations:
            operation.set_params(config)

    def process(self):
        for operation in self.operations:
            operation()

    def compile(self, jit=False):
        if self.ops == np:
            return
        # if self.ops == tf:
        #     tf.function(self.process, jit=jit)


class Mean(Operation):
    def __init__(self, axis):
        super().__init__(
            name="Mean",
            input_data_type=None,
            output_data_type=None,
        )
        self.axis = axis

    def process(self, data):
        return self.ops.mean(data, axis=self.axis)


class Normalize(Operation):
    def __init__(self, output_range=None, input_range=None):
        """Normalize data to a given range.

        Args:
            output_range (Tuple, optional): Range to which data should be mapped.
                Defaults to (0, 1).
            input_range (Tuple, optional): Range of input data. If None, the range
                of the input data will be computed. Defaults to None.
        """
        super().__init__(
            name="Normalize",
            input_data_type=None,
            output_data_type=None,
        )
        self.output_range = output_range
        self.input_range = input_range

    def set_params(self, config):
        self.input_range = config.data.input_range
        self.output_range = config.data.output_range

    def process(self, data):
        if self.output_range is None:
            self.output_range = (0, 1)

        if self.input_range is None:
            minimum = np.min(data)
            maximum = np.max(data)
            self.input_range = (minimum, maximum)
        else:
            a_min, a_max = self.input_range
            data = self.ops.clip(data, a_min, a_max)
        return translate(data, self.input_range, self.output_range)


class LogCompress(Operation):
    def __init__(self, dynamic_range=None):
        super().__init__(
            name="LogCompress",
            input_data_type=None,
            output_data_type=None,
        )
        self.dynamic_range = dynamic_range

    def set_params(self, config):
        self.dynamic_range = config.dynamic_range

    def process(self, data):
        if self.dynamic_range is None:
            self.dynamic_range = (-60, 0)

        data[data == 0] = np.finfo(float).eps
        compressed_data = 20 * np.log10(data)
        compressed_data = self.ops.clip(compressed_data, *self.dynamic_range)
        return compressed_data


class Downsample(Operation):
    def __init__(self, factor: int, phase: int = None, axis: int = -1):
        super().__init__(
            name="Downsample",
            input_data_type=None,
            output_data_type=None,
        )
        self.factor = factor
        self.phase = phase
        self.axis = axis

    def set_params(self, config):
        self.factor = config.scan.downsample

    def process(self, data):
        length = self.ops.shape(data)[self.axis]
        if self.phase is None:
            self.phase = 0
        sample_idx = self.ops.arange(self.phase, length, self.factor)
        return self.ops.take(data, sample_idx, axis=self.axis)


class Companding(Operation):
    def __init__(self, expand=False, comp_type=None, mu=255, A=87.6):
        super().__init__(
            name="Companding",
            input_data_type=None,
            output_data_type=None,
        )
        self.expand = expand
        self.comp_type = comp_type
        self.mu = mu
        self.A = A

    def set_params(self, config):
        self.expand = config.expand
        self.comp_type = config.comp_type
        self.mu = config.mu
        self.A = config.A

    def process(self, data):
        data = self.ops.clip(data, -1, 1)

        def mu_law_compress(x):
            y = (
                self.ops.sign(x)
                * self.ops.log(1 + self.mu * self.ops.abs(x))
                / self.ops.log(1 + self.mu)
            )
            return y

        def mu_law_expand(y):
            x = self.ops.sign(y) * ((1 + self.mu) ** (self.ops.abs(y)) - 1) / self.mu
            return x

        def a_law_compress(x):
            x_sign = self.ops.sign(x)
            x_abs = self.ops.abs(x)
            A_log = self.ops.log(self.A)

            idx_1 = self.ops.where((x_abs >= 0) & (x_abs < (1 / self.A)))
            idx_2 = self.ops.where((x_abs >= (1 / self.A)) & (x_abs <= 1))

            y = x_sign
            y[idx_1] *= self.A * x_abs[idx_1] / (1 + A_log)
            y[idx_2] *= (1 + self.ops.log(self.A * x_abs[idx_2])) / (1 + A_log)
            return y

        def a_law_expand(y):
            y_sign = self.ops.sign(y)
            y_abs = self.ops.abs(y)
            A_log = self.ops.log(self.A)

            idx_1 = self.ops.where((y_abs >= 0) & (y_abs < (1 / (1 + A_log))))
            idx_2 = self.ops.where((y_abs >= (1 / (1 + A_log))) & (y_abs <= 1))

            x = y_sign
            x[idx_1] *= y_abs[idx_1] * (1 + A_log) / self.A
            x[idx_2] *= self.ops.exp(y_abs[idx_2] * (1 + A_log) - 1) / self.A
            return x

        if self.comp_type.lower() == "mu":
            if self.expand:
                array_out = mu_law_expand(data)
            else:
                array_out = mu_law_compress(data)
        elif self.comp_type.lower() == "a":
            if self.expand:
                array_out = a_law_expand(data)
            else:
                array_out = a_law_compress(data)

        return array_out


class EnvelopeDetection(Operation):
    def __init__(self, axis=-1):
        super().__init__(
            name="EnvelopeDetection",
            input_data_type=None,
            output_data_type=None,
        )
        self.axis = axis

    def process(self, data):
        return np.abs(self.ops.hilbert(data, axis=self.axis))
