from abc import ABC, abstractmethod

import numpy as np

from usbmd.utils.checks import get_check
from usbmd.utils.utils import translate
from usbmd.scan import Scan
from usbmd.probes import Probe
from usbmd.utils.config import Config

# import tensorflow as tf

import scipy

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

    def set_params(self, config: Config):
        # Set the arguments using the config dictionary
        pass

    def assign_config_params(self, config: Config):
        # Assign the config parameters to the operation
        pass

    def assign_scan_params(self, scan: Scan):
        # Assign the scan parameters to the operation
        pass

    def assign_probe_params(self, probe: Probe):
        # Assign the probe parameters to the operation
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

    def process(self, data):
        for operation in self.operations:
            data = operation(data)
        return data

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
        self.output_range = None

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
        self.dynamic_range = config.data.dynamic_range

    def process(self, data):
        if self.dynamic_range is None:
            self.dynamic_range = (-60, 0)

        data[data == 0] = np.finfo(float).eps
        compressed_data = 20 * np.log10(data)
        compressed_data = self.ops.clip(compressed_data, *self.dynamic_range)
        return compressed_data


class Downsample(Operation):
    def __init__(self, factor: int = None, phase: int = None, axis: int = -1):
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


class EnvelopeDetect(Operation):
    def __init__(self, axis=-2):
        super().__init__(
            name="EnvelopeDetection",
            input_data_type=None,
            output_data_type=None,
        )
        self.axis = axis

    def process(self, data):
        if data.shape[-1] == 2:
            data = self.ops.take(data, 0, axis=-1) + 1j * self.ops.take(data, 1, axis=-1)
            n_ax = data.shape[self.axis]
            M = 2 ** int(np.ceil(np.log2(n_ax)))
            data = scipy.signal.hilbert(data, N=M, axis=self.axis)
            data = self.ops.take(data, np.arange(n_ax), axis=self.axis)
        return self.ops.abs(data)

class Demodulate(Operation):
    def __init__(self, fs=None, fc=None, bandwidth=None, filter_coeff=None):
        super().__init__(
            name="Demodulate",
            input_data_type=None,
            output_data_type=None,
        )
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.filter_coeff = filter_coeff

    def process(self, data):
        return demodulate(data, self.fs, self.fc, self.bandwidth, self.filter_coeff)

    def assign_scan_params(self, scan):
        self.fs = scan.sampling_frequency
        self.fc = scan.center_frequency
        self.bandwidth = scan.bandwidth_percent

    def assign_probe_params(self, probe):
        pass

    def set_params(self, config):
        self.fs = config.scan.sampling_frequency
        self.fc = config.scan.center_frequency
        # self.bandwidth = config.scan.bandwidth_percent


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
    assert np.isreal(rf_data).all(), "RF must contain real RF signals."

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
