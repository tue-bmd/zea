"""Module containing parameters and classes for different ultrasound probes.

All probes are based on the base :class:`Probe` class.

Supported probes
----------------

- :class:`Probe` -- Base class for all probes
- :class:`Verasonics_l11_4v` -- Verasonics L11-4V linear ultrasound transducer
- :class:`Verasonics_l11_5v` -- Verasonics L11-5V linear ultrasound transducer
- :class:`Esaote_sll1543` -- Esaote SLL1543 linear ultrasound transducer


Example usage
^^^^^^^^^^^^^^

We can initialize a generic probe with the following code:

.. doctest::

    >>> from zea import Probe

    >>> probe = Probe.from_name("generic")
    >>> print(probe.get_parameters())
    {'probe_geometry': None, 'center_frequency': None, 'sampling_frequency': None, 'xlims': None, 'zlims': None, 'n_el': None}
"""  # noqa: E501

import numpy as np

from zea import log
from zea.internal.core import Object, dict_to_tensor
from zea.internal.registry import probe_registry


def create_probe_geometry(n_el, pitch):
    """Create probe geometry based on number of elements and pitch.

    Args:
        n_el (int): Number of elements in the probe.
        pitch (float): Pitch of the elements in the probe.

    Returns:
        np.ndarray: Probe geometry with shape (n_el, 3).
    """
    aperture = (n_el - 1) * pitch
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el).T,
            np.zeros((n_el,)),
            np.zeros((n_el,)),
        ],
        axis=1,
    ).astype(np.float32)
    return probe_geometry


@probe_registry(name="generic")
class Probe(Object):
    """Probe base class. All probes should inherit from this class."""

    # TODO: our data format doesn't store the actual probe center frequency anymore...
    # We should decide how to handle this in the future.
    def __init__(
        self,
        probe_geometry=None,
        center_frequency=None,
        sampling_frequency=None,
        xlims=None,
        zlims=None,
        bandwidth_MHz=None,
        probe_type="linear",
    ):
        """Initialize probe.

        Args:
            probe_geometry (np.ndarray, optional): (n_el, 3) array with element
                positions in meters. Defaults to None.
            center_frequency (float, optional): Center frequency of the probe in Hz.
            sampling_frequency (float, optional): Sampling frequency of the probe in Hz.
            xlims (tuple, optional): Tuple with the limits of the probe in the x
                direction in meters. Defaults to None.
            zlims (tuple, optional): Tuple with the limits of the probe in the z
                direction in meters (depth). Defaults to None.
            bandwidth_MHz (float, optional): Bandwidth of the probe in MHz.
                Defaults to None.
            probe_type (str, optional): Type of probe. Currently only `linear`
                and `phased` probes are supported. Defaults to 'linear'.
        """
        super().__init__()

        self.probe_geometry = probe_geometry
        self.center_frequency = center_frequency
        self.sampling_frequency = sampling_frequency
        self.xlims = xlims
        self.zlims = zlims
        self.bandwidth_MHz = bandwidth_MHz
        self.probe_type = probe_type.lower()

        assert self.probe_type in (
            "linear",
            "phased",
        ), 'Probe type must be either "linear" or "phased"'

        if self.probe_geometry is not None:
            self.n_el = probe_geometry.shape[0]
        else:
            self.n_el = None

        self.filter = None

    def get_parameters(self):
        """Returns a dictionary with default parameters."""
        return {
            "probe_geometry": self.probe_geometry,
            "center_frequency": self.center_frequency,
            "sampling_frequency": self.sampling_frequency,
            "xlims": self.xlims,
            "zlims": self.zlims,
            "n_el": self.n_el,
        }

    @classmethod
    def from_parameters(cls, probe_name: str, parameters: dict) -> "Probe":
        """Instantiate a probe by name, overriding with parameters from dict."""
        if probe_name in probe_registry:
            probe = probe_registry[probe_name]()  # instantiate with class defaults
            for key, value in parameters.items():
                if hasattr(probe, key):
                    setattr(probe, key, value)
                else:
                    raise ValueError(f"Unknown parameter {key} for probe {probe_name}")
            return probe

        # Fallback to generic probe with file parameters
        return probe_registry["generic"](**parameters)

    @classmethod
    def from_name(cls, probe_name, fallback=False, **kwargs) -> "Probe":
        """Create a probe from its name.

        Args:
            probe_name (str): Name of the probe.

        Returns:
            Probe: Probe object.
        """
        try:
            probe_class = probe_registry[probe_name]
        except KeyError as exc:
            if not fallback:
                raise NotImplementedError(f"Probe {probe_name} not implemented.") from exc
            log.warning(f"Probe {probe_name} not implemented, falling back to `generic` probe.")
            probe_class = probe_registry["generic"]

        return probe_class(**kwargs)

    def to_tensor(self, keep_as_is=None):
        """Convert the attributes in the object to tensors."""
        # TODO: merge this with Parameters.to_tensor()
        return dict_to_tensor(self.get_parameters(), keep_as_is=keep_as_is)


@probe_registry(name="verasonics_l11_4v")
class Verasonics_l11_4v(Probe):
    """Verasonics L11-4V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-4V linear ultrasound transducer."""

        n_el = 128
        pitch = 0.3e-3
        probe_geometry = create_probe_geometry(n_el, pitch)
        bandwidth_MHz = 11 - 4

        super().__init__(
            probe_geometry=probe_geometry,
            center_frequency=6.25e6,
            sampling_frequency=4 * 6.25e6,
            xlims=(probe_geometry[0, 0], probe_geometry[-1, 0]),
            zlims=(0.965e-3, 63.58375e-3),
            bandwidth_MHz=bandwidth_MHz,
            probe_type="linear",
        )


@probe_registry(name="verasonics_l11_5v")
class Verasonics_l11_5v(Probe):
    """Verasonics L11-5V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-5V linear ultrasound transducer."""

        n_el = 128
        pitch = 0.3e-3
        probe_geometry = create_probe_geometry(n_el, pitch)
        bandwidth_MHz = 11 - 5

        # elevation_focus = 18e-3
        # sensitivity = -52 +/- 3 dB

        super().__init__(
            probe_geometry=probe_geometry,
            center_frequency=6.25e6,
            sampling_frequency=4 * 6.25e6,
            xlims=(probe_geometry[0, 0], probe_geometry[-1, 0]),
            zlims=(0.965e-3, 63.58375e-3),
            bandwidth_MHz=bandwidth_MHz,
            probe_type="linear",
        )


@probe_registry(name="esaote_sll1543")
class Esaote_sll1543(Probe):
    """Esaote SLL1543 linear ultrasound transducer.

    https://lysis.cc/products/esaote-sl1543
    """

    def __init__(self):
        """Set probe parameters"""

        n_el = 192
        pitch = 0.245 / 1e3
        probe_geometry = create_probe_geometry(n_el, pitch)

        bandwidth_MHz = 13 - 3

        super().__init__(
            probe_geometry=probe_geometry,
            center_frequency=8e6,
            sampling_frequency=65e6,
            xlims=(-15e-3, 15e-3),
            zlims=(0, 40e-3),
            bandwidth_MHz=bandwidth_MHz,
            probe_type="linear",
        )
