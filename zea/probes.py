"""Module containing parameters and classes for different ultrasound probes.

All probes are based on the base :class:`Probe` class.

Supported probes
----------------

- :class:`Probe` -- Base class for all probes
- :class:`Verasonics_l11_4v` -- Verasonics L11-4V linear ultrasound transducer
- :class:`Verasonics_l11_5v` -- Verasonics L11-5V linear ultrasound transducer
- :class:`Esaote_sll1543` -- Esaote SLL1543 linear ultrasound transducer
"""  # noqa: E501

import numpy as np

from zea.data.spec import ProbeSpec
from zea.internal.core import dict_to_tensor
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


class Probe(ProbeSpec):
    def get_parameters(self):
        return {key: getattr(self, key) for key in self.SCHEMA}

    @classmethod
    def from_name(cls, probe_name, **kwargs) -> "Probe":
        """Create a probe from its name.

        Args:
            probe_name (str): Name of the probe.

        Returns:
            Probe: Probe object.
        """
        try:
            probe_class = probe_registry[probe_name]
        except KeyError as exc:
            raise NotImplementedError(f"Probe {probe_name} not implemented.") from exc

        return probe_class(**kwargs)

    def to_tensor(self, keep_as_is=None):
        """Convert the attributes in the object to tensors."""
        # TODO: merge this with Parameters.to_tensor()
        return dict_to_tensor(self.get_parameters(), keep_as_is=keep_as_is)

    def __post_init__(self):
        # Legacy file support
        if self.center_frequency.dtype == np.int32:
            self.center_frequency = self.center_frequency.astype(np.float32)
        super().__post_init__()


@probe_registry(name="verasonics_l11_4v")
class Verasonics_l11_4v(Probe):
    """Verasonics L11-4V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-4V linear ultrasound transducer."""

        probe_geometry = create_probe_geometry(n_el=128, pitch=0.3e-3)
        center_frequency = 6.25e6
        bandwidth_percent = (11 - 4) * 100 / (center_frequency / 1e6)

        super().__init__(
            name="verasonics_l11_4v",
            type="linear",
            center_frequency=center_frequency,
            bandwidth_percent=bandwidth_percent,
            probe_geometry=probe_geometry,
        )


@probe_registry(name="verasonics_l11_5v")
class Verasonics_l11_5v(Probe):
    """Verasonics L11-5V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-5V linear ultrasound transducer."""

        probe_geometry = create_probe_geometry(n_el=128, pitch=0.3e-3)
        center_frequency = 6.25e6
        bandwidth_percent = (11 - 5) * 100 / (center_frequency / 1e6)

        # elevation_focus = 18e-3
        # sensitivity = -52 +/- 3 dB

        super().__init__(
            name="verasonics_l11_5v",
            type="linear",
            center_frequency=center_frequency,
            bandwidth_percent=bandwidth_percent,
            probe_geometry=probe_geometry,
        )


@probe_registry(name="esaote_sll1543")
class Esaote_sll1543(Probe):
    """Esaote SLL1543 linear ultrasound transducer.

    https://lysis.cc/products/esaote-sl1543
    """

    def __init__(self):
        """Set probe parameters"""

        probe_geometry = create_probe_geometry(n_el=192, pitch=0.245 / 1e3)
        center_frequency = 8e6
        bandwidth_percent = (13 - 3) * 100 / (center_frequency / 1e6)

        super().__init__(
            name="esaote_sll1543",
            type="linear",
            center_frequency=center_frequency,
            bandwidth_percent=bandwidth_percent,
            probe_geometry=probe_geometry,
        )
