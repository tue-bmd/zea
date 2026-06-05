"""Ultrasound probe definitions and the base :class:`Probe` class.

A probe describes the physical transducer: element positions, centre frequency,
bandwidth, and properties such as element dimensions and lens geometry.
All probe objects are instances of :class:`Probe`, which inherits validation from
:class:`~zea.data.spec.ProbeSpec`.

There are three ways to obtain a probe:

Loading a built-in probe
^^^^^^^^^^^^^^^^^^^^^^^^

A small set of probes is pre-defined and can be retrieved by name:

.. doctest::

    >>> from zea import Probe
    >>> probe = Probe.from_name("verasonics_l11_4v")
    >>> probe.probe_center_frequency
    np.float32(6250000.0)
    >>> probe.n_el
    128

See :meth:`Probe.from_name` for the full list of registered names.

Built-in probes
~~~~~~~~~~~~~~~

- :class:`Verasonics_l11_4v` -- Verasonics L11-4V linear array
- :class:`Verasonics_l11_5v` -- Verasonics L11-5V linear array
- :class:`Esaote_sll1543` -- Esaote SLL1543 linear array

Loading a probe from a data file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you open a :class:`~zea.data.file.File`, the probe stored in that file is
accessible through the :attr:`~zea.data.file.File.probe` property:

.. doctest::

    >>> from zea import File
    >>> path = (
    ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
    ... )
    >>> with File(path) as f:
    ...     probe = f.probe
    >>> probe.name
    'verasonics_l11_4v'

Defining a custom probe
^^^^^^^^^^^^^^^^^^^^^^^^

Pass any combination of fields from :class:`~zea.data.spec.ProbeSpec` directly
to :class:`Probe`.  Only the fields you provide are validated; everything else
is left as ``None``:

.. doctest::

    >>> import numpy as np
    >>> from zea import Probe
    >>> from zea.probes import create_probe_geometry

    >>> probe = Probe(
    ...     name="my_probe",
    ...     type="linear",
    ...     probe_center_frequency=np.float32(5e6),
    ...     probe_geometry=create_probe_geometry(n_el=64, pitch=0.3e-3),
    ... )
    >>> probe.n_el
    64

You can also register a custom probe class with the
:data:`~zea.internal.registry.probe_registry` decorator so it becomes
retrievable by name — see the built-in classes below as examples.

Saving a probe to a data file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a :class:`Probe` object directly to :meth:`~zea.data.file.File.create`
via the ``probe`` argument, alternatively a simple dictionary of probe
parameters will also work:

.. doctest::

    >>> import numpy as np
    >>> from zea import File, Probe

    >>> n_frames, n_tx, n_el, n_ax = 1, 4, 128, 64
    >>> probe = Probe.from_name("verasonics_l11_4v")
    >>> raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
    >>> scan = {
    ...     "sampling_frequency": np.float32(40e6),
    ...     "center_frequency": np.float32(6.25e6),
    ...     "demodulation_frequency": np.float32(6.25e6),
    ...     "initial_times": np.zeros(n_tx, dtype=np.float32),
    ...     "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
    ...     "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
    ...     "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
    ...     "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
    ...     "polar_angles": np.zeros(n_tx, dtype=np.float32),
    ...     "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
    ... }
    >>> File.create(
    ...     "probe_example.hdf5",
    ...     data={"raw_data": raw},
    ...     scan=scan,
    ...     probe=probe, # dictionary or zea.Probe object
    ...     overwrite=True,
    ... )

.. testcleanup::

    import os
    os.remove("probe_example.hdf5")

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
    # These are not converted to Parameters object
    _NON_PARAMETERS = ("name", "type")

    def get_parameters(self):
        """Return a dict of the probe parameters."""
        return {
            key: getattr(self, key)
            for key in self.SCHEMA
            if getattr(self, key) is not None and key not in Probe._NON_PARAMETERS
        }

    def __repr__(self) -> str:
        parts = []
        if self.name is not None:
            parts.append(f"name='{self.name}'")
        if self.type is not None:
            parts.append(f"type='{self.type}'")
        if self.probe_geometry is not None:
            n_el = self.probe_geometry.shape[0]
            parts.append(f"n_el={n_el}")
        if self.probe_center_frequency is not None:
            parts.append(f"fc={float(self.probe_center_frequency) / 1e6:.2f} MHz")
        if self.probe_bandwidth_percent is not None:
            parts.append(f"bw={float(self.probe_bandwidth_percent):.1f}%")
        if self.element_width is not None:
            parts.append(f"pitch={float(self.element_width) * 1e3:.3f} mm")
        return f"Probe({', '.join(parts)})"

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

    @staticmethod
    def get_pitch(probe_geometry: np.ndarray) -> float | None:
        """Compute the pitch (centre-to-centre element spacing) in metres from
        the probe geometry.

        Returns ``None`` when the pitch is not defined for the geometry (fewer
        than 2 elements, or not a 1-D / linear array). Raises :class:`ValueError`
        when the array looks like a ULA but the element positions are not
        uniformly spaced, to surface likely data errors rather than silently
        returning ``None``.
        """

        n_el = probe_geometry.shape[0]
        if n_el < 2:
            raise ValueError(f"Cannot compute pitch: probe has fewer than 2 elements (n_el={n_el})")

        # Only valid for 1-D (linear) arrangements – all elements must lie on the x-axis
        # (y == 0 and z == 0 for every element).
        if not (np.allclose(probe_geometry[:, 1], 0) and np.allclose(probe_geometry[:, 2], 0)):
            raise ValueError(
                "Cannot compute pitch: probe geometry is not 1-D (linear array). "
                "Element positions must have y=0 and z=0 for all elements."
            )

        spacings = np.diff(probe_geometry[:, 0])
        if not np.allclose(spacings, spacings[0], rtol=1e-3):
            raise ValueError(
                "Cannot compute pitch: element x-positions are not uniformly spaced. "
                f"Min spacing: {spacings.min():.4e} m, max: {spacings.max():.4e} m."
            )

        return float(spacings[0])

    @property
    def pitch(self) -> float | None:
        """Centre-to-centre element spacing in metres, derived from :attr:`probe_geometry`.

        Raises :class:`ValueError` when:

        * :attr:`probe_geometry` is not set,
        * the probe has fewer than 2 elements, or
        * the elements are not arranged along a single axis (not a 1-D / linear array).
        * the spacing is non-uniform (elements are present but clearly not a ULA),
            to surface likely data errors rather than silently returning ``None``.

        """
        if self.probe_geometry is None:
            raise ValueError("Cannot compute pitch: probe_geometry is not set")

        return self.get_pitch(self.probe_geometry)

    @property
    def kerf(self) -> float | None:
        """Gap between elements in metres, derived from :attr:`element_width` and :attr:`pitch`."""
        if self.element_width is not None and self.pitch is not None:
            return self.pitch - self.element_width
        return None


@probe_registry(name="verasonics_l11_4v")
class Verasonics_l11_4v(Probe):
    """Verasonics L11-4V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-4V linear ultrasound transducer."""

        probe_geometry = create_probe_geometry(n_el=128, pitch=0.3e-3)
        center_frequency = 6.25e6
        probe_bandwidth_percent = (11 - 4) * 100 / (center_frequency / 1e6)

        super().__init__(
            name="verasonics_l11_4v",
            type="linear",
            probe_center_frequency=center_frequency,
            probe_bandwidth_percent=probe_bandwidth_percent,
            probe_geometry=probe_geometry,
        )


@probe_registry(name="verasonics_l11_5v")
class Verasonics_l11_5v(Probe):
    """Verasonics L11-5V linear ultrasound transducer."""

    def __init__(self):
        """Verasonics L11-5V linear ultrasound transducer."""

        probe_geometry = create_probe_geometry(n_el=128, pitch=0.3e-3)
        center_frequency = 6.25e6
        probe_bandwidth_percent = (11 - 5) * 100 / (center_frequency / 1e6)

        # elevation_focus = 18e-3
        # sensitivity = -52 +/- 3 dB

        super().__init__(
            name="verasonics_l11_5v",
            type="linear",
            probe_center_frequency=center_frequency,
            probe_bandwidth_percent=probe_bandwidth_percent,
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
        probe_bandwidth_percent = (13 - 3) * 100 / (center_frequency / 1e6)

        super().__init__(
            name="esaote_sll1543",
            type="linear",
            probe_center_frequency=center_frequency,
            probe_bandwidth_percent=probe_bandwidth_percent,
            probe_geometry=probe_geometry,
        )
