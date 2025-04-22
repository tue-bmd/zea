"""Initialize functions for datasets / probes and scans.

- **Author(s)**     : Tristan Stevens
- **Date**          : 28/02/2023
"""

from usbmd.probes import get_probe
from usbmd.registry import probe_registry
from usbmd.scan import Scan
from usbmd.utils import safe_initialize_class


def initialize_scan_from_probe(probe):
    """
    Initializes a Scan object based on the default scan parameters of the given
    probe. Any parameters for which no default values are defined in the probe
    class are initialized with the default arguments in the Scan constructor.

    Args:
        probe (Probe or str): A probe object or probe name.

    Returns:
        Scan: A Scan object that is compatible with the probe.
    """
    if isinstance(probe, str):
        probe = get_probe(probe)

    default_parameters = probe.get_parameters()
    try:
        scan = safe_initialize_class(Scan, **default_parameters)
    except Exception as e:
        raise ValueError(
            f"Could not initialize scan from probe: {probe.__class__.__name__}. "
            f"Found default parameters: {default_parameters.keys()}. "
        ) from e
    return scan


def initialize_scan_from_config(config):
    """
    Defines a scan based on parameters in a config.

    Args:
        config (utils.config.Config): The config object to read parameters from.

    Raises:
        NotImplementedError: This method is not implemented and always raises
        this error.
    """
    raise NotImplementedError
