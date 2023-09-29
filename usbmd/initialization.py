"""Initialize functions for datasets / probes and scans.

- **Author(s)**     : Tristan Stevens
- **Date**          : 28/02/2023
"""
from usbmd.probes import get_probe
from usbmd.registry import dataset_registry, probe_registry
from usbmd.scan import Scan


def get_probe_from_config(config):
    """
    Defines a probe based on parameters in a config.
    Specifically, `data.dataset_name` key is necessary to
    figure out which probe to initialize.

    Args:
        config (Config): The config object to read parameters from.

    """
    dataset_name = config.data.dataset_name

    probe_name = dataset_registry.get_parameter(cls_or_name=dataset_name,
                                                parameter='probe_name')
    probe = probe_registry[probe_name]
    return probe

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

    default_parameters = probe.get_default_scan_parameters()

    scan = Scan(**default_parameters)
    return scan

def initialize_scan_from_config(config):
    """
    Defines a scan based on parameters in a config.

    Args:
        config (Config): The config object to read parameters from.

    Raises:
        NotImplementedError: This method is not implemented and always raises
        this error.
    """
    raise NotImplementedError
