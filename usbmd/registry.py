"""Register classes

- **Author(s)**     : Tristan Stevens, Vincent van de Schaft
- **Date**          : 28/02/2023
"""
from usbmd.utils.registration import RegisterDecorator

# The registry for the datasets linking each dataset to a probe_name and a
# scan_class.
dataset_registry = RegisterDecorator(
    items_to_register=['probe_name', 'scan_class']
)

# The registry for the probes.
probe_registry = RegisterDecorator()

# The registry for the beamformers.
beamformer_registry = RegisterDecorator(
    items_to_register=['name', 'framework']
)