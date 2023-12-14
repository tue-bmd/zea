"""Register classes

- **Author(s)**     : Tristan Stevens, Vincent van de Schaft
- **Date**          : 28/02/2023
"""
from usbmd.utils.registration import RegisterDecorator

# The registry for the datasets linking each dataset to
# a probe_name and a scan_class.
dataset_registry = RegisterDecorator(items_to_register=["probe_name", "scan_class"])

# The registry for the probes.
probe_registry = RegisterDecorator()

# The registry for the beamformers.
# separate registries to avoid dupicate entries
# as torch and tf beamformers cannot coexist in same registry
tf_beamformer_registry = RegisterDecorator(items_to_register=["name", "framework"])

torch_beamformer_registry = RegisterDecorator(items_to_register=["name", "framework"])

post_processing_registry = RegisterDecorator(items_to_register=["name", "framework"])

metrics_registry = RegisterDecorator(
    items_to_register=["name", "framework", "supervised"]
)

checks_registry = RegisterDecorator(items_to_register=["data_type"])
