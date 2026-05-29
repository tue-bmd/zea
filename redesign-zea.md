We have several issues on our current workflow. Currently, parameters are grouped in three classes. `Scan`, `Probe`, `Config`.
Inside the file, we have `scan` and `probe` groups. The Scan class is more than a container for holding the parameters.
It also has complex caching logic and more importantly is able to compute a bunch of derived paramters. For some of these derived
parameters (dependencies as we call them), it needs parameters from the Probe class. This is a bit awkward as the two parameter
containers live side by side inside the file, but in the code they are quite intertwined. On top of that Config class is meant to hold manual
paramaters that might overwrite the ones in the file (contained by the Scan and Probe classes). The Config class also usually holds the
pipeline definition. The goal of this redesign is to have a clear way of working regarding storage and handling of parameters.

The first thing is to rename the Scan class to something more generic, like `Parameters`. This way it is not confused with the
spec classes `ScanSpec` and `ProbeSpec`. These latter two classes are simply containers and definitions (schema) of the parameters that
might be in the file. The `Parameters` class (previously `Scan`) will be the one responsible for holding all the parameters after loading from
file as well as have the ability of caching and computing derived parameters. (One small note is that currently `Scan` class inherits from `Parameters` class, we would have to rename the `Parameters` class to something else as well, even more general).

The second thing is how to deal with the `Config` class. Currently we have a schema for that in `zea.internal.config.validation`, but it is outdated and not really used. We envision that `Config` stores the pipeline definition as it does currently (this works already well), and also holds a parameter key which can be used to overwrite set any parameters in the parameters class.

Something like this:

```yaml
pipeline:
  operations:
    - name: "beamform"
      params:
        beamformer: delay_and_sum
        enable_pfield: false
        num_patches: 200 # static pipeline parameters are defined here
    - name: envelope_detect
    - name: normalize
    - name: log_compress

parameters:
    # any scan or probe parameters here as a flat dictionary
    selected_transmits: all
    grid_size_x: 400
    grid_size_z: 600
    apply_lens_correction: false
    lens_sound_speed: 1000
    lens_thickness: 0.001
    # or any other custom / manual parameters are also allowed
    # these are for instanced passed to the pipeline call
    # these are dynamic pipeline parameters
    my_custom_parameter: 42

# other parameters are also allowed, but are generally ignored by the workflow
# unless a user accesses them manually from the code
another_key: some_value
```

```python
import zea

config = zea.Config.from_path("config.yaml")

file_path = (
    "hf://zeahub/picmus/database/experiments/contrast_speckle/"
    "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
)

with zea.File(file_path) as f:
    # this will load the parameters from the file and store them in the Parameters class
    # it internally grabs the probe and scan groups with ProbeSpec and ScanSpec
    parameters = f.load_parameters()
    # to add your custom parameters, you can just update the parameters object with the config parameters
    parameters.update(config.parameters)
    # or manually set any parameter you want
    parameters.update({"my_custom_parameter": 42})
    # these custom parameters are handled different then validated parameters, which is currently only
    # type of parameter allowed by the Parameters class. Have to find a neat solution for that.
    # these manual parameters are never used in downstream derivation of parameters,
    # but are just stored as is, and for instance passed to a pipeline call. Your custom zea.Operation
    # in a Pipeline might expect some custom parameters, and this is a way to set them.

    # one can do still
    scan = f.scan # this does not return the original Scan class anymore but basic ScanSpec
    probe = f.probe # this returns ProbeSpec (similar as before)
    # however the new way of working and how it should be in the snippets is to just get
    # the parameters class to have a single object that holds the parameters
    # as well as is able to compute derived parameters, do caching, lazy loading, etc.

    # and ofcourse grab our data
    raw_data = f.data.raw_data[:]

```

Then we proceed to the processing and pipeline calls.

Initialize a pipeline as usual.

```python
pipeline = zea.Pipeline.from_config(config) # grabs only config.pipeline part
```

Can ofcourse also have a in-code custom pipeline definition

```python
from zea.ops import (
    Beamform,
    Demodulate,
    Pipeline,
    EnvelopeDetect,
    Normalize,
    LogCompress,
)

operations = [
    Beamform(beamformer="delay_and_sum"),
    EnvelopeDetect(),
    Normalize(),
    LogCompress(),
]

pipeline = Pipeline(operations)

```

One can use the :meth:`Pipeline.prepare_parameters` method to convert the Parmaeters class into a flat dictionary of tensors that can be directly passed to the pipeline. Additionally you can inject manual parameters here again as a dictionary (for instance from the config).

```python
tensors = pipeline.prepare_parameters(parameters, **{"my_custom_parameter": 42}, bandwidth=config.parameters.bandwidth)

inputs = {"data": raw_data, **tensors}
outputs = pipeline(**inputs)

data_out = outputs[pipeline.output_key]
```

### other things

- since we merge probe and scan parameters into Parameters class upon file.load_parameters(), there can be no conflicting parameters in the probe and scan groups. One that is already conflicting is probe.center_frequency and scan.center_frequency. Let's rename to probe.probe_center_frequency. We need to add a test for this so people won't add parameters in the future that are conflicting between probe and scan groups. After intiialization you can update Parameters (with for instance Config parameters) these overwrite the ones currently in the Parameters class.
- even though probe is for all tracks, every track need to have a duplicate of probe as well so that track.parameters is the same as file.parameters (each track will have a different scan but same probe, but also thus a different Parameters class as well).
- after all relevant docs and tests should be updated, as well as example scripts.
- we probably want to derive the Parameters keys from the ScanSpec and ProbeSpec, so that we have a single source of truth for the parameters that are expected to be in the file. Additionally we need to extend Parameter to hold custom parameters (and generally ignore them, just as a container).

Instead of:

```python
with zea.File(file_path) as f:
    scan = f.scan
    probe = f.probe

```

we do

```python
with zea.File(file_path) as f:
    parameters = f.load_parameters()
```

Note that this is a design document, there might be coding details or bugs (i didn't run the code), it is mainly for drafting the new design and workflow.
