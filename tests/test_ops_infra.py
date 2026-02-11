"""Tests for the Operation and Pipeline classes in ops.py"""

import inspect
import json

import keras
import numpy as np
import pytest

from zea import func, ops
from zea.beamform.delays import compute_t0_delays_planewave
from zea.config import Config
from zea.internal.core import DEFAULT_DYNAMIC_RANGE, DataTypes
from zea.internal.registry import ops_registry
from zea.ops.pipeline import pipeline_from_config, pipeline_from_json, pipeline_from_yaml
from zea.probes import Probe
from zea.scan import Scan

from . import DEFAULT_TEST_SEED

"""Some operations for testing"""


@ops_registry("multiply")
class MultiplyOperation(ops.Operation):
    """Multiply Operation for testing purposes."""

    def __init__(self, useless_parameter: int = None, **kwargs):
        super().__init__(**kwargs)
        self.useless_parameter = useless_parameter

    def call(self, x, y):
        """
        Multiplies the input x by the specified factor.
        """

        return {"x": keras.ops.multiply(x, y)}


@ops_registry("add")
class AddOperation(ops.Operation):
    """Add Operation for testing purposes."""

    def call(self, x, y):
        """
        Adds the result from MultiplyOperation with y.
        """
        # print(f"Processing AddOperation: result={result}, y={y}")
        return {"z": keras.ops.add(x, y)}


@ops_registry("add_transmits")
class AddTransmitsOperation(ops.Operation):
    """Add Transmits Operation for testing purposes."""

    def call(self, x, n_tx):
        return {"z": keras.ops.add(x, n_tx)}


@ops_registry("large_matrix_multiplication")
class LargeMatrixMultiplicationOperation(ops.Operation):
    """Large Matrix Multiplication Operation for testing purposes."""

    def call(self, matrix_a, matrix_b):
        """
        Performs large matrix multiplication using Keras ops.
        """
        # print("Processing LargeMatrixMultiplicationOperation...")
        # Perform matrix multiplication
        result = keras.ops.matmul(matrix_a, matrix_b)
        result2 = keras.ops.matmul(result, matrix_a)
        result3 = keras.ops.matmul(result2, matrix_b)
        return {"matrix_result": result3}


@ops_registry("elementwise_matrix_operation")
class ElementwiseMatrixOperation(ops.Operation):
    """Elementwise Matrix Operation for testing purposes."""

    def call(self, matrix, scalar):
        """
        Performs elementwise operations on a matrix (adds and multiplies by scalar).
        """
        # print("Processing ElementwiseMatrixOperation...")
        # Perform elementwise addition and multiplication
        result = keras.ops.add(matrix, scalar)
        result = keras.ops.multiply(result, scalar)
        return {"elementwise_result": result}


@pytest.fixture
def test_operation():
    """Returns a MultiplyOperation instance."""
    return AddOperation(cache_inputs=True, cache_outputs=True, jit_compile=False)


@pytest.fixture
def pipeline_config():
    """Returns a test pipeline configuration."""
    return {
        "operations": [
            {"name": "multiply", "params": {}},
            {"name": "add", "params": {}},
        ]
    }


@pytest.fixture
def pipeline_config_with_params():
    """Returns a test pipeline configuration with parameters."""
    return {
        "operations": [
            {"name": "multiply", "params": {"useless_parameter": 10}},
            {"name": "add"},
        ]
    }


@pytest.fixture
def default_pipeline_config():
    """Config for default pipeline"""
    return {
        "operations": [
            {"name": "simulate_rf"},
            {"name": "demodulate"},
            {"name": "tof_correction"},
            {"name": "pfield_weighting"},
            {"name": "delay_and_sum"},
            {"name": "reshape_grid"},
            {"name": "envelope_detect"},
            {"name": "normalize"},
            {"name": "log_compress"},
        ]
    }


@pytest.fixture
def patched_pipeline_config():
    """Config for patch-wise default pipeline"""
    return {
        "operations": [
            {"name": "simulate_rf"},
            {"name": "demodulate"},
            {
                "name": "beamform",
                "params": {
                    "beamformer": "delay_and_sum",
                    "num_patches": 15,
                    "enable_pfield": True,
                },
            },
            {"name": "envelope_detect"},
            {"name": "normalize"},
            {"name": "log_compress"},
        ]
    }


@pytest.fixture
def branched_pipeline_config():
    """Returns a configuration for a BranchedPipeline."""
    return {
        "operations": [
            {
                "name": "branched_pipeline",
                "params": {"merge_strategy": "flatten"},
                "branches": {
                    "branch_1": [
                        {"name": "simulate_rf"},
                        {"name": "demodulate"},
                        {"name": "tof_correction"},
                        {"name": "pfield_weighting"},
                        {"name": "delay_and_sum"},
                    ],
                    "branch_2": [
                        {"name": "simulate_rf"},
                        {"name": "demodulate"},
                        {"name": "tof_correction"},
                        {"name": "pfield_weighting"},
                        {"name": "delay_and_sum"},
                    ],
                },
            },
            {"name": "reshape_grid"},
            {"name": "envelope_detect"},
            {"name": "normalize"},
            {"name": "log_compress"},
        ]
    }


def validate_branched_pipeline(pipeline):
    """Validates the branched pipeline."""
    assert len(pipeline.operations) == 5
    assert hasattr(pipeline.operations[0], "branches")
    assert isinstance(pipeline.operations[1], ops.ReshapeGrid)
    assert isinstance(pipeline.operations[2], ops.EnvelopeDetect)
    assert isinstance(pipeline.operations[3], ops.Normalize)
    assert isinstance(pipeline.operations[4], ops.LogCompress)

    branch_1 = pipeline.operations[0].branches["branch_1"]
    branch_2 = pipeline.operations[0].branches["branch_2"]

    for branch in [branch_1, branch_2]:
        assert isinstance(branch[0], ops.Simulate)
        assert isinstance(branch[1], ops.Demodulate)
        assert isinstance(branch[2], ops.TOFCorrection)
        assert isinstance(branch[3], ops.PfieldWeighting)
        assert isinstance(branch[4], ops.DelayAndSum)


@pytest.fixture
def default_pipeline():
    """Returns a default pipeline for ultrasound simulation."""
    pipeline = ops.Pipeline.from_default(num_patches=1, jit_options=None)
    pipeline.prepend(ops.Simulate())
    pipeline.append(ops.Normalize(input_range=DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))
    return pipeline


@pytest.fixture
def patched_pipeline():
    """Returns a pipeline for ultrasound simulation where the beamforming happens patch-wise."""
    pipeline = ops.Pipeline.from_default(jit_options=None)
    pipeline.prepend(ops.Simulate())
    pipeline.append(ops.Normalize(input_range=DEFAULT_DYNAMIC_RANGE, output_range=(0, 255)))
    return pipeline


def test_pipeline_modification():
    """Tests if modifying the pipeline updates callable layers correctly."""
    # set timed to True to ensure _callable_layers is used
    # basically this makes sure that the pipeline is reinitialized
    pipeline = ops.Pipeline.from_default(jit_options=None, with_batch_dim=False, timed=True)
    pipeline.prepend(ops.Simulate())
    assert len(pipeline._callable_layers) == len(pipeline.operations)
    pipeline.append(ops.Normalize())
    assert len(pipeline._callable_layers) == len(pipeline.operations)
    pipeline.insert(2, ops.Identity())
    assert len(pipeline._callable_layers) == len(pipeline.operations)


def test_operation_initialization(test_operation):
    """Tests initialization of an Operation."""
    assert test_operation.cache_inputs is True
    assert test_operation.cache_outputs is True
    assert test_operation._jit_compile is False
    assert test_operation._input_cache == {}
    assert test_operation._output_cache == {}


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_validation(test_operation, jit_compile):
    """Tests input validation and handling of unexpected keys."""
    test_operation.set_jit(jit_compile)
    outputs = test_operation(x=5, y=3, other=10)
    assert outputs["other"] == 10
    assert outputs["z"] == 8


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_output_caching(test_operation, jit_compile):
    """Tests output caching behavior."""
    test_operation.set_jit(jit_compile)
    output1 = test_operation(x=5, y=3)
    output2 = test_operation(x=5, y=3)
    assert output1 == output2
    output3 = test_operation(x=5, y=4)
    assert output1 != output3


@pytest.mark.parametrize("jit_compile", [True, False])
def test_operation_input_caching(test_operation, jit_compile):
    """Tests input caching behavior."""
    test_operation.set_jit(jit_compile)
    test_operation.set_input_cache(input_cache={"x": 10})
    result = test_operation(y=5)
    assert result["z"] == 15


def test_operation_jit_compilation():
    """Ensures JIT compilation works."""
    op = AddOperation(jit_compile=True)
    assert callable(op.call)


def test_operation_cache_persistence():
    """Tests persistence of output cache."""
    op = AddOperation(cache_outputs=True)
    result1 = op(x=5, y=3)
    assert result1["z"] == 8
    assert len(op._output_cache) == 1
    result2 = op(x=5, y=3)
    assert result2 == result1
    assert len(op._output_cache) == 1


def test_string_representation(verbose=False):
    """Print the string representation of the Pipeline"""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)
    if verbose:
        print(str(pipeline))
    assert str(pipeline) == "MultiplyOperation -> AddOperation"


"""Pipeline Class Tests"""


def test_pipeline_initialization():
    """Tests initialization of a Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)
    assert len(pipeline.operations) == 2
    assert isinstance(pipeline.operations[0], MultiplyOperation)
    assert isinstance(pipeline.operations[1], AddOperation)


def test_pipeline_call():
    """Tests the call method of the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)
    result = pipeline(x=2, y=3)
    assert result["z"] == 9  # (2 * 3) + 3


def test_pipeline_with_large_matrix_multiplication():
    """Tests the Pipeline with a large matrix multiplication operation."""
    operations = [LargeMatrixMultiplicationOperation()]
    pipeline = ops.Pipeline(operations=operations)
    matrix_a = keras.random.normal(shape=(512, 512))
    matrix_b = keras.random.normal(shape=(512, 512))
    result = pipeline(matrix_a=matrix_a, matrix_b=matrix_b)
    assert result["matrix_result"].shape == (512, 512)


def test_pipeline_with_elementwise_operation():
    """Tests the Pipeline with an elementwise matrix operation."""
    operations = [ElementwiseMatrixOperation()]
    pipeline = ops.Pipeline(operations=operations)
    matrix = keras.random.normal(shape=(512, 512))
    scalar = 2
    result = pipeline(matrix=matrix, scalar=scalar)
    assert result["elementwise_result"].shape == (512, 512)


def test_pipeline_jit_options():
    """Tests the JIT options for the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations, jit_options="pipeline")
    assert callable(pipeline.call)

    pipeline = ops.Pipeline(operations=operations, jit_options="ops")
    for operation in pipeline.operations:
        assert operation._jit_compile is True

    pipeline = ops.Pipeline(operations=operations, jit_options=None)
    for operation in pipeline.operations:
        assert operation._jit_compile is False


def test_pipeline_set_params():
    """Tests setting parameters for the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)
    pipeline.set_params(x=5, y=3)
    params = pipeline.get_params()
    assert params["x"] == 5
    assert params["y"] == 3


def test_pipeline_get_params_per_operation():
    """Tests getting parameters per operation in the Pipeline."""
    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)
    pipeline.set_params(x=5, y=3)
    params = pipeline.get_params(per_operation=True)
    assert params[0]["x"] == 5
    assert params[1]["y"] == 3


def test_pipeline_validation():
    """Tests the validation of the Pipeline."""
    operations = [
        MultiplyOperation(output_data_type=DataTypes.RAW_DATA),
        AddOperation(input_data_type=DataTypes.RAW_DATA),
    ]
    _ = ops.Pipeline(operations=operations)

    operations = [
        MultiplyOperation(output_data_type=DataTypes.RAW_DATA),
        AddOperation(input_data_type=DataTypes.IMAGE),
    ]
    with pytest.raises(ValueError):
        _ = ops.Pipeline(operations=operations)


def test_pipeline_with_scan_probe_config():
    """Tests the Pipeline with Scan, Probe, and Config objects as inputs."""

    probe = Probe.from_name("generic")
    scan = Scan(
        n_tx=128,
        n_ax=256,
        n_el=128,
        n_ch=2,
        center_frequency=5.0,
        sampling_frequency=5.0,
        xlims=(-2e-3, 2e-3),
    )

    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)

    parameters = pipeline.prepare_parameters(probe, scan)
    result = pipeline(**parameters, x=2, y=3)

    assert "z" in result
    assert "probe_geometry" in result  # Check if we parsed the probe object correctly
    assert "n_tx" not in result  # n_tx is not needed in the pipeline

    # Now let's use n_tx, such that it has to be in the pipeline
    pipeline.append(AddTransmitsOperation())
    parameters = pipeline.prepare_parameters(probe, scan)
    result = pipeline(**parameters, x=2, y=3)

    assert "z" in result
    assert "probe_geometry" in result  # Check if we parsed the probe object correctly
    assert "n_tx" in result  # now we actually need to have n_tx in the result


"""Pipeline build from config / json tests"""


def validate_basic_pipeline(pipeline, with_params=False):
    """Validates a basic pipeline."""
    assert len(pipeline.operations) == 2
    assert isinstance(pipeline.operations[0], MultiplyOperation)
    assert isinstance(pipeline.operations[1], AddOperation)
    if with_params:
        assert pipeline.operations[0].useless_parameter == 10

    result = pipeline(x=2, y=3)
    assert result["z"] == 9  # (2 * 3) + 3


def validate_default_pipeline(pipeline, patched=False):
    """Validates the default pipeline."""
    assert isinstance(pipeline.operations[0], ops.Simulate)
    assert isinstance(pipeline.operations[1], ops.Demodulate)

    if not patched:
        assert isinstance(pipeline.operations[2], ops.TOFCorrection)
        assert isinstance(pipeline.operations[3], ops.PfieldWeighting)
        assert isinstance(pipeline.operations[4], ops.DelayAndSum)
        assert isinstance(pipeline.operations[5], ops.ReshapeGrid)
        assert isinstance(pipeline.operations[6], ops.EnvelopeDetect)
        assert isinstance(pipeline.operations[7], ops.Normalize)
        assert isinstance(pipeline.operations[8], ops.LogCompress)
    else:
        beamform = pipeline.operations[2]
        assert hasattr(beamform, "operations")
        assert isinstance(beamform.operations[0].operations[0], ops.TOFCorrection)
        assert isinstance(beamform.operations[0].operations[1], ops.PfieldWeighting)
        assert isinstance(beamform.operations[0].operations[2], ops.DelayAndSum)
        assert isinstance(pipeline.operations[3], ops.EnvelopeDetect)
        assert isinstance(pipeline.operations[4], ops.Normalize)
        assert isinstance(pipeline.operations[5], ops.LogCompress)


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_default_pipeline_from_json(config_fixture, request):
    """Tests building a default pipeline from a JSON string."""
    config = request.getfixturevalue(config_fixture)
    json_string = json.dumps(config)
    pipeline = pipeline_from_json(json_string, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(pipeline)
    else:
        validate_default_pipeline(pipeline, patched=config_fixture == "patched_pipeline_config")


@pytest.mark.parametrize("config_fixture", ["pipeline_config", "pipeline_config_with_params"])
def test_pipeline_from_config(config_fixture, request):
    """Tests building a dummy pipeline from a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    validate_basic_pipeline(pipeline, with_params=config_fixture == "pipeline_config_with_params")


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_default_pipeline_from_config(config_fixture, request):
    """Tests building a default pipeline from a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(pipeline)
    else:
        validate_default_pipeline(pipeline, patched=config_fixture == "patched_pipeline_config")


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_config(config_fixture, request):
    """Tests converting a pipeline to a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    # Convert the pipeline back to a Config object
    new_config = pipeline.to_config()

    # Create a new pipeline from the new Config object
    new_pipeline = pipeline_from_config(new_config, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(new_pipeline, patched=config_fixture == "patched_pipeline_config")


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_json(config_fixture, request):
    """Tests converting a pipeline to a JSON string."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    # Convert the pipeline to a JSON string
    json_string = pipeline.to_json()

    # Create a new pipeline from the JSON string
    new_pipeline = pipeline_from_json(json_string, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(new_pipeline, patched=config_fixture == "patched_pipeline_config")


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_yaml(config_fixture, request, tmp_path):
    """Tests converting a pipeline to a YAML file (in tmp directory), and then loading it back."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = pipeline_from_config(config, jit_options=None)

    # Write pipeline to a YAML file in the temporary directory
    path = tmp_path / "tmp_pipeline.yaml"
    pipeline.to_yaml(path)

    # Load the pipeline from the YAML file
    new_pipeline = pipeline_from_yaml(path, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(new_pipeline, patched=config_fixture == "patched_pipeline_config")


def get_probe():
    """Returns a probe for ultrasound simulation tests."""
    n_el = 128
    aperture = 30e-3
    probe_geometry = np.stack(
        [
            np.linspace(-aperture / 2, aperture / 2, n_el),
            np.zeros(n_el),
            np.zeros(n_el),
        ],
        axis=1,
    )

    return Probe(
        probe_geometry=probe_geometry,
        center_frequency=3.125e6,
        sampling_frequency=12.5e6,
    )


@pytest.fixture
def ultrasound_probe():
    """Returns a probe for ultrasound simulation tests."""
    return get_probe()


def get_scan(ultrasound_probe, grid_size_x=None, grid_size_z=None):
    """Returns a scan for ultrasound simulation tests.

    Note these parameters are not really realistic, but are used for testing purposes.
    """
    n_el = ultrasound_probe.n_el
    n_tx = 2
    n_ax = 100

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = ultrasound_probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180
    sound_speed = 1540.0
    focus_distances = np.ones(n_tx) * np.inf
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry, polar_angles=angles, sound_speed=sound_speed
    )

    return Scan(
        grid_size_x=grid_size_x,
        grid_size_z=grid_size_z,
        n_tx=n_tx,
        n_ax=n_ax,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency / 100,
        sampling_frequency=ultrasound_probe.sampling_frequency / 100,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        apply_lens_correction=False,
        sound_speed=sound_speed,
        lens_sound_speed=1000.0,
        lens_thickness=1e-3,
        initial_times=np.ones((n_tx,)) * 1e-6,
        attenuation_coef=0.2,
        n_ch=1,
        selected_transmits="all",
        focus_distances=focus_distances,
        polar_angles=angles,
        xlims=(-15e-3, 15e-3),
        zlims=(0, 35e-3),
    )


@pytest.fixture
def ultrasound_scan(ultrasound_probe):
    """Returns a scan for ultrasound simulation tests."""
    return get_scan(ultrasound_probe, grid_size_x=20, grid_size_z=20)


def get_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests.
    Has a batch dimension of 1."""
    scat_x, scat_z = np.meshgrid(
        np.linspace(-10e-3, 10e-3, 5),
        np.linspace(10e-3, 30e-3, 5),
        indexing="ij",
    )
    scat_x, scat_z = np.ravel(scat_x), np.ravel(scat_z)
    # scat_x, scat_z = np.array([-10e-3, 0e-3]), np.array([10e-3, 20e-3])
    n_scat = len(scat_x)
    scat_positions = np.stack(
        [
            scat_x,
            np.zeros_like(scat_x),
            scat_z,
        ],
        axis=1,
    )
    scat_positions = np.expand_dims(scat_positions, axis=0)  # add batch dimension

    return {
        "positions": scat_positions.astype(np.float32),
        "magnitudes": np.ones((1, n_scat), dtype=np.float32),
        "n_scat": n_scat,
    }


@pytest.fixture
def ultrasound_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests."""
    return get_scatterers()


@pytest.mark.parametrize(
    "with_batch_dim",
    [False, True],
)
def test_simulator(ultrasound_probe, ultrasound_scan, ultrasound_scatterers, with_batch_dim):
    """Tests the simulator operation."""
    pipeline = ops.Pipeline([ops.Simulate()], with_batch_dim=with_batch_dim)
    parameters = pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)

    if not with_batch_dim:
        # remove batch_dim of scatterers for pipeline without batch dimension
        ultrasound_scatterers["positions"] = ultrasound_scatterers["positions"][0]
        ultrasound_scatterers["magnitudes"] = ultrasound_scatterers["magnitudes"][0]

    output = pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )
    # assert output shape with batch dimension if with_batch_dim else without
    expected_shape = (ultrasound_scan.n_tx, ultrasound_scan.n_ax, ultrasound_scan.n_el, 1)
    expected_shape = (1,) + expected_shape if with_batch_dim else expected_shape
    assert output["data"].shape == expected_shape


@pytest.mark.heavy
def test_default_ultrasound_pipeline(
    default_pipeline,
    patched_pipeline,
    ultrasound_probe,
    ultrasound_scan,
    ultrasound_scatterers,
):
    """Tests the default ultrasound pipeline."""
    # all dynamic parameters are set in the call method of the operations
    # or equivalently in the pipeline call (which is passed to the operations)
    parameters = default_pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)
    output_default = default_pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )

    parameters = patched_pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)

    output_patched = patched_pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )

    for output in [output_default, output_patched]:
        # Check that the pipeline produced the expected outputs
        assert "data" in output
        assert output["data"].shape[0] == 1  # Batch dimension
        # Verify the normalized image has values between 0 and 255
        assert np.nanmin(output["data"]) >= 0.0
        assert np.nanmax(output["data"]) <= 255.0

    np.testing.assert_allclose(
        output_default["data"] / np.max(output_default["data"]),
        output_patched["data"] / np.max(output_patched["data"]),
        rtol=1e-3,
        atol=1e-3,
    )


def test_pipeline_parameter_tracing(ultrasound_scan: Scan):
    """Tests that the pipeline can run without parameters that are not needed as input because they
    are computed inside the pipeline."""

    pipeline = ops.Pipeline([ops.Demodulate(), ops.TOFCorrection()])
    ultrasound_scan._params.pop("n_ch", None)  # remove a parameter that is not needed
    ultrasound_scan._params.pop("demodulation_frequency", None)
    params = pipeline.prepare_parameters(scan=ultrasound_scan)
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    data = rng.standard_normal(
        (1, ultrasound_scan.n_tx, ultrasound_scan.n_ax, ultrasound_scan.n_el, 1)
    )
    output = pipeline(data=data, **params)
    assert "demodulation_frequency" in output


def test_ops_pass_positional_arg():
    """Test that passing positional arguments to Operation raises a custom error."""
    op = AddOperation()
    with pytest.raises(TypeError) as excinfo:
        op(1, 2)
    assert "Positional arguments are not allowed." in str(excinfo.value)
    op = ops.Lambda(lambda x: x + 1)
    with pytest.raises(TypeError) as excinfo:
        op(1)
    assert "Positional arguments are not allowed." in str(excinfo.value)


def test_registry():
    """Test that all Operations are registered in ops_registry."""

    classes = inspect.getmembers(ops, inspect.isclass)
    for _, _class in classes:
        if _class.__module__.startswith("zea.ops."):
            # Skip abstract base classes and base Operation classes
            if inspect.isabstract(_class) or _class.__name__ in [
                "Operation",
                "MissingKerasOps",
            ]:
                continue
            ops_registry.get_name(_class)  # this raises an error if the class is not registered


def _get_defined_names_from_submodules(parent_module, submodule_names, exclude_private=True):
    """Get all function and class names defined in specific submodules.

    This inspects the actual submodule files, not what's imported into the parent,
    so it will catch items that should be exported but aren't imported yet.

    Args:
        parent_module: The parent module object
        submodule_names: List of submodule names to inspect (e.g., ['tensor', 'ultrasound'])
        exclude_private: Whether to exclude names starting with underscore

    Returns:
        set: Set of names defined in the submodules
    """
    import importlib

    defined_names = set()
    parent_name = parent_module.__name__

    for submodule_name in submodule_names:
        # Import the submodule directly
        full_module_name = f"{parent_name}.{submodule_name}"
        submodule = importlib.import_module(full_module_name)

        # Get all members defined in this specific submodule
        for name, obj in inspect.getmembers(submodule):
            # Skip private names if requested
            if exclude_private and name.startswith("_"):
                continue

            # Check if it's a function or class
            if inspect.isfunction(obj) or inspect.isclass(obj):
                # Only include if it's defined in this specific submodule
                if hasattr(obj, "__module__") and obj.__module__ == full_module_name:
                    defined_names.add(name)

    return defined_names


def _check_exports(module, module_name, defined_names, exported_names, file_path):
    """Check that all defined names are both importable and exported in __all__.

    Args:
        module: The module object to check imports from
        module_name: Name of the module for error messages
        defined_names: Set of names that should be exported
        exported_names: Set of names in __all__
        file_path: Path to the __init__.py file for error messages
    """
    # Check if items are in __all__
    missing_in_all = defined_names - exported_names

    # Check if items are actually importable from the module
    missing_imports = []
    for name in defined_names:
        if not hasattr(module, name):
            missing_imports.append(name)

    # Report errors
    errors = []
    if missing_in_all:
        errors.append(f"Not in __all__: {sorted(missing_in_all)}")
    if missing_imports:
        errors.append(f"Not imported: {sorted(missing_imports)}")

    if errors:
        error_msg = (
            f"The following items are not properly exported from {module_name}:\n"
            + "\n".join(f"  - {err}" for err in errors)
            + f"\nPlease add them to both the imports and __all__ list in {file_path}"
        )
        pytest.fail(error_msg)


def test_all_operations_exported():
    """Test that all registered Operation classes are exported in zea.ops.__all__."""
    # Get all registered operation classes from the registry
    registered_ops = set()
    for name in ops_registry.registry:
        op_class = ops_registry[name]
        # Only check Operation subclasses that are defined in zea.ops
        # Skip keras_ops (they're exported via the keras_ops module)
        if (
            inspect.isclass(op_class)
            and issubclass(op_class, ops.Operation)
            and op_class.__module__.startswith("zea.ops.")
            and op_class.__module__ != "zea.ops.keras_ops"
            and op_class.__name__ not in ["Operation", "ImageOperation", "MissingKerasOps"]
        ):
            registered_ops.add(op_class.__name__)

    # Check that all registered operations are both imported and in __all__
    _check_exports(
        module=ops,
        module_name="zea.ops",
        defined_names=registered_ops,
        exported_names=set(ops.__all__),
        file_path="zea/ops/__init__.py",
    )


def test_all_functions_exported():
    """Test that all functions defined in zea.func submodules are exported in zea.func.__all__."""
    # Get all functions defined in the actual submodule files (tensor.py, ultrasound.py)
    # This will catch functions that should be exported but aren't imported yet
    defined_funcs = _get_defined_names_from_submodules(
        parent_module=func, submodule_names=["tensor", "ultrasound"], exclude_private=True
    )

    # Check that all defined functions are both imported and in __all__
    _check_exports(
        module=func,
        module_name="zea.func",
        defined_names=defined_funcs,
        exported_names=set(func.__all__),
        file_path="zea/func/__init__.py",
    )
