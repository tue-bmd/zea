"""Tests for the Operation and Pipeline classes in ops.py"""

# pylint: disable=arguments-differ, abstract-class-instantiated, pointless-string-statement

import json

import keras
import numpy as np
import pytest

from usbmd import ops
from usbmd.beamform.delays import compute_t0_delays_planewave
from usbmd.config.config import Config
from usbmd.internal.core import DataTypes
from usbmd.internal.registry import ops_registry
from usbmd.probes import Dummy, Probe
from usbmd.scan import Scan

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
            {"name": "tof_correction", "params": {"apply_phase_rotation": True}},
            {"name": "pfield_weighting"},
            {"name": "delay_and_sum"},
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
                "name": "patched_grid",
                "params": {"num_patches": 15},
                "operations": [
                    {
                        "name": "tof_correction",
                        "params": {"apply_phase_rotation": True},
                    },
                    {"name": "pfield_weighting"},
                    {"name": "delay_and_sum"},
                ],
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
                        {
                            "name": "tof_correction",
                            "params": {"apply_phase_rotation": True},
                        },
                        {"name": "pfield_weighting"},
                        {"name": "delay_and_sum"},
                    ],
                    "branch_2": [
                        {"name": "simulate_rf"},
                        {"name": "demodulate"},
                        {
                            "name": "tof_correction",
                            "params": {"apply_phase_rotation": False},
                        },
                        {"name": "pfield_weighting"},
                        {"name": "delay_and_sum"},
                    ],
                },
            },
            {"name": "envelope_detect"},
            {"name": "normalize"},
            {"name": "log_compress"},
        ]
    }


def validate_branched_pipeline(pipeline):
    """Validates the branched pipeline."""
    assert len(pipeline.operations) == 4
    assert hasattr(pipeline.operations[0], "branches")
    assert isinstance(pipeline.operations[1], ops.EnvelopeDetect)
    assert isinstance(pipeline.operations[2], ops.Normalize)
    assert isinstance(pipeline.operations[3], ops.LogCompress)

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
    pipeline.append(
        ops.Normalize(input_range=ops.DEFAULT_DYNAMIC_RANGE, output_range=(0, 255))
    )
    return pipeline


@pytest.fixture
def patched_pipeline():
    """Returns a pipeline for ultrasound simulation where the beamforming happens patch-wise."""
    pipeline = ops.Pipeline.from_default(jit_options=None)
    pipeline.prepend(ops.Simulate())
    pipeline.append(
        ops.Normalize(input_range=ops.DEFAULT_DYNAMIC_RANGE, output_range=(0, 255))
    )
    return pipeline


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

    probe = Dummy()
    scan = Scan(
        n_tx=128,
        n_ax=256,
        n_el=128,
        n_ch=2,
        center_frequency=5.0,
        sampling_frequency=5.0,
        xlims=(-2e-3, 2e-3),
    )

    # TODO: Add Config object as input to the Pipeline, currently config is not an Object

    operations = [MultiplyOperation(), AddOperation()]
    pipeline = ops.Pipeline(operations=operations)

    parameters = pipeline.prepare_parameters(probe, scan)
    result = pipeline(**parameters, x=2, y=3)

    assert "z" in result
    assert "n_tx" in result  # Check if we parsed the scan object correctly
    assert "probe_geometry" in result  # Check if we parsed the probe object correctly


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
        assert isinstance(pipeline.operations[5], ops.EnvelopeDetect)
        assert isinstance(pipeline.operations[6], ops.Normalize)
        assert isinstance(pipeline.operations[7], ops.LogCompress)
    else:
        patched_grid = pipeline.operations[2]
        assert hasattr(patched_grid, "operations")
        assert isinstance(patched_grid.operations[0], ops.TOFCorrection)
        assert isinstance(patched_grid.operations[1], ops.PfieldWeighting)
        assert isinstance(patched_grid.operations[2], ops.DelayAndSum)

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
    pipeline = ops.pipeline_from_json(json_string, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(pipeline)
    else:
        validate_default_pipeline(
            pipeline, patched=config_fixture == "patched_pipeline_config"
        )


@pytest.mark.parametrize(
    "config_fixture", ["pipeline_config", "pipeline_config_with_params"]
)
def test_pipeline_from_config(config_fixture, request):
    """Tests building a dummy pipeline from a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = ops.pipeline_from_config(config, jit_options=None)

    validate_basic_pipeline(
        pipeline, with_params=config_fixture == "pipeline_config_with_params"
    )


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_default_pipeline_from_config(config_fixture, request):
    """Tests building a default pipeline from a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = ops.pipeline_from_config(config, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(pipeline)
    else:
        validate_default_pipeline(
            pipeline, patched=config_fixture == "patched_pipeline_config"
        )


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_config(config_fixture, request):
    """Tests converting a pipeline to a Config object."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = ops.pipeline_from_config(config, jit_options=None)

    # Convert the pipeline back to a Config object
    new_config = pipeline.to_config()

    # Create a new pipeline from the new Config object
    new_pipeline = ops.pipeline_from_config(new_config, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(
            new_pipeline, patched=config_fixture == "patched_pipeline_config"
        )


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_json(config_fixture, request):
    """Tests converting a pipeline to a JSON string."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = ops.pipeline_from_config(config, jit_options=None)

    # Convert the pipeline to a JSON string
    json_string = pipeline.to_json()

    # Create a new pipeline from the JSON string
    new_pipeline = ops.pipeline_from_json(json_string, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(
            new_pipeline, patched=config_fixture == "patched_pipeline_config"
        )


@pytest.mark.parametrize(
    "config_fixture",
    ["default_pipeline_config", "patched_pipeline_config", "branched_pipeline_config"],
)
def test_pipeline_to_yaml(config_fixture, request, tmp_path):
    """Tests converting a pipeline to a YAML file (in tmp directory), and then loading it back."""
    config_dict = request.getfixturevalue(config_fixture)
    config = Config(**config_dict)
    pipeline = ops.pipeline_from_config(config, jit_options=None)

    # Write pipeline to a YAML file in the temporary directory
    path = tmp_path / "tmp_pipeline.yaml"
    pipeline.to_yaml(path)

    # Load the pipeline from the YAML file
    new_pipeline = ops.pipeline_from_yaml(path, jit_options=None)

    if config_fixture == "branched_pipeline_config":
        validate_branched_pipeline(new_pipeline)
    else:
        validate_default_pipeline(
            new_pipeline, patched=config_fixture == "patched_pipeline_config"
        )


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


def get_scan(ultrasound_probe, Nx=None, Nz=None):
    """Returns a scan for ultrasound simulation tests."""
    n_el = ultrasound_probe.n_el
    n_tx = 2
    n_ax = 513

    tx_apodizations = np.ones((n_tx, n_el)) * np.hanning(n_el)[None]
    probe_geometry = ultrasound_probe.probe_geometry

    angles = np.linspace(10, -10, n_tx) * np.pi / 180
    sound_speed = 1540.0
    focus_distances = np.ones(n_tx) * np.inf
    t0_delays = compute_t0_delays_planewave(
        probe_geometry=probe_geometry, polar_angles=angles, sound_speed=sound_speed
    )

    return Scan(
        Nx=Nx,
        Nz=Nz,
        n_tx=n_tx,
        n_ax=n_ax,
        n_el=n_el,
        center_frequency=ultrasound_probe.center_frequency,
        sampling_frequency=ultrasound_probe.sampling_frequency,
        probe_geometry=probe_geometry,
        t0_delays=t0_delays,
        tx_apodizations=tx_apodizations,
        element_width=np.linalg.norm(probe_geometry[1] - probe_geometry[0]),
        apply_lens_correction=False,
        sound_speed=sound_speed,
        lens_sound_speed=1000,
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
    return get_scan(ultrasound_probe, Nx=100, Nz=100)


def get_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests."""
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

    return {
        "positions": scat_positions.astype(np.float32),
        "magnitudes": np.ones(n_scat, dtype=np.float32),
        "n_scat": n_scat,
    }


@pytest.fixture
def ultrasound_scatterers():
    """Returns scatterer positions and magnitudes for ultrasound simulation tests."""
    return get_scatterers()


def test_simulator(ultrasound_probe, ultrasound_scan, ultrasound_scatterers):
    """Tests the simulator operation."""
    pipeline = ops.Pipeline([ops.Simulate()])
    parameters = pipeline.prepare_parameters(ultrasound_probe, ultrasound_scan)

    output = pipeline(
        **parameters,
        scatterer_positions=ultrasound_scatterers["positions"],
        scatterer_magnitudes=ultrasound_scatterers["magnitudes"],
    )

    assert output["data"].shape == (
        1,
        ultrasound_scan.n_tx,
        ultrasound_scan.n_ax,
        ultrasound_scan.n_el,
        1,
    )


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
