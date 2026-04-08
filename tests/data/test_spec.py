from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from zea.data import spec as spec_module
from zea.data.file import File
from zea.data.spec import FileSpec, Scan, Segmentation, Spec


def test_segmentation_spec():
    # Correct usage
    pixels = np.zeros((10, 256, 256, 1), dtype=np.uint8)
    labels = np.array(["background", "label1", "label2", "label3"], dtype=np.str_)
    extent = np.array([0.0, 1.0, 0.0, 1.0, -1.0, 0.0], dtype=np.float32)
    segmentation = Segmentation(pixels=pixels, labels=labels, extent=extent)
    assert segmentation.pixels.shape == (10, 256, 256, 1)
    assert segmentation.labels.shape == (4,)
    assert segmentation.extent.shape == (6,)

    # Incorrect usage: pixel values do not correspond to labels
    pixels_invalid = np.array([[[[0], [1]], [[2], [3]]], [[[4], [5]], [[6], [7]]]], dtype=np.uint8)
    with pytest.raises(
        ValueError, match="Segmentation pixels contain values that do not correspond to any label"
    ):
        Segmentation(pixels=pixels_invalid, labels=labels, extent=extent)


def _scan_minimal(n_frames: int = 3, n_tx: int = 2, n_el: int = 4):
    return {
        "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
        "sampling_frequency": np.float32(30e6),
        "center_frequency": np.float32(5e6),
        "demodulation_frequency": np.float32(5e6),
        "initial_times": np.zeros((n_tx,), dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.zeros((n_tx,), dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros((n_tx,), dtype=np.float32),
        "azimuth_angles": np.zeros((n_tx,), dtype=np.float32),
        "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32),
    }


@pytest.fixture
def dataset_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    return FileSpec(
        data={
            "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
            "image": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "segmentation": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "labels": np.array(["background", "tissue"], dtype=np.str_),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "sos_map": {
                "pixels": np.full((n_frames, 16, 12, 1), 1540.0, dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "strain": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "swe": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
            "tissue_doppler": {
                "pixels": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
                "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
            },
        },
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        metadata={
            "subject": {
                "type": "human",
                "age": np.uint8(42),
                "sex": "f",
                "fat_percentage": np.float32(17.5),
            },
            "credit": "example-lab",
            "probe_orientation": {
                "pose": np.zeros((25, 6), dtype=np.float32),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(50.0),
            },
            "voice_narration": {
                "samples": np.zeros((100, 1), dtype=np.uint8),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(8000.0),
            },
            "ecg": {
                "samples": np.zeros((100, 1), dtype=np.uint8),
                "offset": np.float32(0.0),
                "sampling_frequency": np.float32(250.0),
            },
            "text_report": "normal acquisition",
            "annotations": {
                "anatomy": "heart",
                "view": np.array(["plax", "plax", "psax"], dtype=np.str_),
                "label": np.array(["normal", "normal", "normal"], dtype=np.str_),
                "image_quality": "high",
            },
        },
        metrics={
            "common_midpoint_phase_error": np.zeros((n_frames,), dtype=np.float32),
            "coherence_factor": np.ones((n_frames,), dtype=np.float32),
        },
    )


def test_dataset_spec(dataset_spec):
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    assert dataset_spec.data.raw_data.shape == (n_frames, n_tx, n_ax, n_el, n_ch)
    assert dataset_spec.scan.t0_delays.shape == (n_tx, n_el)
    assert dataset_spec.metadata.annotations.view.shape == (n_frames,)
    assert dataset_spec.metrics.coherence_factor.shape == (n_frames,)


def test_saving_and_loading(tmp_path, dataset_spec: FileSpec):
    # Save the dataset
    save_path = tmp_path / "test_dataset.hdf5"
    dataset_spec.save(save_path)

    with File(save_path) as loaded_dataset:
        # Check that the loaded data matches the original
        assert np.array_equal(loaded_dataset["data"]["raw_data"], dataset_spec.data.raw_data)
        assert np.array_equal(loaded_dataset["scan"]["t0_delays"], dataset_spec.scan.t0_delays)
        assert np.array_equal(
            loaded_dataset["metadata"]["annotations"]["view"].asstr()[()],
            dataset_spec.metadata.annotations.view,
        )
        assert np.array_equal(
            loaded_dataset["metrics"]["coherence_factor"], dataset_spec.metrics.coherence_factor
        )


def test_scan_requires_required_fields():
    scan = _scan_minimal()
    scan.pop("demodulation_frequency")

    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'demodulation_frequency'"
    ):
        Scan(**scan)


def test_scan_dimension_count_consistency():
    scan = _scan_minimal(n_tx=2)
    scan["initial_times"] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension 'n_tx' has inconsistent sizes"):
        Scan(**scan)


def test_optional_fields_can_be_omitted():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        metadata={},
        metrics={},
    )

    assert dataset.metadata.subject is None
    assert dataset.metrics.common_midpoint_phase_error is None


def test_scan_accepts_float_inputs_and_casts_to_float32():
    scan = _scan_minimal()
    scan["sampling_frequency"] = np.float64(30e6)
    scan["center_frequency"] = np.array([5e6, 6e6], dtype=np.float64)
    scan["demodulation_frequency"] = np.float64(5e6)
    scan["initial_times"] = np.zeros((2,), dtype=np.float64)
    scan["t0_delays"] = np.zeros((2, 4), dtype=np.float64)

    scan_spec = Scan(**scan)

    assert np.dtype(scan_spec.sampling_frequency.dtype) == np.dtype(
        Scan.SCHEMA["sampling_frequency"]["dtype"]
    )
    assert scan_spec.center_frequency.dtype == np.dtype(Scan.SCHEMA["center_frequency"]["dtype"])
    assert np.dtype(scan_spec.demodulation_frequency.dtype) == np.dtype(
        Scan.SCHEMA["demodulation_frequency"]["dtype"]
    )
    assert scan_spec.initial_times.dtype == np.dtype(Scan.SCHEMA["initial_times"]["dtype"])
    assert scan_spec.t0_delays.dtype == np.dtype(Scan.SCHEMA["t0_delays"]["dtype"])


def test_dataset_builder_accepts_float_raw_data_and_casts_to_float32():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float64)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        metadata={},
        metrics={},
    )

    assert dataset.data.raw_data.dtype == np.float32


def test_dataset_builder_dimension_consistency_across_nested_specs():
    n_frames_data, n_frames_scan = 3, 4
    n_tx, n_el, n_ax, n_ch = 2, 4, 8, 1

    scan = {
        "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
        "sampling_frequency": np.float32(30e6),
        "center_frequency": np.float32(5e6),
        "demodulation_frequency": np.float32(5e6),
        "initial_times": np.zeros((n_tx,), dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.zeros((n_tx,), dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros((n_tx,), dtype=np.float32),
        "azimuth_angles": np.zeros((n_tx,), dtype=np.float32),
        "time_to_next_transmit": np.ones((n_frames_scan, n_tx), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="Dimension 'n_frames' has inconsistent sizes"):
        FileSpec(
            data={"raw_data": np.zeros((n_frames_data, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=scan,
            metadata={},
            metrics={},
        )


def test_metadata_dict_with_unknown_field():
    """Test that passing a metadata dict with a custom/unknown field raises TypeError."""
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            metadata={"custom_field": "this field does not exist in Metadata spec"},
            metrics={},
        )


def test_schema_keys_match_dataclass_fields_for_all_specs():
    """Test that all Spec subclasses have SCHEMA keys that exactly match their dataclass fields."""
    spec_classes = []
    for obj in vars(spec_module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Spec)
            and obj is not Spec
            and is_dataclass(obj)
        ):
            spec_classes.append(obj)

    assert spec_classes, "No dataclass Spec subclasses found in zea.data.spec"

    for cls in spec_classes:
        dataclass_field_names = {field.name for field in fields(cls)}
        schema_field_names = set(cls.SCHEMA.keys())

        missing_in_schema = dataclass_field_names - schema_field_names
        extra_in_schema = schema_field_names - dataclass_field_names

        assert not missing_in_schema and not extra_in_schema, (
            f"{cls.__name__} SCHEMA mismatch. "
            f"Missing in SCHEMA: {sorted(missing_in_schema)}; "
            f"Extra in SCHEMA: {sorted(extra_in_schema)}"
        )
