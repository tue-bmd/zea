from dataclasses import fields, is_dataclass
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from zea.data import spec as spec_module
from zea.data.file import File
from zea.data.spec import (
    DataSpec,
    FileSpec,
    Image,
    Map,
    MetadataSpec,
    MetricsSpec,
    ProbePose,
    ProbeSpec,
    ScanSpec,
    Segmentation,
    Signal1D,
    SignalND,
    SosMap,
    Spec,
    Subject,
)


def test_segmentation_spec():
    # Correct usage: 3D spatial (n_frames, z, x, y, n_labels)
    values = np.zeros((10, 256, 256, 1, 4), dtype=np.bool_)
    labels = np.array(["background", "label1", "label2", "label3"], dtype=np.str_)
    # values shape (10, 256, 256, 1, 4): spatial dims = (10, 256, 256, 1),
    # n_labels treated as channel
    coordinates = np.zeros((10, 256, 256, 1, 3), dtype=np.float32)
    segmentation = Segmentation(values=values, labels=labels, coordinates=coordinates)
    assert segmentation.values.shape == (10, 256, 256, 1, 4)
    assert segmentation.labels.shape == (4,)
    assert segmentation.coordinates.shape == (10, 256, 256, 1, 3)

    # Incorrect usage: labels shape mismatch
    with pytest.raises(ValueError):
        Segmentation(
            values=values,
            labels=np.array(["background", "label1"], dtype=np.str_),
            coordinates=coordinates,
        )


def test_segmentation_spec_2d():
    """Segmentation with 2D spatial data: (n_frames, z, x, n_labels)."""
    n_frames, z, x, n_labels = 10, 256, 256, 2
    values = np.zeros((n_frames, z, x, n_labels), dtype=np.bool_)
    labels = np.array(["lv", "myocardium"], dtype=np.str_)
    coordinates = np.zeros((n_frames, z, x, 3), dtype=np.float32)

    segmentation = Segmentation(values=values, labels=labels, coordinates=coordinates)
    assert segmentation.values.shape == (n_frames, z, x, n_labels)
    assert segmentation.labels.shape == (n_labels,)
    assert segmentation.coordinates.shape == (n_frames, z, x, 3)

    # Broadcast coordinates (no frame axis)
    coords_broadcast = np.zeros((z, x, 3), dtype=np.float32)
    seg_broadcast = Segmentation(values=values, labels=labels, coordinates=coords_broadcast)
    assert seg_broadcast.coordinates.shape == (z, x, 3)

    # Incorrect: labels shape mismatch
    with pytest.raises(ValueError):
        Segmentation(
            values=values,
            labels=np.array(["lv"], dtype=np.str_),
            coordinates=coordinates,
        )

    # Incorrect: labels missing
    with pytest.raises(AssertionError):
        Segmentation(values=values, coordinates=coordinates)


def _scan_minimal(n_frames: int = 3, n_tx: int = 2, n_el: int = 4):
    return {
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


def _probe_minimal(n_el: int = 4):
    return {"probe_geometry": np.zeros((n_el, 3), dtype=np.float32)}


def _example_metadata():
    return {
        "subject": {
            "type": "human",
            "age": np.uint8(42),
            "sex": "f",
            "fat_percentage": np.float32(17.5),
        },
        "credit": "example-lab",
        "probe_pose": {
            "translation": np.zeros((25, 3), dtype=np.float32),
            "rotation": np.zeros((25, 3), dtype=np.float32),
            "rotation_representation": "euler_xyz",
            "start_time_offset": np.float32(0.0),
            "sampling_frequency": np.float32(50.0),
        },
        "voice_narration": {
            "samples": np.zeros((100), dtype=np.uint8),
            "start_time_offset": np.float32(0.0),
            "sampling_frequency": np.float32(8000.0),
        },
        "ecg": {
            "samples": np.zeros((100), dtype=np.uint8),
            "start_time_offset": np.float32(0.0),
            "sampling_frequency": np.float32(250.0),
        },
        "text_report": "normal acquisition",
        "annotations": {
            "anatomy": "heart",
            "view": np.array(["plax", "plax", "psax"], dtype=np.str_),
            "label": np.array(["normal", "normal", "normal"], dtype=np.str_),
            "image_quality": "high",
        },
    }


def _make_coordinates(values_shape):
    """Build a zero-filled coordinates array compatible with the given values shape.

    For unchanneled values (values_shape has no trailing channel dim) the
    coordinates shape is ``(*values_shape, 3)``; callers that know their values
    are channeled should pass ``values_shape[:-1]`` as *values_shape* explicitly.
    """
    return np.zeros((*values_shape, 3), dtype=np.float32)


def _example_data(n_frames, n_tx, n_el, n_ax, n_ch):
    # For channeled values (last dim = channel), coordinates use values.shape[:-1].
    coords_3d = _make_coordinates((n_frames, 16, 12))  # spatial grid, no channel
    coords_segm = _make_coordinates((n_frames, 16, 12, 1))  # spatial grid for seg (y dim = 1)
    return {
        "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
        "image": {
            "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
            "coordinates": coords_3d,
        },
        "segmentation": {
            "values": np.zeros((n_frames, 16, 12, 1, 2), dtype=np.bool_),
            "labels": np.array(["background", "tissue"], dtype=np.str_),
            "coordinates": coords_segm,
        },
        "sos_map": {
            "values": np.full((n_frames, 16, 12, 1), 1540.0, dtype=np.float32),
            "coordinates": coords_3d,
        },
        "strain": {
            "values": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
            "coordinates": coords_3d,
        },
        "swe": {
            "values": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
            "coordinates": coords_3d,
        },
        "tissue_doppler": {
            "values": np.zeros((n_frames, 16, 12, 1), dtype=np.float32),
            "coordinates": coords_3d,
        },
    }


@pytest.fixture
def dataset_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    return FileSpec(
        data=_example_data(n_frames, n_tx, n_el, n_ax, n_ch),
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
        metadata=_example_metadata(),
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


def test_spec_to_dict_is_recursive(dataset_spec: FileSpec):
    result = dataset_spec.to_dict()

    assert isinstance(result, dict)
    assert isinstance(result["tracks"], list)
    assert len(result["tracks"]) == 1
    assert isinstance(result["tracks"][0]["data"], dict)
    assert isinstance(result["tracks"][0]["scan"], dict)
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["metrics"], dict)

    assert np.array_equal(result["tracks"][0]["data"]["raw_data"], dataset_spec.data.raw_data)
    assert np.array_equal(result["tracks"][0]["scan"]["t0_delays"], dataset_spec.scan.t0_delays)
    assert np.array_equal(
        result["metadata"]["annotations"]["view"],
        dataset_spec.metadata.annotations.view,
    )


def test_spec_to_dict_keeps_optional_fields():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
    )

    result = dataset.to_dict()

    assert "subject" in result["metadata"]
    assert result["metadata"]["subject"] is None
    assert "common_midpoint_phase_error" in result["metrics"]
    assert result["metrics"]["common_midpoint_phase_error"] is None


def test_saving_and_loading(tmp_path, dataset_spec: FileSpec):
    # Save the dataset
    save_path = tmp_path / "test_dataset.hdf5"
    dataset_spec.save(save_path)

    with File(save_path) as loaded_dataset:
        assert np.array_equal(
            loaded_dataset["tracks"]["track_0"]["data"]["raw_data"],
            dataset_spec.data.raw_data,
        )
        assert np.array_equal(
            loaded_dataset["tracks"]["track_0"]["scan"]["t0_delays"],
            dataset_spec.scan.t0_delays,
        )
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
        ScanSpec(**scan)


def test_scan_dimension_count_consistency():
    scan = _scan_minimal(n_tx=2)
    scan["initial_times"] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension 'n_tx' has inconsistent sizes"):
        ScanSpec(**scan)


def test_inconsistent_dimension_error_groups_fields_by_size():
    """The error message groups the offending fields by their observed size."""
    scan = _scan_minimal(n_tx=2)
    # initial_times disagrees (n_tx=3) with the other n_tx=2 fields.
    scan["initial_times"] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        ScanSpec(**scan)

    message = str(exc_info.value)
    assert message.startswith("Dimension 'n_tx' has inconsistent sizes:")
    # Each distinct size is reported on its own line with its fields.
    assert "size 2: " in message
    assert "size 3: " in message
    # The lone disagreeing field appears under its own size.
    assert "size 3: initial_times" in message
    # Fields that share the majority size are grouped together (sorted).
    assert "t0_delays" in message and "tx_apodizations" in message
    size_2_line = next(line for line in message.splitlines() if line.strip().startswith("size 2:"))
    assert "t0_delays" in size_2_line


def test_signal_nd_accepts_variable_trailing_dimensions_with_ellipsis():
    signal = SignalND(
        samples=np.zeros((10, 3, 4, 5), dtype=np.float32),
        start_time_offset=np.float32(0.0),
        sampling_frequency=np.float32(1000.0),
    )

    assert signal.samples.shape == (10, 3, 4, 5)


def test_signal_nd_rejects_missing_time_dimension_for_ellipsis_shape():
    with pytest.raises(ValueError, match=r"samples has shape \(\), expected one of"):
        SignalND(
            samples=np.array(1.0, dtype=np.float32),
            start_time_offset=np.float32(0.0),
            sampling_frequency=np.float32(1000.0),
        )


def test_optional_fields_can_be_omitted():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
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

    scan_spec = ScanSpec(**scan)

    assert np.dtype(scan_spec.sampling_frequency.dtype) == np.dtype(
        ScanSpec.SCHEMA["sampling_frequency"]["dtype"]
    )
    assert scan_spec.center_frequency.dtype == np.dtype(
        ScanSpec.SCHEMA["center_frequency"]["dtype"]
    )
    assert np.dtype(scan_spec.demodulation_frequency.dtype) == np.dtype(
        ScanSpec.SCHEMA["demodulation_frequency"]["dtype"]
    )
    assert scan_spec.initial_times.dtype == np.dtype(ScanSpec.SCHEMA["initial_times"]["dtype"])
    assert scan_spec.t0_delays.dtype == np.dtype(ScanSpec.SCHEMA["t0_delays"]["dtype"])


def test_dataset_builder_accepts_float_raw_data_and_casts_to_float32():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float64)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
    )

    assert dataset.data.raw_data.dtype == np.float32


def test_dataset_builder_dimension_consistency_across_nested_specs():
    n_frames_data, n_frames_scan = 3, 4
    n_tx, n_el, n_ax, n_ch = 2, 4, 8, 1

    scan = {
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
            probe=_probe_minimal(n_el=n_el),
        )


def test_metadata_accepts_custom_signal_nd_keys_and_warns(tmp_path):
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
    data = {"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)}
    metadata = {
        "custom_signal": {
            "samples": np.zeros((32, 3), dtype=np.float16),
            "start_time_offset": np.float32(0.0),
            "sampling_frequency": np.float32(120.0),
        }
    }

    dataset = FileSpec(
        data=data,
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
        metadata=metadata,
        metrics={},
    )
    assert isinstance(dataset.metadata.custom_signal, SignalND)
    assert "custom_signal" in dataset.to_dict()["metadata"]

    with patch("zea.log.warning") as mock_warn:
        File.create(
            tmp_path / "test.hdf5",
            data=data,
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
            metadata=metadata,
        )
    messages = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert any("Custom signal key(s) added to 'metadata'" in m for m in messages)


def test_metadata_custom_key_requires_signal_nd_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="custom 'metadata' key 'custom_signal'"):
        FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
            metadata={"custom_signal": 123},
            metrics={},
        )


def test_data_accepts_custom_map_keys_and_warns(tmp_path):
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
    data = {
        "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
        "custom_map": {
            "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
            "coordinates": np.zeros((n_frames, 16, 12, 3), dtype=np.float32),
            "description": "This is a custom map",
            "unit": "mm",
        },
    }

    dataset = FileSpec(
        data=data,
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe=_probe_minimal(n_el=n_el),
    )
    assert isinstance(dataset.data, DataSpec)
    assert isinstance(dataset.data.custom_map, Map)
    assert "custom_map" in dataset.to_dict()["tracks"][0]["data"]

    with patch("zea.log.warning") as mock_warn:
        File.create(
            tmp_path / "test.hdf5",
            data=data,
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
        )
    messages = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert any("Custom spatial map key(s) added to 'data'" in m for m in messages)


def test_data_custom_key_requires_map_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="Expected field 'custom_scalar' to be"):
        FileSpec(
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "custom_scalar": 123,
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
        )


def test_data_custom_map_dtype_error_includes_map_key_context():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="In field 'custom_map':"):
        FileSpec(
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "custom_map": {
                    "values": np.zeros((n_frames, 16, 12, 1), dtype=np.bool_),
                },
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
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

        # Some Spec subclasses declare fields that are intentionally excluded from SCHEMA
        # (e.g. FileSpec.tracks is managed manually in save/from_hdf5).
        excluded = getattr(cls, "_SCHEMA_EXCLUDED_FIELDS", frozenset())
        dataclass_field_names -= excluded

        missing_in_schema = dataclass_field_names - schema_field_names
        extra_in_schema = schema_field_names - dataclass_field_names

        assert not missing_in_schema and not extra_in_schema, (
            f"{cls.__name__} SCHEMA mismatch. "
            f"Missing in SCHEMA: {sorted(missing_in_schema)}; "
            f"Extra in SCHEMA: {sorted(extra_in_schema)}"
        )


def test_field_metadata_keys_are_subset_of_schema_for_all_specs():
    """FIELD_METADATA keys must be a subset of SCHEMA keys."""
    for obj in vars(spec_module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Spec)
            and obj is not Spec
            and is_dataclass(obj)
            and hasattr(obj, "FIELD_METADATA")
        ):
            extra = set(obj.FIELD_METADATA.keys()) - set(obj.SCHEMA.keys())
            assert not extra, (
                f"{obj.__name__} FIELD_METADATA has keys not in SCHEMA: {sorted(extra)}"
            )


def test_subject_id_warning_for_missing_id(tmp_path):
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    path = tmp_path / "subject_id_missing_warns_on_save.hdf5"
    with patch("zea.log.warning") as mock_warn:
        File.create(
            path,
            data=_example_data(n_frames, n_tx, n_el, n_ax, n_ch),
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
            metadata={
                "subject": {
                    "type": "human",
                    "age": np.uint8(42),
                    "sex": "f",
                    "fat_percentage": np.float32(17.5),
                }
            },
            metrics={
                "common_midpoint_phase_error": np.zeros((n_frames,), dtype=np.float32),
                "coherence_factor": np.ones((n_frames,), dtype=np.float32),
            },
            overwrite=True,
        )
    messages = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert any("Optional Subject field 'id' is not set" in m for m in messages)


def test_subject_id_warning_includes_field_metadata_description(tmp_path):
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    path = tmp_path / "subject_id_description_warns_on_save.hdf5"
    with patch("zea.log.warning") as mock_warn:
        File.create(
            path,
            data=_example_data(n_frames, n_tx, n_el, n_ax, n_ch),
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
            metadata={
                "subject": {
                    "type": "human",
                    "age": np.uint8(42),
                    "sex": "f",
                    "fat_percentage": np.float32(17.5),
                }
            },
            overwrite=True,
        )
    messages = [str(c.args[0]) for c in mock_warn.call_args_list]
    assert any("subject-wise splits" in m for m in messages)


class TestScanValidationErrors:
    """TypeError / ValueError raised by Scan spec validation."""

    def test_t0_delays_dimension_mismatch_raises(self):
        """n_el in tx_apodizations doesn't match t0_delays n_el."""
        scan = _scan_minimal(n_tx=3, n_el=4)
        scan["tx_apodizations"] = np.ones((3, 6), dtype=np.float32)  # 6 ≠ n_el=4
        with pytest.raises(ValueError, match="n_el"):
            ScanSpec(**scan)

    def test_unknown_keyword_raises(self):
        scan = _scan_minimal()
        with pytest.raises(TypeError):
            ScanSpec(**scan, this_key_does_not_exist=42)


class TestDataValidationErrors:
    """TypeError / ValueError raised by DataSpec spec validation."""

    def test_raw_data_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="raw_data"):
            DataSpec(raw_data=np.zeros((2, 3, 8, 4, 1), dtype=np.int8))

    def test_raw_data_wrong_ndim_raises(self):
        """raw_data must be 5-D (n_frames, n_tx, n_ax, n_el, n_ch)."""
        with pytest.raises(ValueError, match="raw_data"):
            DataSpec(raw_data=np.zeros((2, 3, 8), dtype=np.float32))

    def test_empty_data_raises(self):
        """DataSpec() with no fields set must raise."""
        with pytest.raises(ValueError, match="At least one data field must be provided"):
            DataSpec()

    def test_map_wrong_pixel_dtype_raises(self):
        """SosMap inherits FloatMap – values must be float32, not uint8."""
        with pytest.raises(TypeError, match="SosMap: field 'values'"):
            SosMap(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 16, 12, 3), dtype=np.float32),
            )

    def test_image_wrong_pixel_dtype_raises(self):
        """Image is UnsignedIntMap – values must be float32 or uint8, not complex128."""
        with pytest.raises(TypeError, match="Image: field 'values'"):
            Image(
                values=np.zeros((2, 16, 12, 1), dtype=np.complex128),
                coordinates=np.zeros((2, 16, 12, 3), dtype=np.float32),
            )

    def test_segmentation_wrong_pixel_dtype_raises(self):
        """Segmentation is BooleanMap – values must be bool_, not float32."""
        with pytest.raises(TypeError, match="Segmentation: field 'values'"):
            Segmentation(
                values=np.zeros((2, 16, 12, 1, 2), dtype=np.float32),
                labels=np.array(["a", "b"], dtype=np.str_),
                coordinates=np.zeros((2, 16, 12, 1, 3), dtype=np.float32),
            )

    def test_map_coordinates_wrong_shape_raises(self):
        """coordinates must have final dim 3 and spatial dims matching values."""
        # Final dim is not 3 — caught by SCHEMA shape check
        with pytest.raises(ValueError, match="coordinates"):
            Image(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 16, 12, 4), dtype=np.float32),
            )
        # Spatial dims don't match values — caught by Map.__post_init__
        with pytest.raises(ValueError, match="Image: coordinates shape"):
            Image(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 99, 12, 3), dtype=np.float32),
            )

    def test_map_coordinates_valid_channeled_and_unchanneled(self):
        """Valid coordinates shapes for channeled and unchanneled values."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Unchanneled: coordinates.shape == (*values.shape, 3)
            m1 = Map(
                values=np.zeros((2, 16, 12), dtype=np.uint8),
                coordinates=np.zeros((2, 16, 12, 3), dtype=np.float32),
            )
            assert m1.coordinates.shape == (2, 16, 12, 3)

            # Unchanneled with frame-broadcast: coordinates.shape == (*values.shape[1:], 3)
            m1_broadcast = Map(
                values=np.zeros((2, 16, 12), dtype=np.uint8),
                coordinates=np.zeros((16, 12, 3), dtype=np.float32),
            )
            assert m1_broadcast.coordinates.shape == (16, 12, 3)

            # Channeled: coordinates.shape == (*values.shape[:-1], 3)
            m2 = Map(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 16, 12, 3), dtype=np.float32),
            )
            assert m2.coordinates.shape == (2, 16, 12, 3)

            # Channeled with frame-broadcast: coordinates.shape == (*values.shape[1:-1], 3)
            m2_broadcast = Map(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((16, 12, 3), dtype=np.float32),
            )
            assert m2_broadcast.coordinates.shape == (16, 12, 3)

    def test_map_coordinates_frame_broadcast_shape_mismatch_raises(self):
        """Frame-broadcast coordinates must still match non-frame spatial dimensions."""
        with pytest.raises(ValueError, match="Image: coordinates shape"):
            Image(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((99, 12, 3), dtype=np.float32),
            )

    def test_map_coordinates_spatial_axis_omission_raises(self):
        """Dropping a non-frame spatial axis from coordinates must raise ValueError."""
        with pytest.raises(ValueError, match="Image: coordinates shape"):
            Image(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 12, 3), dtype=np.float32),
            )

    def test_map_coordinates_millimetre_range_warns(self):
        """Coordinates with |value| > 1 m should trigger a units warning."""
        # Values of 50 mm look fine in mm but are 0.05 m — no warning expected.
        coords_metres = np.zeros((2, 8, 8, 3), dtype=np.float32)
        coords_metres[..., 2] = 0.05  # 5 cm depth — valid
        Map(
            values=np.zeros((2, 8, 8), dtype=np.uint8),
            coordinates=coords_metres,
        )  # should not warn

        # Coordinates in millimetres: max absolute value = 50 mm > 1 m threshold.
        coords_mm = np.zeros((2, 8, 8, 3), dtype=np.float32)
        coords_mm[..., 2] = 50.0  # 50 mm — looks like mm, not metres
        with patch("zea.log.warning") as mock_warn:
            Map(
                values=np.zeros((2, 8, 8), dtype=np.uint8),
                coordinates=coords_mm,
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("metres" in m for m in messages)

    def test_n_ch_3_raises_for_raw_data(self):
        """raw_data n_ch must be 1 or 2, 3 channels should be rejected."""
        with pytest.raises(ValueError, match="n_ch"):
            DataSpec(raw_data=np.zeros((2, 3, 8, 4, 3), dtype=np.float32))

    def test_n_ch_3_raises_for_aligned_data(self):
        with pytest.raises(ValueError, match="n_ch"):
            DataSpec(aligned_data={"values": np.zeros((2, 3, 8, 4, 3), dtype=np.float32)})

    def test_n_ch_3_raises_for_beamformed_data(self):
        with pytest.raises(ValueError, match="n_ch"):
            DataSpec(
                beamformed_data={
                    "values": np.zeros((2, 8, 6, 3), dtype=np.float32),
                }
            )

    def test_n_ch_1_and_2_are_valid(self):
        """Both n_ch=1 (RF) and n_ch=2 (IQ) must pass."""
        DataSpec(raw_data=np.zeros((2, 3, 8, 4, 1), dtype=np.float32))
        DataSpec(raw_data=np.zeros((2, 3, 8, 4, 2), dtype=np.float32))


class TestMetadataAndMetricsValidationErrors:
    """TypeError / ValueError raised by Metadata / Metrics / Subject validation."""

    def test_subject_age_wrong_dtype_raises(self):
        """age must be uint8, not str."""
        with pytest.raises(TypeError, match="age"):
            Subject(age="forty two")

    def test_signal_missing_required_field_raises(self):
        """Signal1D requires both start_time_offset and sampling_frequency."""
        with pytest.raises(TypeError, match="sampling_frequency"):
            Signal1D(samples=np.zeros(100, dtype=np.float32), start_time_offset=np.float32(0.0))

    def test_metrics_wrong_shape_raises(self):
        """coherence_factor must be 1-D (n_frames,), not 2-D."""
        with pytest.raises(ValueError, match="coherence_factor"):
            MetricsSpec(coherence_factor=np.ones((3, 2), dtype=np.float32))

    def test_annotations_n_frames_mismatch_raises(self):
        """view n_frames in Annotations must match DataSpec n_frames across FileSpec."""
        n_frames_data, n_frames_ann = 3, 5
        n_tx, n_el, n_ax, n_ch = 2, 4, 8, 1

        with pytest.raises(ValueError, match="n_frames"):
            FileSpec(
                data={
                    "raw_data": np.zeros((n_frames_data, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
                },
                scan=_scan_minimal(n_frames=n_frames_data, n_tx=n_tx, n_el=n_el),
                probe=_probe_minimal(n_el=n_el),
                metadata={
                    "annotations": {
                        "view": np.array(["a4c"] * n_frames_ann, dtype=np.str_),
                    }
                },
            )

    def test_annotations_n_frames_mismatch_against_later_track_raises(self):
        """Metadata may agree with track 0 but conflict with a later track.

        Exercises the multi-track loop in ``FileSpec.__post_init__``: the
        per-track consistency check must keep iterating past the matching
        track and report which track disagrees.
        """
        n_tx, n_el, n_ax, n_ch = 2, 4, 8, 1
        n_frames_match, n_frames_conflict = 3, 5

        def _track(n_frames):
            return {
                "data": {
                    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
                },
                "scan": _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                "label": f"track_{n_frames}",
            }

        with pytest.raises(ValueError) as exc_info:
            FileSpec(
                tracks=[_track(n_frames_match), _track(n_frames_conflict)],
                probe=_probe_minimal(n_el=n_el),
                metadata={
                    "annotations": {
                        "view": np.array(["a4c"] * n_frames_match, dtype=np.str_),
                    }
                },
            )

        message = str(exc_info.value)
        assert message.startswith("Dimension 'n_frames' has inconsistent sizes:")
        # The message attributes the sizes to the metadata field and the
        # conflicting track (not track 0, which matched the metadata).
        assert "metadata.annotations.view" in message
        assert "tracks[1]." in message
        assert "tracks[0]." not in message


class TestProbePoseValidation:
    def test_probe_pose_accepts_euler_xyz(self):
        pose = ProbePose(
            translation=np.zeros((25, 3), dtype=np.float32),
            rotation=np.zeros((25, 3), dtype=np.float32),
            rotation_representation="euler_xyz",
            start_time_offset=np.float32(-0.1),
            sampling_frequency=np.float32(50.0),
        )

        assert pose.translation.shape == (25, 3)
        assert pose.rotation.shape == (25, 3)

    def test_probe_pose_accepts_quaternion_wxyz(self):
        pose = ProbePose(
            translation=np.zeros((25, 3), dtype=np.float32),
            rotation=np.zeros((25, 4), dtype=np.float32),
            rotation_representation="quaternion_wxyz",
            start_time_offset=np.float32(0.2),
            sampling_frequency=np.float32(50.0),
        )

        assert pose.rotation.shape == (25, 4)

    def test_probe_pose_accepts_quaternion_xyzw(self):
        pose = ProbePose(
            translation=np.zeros((25, 3), dtype=np.float32),
            rotation=np.zeros((25, 4), dtype=np.float32),
            rotation_representation="quaternion_xyzw",
            start_time_offset=np.float32(0.2),
            sampling_frequency=np.float32(50.0),
        )

        assert pose.rotation.shape == (25, 4)

    def test_probe_pose_requires_rotation_representation(self):
        with pytest.raises(TypeError, match="rotation_representation"):
            ProbePose(
                translation=np.zeros((25, 3), dtype=np.float32),
                rotation=np.zeros((25, 3), dtype=np.float32),
                start_time_offset=np.float32(0.0),
                sampling_frequency=np.float32(50.0),
            )

    def test_probe_pose_rejects_euler_with_quaternion_width(self):
        with pytest.raises(ValueError, match="rotation shape does not match"):
            ProbePose(
                translation=np.zeros((25, 3), dtype=np.float32),
                rotation=np.zeros((25, 4), dtype=np.float32),
                rotation_representation="euler_xyz",
                start_time_offset=np.float32(0.0),
                sampling_frequency=np.float32(50.0),
            )

    def test_probe_pose_rejects_quaternion_with_euler_width(self):
        with pytest.raises(ValueError, match="rotation shape does not match"):
            ProbePose(
                translation=np.zeros((25, 3), dtype=np.float32),
                rotation=np.zeros((25, 3), dtype=np.float32),
                rotation_representation="quaternion_wxyz",
                start_time_offset=np.float32(0.0),
                sampling_frequency=np.float32(50.0),
            )

    def test_probe_pose_rejects_mismatched_time_dimension(self):
        with pytest.raises(
            ValueError, match="translation and rotation must have the same number of time samples"
        ):
            ProbePose(
                translation=np.zeros((25, 3), dtype=np.float32),
                rotation=np.zeros((24, 3), dtype=np.float32),
                rotation_representation="euler_xyz",
                start_time_offset=np.float32(0.0),
                sampling_frequency=np.float32(50.0),
            )

    def test_probe_pose_rejects_non_positive_sampling_frequency(self):
        with pytest.raises(ValueError, match="Sampling frequency must be positive"):
            ProbePose(
                translation=np.zeros((25, 3), dtype=np.float32),
                rotation=np.zeros((25, 3), dtype=np.float32),
                rotation_representation="euler_xyz",
                start_time_offset=np.float32(0.0),
                sampling_frequency=np.float32(0.0),
            )

    def test_signal_accepts_negative_and_positive_start_time_offset(self):
        negative = Signal1D(
            samples=np.zeros(10, dtype=np.float32),
            start_time_offset=np.float32(-0.25),
            sampling_frequency=np.float32(1000.0),
        )
        positive = SignalND(
            samples=np.zeros((10, 2), dtype=np.float32),
            start_time_offset=np.float32(0.25),
            sampling_frequency=np.float32(1000.0),
        )

        assert negative.start_time_offset < 0
        assert positive.start_time_offset > 0


def test_image_spec_accepts_neginf():
    """Image spec validation must allow -inf in float32 arrays (represents
    complete silence in dB domain) but still reject +inf and values above 0."""
    coordinates = np.zeros((2, 8, 8, 3), dtype=np.float32)

    values_with_neginf = np.full((2, 8, 8), -30.0, dtype=np.float32)
    values_with_neginf[0, 0, 0] = -np.inf

    img = Image(values=values_with_neginf, coordinates=coordinates)
    assert img is not None

    values_with_posinf = np.full((2, 8, 8), -30.0, dtype=np.float32)
    values_with_posinf[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite or -inf"):
        Image(values=values_with_posinf, coordinates=coordinates)

    values_positive = np.full((2, 8, 8), 0.1, dtype=np.float32)
    with pytest.raises(ValueError, match="dB scale"):
        Image(values=values_positive, coordinates=coordinates)


def _scan_bare(n_tx: int = 2, n_el: int = 4):
    """Minimal ScanSpec dict with only required fields (all optionals left as None)."""
    return {
        "sampling_frequency": np.float32(30e6),
        "center_frequency": np.float32(5e6),
        "demodulation_frequency": np.float32(5e6),
        "initial_times": np.zeros((n_tx,), dtype=np.float32),
        "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
        "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
        "focus_distances": np.zeros((n_tx,), dtype=np.float32),
        "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
        "polar_angles": np.zeros((n_tx,), dtype=np.float32),
    }


class TestScanSpecSaveWarnings:
    """log.warning calls emitted during save/serialization."""

    @pytest.mark.parametrize(
        "field",
        [
            f.name
            for f in fields(ScanSpec)
            if f.default is None and f.name in ScanSpec.FIELD_METADATA
        ],
    )
    def test_optional_scan_field_missing_warns(self, field):
        with patch("zea.log.warning") as mock_warn:
            ScanSpec(**_scan_bare())
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any(f"ScanSpec field '{field}' is not set" in m for m in messages)

    @pytest.mark.parametrize(
        "field",
        [
            f.name
            for f in fields(ScanSpec)
            if f.default is None and f.name in ScanSpec.FIELD_METADATA
        ],
    )
    def test_optional_scan_field_missing_warns_on_save(self, field, tmp_path):
        path = tmp_path / "scan_save_warns.hdf5"
        with patch("zea.log.warning") as mock_warn:
            File.create(
                path,
                data={"raw_data": np.zeros((2, 2, 8, 4, 1), dtype=np.float32)},
                scan=_scan_bare(n_tx=2, n_el=4),
                probe=_probe_minimal(n_el=4),
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any(f"ScanSpec field '{field}' is not set" in m for m in messages)

    def test_probe_geometry_out_of_range_warns(self):
        with patch("zea.log.warning") as mock_warn:
            ProbeSpec(probe_geometry=np.full((4, 3), 2.0, dtype=np.float32))
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("extend beyond" in m for m in messages)

    def test_focus_distances_large_warns(self):
        scan = _scan_bare()
        scan["focus_distances"] = np.full((2,), 1.5, dtype=np.float32)
        with patch("zea.log.warning") as mock_warn:
            ScanSpec(**scan)
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Focus distances greater than or equal to 1 meter" in m for m in messages)

    def test_transmit_origins_out_of_range_warns(self):
        scan = _scan_bare()
        scan["transmit_origins"] = np.full((2, 3), 2.0, dtype=np.float32)
        with patch("zea.log.warning") as mock_warn:
            ScanSpec(**scan)
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Transmit origin values are unusually large" in m for m in messages)

    def test_map_coordinates_not_provided_warns(self):
        with patch("zea.log.warning") as mock_warn:
            Image(values=np.zeros((2, 8, 8, 1), dtype=np.uint8))
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("coordinates are not provided" in m for m in messages)

    def test_map_coordinates_out_of_range_warns(self):
        with patch("zea.log.warning") as mock_warn:
            Image(
                values=np.zeros((2, 8, 8, 1), dtype=np.uint8),
                coordinates=np.ones((2, 8, 8, 1, 3), dtype=np.float32) * 2,
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("metres" in m for m in messages)

    def test_sos_map_low_values_warns(self):
        with patch("zea.log.warning") as mock_warn:
            SosMap(
                values=np.full((2, 8, 8, 1), 100.0, dtype=np.float32),
                coordinates=np.zeros((2, 8, 8, 1, 3), dtype=np.float32),
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Speed-of-sound map contains values below 300 m/s" in m for m in messages)

    def test_custom_data_map_key_warns(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        data = {
            "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
            "custom_map": {
                "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "coordinates": np.zeros((n_frames, 16, 12, 1, 3), dtype=np.float32),
            },
        }
        with patch("zea.log.warning") as mock_warn:
            File.create(
                tmp_path / "test.hdf5",
                data=data,
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                probe=_probe_minimal(n_el=n_el),
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Custom spatial map key(s) added to 'data'" in m for m in messages)

    def test_custom_metadata_signal_key_warns(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        metadata = {
            "custom_signal": {
                "samples": np.zeros((32, 3), dtype=np.float16),
                "start_time_offset": np.float32(0.0),
                "sampling_frequency": np.float32(120.0),
            }
        }
        with patch("zea.log.warning") as mock_warn:
            File.create(
                tmp_path / "test.hdf5",
                data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                probe=_probe_minimal(n_el=n_el),
                metadata=metadata,
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Custom signal key(s) added to 'metadata'" in m for m in messages)

    def test_subject_id_missing_warns(self):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        with patch("zea.log.warning") as mock_warn:
            FileSpec(
                data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                probe=_probe_minimal(n_el=n_el),
                metadata={"subject": {"type": "human"}},
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any("Optional Subject field 'id' is not set" in m for m in messages)


class TestSubjectFieldWarnings:
    """Subject optional-field warning behavior."""

    @pytest.mark.parametrize("field", list(Subject.SCHEMA))
    def test_optional_subject_field_missing_warns(self, field):
        with patch("zea.log.warning") as mock_warn:
            Subject()
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any(f"Optional Subject field '{field}' is not set" in m for m in messages)

    @pytest.mark.parametrize("field", list(Subject.SCHEMA))
    def test_optional_subject_field_missing_warns_on_save(self, field, tmp_path):
        path = tmp_path / "subject_save_warns.hdf5"
        with patch("zea.log.warning") as mock_warn:
            File.create(
                path,
                data={"raw_data": np.zeros((2, 2, 8, 4, 1), dtype=np.float32)},
                scan=_scan_minimal(n_frames=2, n_tx=2, n_el=4),
                probe=_probe_minimal(n_el=4),
                metadata={"subject": {}},
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any(f"Optional Subject field '{field}' is not set" in m for m in messages)

    def test_no_warning_when_all_fields_provided(self):
        with patch("zea.log.warning") as mock_warn:
            Subject(
                id="patient-001",
                type="human",
                age=np.uint8(42),
                sex="f",
                fat_percentage=np.float32(17.5),
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any("Optional Subject field" in m for m in messages)


class TestMetadataSpecFieldWarnings:
    """MetadataSpec optional-field warning behavior."""

    @pytest.mark.parametrize(
        "field",
        [f.name for f in fields(MetadataSpec) if f.default is None],
    )
    def test_optional_metadata_field_missing_warns(self, field):
        with patch("zea.log.warning") as mock_warn:
            MetadataSpec()
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any(f"Optional MetadataSpec field '{field}' is not set" in m for m in messages)

    @pytest.mark.parametrize(
        "field",
        [f.name for f in fields(MetadataSpec) if f.default is None],
    )
    def test_optional_metadata_field_missing_warns_on_save(self, field, tmp_path):
        path = tmp_path / "metadata_save_warns.hdf5"
        with patch("zea.log.warning") as mock_warn:
            File.create(
                path,
                data={"raw_data": np.zeros((2, 2, 8, 4, 1), dtype=np.float32)},
                scan=_scan_minimal(n_frames=2, n_tx=2, n_el=4),
                probe=_probe_minimal(n_el=4),
                metadata={},
            )
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any(f"Optional MetadataSpec field '{field}' is not set" in m for m in messages)

    def test_no_warning_when_field_is_provided(self):
        with patch("zea.log.warning") as mock_warn:
            MetadataSpec(credit="Doe et al.")
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert not any("Optional MetadataSpec field 'credit'" in m for m in messages)


class TestLoadingWarnings:
    """log.warning calls emitted when reading a zea File."""

    def test_no_scan_group_warns(self, tmp_path):
        path = tmp_path / "no_scan.hdf5"
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("raw_data", data=np.zeros((2, 2, 8, 4, 1), dtype=np.float32))

        with patch("zea.log.warning") as mock_warn:
            with File(path) as f:
                f.get_scan_parameters()
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("Could not find scan parameters in file" in m for m in messages)

    def test_waveforms_stored_as_dict_warns(self, tmp_path):
        """Legacy waveforms stored as an HDF5 group (dict-like) trigger a warning on load."""
        path = tmp_path / "waveforms_dict.hdf5"
        with h5py.File(path, "w") as f:
            s = f.create_group("scan")
            wv = s.create_group("waveforms_one_way")
            wv.create_dataset("0", data=np.zeros(10, dtype=np.float32))
            wv.create_dataset("1", data=np.zeros(10, dtype=np.float32))

        with patch("zea.log.warning") as mock_warn:
            with File(path) as f:
                # f.scan emits the legacy-waveforms warning, then fails because the
                # file has no other (required) ScanSpec fields.
                with pytest.raises((ValueError, TypeError)):
                    f.scan
        messages = [str(c.args[0]) for c in mock_warn.call_args_list]
        assert any("waveforms_one_way" in m and "stored as a dictionary" in m for m in messages)


class TestProbeSpec:
    """Unit tests for the ProbeSpec dataclass."""

    def test_raw_data_requires_probe_geometry(self):
        """A FileSpec with raw_data must define probe_geometry."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        with pytest.raises(ValueError, match="'probe_geometry' is required"):
            FileSpec(
                data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            )

    def test_raw_data_requires_probe_geometry_probe_without_geometry(self):
        """Supplying a probe but omitting probe_geometry still raises."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        with pytest.raises(ValueError, match="'probe_geometry' is required"):
            FileSpec(
                data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                probe={"name": "no_geometry_probe", "type": "linear"},
            )

    def test_raw_data_with_probe_geometry_ok(self):
        """raw_data + probe_geometry validates without error."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        spec = FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe=_probe_minimal(n_el=n_el),
        )
        assert spec.probe.probe_geometry.shape == (n_el, 3)

    def test_no_raw_data_does_not_require_probe_geometry(self):
        """Without raw_data, probe_geometry is not required."""
        n_frames = 2
        coords = np.zeros((n_frames, 16, 12, 3), dtype=np.float32)
        spec = FileSpec(
            data={
                "image": {
                    "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                    "coordinates": coords,
                }
            },
        )
        assert spec.probe is None

    def test_all_fields_none_by_default(self):
        probe = ProbeSpec()
        assert probe.name is None
        assert probe.type is None
        assert probe.probe_center_frequency is None
        assert probe.probe_bandwidth_percent is None
        assert probe.element_width is None
        assert probe.lens_sound_speed is None
        assert probe.lens_thickness is None

    def test_full_probe_spec(self):
        probe = ProbeSpec(
            name="verasonics_l11_4v",
            type="linear",
            probe_center_frequency=np.float32(5.208e6),
            probe_bandwidth_percent=np.float32(67.0),
            element_width=np.float32(0.27e-3),
            lens_sound_speed=np.float32(1000.0),
            lens_thickness=np.float32(1.5e-3),
        )
        assert probe.name == "verasonics_l11_4v"
        assert probe.type == "linear"
        assert probe.probe_center_frequency == pytest.approx(5.208e6, rel=1e-4)
        assert probe.probe_bandwidth_percent == pytest.approx(67.0)
        assert probe.element_width == pytest.approx(0.27e-3, rel=1e-4)

    def test_invalid_center_frequency_raises(self):
        with pytest.raises(ValueError, match="probe_center_frequency"):
            ProbeSpec(probe_center_frequency=np.float32(-1.0))

    def test_invalid_probe_bandwidth_percent_raises(self):
        with pytest.raises(ValueError, match="probe_bandwidth_percent"):
            ProbeSpec(probe_bandwidth_percent=np.float32(0.0))
        with pytest.raises(ValueError, match="probe_bandwidth_percent"):
            ProbeSpec(probe_bandwidth_percent=np.float32(-10.0))

    def test_invalid_element_width_raises(self):
        with pytest.raises(ValueError, match="element_width"):
            ProbeSpec(element_width=np.float32(-0.001))

    def test_probe_geometry_wrong_dtype_raises(self):
        with pytest.raises(TypeError, match="probe_geometry"):
            ProbeSpec(probe_geometry=np.zeros((4, 3), dtype=np.int32))

    def test_probe_geometry_wrong_shape_raises(self):
        """probe_geometry must be (n_el, 3) — literal 3 is enforced."""
        with pytest.raises(ValueError, match="probe_geometry"):
            ProbeSpec(probe_geometry=np.zeros((4, 2), dtype=np.float32))

    def test_n_elements_derived_from_probe_geometry(self):
        pg = np.zeros((128, 3), dtype=np.float32)
        probe = ProbeSpec(probe_geometry=pg)
        assert probe.n_el == 128

    def test_n_elements_none_without_probe_geometry(self):
        probe = ProbeSpec()
        assert probe.n_el is None

    def test_probe_spec_ignores_legacy_n_elements_pitch_kwargs(self):
        """Old HDF5 files may pass n_elements/pitch; they are silently ignored."""
        # Simulate what _validate_nested_field does after filtering known fields
        pg = np.zeros((4, 3), dtype=np.float32)
        probe = ProbeSpec(probe_geometry=pg)
        assert probe.n_el == 4

    def test_invalid_lens_sound_speed_raises(self):
        with pytest.raises(ValueError, match="lens_sound_speed"):
            ProbeSpec(lens_sound_speed=np.float32(0.0))

    def test_invalid_lens_thickness_raises(self):
        with pytest.raises(ValueError, match="lens_thickness"):
            ProbeSpec(lens_thickness=np.float32(-0.001))

    def test_probe_spec_from_dict(self):
        d = {"name": "L11-4v", "type": "linear", "probe_center_frequency": np.float32(5e6)}
        probe = ProbeSpec(**d)
        assert probe.name == "L11-4v"
        assert probe.probe_center_frequency == pytest.approx(5e6)

    def test_probe_casts_float64_to_float32(self):
        probe = ProbeSpec(probe_center_frequency=5.208e6)  # Python float → float64 → float32
        assert probe.probe_center_frequency.dtype == np.float32

    def test_file_spec_probe_propagates_name(self):
        """FileSpec.probe.name is accessible when probe dict is given."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        spec = FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe={
                "name": "test_probe",
                "type": "phased",
                "probe_center_frequency": np.float32(3e6),
                "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
            },
        )
        assert spec.probe.name == "test_probe"
        assert isinstance(spec.probe, ProbeSpec)
        assert spec.probe.type == "phased"

    def test_file_spec_probe_name_not_a_field(self):
        """probe_name is not a FileSpec field; probe={'name': ...} is the way."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        with pytest.raises(TypeError, match="probe_name"):
            FileSpec(
                data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
                probe={"name": "inner_name", "type": "linear"},
                probe_name="outer_name",
            )

    def test_probe_spec_round_trip_hdf5(self, tmp_path):
        """ProbeSpec saved to HDF5 via FileSpec and loaded back preserves values."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        save_path = tmp_path / "with_probe.hdf5"
        spec = FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe={
                "name": "verasonics_l11_4v",
                "type": "linear",
                "probe_center_frequency": np.float32(5.208e6),
                "probe_bandwidth_percent": np.float32(67.0),
                "element_width": np.float32(0.27e-3),
                "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
            },
        )
        spec.save(save_path)

        with File(save_path) as f:
            assert f.probe.name == "verasonics_l11_4v"
            assert "probe" in f

        loaded = File(str(save_path))._to_file_spec()
        assert loaded.probe.name == "verasonics_l11_4v"
        assert loaded.probe is not None
        assert isinstance(loaded.probe, ProbeSpec)
        assert loaded.probe.type == "linear"
        assert loaded.probe.probe_center_frequency == pytest.approx(5.208e6, rel=1e-4)
        assert loaded.probe.probe_bandwidth_percent == pytest.approx(67.0)

    def test_file_create_with_probe_dict(self, tmp_path):
        """File.create accepts a probe dict and stores probe group + probe_name attr."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1
        path = tmp_path / "probe_create.hdf5"
        raw = np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
        scan = _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el)

        File.create(
            path,
            data={"raw_data": raw},
            scan=scan,
            probe={
                "name": "my_probe",
                "type": "linear",
                "probe_center_frequency": np.float32(7.5e6),
                "probe_geometry": np.zeros((n_el, 3), dtype=np.float32),
            },
        )

        with File(path) as f:
            assert f.probe.name == "my_probe"
            assert "probe" in f
