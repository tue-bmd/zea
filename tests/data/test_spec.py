from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from zea.data import spec as spec_module
from zea.data.file import File
from zea.data.spec import (
    DataSpec,
    FileSpec,
    Image,
    Map,
    MetricsSpec,
    ProbePose,
    ScanSpec,
    Segmentation,
    Signal1D,
    SignalND,
    SosMap,
    Spec,
    Subject,
)


def test_segmentation_spec():
    # Correct usage
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
    assert isinstance(result["data"], dict)
    assert isinstance(result["scan"], dict)
    assert isinstance(result["metadata"], dict)
    assert isinstance(result["metrics"], dict)

    assert np.array_equal(result["data"]["raw_data"], dataset_spec.data.raw_data)
    assert np.array_equal(result["scan"]["t0_delays"], dataset_spec.scan.t0_delays)
    assert np.array_equal(
        result["metadata"]["annotations"]["view"],
        dataset_spec.metadata.annotations.view,
    )


def test_spec_to_dict_keeps_optional_fields():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    dataset = FileSpec(
        data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
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
        ScanSpec(**scan)


def test_scan_dimension_count_consistency():
    scan = _scan_minimal(n_tx=2)
    scan["initial_times"] = np.zeros((3,), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension 'n_tx' has inconsistent sizes"):
        ScanSpec(**scan)


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
        )


def test_metadata_accepts_custom_signal_nd_keys_and_warns():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.warns(match="Custom signal key\(s\) added to 'metadata'"):
        dataset = FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            metadata={
                "custom_signal": {
                    "samples": np.zeros((32, 3), dtype=np.float16),
                    "start_time_offset": np.float32(0.0),
                    "sampling_frequency": np.float32(120.0),
                }
            },
            metrics={},
        )

    assert isinstance(dataset.metadata.custom_signal, SignalND)
    assert "custom_signal" in dataset.to_dict()["metadata"]


def test_metadata_custom_key_requires_signal_nd_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="Expected field 'custom_signal' to be"):
        FileSpec(
            data={"raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            metadata={"custom_signal": 123},
            metrics={},
        )


def test_data_accepts_custom_map_keys_and_warns():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.warns(match="Custom spatial map key\(s\) added to 'data'"):
        dataset = FileSpec(
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "custom_map": {
                    "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                    "coordinates": np.zeros((n_frames, 16, 12, 3), dtype=np.float32),
                    "description": "This is a custom map",
                    "unit": "mm",
                },
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        )

    assert isinstance(dataset.data, DataSpec)
    assert isinstance(dataset.data.custom_map, Map)
    assert "custom_map" in dataset.to_dict()["data"]


def test_data_custom_key_requires_map_spec():
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

    with pytest.raises(TypeError, match="Expected field 'custom_scalar' to be"):
        FileSpec(
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "custom_scalar": 123,
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
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


def test_subject_id_warning_for_missing_id():
    n_frames, n_tx, n_el, n_ax, n_ch = 3, 2, 4, 8, 1

    with pytest.warns(
        match="Subject ID is not provided; please consider adding an ID for better traceability"
    ):
        FileSpec(
            data=_example_data(n_frames, n_tx, n_el, n_ax, n_ch),
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
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
        )


class TestScanValidationErrors:
    """TypeError / ValueError raised by Scan spec validation."""

    def test_probe_geometry_wrong_dtype_raises(self):
        scan = _scan_minimal()
        scan["probe_geometry"] = np.zeros((4, 3), dtype=np.int32)
        with pytest.raises(TypeError, match="probe_geometry"):
            ScanSpec(**scan)

    def test_probe_geometry_wrong_shape_raises(self):
        """probe_geometry must be (n_el, 3) — literal 3 is enforced."""
        scan = _scan_minimal(n_el=4)
        scan["probe_geometry"] = np.zeros((4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="probe_geometry"):
            ScanSpec(**scan)

    def test_t0_delays_dimension_mismatch_raises(self):
        """n_el in t0_delays doesn't match probe_geometry n_el."""
        scan = _scan_minimal(n_tx=3, n_el=4)
        scan["t0_delays"] = np.zeros((3, 6), dtype=np.float32)  # 6 ≠ n_el=4
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

            # Channeled: coordinates.shape == (*values.shape[:-1], 3)
            m2 = Map(
                values=np.zeros((2, 16, 12, 1), dtype=np.uint8),
                coordinates=np.zeros((2, 16, 12, 3), dtype=np.float32),
            )
            assert m2.coordinates.shape == (2, 16, 12, 3)

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
        with pytest.warns(match="metres"):
            Map(
                values=np.zeros((2, 8, 8), dtype=np.uint8),
                coordinates=coords_mm,
            )

    def test_n_ch_3_raises_for_raw_data(self):
        """raw_data n_ch must be 1 or 2, 3 channels should be rejected."""
        with pytest.raises(ValueError, match="n_ch"):
            DataSpec(raw_data=np.zeros((2, 3, 8, 4, 3), dtype=np.float32))

    def test_n_ch_3_raises_for_aligned_data(self):
        with pytest.raises(ValueError, match="n_ch"):
            DataSpec(aligned_data=np.zeros((2, 3, 8, 4, 3), dtype=np.float32))

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
                metadata={
                    "annotations": {
                        "view": np.array(["a4c"] * n_frames_ann, dtype=np.str_),
                    }
                },
            )


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
