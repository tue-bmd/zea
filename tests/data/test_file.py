"""Tests for the File module."""

import numpy as np
import pytest

from zea.data.file import File, GroupProxy, dict_to_sorted_list, load_file
from zea.data.spec import FileSpec, Image, Segmentation
from zea.probes import Probe
from zea.scan import Scan

from . import generate_example_dataset

# Dummy extent for map-based data types in tests
_TEST_EXTENT = np.array([-0.02, 0.02, 0, 0, -0.03, 0], dtype=np.float32)


def _make_map(values):
    """Wrap values into a Map-compatible dict."""
    return {"values": values, "extent": _TEST_EXTENT}


@pytest.fixture
def h5_filepath(tmp_path):
    """Create path for the H5 test file."""
    path = tmp_path / "dummy_dataset.hdf5"
    yield path


@pytest.fixture
def simple_h5_file(h5_filepath):
    """Create a simple H5 file with only attributes."""
    with File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.attrs["dummy_attr2"] = "dummy_value2"
        dataset.attrs["dummy_attr3"] = ["dummy_value3"]
    yield h5_filepath


@pytest.fixture
def complex_h5_file(h5_filepath):
    """Create an H5 file with attributes and datasets."""
    with File(h5_filepath, "w") as dataset:
        dataset.attrs["dummy_attr"] = "dummy_value"
        dataset.create_dataset("dummy_dataset", data=np.random.randn(10, 20))
        dataset.create_dataset("dummy_dataset2", data=np.arange(5))
    yield h5_filepath


def test_basic_properties(simple_h5_file):
    """Test basic properties of File class."""

    with File(simple_h5_file) as file:
        assert file.attrs["dummy_attr"] == "dummy_value"

        # Get length of file (should be 0 as there are no datasets)
        assert len(file) == 0


def test_with_datasets(complex_h5_file):
    """Test File features with datasets."""
    with File(complex_h5_file) as file:
        # Get length of file
        assert len(file) == 2

        # Get shape of file
        assert file.shape("dummy_dataset") == (10, 20)

        # Get keys in file
        assert list(file.keys()) == ["dummy_dataset", "dummy_dataset2"]


def test_recursively_load_dict(complex_h5_file):
    """Test recursively loading dict contents from group."""

    with File(complex_h5_file) as file:
        dict_contents = file.recursively_load_dict_contents_from_group("/")
        assert list(dict_contents.keys()) == ["dummy_dataset", "dummy_dataset2"]
        assert dict_contents["dummy_dataset"].shape == (10, 20)
        assert dict_contents["dummy_dataset2"].shape == (5,)
        assert np.array_equal(dict_contents["dummy_dataset2"], np.arange(5))


def test_print_hdf5_attrs(complex_h5_file, capsys):
    """Test printing HDF5 attributes."""

    with File(complex_h5_file) as file:
        file.summary()

    captured = capsys.readouterr()
    assert "dummy_attr" in captured.out


def test_file_attributes():
    """Test file attributes."""

    DATASET_PATH = (
        "hf://zeahub/picmus/database/simulation/contrast_speckle/contrast_speckle_simu_dataset_iq"
    )

    FILE_NAME = "contrast_speckle_simu_dataset_iq.hdf5"
    FILE_PATH = DATASET_PATH + "/" + FILE_NAME
    FILE_N_FRAMES = 1
    FILE_PROBE_NAME = "verasonics_l11_4v"

    with File(FILE_PATH) as file:
        assert file.name == FILE_NAME, "File name should match expected value"
        assert file.n_frames == FILE_N_FRAMES, "Number of frames should match expected value"
        assert file.probe_name == FILE_PROBE_NAME, "Probe name should match expected value"
        assert isinstance(file.probe(), Probe), "Probe should be an instance of Probe class"
        assert isinstance(file.scan(), Scan), "Scan should be an instance of Scan class"

        file.validate()


def test_load_file_function(dummy_file):
    """Test the load_file function."""

    selected_transmits = [0, 2, 4]
    data, scan, probe = load_file(dummy_file, indices=(slice(2), selected_transmits))

    assert data.shape[0] == 2, "Data should have 2 frames"
    assert data.shape[1] == 3, "Data should have 3 selected transmits"
    assert isinstance(scan, Scan), "Scan should be an instance of Scan class"
    assert isinstance(probe, Probe), "Probe should be an instance of Probe class"
    assert scan.selected_transmits == selected_transmits, (
        "Selected transmits should match expected value"
    )


def test_dict_to_sorted_list():
    """Test dict_to_sorted_list utility function."""

    test_dict = {"b": 2, "a": 1, "c": 3}
    sorted_list = dict_to_sorted_list(test_dict)

    assert sorted_list == [1, 2, 3], "The sorted list should be [1, 2, 3]"

    assert dict_to_sorted_list({}) == [], "The sorted list of an empty dict should be []"


def _scan_minimal(n_frames=3, n_tx=2, n_el=4):
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
def spec_file(tmp_path):
    """Create a spec-format HDF5 file via FileSpec.save()."""
    n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
    raw = np.random.randn(n_frames, n_tx, n_ax, n_el, n_ch).astype(np.float32)
    env = np.random.randn(n_frames, 16, 12).astype(np.float32)

    fspec = FileSpec(
        data={"raw_data": raw, "envelope_data": _make_map(env)},
        scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        probe_name="test_probe",
        description="spec format test file",
    )
    path = tmp_path / "spec_format.hdf5"
    fspec.save(str(path))
    return str(path), fspec, raw, env


class TestGroupProxy:
    def test_attribute_access_returns_dataset(self, spec_file):
        path, _, raw, _ = spec_file
        import h5py as _h5py

        with File(path) as f:
            ds = f.data.raw_data
            assert isinstance(ds, _h5py.Dataset)
            assert ds.shape == raw.shape

    def test_slicing_loads_subset(self, spec_file):
        path, _, raw, _ = spec_file

        with File(path) as f:
            loaded = f.data.raw_data[:, :2]
            np.testing.assert_array_equal(loaded, raw[:, :2])

    def test_nested_group_access(self, tmp_path):
        """Groups within 'data' are also proxied."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        fspec = FileSpec(
            data={
                "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                "image": {
                    "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                    "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
                },
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        )
        path = tmp_path / "nested.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            proxy = f.data.image
            assert isinstance(proxy, GroupProxy)
            assert proxy.values.shape == (n_frames, 16, 12, 1)

    def test_missing_key_raises_attribute_error(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            with pytest.raises(AttributeError, match="No key 'nonexistent'"):
                f.data.nonexistent

    def test_keys_and_contains(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            assert "raw_data" in f.data
            assert "envelope_data" in f.data
            assert "nothing_here" not in f.data
            assert set(f.data.keys()) >= {"raw_data", "envelope_data"}

    def test_dir_lists_children(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            d = dir(f.data)
            assert "raw_data" in d
            assert "envelope_data" in d


class TestFileDataProperty:
    def test_data_property_returns_group_proxy(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            assert isinstance(f.data, GroupProxy)

    def test_data_property_raises_when_no_data_group(self, simple_h5_file):
        with File(simple_h5_file) as f:
            with pytest.raises(KeyError, match="No 'data' group"):
                f.data


class TestValidateSpec:
    def test_round_trip(self, spec_file):
        """Save via FileSpec, re-open, validate_spec() returns equivalent object."""
        path, original_spec, raw, env = spec_file

        with File(path) as f:
            loaded_spec = f.validate_spec()

        np.testing.assert_array_equal(loaded_spec.data.raw_data, raw)
        np.testing.assert_array_equal(loaded_spec.data.envelope_data.values, env)
        assert loaded_spec.probe_name == "test_probe"
        assert loaded_spec.description == "spec format test file"

    def test_validate_spec_on_complete_legacy_file(self, tmp_path):
        """validate_spec() succeeds on legacy files that have all required scan
        fields plus the extra scalar datasets (n_frames, n_tx, etc.)."""
        import h5py

        path = tmp_path / "complete_legacy.hdf5"
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        raw = np.random.randn(n_frames, n_tx, n_ax, n_el, n_ch).astype(np.float32)

        with h5py.File(path, "w") as f:
            # Legacy root attrs
            f.attrs["probe"] = "legacy_probe"
            f.attrs["description"] = "legacy file"

            # Data group with flat image (legacy format)
            g = f.create_group("data")
            g.attrs["description"] = "data group"
            ds = g.create_dataset("raw_data", data=raw)
            ds.attrs["unit"] = "-"
            ds.attrs["description"] = "raw data"
            img = np.zeros((n_frames, 16, 12), dtype=np.float32)
            ds_img = g.create_dataset("image", data=img)
            ds_img.attrs["unit"] = "-"
            ds_img.attrs["description"] = "image"

            # Scan group with all required fields PLUS legacy scalar extras
            s = f.create_group("scan")
            s.attrs["description"] = "scan group"

            def _add(name, data, unit="-", desc=""):
                ds = s.create_dataset(name, data=np.asarray(data))
                ds.attrs["unit"] = unit
                ds.attrs["description"] = desc

            _add("probe_geometry", np.zeros((n_el, 3), dtype=np.float32), "m")
            _add("sampling_frequency", np.float32(30e6), "Hz")
            _add("center_frequency", np.float32(5e6), "Hz")
            _add("demodulation_frequency", np.float32(5e6), "Hz")
            _add("initial_times", np.zeros(n_tx, dtype=np.float32), "s")
            _add("t0_delays", np.zeros((n_tx, n_el), dtype=np.float32), "s")
            _add("tx_apodizations", np.ones((n_tx, n_el), dtype=np.float32))
            _add("focus_distances", np.zeros(n_tx, dtype=np.float32), "m")
            _add("transmit_origins", np.zeros((n_tx, 3), dtype=np.float32), "m")
            _add("polar_angles", np.zeros(n_tx, dtype=np.float32), "rad")
            _add("azimuth_angles", np.zeros(n_tx, dtype=np.float32), "rad")
            _add("time_to_next_transmit", np.ones((n_frames, n_tx), dtype=np.float32), "s")
            # Legacy scalar fields NOT in Scan.SCHEMA
            _add("n_frames", np.int64(n_frames))
            _add("n_tx", np.int64(n_tx))
            _add("n_ax", np.int64(n_ax))
            _add("n_el", np.int64(n_el))
            _add("n_ch", np.int64(n_ch))

        with File(path) as f:
            spec = f.validate_spec()
            assert isinstance(spec, FileSpec)
            np.testing.assert_array_equal(spec.data.raw_data, raw)
            # Legacy flat image should be skipped, not cause a crash
            assert spec.data.image is None
            # probe attr mapped to probe_name
            assert spec.probe_name == "legacy_probe"

    def test_validate_spec_raises_on_incomplete_legacy_file(self, tmp_path):
        """validate_spec() raises on legacy files missing required scan fields."""
        import h5py

        path = tmp_path / "incomplete_legacy.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs["probe"] = "test_probe"
            g = f.create_group("data")
            g.create_dataset("raw_data", data=np.zeros((1, 2, 8, 4, 1), dtype=np.float32))
            # Scan group with only a subset of required fields (incomplete)
            s = f.create_group("scan")
            s.create_dataset("probe_geometry", data=np.zeros((4, 3), dtype=np.float32))
            s.create_dataset("sampling_frequency", data=np.float32(40e6))

        with File(str(path)) as f:
            with pytest.raises(TypeError, match="missing.*required"):
                f.validate_spec()

    def test_validate_spec_passes_for_custom_map_key(self, tmp_path):
        """A file saved with a custom map key in 'data' should pass validate_spec()."""
        import warnings

        from zea.data.spec import FileSpec

        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fspec = FileSpec(
                data={
                    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                    "custom_map": {
                        "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                        "extent": np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32),
                    },
                },
                scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            )

        path = tmp_path / "custom_map.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            loaded = f.validate_spec()

        assert loaded.data.custom_map is not None
        np.testing.assert_array_equal(loaded.data.custom_map.values, fspec.data.custom_map.values)


class TestFieldMetadataAttrs:
    def test_unit_and_description_written(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            rd_ds = f["data"]["raw_data"]
            assert rd_ds.attrs["unit"] == "-"
            assert rd_ds.attrs["description"] != ""

            # Check scan field metadata
            pg_ds = f["scan"]["probe_geometry"]
            assert pg_ds.attrs["unit"] == "m"

    def test_scan_field_metadata_matches_spec(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            for key in f["scan"].keys():
                ds = f["scan"][key]
                assert "unit" in ds.attrs, f"Missing 'unit' on scan/{key}"
                assert "description" in ds.attrs, f"Missing 'description' on scan/{key}"


class TestProbeNameCompat:
    def test_probe_name_from_spec_format(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            assert f.probe_name == "test_probe"

    def test_probe_name_from_legacy_format(self, dummy_file):
        """Legacy files use 'probe' attr; File.probe_name handles both."""
        with File(dummy_file) as f:
            assert f.probe_name == "generic"


class TestImageOnlyFile:
    def test_image_only_spec_file(self, tmp_path):
        """FileSpec and File work for files with only image data (no raw_data)."""
        n_frames = 2
        fspec = FileSpec(
            data={
                "image": {
                    "values": np.zeros((n_frames, 32, 24, 1), dtype=np.uint8),
                    "extent": np.array([0, 0.05, 0, 0.04, -0.04, -0.01], dtype=np.float32),
                },
            },
            scan=_scan_minimal(n_frames=n_frames),
        )
        path = tmp_path / "image_only.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            assert "image" in f.data
            proxy = f.data.image
            assert isinstance(proxy, GroupProxy)
            assert proxy.values.shape[0] == n_frames

    def test_envelope_only_spec_file(self, tmp_path):
        """File with only envelope_data (no raw_data)."""
        n_frames = 4
        fspec = FileSpec(
            data={"envelope_data": _make_map(np.ones((n_frames, 32, 24), dtype=np.float32))},
            scan=_scan_minimal(n_frames=n_frames),
        )
        path = tmp_path / "envelope_only.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            loaded_spec = f.validate_spec()
            assert loaded_spec.data.raw_data is None
            assert loaded_spec.data.envelope_data is not None


class TestAllPipelineDataTypes:
    def test_all_pipeline_fields(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        data_dict = {
            "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
            "aligned_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
            "beamformed_data": _make_map(np.zeros((n_frames, 16, 12, n_ch), dtype=np.float32)),
            "envelope_data": _make_map(np.zeros((n_frames, 16, 12), dtype=np.float32)),
            "image_sc": _make_map(np.zeros((n_frames, 32, 24), dtype=np.uint8)),
        }
        fspec = FileSpec(
            data=data_dict,
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe_name="all_pipeline",
        )
        path = tmp_path / "all_pipeline.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            loaded = f.validate_spec()
            assert loaded.data.raw_data is not None
            assert loaded.data.aligned_data is not None
            assert loaded.data.beamformed_data is not None
            assert loaded.data.envelope_data is not None
            assert loaded.data.image_sc is not None


class TestSlicing:
    """Verify that data can be sliced via GroupProxy without loading the full array."""

    @pytest.fixture
    def sliceable_file(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 4, 5, 6, 16, 2
        raw = np.random.randn(n_frames, n_tx, n_ax, n_el, n_ch).astype(np.float32)
        env = np.random.randn(n_frames, 32, 24).astype(np.float32)
        path = tmp_path / "sliceable.hdf5"
        f = File.create(
            path,
            data={"raw_data": raw, "envelope_data": _make_map(env)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe_name="slice_test",
        )
        f.close()
        return str(path), raw, env

    def test_single_frame(self, sliceable_file):
        path, raw, _ = sliceable_file
        with File(path) as f:
            frame0 = f.data.raw_data[0]
            np.testing.assert_array_equal(frame0, raw[0])

    def test_frame_range(self, sliceable_file):
        path, raw, _ = sliceable_file
        with File(path) as f:
            first_two = f.data.raw_data[:2]
            np.testing.assert_array_equal(first_two, raw[:2])

    def test_transmit_selection(self, sliceable_file):
        """Select specific transmits: f.data.raw_data[:, [0, 2, 4]]."""
        path, raw, _ = sliceable_file
        with File(path) as f:
            selected = f.data.raw_data[:, [0, 2, 4]]
            np.testing.assert_array_equal(selected, raw[:, [0, 2, 4]])

    def test_combined_frame_and_transmit(self, sliceable_file):
        path, raw, _ = sliceable_file
        with File(path) as f:
            subset = f.data.raw_data[1:3, :2]
            np.testing.assert_array_equal(subset, raw[1:3, :2])

    def test_envelope_slice(self, sliceable_file):
        path, _, env = sliceable_file
        with File(path) as f:
            cropped = f.data.envelope_data.values[:, 8:16, 4:12]
            np.testing.assert_array_equal(cropped, env[:, 8:16, 4:12])

    def test_ellipsis_slice(self, sliceable_file):
        path, raw, _ = sliceable_file
        with File(path) as f:
            last_channel = f.data.raw_data[..., -1]
            np.testing.assert_array_equal(last_channel, raw[..., -1])


class TestSpatialData:
    """Test saving + reading spatial maps that include values + extent."""

    @pytest.fixture
    def spatial_file(self, tmp_path):
        n_frames = 3
        img_values = np.random.randint(0, 255, (n_frames, 64, 48, 1), dtype=np.uint8)
        img_extent = np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32)
        seg_values = np.random.choice([True, False], (n_frames, 64, 48, 1, 2)).astype(np.bool_)
        seg_labels = np.array(["background", "lumen"], dtype=np.str_)
        seg_extent = np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32)
        sos_values = np.full((n_frames, 64, 48, 1), 1540.0, dtype=np.float32)
        sos_extent = np.array([0.0, 0.05, 0.0, 0.04, -0.04, -0.01], dtype=np.float32)

        path = tmp_path / "spatial.hdf5"
        f = File.create(
            path,
            data={
                "envelope_data": _make_map(np.ones((n_frames, 32, 24), dtype=np.float32)),
                "image": {"values": img_values, "extent": img_extent},
                "segmentation": {
                    "values": seg_values,
                    "labels": seg_labels,
                    "extent": seg_extent,
                },
                "sos_map": {"values": sos_values, "extent": sos_extent},
            },
            scan=_scan_minimal(n_frames=n_frames),
            probe_name="spatial_test",
        )
        f.close()
        return (
            str(path),
            img_values,
            img_extent,
            seg_values,
            seg_labels,
            sos_values,
        )

    def test_image_group_structure(self, spatial_file):
        path, img_values, img_extent, *_ = spatial_file
        with File(path) as f:
            proxy = f.data.image
            assert isinstance(proxy, GroupProxy)
            assert "values" in proxy
            assert "extent" in proxy

    def test_image_values_read(self, spatial_file):
        path, img_values, *_ = spatial_file
        with File(path) as f:
            np.testing.assert_array_equal(f.data.image.values[()], img_values)

    def test_image_values_slice(self, spatial_file):
        path, img_values, *_ = spatial_file
        with File(path) as f:
            frame0 = f.data.image.values[0]
            np.testing.assert_array_equal(frame0, img_values[0])

    def test_segmentation_values_and_labels(self, spatial_file):
        path, _, _, seg_values, seg_labels, _ = spatial_file
        with File(path) as f:
            np.testing.assert_array_equal(f.data.segmentation.values[()], seg_values)
            loaded_labels = f.data.segmentation.labels.asstr()[()]
            np.testing.assert_array_equal(loaded_labels, seg_labels)

    def test_sos_map_values(self, spatial_file):
        path, *_, sos_values = spatial_file
        with File(path) as f:
            np.testing.assert_allclose(f.data.sos_map.values[()], sos_values, atol=1e-6)

    def test_spatial_round_trip_via_validate_spec(self, spatial_file):
        path, img_values, img_extent, seg_values, seg_labels, sos_values = spatial_file
        with File(path) as f:
            spec = f.validate_spec()

        assert isinstance(spec.data.image, Image)
        np.testing.assert_array_equal(spec.data.image.values, img_values)
        np.testing.assert_array_equal(spec.data.image.extent, img_extent)
        assert isinstance(spec.data.segmentation, Segmentation)
        np.testing.assert_array_equal(spec.data.segmentation.values, seg_values)
        np.testing.assert_array_equal(spec.data.segmentation.labels, seg_labels)


class TestFileCreate:
    def test_create_returns_readable_file(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        raw = np.ones((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
        path = tmp_path / "created.hdf5"

        f = File.create(
            path,
            data={"raw_data": raw},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe_name="create_test",
            description="created via File.create",
        )
        assert f.mode == "r"
        np.testing.assert_array_equal(f.data.raw_data[()], raw)
        assert f.probe_name == "create_test"
        f.close()

    def test_create_raises_on_existing_file(self, tmp_path):
        path = tmp_path / "exists.hdf5"
        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((2, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=2),
        ).close()

        with pytest.raises(FileExistsError):
            File.create(
                path,
                data={"envelope_data": _make_map(np.ones((2, 8, 6), dtype=np.float32))},
                scan=_scan_minimal(n_frames=2),
            )

    def test_create_overwrite(self, tmp_path):
        path = tmp_path / "overwrite.hdf5"
        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((2, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=2),
        ).close()

        # Should succeed with overwrite=True
        f = File.create(
            path,
            data={"envelope_data": _make_map(np.zeros((3, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=3),
            overwrite=True,
        )
        assert f.data.envelope_data.values.shape[0] == 3
        f.close()

    def test_create_validates_before_writing(self, tmp_path):
        """Bad shape should be caught before any file is created."""
        path = tmp_path / "bad.hdf5"
        with pytest.raises((TypeError, ValueError)):
            File.create(
                path,
                # raw_data needs 5 dims, giving 3 should fail
                data={"raw_data": np.ones((2, 8, 4), dtype=np.float32)},
                scan=_scan_minimal(n_frames=2, n_tx=3, n_el=4),
            )
        assert not path.exists()


class TestMetadataMetricsAccessors:
    """Tests for File.metadata() and File.metrics() accessors."""

    def test_metadata_round_trip(self, tmp_path):
        n_frames, n_tx, n_el = 2, 3, 4
        path = tmp_path / "meta.hdf5"
        rng = np.random.default_rng(0)

        metadata = {
            "subject": {"id": "patient_01", "type": "human", "age": np.uint8(30), "sex": "f"},
            "credit": "Test Lab",
            "probe_pose": {
                "translation": np.zeros((25, 3), dtype=np.float32),
                "rotation": np.zeros((25, 4), dtype=np.float32),
                "rotation_representation": "quaternion_wxyz",
                "start_time_offset": np.float32(-0.1),
                "sampling_frequency": np.float32(50.0),
            },
            "ecg": {
                "samples": rng.standard_normal(100).astype(np.float32),
                "start_time_offset": np.float32(0.0),
                "sampling_frequency": np.float32(500.0),
            },
            "annotations": {
                "view": np.array(["a4c"] * n_frames, dtype=np.str_),
                "label": np.array(["normal"] * n_frames, dtype=np.str_),
            },
        }

        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((n_frames, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            metadata=metadata,
        ).close()

        with File(path) as f:
            meta = f.metadata()
            assert meta.subject.id == "patient_01"
            assert meta.subject.age == 30
            assert meta.credit == "Test Lab"
            assert meta.probe_pose.translation.shape == (25, 3)
            assert meta.probe_pose.rotation.shape == (25, 4)
            assert meta.probe_pose.rotation_representation == "quaternion_wxyz"
            assert meta.probe_pose.start_time_offset == np.float32(-0.1)
            assert meta.ecg.samples.shape == (100,)
            assert meta.ecg.start_time_offset == np.float32(0.0)
            np.testing.assert_array_equal(meta.annotations.view, ["a4c"] * n_frames)

    def test_metrics_round_trip(self, tmp_path):
        n_frames, n_tx, n_el = 2, 3, 4
        path = tmp_path / "metrics.hdf5"
        cf = np.array([0.8, 0.9], dtype=np.float32)

        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((n_frames, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            metrics={"coherence_factor": cf},
        ).close()

        with File(path) as f:
            met = f.metrics()
            np.testing.assert_array_almost_equal(met.coherence_factor, cf)

    def test_metadata_raises_when_missing(self, tmp_path):
        """File without a metadata group raises KeyError."""
        path = tmp_path / "no_meta.hdf5"
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("dummy", data=[1])

        with File(path) as f:
            with pytest.raises(KeyError, match="metadata"):
                f.metadata()

    def test_metrics_raises_when_missing(self, tmp_path):
        """File without a metrics group raises KeyError."""
        path = tmp_path / "no_metrics.hdf5"
        import h5py

        with h5py.File(path, "w") as f:
            f.create_dataset("dummy", data=[1])

        with File(path) as f:
            with pytest.raises(KeyError, match="metrics"):
                f.metrics()


class TestZeaVersion:
    """Tests for the zea_version attribute written by File.create()."""

    def test_version_written_on_create(self, tmp_path):
        """File.create() stores a non-empty zea_version root attribute."""
        import zea

        path = tmp_path / "versioned.hdf5"
        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((2, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=2),
        ).close()

        with File(path) as f:
            assert f.zea_version == zea.__version__

    def test_legacy_file_has_no_version(self, tmp_path):
        """A hand-crafted file without the zea_version attr is treated as legacy."""
        import h5py

        path = tmp_path / "no_version.hdf5"
        with h5py.File(path, "w") as f:
            f.create_group("data")

        with File(path) as f:
            assert f.zea_version is None

    def test_validate_does_not_load_data(self, tmp_path):
        """validate() succeeds without loading array data (lightweight path)."""
        path = tmp_path / "validate_light.hdf5"
        File.create(
            path,
            data={"envelope_data": _make_map(np.ones((2, 8, 6), dtype=np.float32))},
            scan=_scan_minimal(n_frames=2),
        ).close()

        with File(path) as f:
            result = f.validate()
        assert result["status"] == "success"

    def test_validate_and_validate_spec_are_independent(self, tmp_path):
        """validate() does structural check; validate_spec() does full schema check."""
        path = tmp_path / "both.hdf5"
        n_frames, n_tx, n_el = 2, 3, 4
        File.create(
            path,
            data={"raw_data": np.ones((n_frames, n_tx, 8, n_el, 1), dtype=np.float32)},
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe_name="test_probe",
        ).close()

        with File(path) as f:
            # validate() returns a simple status dict
            assert f.validate() == {"status": "success"}
            # validate_spec() returns a rich FileSpec object
            spec = f.validate_spec()
            assert isinstance(spec, FileSpec)
            assert spec.data.raw_data.shape[0] == n_frames

    def test_legacy_file_validate_passes(self, tmp_path):
        """validate() works on a legacy file (no zea_version) that has image-only data
        (no scan group required for image-only legacy files)."""
        import h5py

        path = tmp_path / "legacy.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs["probe"] = "legacy_probe"
            g = f.create_group("data")
            # image-only legacy file: no scan group needed
            g.create_dataset("image_sc", data=np.zeros((2, 8, 6), dtype=np.float32))

        with File(path) as f:
            assert f.validate() == {"status": "success"}


def test_load_file_image_type(tmp_path):
    """load_file with data_type='image' must return the values array, not crash
    trying to slice an h5py.Group directly."""
    path = tmp_path / "with_image.hdf5"
    generate_example_dataset(
        path,
        add_optional_dtypes=True,
        n_frames=2,
        grid_size_z=8,
        grid_size_x=8,
        image_dtype=np.uint8,
    )

    data, scan, probe = load_file(path, data_type="image")
    assert isinstance(data, np.ndarray), "load_file should return ndarray for image type"
    assert data.shape[0] == 2, "should load all 2 frames"
