"""Tests for the File module."""

import warnings
from unittest.mock import patch

import h5py
import numpy as np
import pytest

import zea
from zea.data.data_format import generate_zea_dataset
from zea.data.file import File, Track, _GroupProxy, _StringDataset, load_file
from zea.data.legacy_file import dict_to_sorted_list
from zea.data.spec import FileSpec, Image, ScanSpec, Segmentation
from zea.probes import Probe
from zea.scan import Parameters

from . import generate_example_dataset


def _make_map(values):
    """Wrap values into a Map-compatible dict."""
    return {"values": values, "coordinates": np.zeros((*values.shape, 3), dtype=np.float32)}


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
        assert file.probe.name == FILE_PROBE_NAME, "Probe name should match expected value"
        assert isinstance(file.probe, Probe), "Probe should be an instance of Probe class"
        # load_parameters tolerates legacy files missing some spec fields and
        # returns a full (derivable) Parameters object.
        assert isinstance(file.load_parameters(), Parameters), (
            "load_parameters should return a Parameters object"
        )

        file.validate()


def test_image_only_dataset_load_parameters(tmp_path):
    """Image-only datasets carry no probe (or scan) group.

    ``File.probe`` should return an empty Probe rather than raising, and
    ``load_parameters`` should still return a Parameters object.
    """
    n_frames = 2
    fspec = FileSpec(
        data={
            "image": {
                "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                "coordinates": np.zeros((n_frames, 16, 12, 3), dtype=np.float32),
            },
        },
    )
    path = tmp_path / "image_only.hdf5"
    fspec.save(str(path))

    with File(path) as f:
        assert "probe" not in f.keys(), "Image-only file should have no probe group"
        assert f.scan is None, "Image-only file should have no scan group"

        probe = f.probe
        assert isinstance(probe, Probe), "probe should be an (empty) Probe instance"
        assert probe.get_parameters() == {}, "Empty probe should have no parameters"

        assert isinstance(f.load_parameters(), Parameters), (
            "load_parameters should return a Parameters object for image-only files"
        )


def test_load_file_function(dummy_file):
    """Test the load_file function."""
    selected_transmits = [0, 2, 4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data, scan = load_file(dummy_file, indices=(slice(2), selected_transmits))

    assert data.shape[0] == 2, "Data should have 2 frames"
    assert data.shape[1] == 3, "Data should have 3 selected transmits"
    assert isinstance(scan, Parameters), "load_file should return a Parameters object"
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
        probe={"name": "test_probe"},
        description="spec format test file",
    )
    path = tmp_path / "spec_format.hdf5"
    fspec.save(str(path))
    return str(path), fspec, raw, env


class TestStringDataset:
    def test_array_slice_decodes_bytes(self, tmp_path):
        path = tmp_path / "str_test.hdf5"
        labels = np.array(["foo", "bar", "baz"])
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=labels.astype(bytes))

        with h5py.File(path, "r") as f:
            ds = _StringDataset(f["labels"])
            result = ds[:]
            assert result.dtype.kind == "U"  # Unicode string dtype
            np.testing.assert_array_equal(result, labels)

    def test_scalar_access_returns_str(self, tmp_path):
        path = tmp_path / "str_scalar.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("s", data=np.bytes_(b"hello"))

        with h5py.File(path, "r") as f:
            ds = _StringDataset(f["s"])
            result = ds[()]
            assert isinstance(result, str)
            assert result == "hello"

    def test_len_and_repr(self, tmp_path):
        path = tmp_path / "str_len.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=np.array([b"a", b"b"]))

        with h5py.File(path, "r") as f:
            ds = _StringDataset(f["labels"])
            assert len(ds) == 2
            assert "StringDataset" in repr(ds)

    def test_getattr_delegates_to_dataset(self, tmp_path):
        path = tmp_path / "str_attr.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=np.array([b"x"]))

        with h5py.File(path, "r") as f:
            ds = _StringDataset(f["labels"])
            assert ds.shape == (1,)

    def test_auto_wrapped_via_group_proxy(self, tmp_path):
        """GroupProxy should auto-wrap string datasets in _StringDataset."""
        path = tmp_path / "proxy_str.hdf5"
        with h5py.File(path, "w") as f:
            grp = f.create_group("data")
            grp.create_dataset("labels", data=np.array([b"ED", b"ES"]))

        with h5py.File(path, "r") as f:
            proxy = _GroupProxy(f["data"])
            result = proxy.labels[:]
            assert isinstance(proxy.labels, _StringDataset)
            np.testing.assert_array_equal(result, np.array(["ED", "ES"]))

    def test_string_labels_decoded_via_zea_file(self, tmp_path):
        """Segmentation labels written via File.create should be auto-decoded to strings."""
        n_frames = 2
        seg_labels = np.array(["background", "lumen"], dtype=np.str_)
        seg_values = np.zeros((n_frames, 8, 8, 2), dtype=np.bool_)
        path = tmp_path / "seg_str.hdf5"
        File.create(
            path,
            data={
                "segmentation": {
                    "values": seg_values,
                    "labels": seg_labels,
                    "coordinates": np.zeros((8, 8, 3), dtype=np.float32),
                },
            },
            scan=_scan_minimal(n_frames=n_frames),
            probe={"name": "test"},
        ).close()

        with File(path) as f:
            labels_ds = f.data.segmentation.labels
            assert isinstance(labels_ds, _StringDataset)
            result = labels_ds[:]
            assert result.dtype.kind == "U"
            np.testing.assert_array_equal(result, seg_labels)


class TestGroupProxy:
    def test_attribute_access_returns_dataset(self, spec_file):
        path, _, raw, _ = spec_file
        with File(path) as f:
            ds = f.data.raw_data
            assert isinstance(ds, h5py.Dataset)
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
                    "coordinates": np.zeros((n_frames, 16, 12, 3), dtype=np.float32),
                },
            },
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
        )
        path = tmp_path / "nested.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            proxy = f.data.image
            assert isinstance(proxy, _GroupProxy)
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

    def test_repr_delegates_to_h5py(self, spec_file):
        """_GroupProxy repr shows the underlying HDF5 group path."""
        path, *_ = spec_file
        with File(path) as f:
            r = repr(f.data)
        assert "HDF5 group" in r
        assert "data" in r


class TestFileDataProperty:
    def test_data_property_returns_group_proxy(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            assert isinstance(f.data, _GroupProxy)

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
        assert loaded_spec.probe.name == "test_probe"
        assert loaded_spec.description == "spec format test file"

    def test_validate_spec_on_complete_legacy_file(self, tmp_path):
        """validate_spec() succeeds on legacy files that have all required scan
        fields plus the extra scalar datasets (n_frames, n_tx, etc.)."""
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
            # Legacy flat image is now wrapped as Map with values; coordinates is None
            assert spec.data.image is not None
            np.testing.assert_array_equal(spec.data.image.values, img)
            assert spec.data.image.coordinates is None
            # probe attr mapped to probe.name
            assert spec.probe.name == "legacy_probe"

    def test_validate_spec_raises_on_incomplete_legacy_file(self, tmp_path):
        """validate_spec() raises on legacy files missing required scan fields."""
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
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 2, 4, 8, 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fspec = FileSpec(
                data={
                    "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32),
                    "custom_map": {
                        "values": np.zeros((n_frames, 16, 12, 1), dtype=np.uint8),
                        "coordinates": np.zeros((n_frames, 16, 12, 3), dtype=np.float32),
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
            rd_ds = f.data.raw_data
            assert rd_ds.attrs["unit"] == "-"
            assert rd_ds.attrs["description"] != ""

            # Check scan field metadata (t0_delays is always present)
            td_ds = f._scan_h5_group["t0_delays"]
            assert td_ds.attrs["unit"] == "s"

    def test_scan_field_metadata_matches_spec(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            for key in f._scan_h5_group.keys():
                ds = f._scan_h5_group[key]
                assert "unit" in ds.attrs, f"Missing 'unit' on scan/{key}"
                assert "description" in ds.attrs, f"Missing 'description' on scan/{key}"


class TestProbeNameCompat:
    def test_probe_name_from_spec_format(self, spec_file):
        path, *_ = spec_file

        with File(path) as f:
            assert f.probe.name == "test_probe"

    def test_probe_name_from_legacy_format(self, dummy_file):
        """Legacy files use 'probe' attr; File.probe_name handles both."""
        with File(dummy_file) as f:
            assert f.probe.name == "generic"


class TestImageOnlyFile:
    def test_image_only_spec_file(self, tmp_path):
        """FileSpec and File work for files with only image data (no raw_data)."""
        n_frames = 2
        fspec = FileSpec(
            data={
                "image": {
                    "values": np.zeros((n_frames, 32, 24, 1), dtype=np.uint8),
                    "coordinates": np.zeros((n_frames, 32, 24, 3), dtype=np.float32),
                },
            },
            scan=_scan_minimal(n_frames=n_frames),
        )
        path = tmp_path / "image_only.hdf5"
        fspec.save(str(path))

        with File(path) as f:
            assert "image" in f.data
            proxy = f.data.image
            assert isinstance(proxy, _GroupProxy)
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
            "aligned_data": {
                "values": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
            },
            "beamformed_data": _make_map(np.zeros((n_frames, 16, 12, n_ch), dtype=np.float32)),
            "envelope_data": _make_map(np.zeros((n_frames, 16, 12), dtype=np.float32)),
            "image_sc": _make_map(np.zeros((n_frames, 32, 24), dtype=np.uint8)),
        }
        fspec = FileSpec(
            data=data_dict,
            scan=_scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el),
            probe={"name": "all_pipeline"},
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
            probe={"name": "slice_test"},
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
    """Test saving + reading spatial maps that include values + coordinates."""

    @pytest.fixture
    def spatial_file(self, tmp_path):
        n_frames = 3
        img_values = np.random.randint(0, 255, (n_frames, 64, 48, 1), dtype=np.uint8)
        img_coordinates = np.zeros((n_frames, 64, 48, 3), dtype=np.float32)
        seg_values = np.random.choice([True, False], (n_frames, 64, 48, 1, 2)).astype(np.bool_)
        seg_labels = np.array(["background", "lumen"], dtype=np.str_)
        seg_coordinates = np.zeros((n_frames, 64, 48, 1, 3), dtype=np.float32)
        sos_values = np.full((n_frames, 64, 48, 1), 1540.0, dtype=np.float32)
        sos_coordinates = np.zeros((n_frames, 64, 48, 3), dtype=np.float32)

        path = tmp_path / "spatial.hdf5"
        f = File.create(
            path,
            data={
                "envelope_data": _make_map(np.ones((n_frames, 32, 24), dtype=np.float32)),
                "image": {"values": img_values, "coordinates": img_coordinates},
                "segmentation": {
                    "values": seg_values,
                    "labels": seg_labels,
                    "coordinates": seg_coordinates,
                },
                "sos_map": {"values": sos_values, "coordinates": sos_coordinates},
            },
            scan=_scan_minimal(n_frames=n_frames),
            probe={"name": "spatial_test"},
        )
        f.close()
        return (
            str(path),
            img_values,
            img_coordinates,
            seg_values,
            seg_labels,
            sos_values,
        )

    def test_image_group_structure(self, spatial_file):
        path, img_values, img_coordinates, *_ = spatial_file
        with File(path) as f:
            proxy = f.data.image
            assert isinstance(proxy, _GroupProxy)
            assert "values" in proxy
            assert "coordinates" in proxy

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
            loaded_labels = f.data.segmentation.labels[:]
            np.testing.assert_array_equal(loaded_labels, seg_labels)

    def test_sos_map_values(self, spatial_file):
        path, *_, sos_values = spatial_file
        with File(path) as f:
            np.testing.assert_allclose(f.data.sos_map.values[()], sos_values, atol=1e-6)

    def test_spatial_round_trip_via_validate_spec(self, spatial_file):
        path, img_values, img_coordinates, seg_values, seg_labels, sos_values = spatial_file
        with File(path) as f:
            spec = f.validate_spec()

        assert isinstance(spec.data.image, Image)
        np.testing.assert_array_equal(spec.data.image.values, img_values)
        np.testing.assert_array_equal(spec.data.image.coordinates, img_coordinates)
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
            probe={"name": "create_test"},
            description="created via File.create",
        )
        assert f.mode == "r"
        np.testing.assert_array_equal(f.data.raw_data[()], raw)
        assert f.probe.name == "create_test"
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
            meta = f.metadata
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
            met = f.metrics
            np.testing.assert_array_almost_equal(met.coherence_factor, cf)

    def test_metadata_raises_when_missing(self, tmp_path):
        """File without a metadata group raises KeyError."""
        path = tmp_path / "no_meta.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("dummy", data=[1])

        with File(path) as f:
            with pytest.raises(KeyError, match="metadata"):
                _ = f.metadata

    def test_metrics_raises_when_missing(self, tmp_path):
        """File without a metrics group raises KeyError."""
        path = tmp_path / "no_metrics.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("dummy", data=[1])

        with File(path) as f:
            with pytest.raises(KeyError, match="metrics"):
                _ = f.metrics


class TestZeaVersion:
    """Tests for the zea_version attribute written by File.create()."""

    def test_version_written_on_create(self, tmp_path):
        """File.create() stores a non-empty zea_version root attribute."""
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
        path = tmp_path / "no_version.hdf5"
        with h5py.File(path, "w") as f:
            f.create_group("data")

        with File(path) as f:
            assert f.zea_version is None

    def test_legacy_warning_no_version(self, tmp_path):
        """Opening a file with no zea_version emits a legacy warning."""
        path = tmp_path / "no_version.hdf5"
        with h5py.File(path, "w") as f:
            f.create_group("data")

        with patch("zea.data.file.log.warning") as mock_warn:
            with File(path):
                pass
        mock_warn.assert_called_once()
        assert "legacy" in mock_warn.call_args.args[0].lower()

    def test_legacy_warning_old_version(self, tmp_path):
        """Opening a file with zea_version < 0.1.0 emits a legacy warning."""
        path = tmp_path / "old_version.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs["zea_version"] = "0.0.13"

        with patch("zea.data.file.log.warning") as mock_warn:
            with File(path):
                pass
        mock_warn.assert_called_once()
        assert "legacy" in mock_warn.call_args.args[0].lower()

    def test_no_legacy_warning_current_version(self, tmp_path):
        """Opening a file with zea_version >= 0.1.0 does not emit a legacy warning."""
        path = tmp_path / "current.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs["zea_version"] = "0.1.0"

        with patch("zea.data.file.log.warning") as mock_warn:
            with File(path):
                pass
        mock_warn.assert_not_called()

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
            probe={"name": "test_probe"},
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        data, scan = load_file(path, data_type="image")
    assert isinstance(data, np.ndarray), "load_file should return ndarray for image type"
    assert data.shape[0] == 2, "should load all 2 frames"


# ---------------------------------------------------------------------------
# Helpers shared by multi-track tests
# ---------------------------------------------------------------------------


def _make_two_track_spec(tmp_path, n_frames=2, n_tx=3, n_el=4, n_ax=8, n_ch=1):
    """Build and save a two-track file via File.create; return (path, raw_a, raw_b)."""
    raw_a = np.arange(n_frames * n_tx * n_ax * n_el * n_ch, dtype=np.float32).reshape(
        n_frames, n_tx, n_ax, n_el, n_ch
    )
    raw_b = raw_a * 2

    scan = _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el)
    path = tmp_path / "two_tracks.hdf5"
    f = File.create(
        path,
        tracks=[
            {"data": {"raw_data": raw_a}, "scan": scan, "label": "track_a"},
            {"data": {"raw_data": raw_b}, "scan": scan, "label": "track_b"},
        ],
        probe={"name": "two_track_probe"},
    )
    f.close()
    return path, raw_a, raw_b


class TestRepr:
    """Tests for __repr__ / __str__ of File, Track, _StringDataset."""

    def test_file_repr_single_track(self, tmp_path):
        """Single-track file repr shows filename, mode and '1 track'."""
        path = tmp_path / "single.hdf5"
        File.create(
            path,
            data={"raw_data": np.zeros((1, 2, 8, 4, 1), dtype=np.float32)},
            scan=_scan_minimal(n_frames=1, n_tx=2, n_el=4),
        ).close()
        with File(path) as f:
            r = repr(f)
        assert r.startswith('<File "')
        assert "single.hdf5" in r
        assert "mode r" in r
        assert "1 track" in r

    def test_file_repr_multi_track_with_labels(self, tmp_path):
        """Multi-track repr includes track count and label names."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            r = repr(f)
        assert "2 tracks" in r
        assert '"track_a"' in r
        assert '"track_b"' in r

    def test_file_str_equals_repr(self, tmp_path):
        path = tmp_path / "s.hdf5"
        File.create(
            path,
            data={"raw_data": np.zeros((1, 2, 8, 4, 1), dtype=np.float32)},
            scan=_scan_minimal(n_frames=1, n_tx=2, n_el=4),
        ).close()
        with File(path) as f:
            assert repr(f) == str(f)

    def test_track_repr_with_label(self, tmp_path):
        """Track repr shows index, label, and data keys."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            r = repr(f.tracks[0])
        assert r.startswith("<Track[0]")
        assert '"track_a"' in r
        assert "data=" in r
        assert "raw_data" in r

    def test_track_repr_without_label(self, tmp_path):
        """Track repr omits label part when track has no label."""
        path = tmp_path / "nolabel.hdf5"
        File.create(
            path,
            data={"raw_data": np.zeros((1, 2, 8, 4, 1), dtype=np.float32)},
            scan=_scan_minimal(n_frames=1, n_tx=2, n_el=4),
        ).close()
        with File(path) as f:
            r = repr(f.tracks[0])
        assert "<Track[0]" in r
        assert "data=" in r
        # no spurious quote from a missing label
        assert r.count('"') == 0 or r.startswith("<Track[0] data=")

    def test_string_dataset_repr(self, tmp_path):
        """_StringDataset repr mentions shape and str dtype."""
        path = tmp_path / "sd.hdf5"
        with h5py.File(path, "w") as f:
            f.create_dataset("labels", data=np.array([b"a", b"b"]))
        with h5py.File(path, "r") as f:
            ds = _StringDataset(f["labels"])
            r = repr(ds)
        assert "StringDataset" in r
        assert "shape" in r
        assert "str" in r


class TestMultiTrackFile:
    """Tests for File.tracks, Track, and single-track guards."""

    # ------------------------------------------------------------------
    # File.tracks property
    # ------------------------------------------------------------------

    def test_tracks_returns_list_of_track_proxies(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            tracks = f.tracks
        assert len(tracks) == 2
        assert all(isinstance(t, Track) for t in tracks)

    def test_tracks_single_track_file_returns_one_proxy(self, tmp_path):
        """A single-track new-format file exposes one Track."""
        raw = np.zeros((2, 3, 8, 4, 1), dtype=np.float32)
        path = tmp_path / "single_track.hdf5"
        f = File.create(
            path,
            data={"raw_data": raw},
            scan=_scan_minimal(n_frames=2, n_tx=3, n_el=4),
        )
        f.close()

        with File(path) as f:
            tracks = f.tracks
        assert len(tracks) == 1
        assert isinstance(tracks[0], Track)

    def test_tracks_raises_for_legacy_flat_file(self, tmp_path):
        """Legacy files (no tracks/ group) raise AttributeError on .tracks."""
        import h5py

        path = tmp_path / "legacy.hdf5"
        with h5py.File(path, "w") as f:
            g = f.create_group("data")
            g.create_dataset("raw_data", data=np.zeros((1, 2, 8, 4, 1), dtype=np.float32))

        with File(path) as f:
            with pytest.raises(AttributeError, match="legacy"):
                _ = f.tracks

    # ------------------------------------------------------------------
    # Track.data and Track.scan
    # ------------------------------------------------------------------

    def test_track_data_returns_correct_array(self, tmp_path):
        path, raw_a, raw_b = _make_two_track_spec(tmp_path)
        with File(path) as f:
            tracks = f.tracks
            loaded_a = tracks[0].data.raw_data[:]
            loaded_b = tracks[1].data.raw_data[:]
        np.testing.assert_array_equal(loaded_a, raw_a)
        np.testing.assert_array_equal(loaded_b, raw_b)

    def test_track_data_is_group_proxy(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            assert isinstance(f.tracks[0].data, _GroupProxy)

    def test_track_scan_returns_scan_object(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            scan = f.tracks[0].scan
        assert isinstance(scan, ScanSpec)

    def test_track_scan_kwargs_override(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            scan = f.tracks[0].load_parameters(sound_speed=np.float32(1480.0))
        assert float(scan.sound_speed) == pytest.approx(1480.0)

    def test_track_repr(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            r = repr(f.tracks[1])
        assert r.startswith("<Track[1]")
        assert "data=" in r

    def test_track_repr_includes_label(self, tmp_path):
        """repr(track) includes the label when one is set."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            r = repr(f.tracks[0])
        assert '"track_a"' in r

    # ------------------------------------------------------------------
    # Track.label, File.track_labels, File.get_track
    # ------------------------------------------------------------------

    def test_track_label_roundtrip(self, tmp_path):
        """Labels written to HDF5 are read back correctly on each Track."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            assert f.tracks[0].label == "track_a"
            assert f.tracks[1].label == "track_b"

    def test_track_labels_property(self, tmp_path):
        """File.track_labels returns labels in acquisition order."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            assert f.track_labels == ["track_a", "track_b"]

    def test_get_track_returns_correct_track(self, tmp_path):
        """File.get_track returns the track whose label matches."""
        path, raw_a, raw_b = _make_two_track_spec(tmp_path)
        with File(path) as f:
            t = f.get_track("track_b")
            assert t.label == "track_b"
            np.testing.assert_array_equal(t.data.raw_data[:], raw_b)

    def test_get_track_missing_label_raises(self, tmp_path):
        """File.get_track raises KeyError with available labels in the message."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            with pytest.raises(KeyError, match="track_a"):
                f.get_track("nonexistent")

    def test_filespec_multi_track_missing_label_raises(self):
        """FileSpec raises ValueError when any track in a multi-track file has no label."""
        raw = np.zeros((1, 2, 8, 4, 1), dtype=np.float32)
        scan = _scan_minimal(n_frames=1, n_tx=2, n_el=4)
        with pytest.raises(ValueError, match="label"):
            FileSpec(
                tracks=[
                    {"data": {"raw_data": raw}, "scan": scan, "label": "track_a"},
                    {"data": {"raw_data": raw}, "scan": scan},  # missing label
                ]
            )

    def test_single_track_label_is_optional(self, tmp_path):
        """A single-track file does not require a label."""
        raw = np.zeros((2, 3, 8, 4, 1), dtype=np.float32)
        path = tmp_path / "single_no_label.hdf5"
        File.create(
            path,
            data={"raw_data": raw},
            scan=_scan_minimal(n_frames=2, n_tx=3, n_el=4),
        )
        with File(path) as f:
            assert f.tracks[0].label is None

    # ------------------------------------------------------------------
    # Guards on File.data and File.scan() for multi-track files
    # ------------------------------------------------------------------

    def test_file_data_raises_for_multi_track(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            with pytest.raises(AttributeError, match="2 tracks"):
                _ = f.data

    def test_file_scan_raises_for_multi_track(self, tmp_path):
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            with pytest.raises(AttributeError, match="2 tracks"):
                _ = f.scan

    def test_error_message_mentions_tracks_property(self, tmp_path):
        """The error on file.data tells the user to use file.tracks."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            with pytest.raises(AttributeError, match="file.tracks"):
                _ = f.data
        with File(path) as f:
            with pytest.raises(AttributeError, match="file.tracks"):
                _ = f.scan

    # ------------------------------------------------------------------
    # Single-track files: backwards-compatible access still works
    # ------------------------------------------------------------------

    def test_single_track_data_still_works(self, tmp_path):
        """file.data works unchanged for single-track new-format files."""
        raw = np.ones((2, 3, 8, 4, 1), dtype=np.float32)
        path = tmp_path / "single.hdf5"
        File.create(
            path,
            data={"raw_data": raw},
            scan=_scan_minimal(n_frames=2, n_tx=3, n_el=4),
        ).close()

        with File(path) as f:
            np.testing.assert_array_equal(f.data.raw_data[:], raw)

    def test_single_track_scan_still_works(self, tmp_path):
        """file.scan() works unchanged for single-track new-format files."""
        path = tmp_path / "single_scan.hdf5"
        File.create(
            path,
            data={"raw_data": np.zeros((2, 3, 8, 4, 1), dtype=np.float32)},
            scan=_scan_minimal(n_frames=2, n_tx=3, n_el=4),
        ).close()

        with File(path) as f:
            scan = f.scan
        assert isinstance(scan, ScanSpec)
        assert scan.n_tx == 3

    # ------------------------------------------------------------------
    # Probe: file-level access and track isolation
    # ------------------------------------------------------------------

    def test_track_scan_includes_file_level_probe_geometry(self, tmp_path):
        """probe_geometry from the file-level probe group is merged into track.load_parameters()."""
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        geom = np.arange(n_el * 3, dtype=np.float32).reshape(n_el, 3) * 1e-3
        scan = _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el)
        path = tmp_path / "probe_geom.hdf5"
        File.create(
            path,
            tracks=[
                {
                    "data": {
                        "raw_data": np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
                    },
                    "scan": scan,
                    "label": "track_a",
                },
                {
                    "data": {
                        "raw_data": np.ones((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
                    },
                    "scan": scan,
                    "label": "track_b",
                },
            ],
            probe={"probe_geometry": geom},
        ).close()

        with File(path) as f:
            for track in f.tracks:
                np.testing.assert_array_equal(track.load_parameters().probe_geometry, geom)

    def test_track_has_no_probe_attribute(self, tmp_path):
        """Track exposes no .probe attribute; probe is accessed via File.probe."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            track = f.tracks[0]
            with pytest.raises(AttributeError):
                _ = track.probe

    # ------------------------------------------------------------------
    # Dict-format track inputs
    # ------------------------------------------------------------------

    def test_multi_track_from_dicts(self, tmp_path):
        n_frames, n_tx, n_el, n_ax, n_ch = 2, 3, 4, 8, 1
        raw_a = np.zeros((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
        raw_b = np.ones((n_frames, n_tx, n_ax, n_el, n_ch), dtype=np.float32)
        scan = _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el)

        path = tmp_path / "dict_tracks.hdf5"
        File.create(
            path,
            tracks=[
                {"data": {"raw_data": raw_a}, "scan": scan, "label": "track_a"},
                {"data": {"raw_data": raw_b}, "scan": scan, "label": "track_b"},
            ],
        ).close()

        with File(path) as f:
            tracks = f.tracks
            assert len(tracks) == 2
            np.testing.assert_array_equal(tracks[0].data.raw_data[:], raw_a)
            np.testing.assert_array_equal(tracks[1].data.raw_data[:], raw_b)

    # ------------------------------------------------------------------
    # track_schedule: storage and retrieval
    # ------------------------------------------------------------------

    def _make_scheduled_file(self, tmp_path, n_frames=2, n_tx_a=3, n_tx_b=2, n_el=4, n_ax=8):
        """Two-track file with an interleaved track_schedule and distinct t2nt values."""
        n_ch = 1
        raw_a = np.zeros((n_frames, n_tx_a, n_ax, n_el, n_ch), dtype=np.float32)
        raw_b = np.ones((n_frames, n_tx_b, n_ax, n_el, n_ch), dtype=np.float32)

        dt_a = np.full((n_frames, n_tx_a), 0.1, dtype=np.float32)
        dt_b = np.full((n_frames, n_tx_b), 0.05, dtype=np.float32)

        scan_a = _scan_minimal(n_frames=n_frames, n_tx=n_tx_a, n_el=n_el)
        scan_a["time_to_next_transmit"] = dt_a
        scan_b = _scan_minimal(n_frames=n_frames, n_tx=n_tx_b, n_el=n_el)
        scan_b["time_to_next_transmit"] = dt_b

        # Per-frame interleaving: a0 b0 a1 b1 a2, tiled for all n_frames
        schedule = np.tile(np.array([0, 1, 0, 1, 0], dtype=np.int32), n_frames)

        path = tmp_path / "scheduled.hdf5"
        File.create(
            path,
            tracks=[
                {"data": {"raw_data": raw_a}, "scan": scan_a, "label": "track_a"},
                {"data": {"raw_data": raw_b}, "scan": scan_b, "label": "track_b"},
            ],
            track_schedule=schedule,
        ).close()
        return path, schedule, dt_a, dt_b

    def test_track_schedule_stored_and_loaded(self, tmp_path):
        """File.track_schedule returns the stored int32 array."""
        path, schedule, *_ = self._make_scheduled_file(tmp_path)
        with File(path) as f:
            loaded = f.track_schedule
        assert loaded is not None
        np.testing.assert_array_equal(loaded, schedule)
        assert loaded.dtype == np.int32

    def test_track_schedule_none_when_absent(self, tmp_path):
        """File.track_schedule returns None for files without a schedule."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            assert f.track_schedule is None

    def test_track_schedule_invalid_indices_raises(self, tmp_path):
        """FileSpec raises ValueError when schedule indices exceed track count."""
        raw = np.zeros((1, 2, 8, 4, 1), dtype=np.float32)
        scan = _scan_minimal(n_frames=1, n_tx=2, n_el=4)
        schedule_bad = np.array([0, 1, 2], dtype=np.int32)  # index 2 out of range for 2 tracks

        with pytest.raises(ValueError, match="track_schedule"):
            FileSpec(
                tracks=[
                    {"data": {"raw_data": raw}, "scan": scan, "label": "track_a"},
                    {"data": {"raw_data": raw}, "scan": scan, "label": "track_b"},
                ],
                track_schedule=schedule_bad,
            )

    def test_track_schedule_valid_does_not_raise(self, tmp_path):
        """FileSpec accepts a schedule whose indices are all in range."""
        raw = np.zeros((1, 2, 8, 4, 1), dtype=np.float32)
        scan = _scan_minimal(n_frames=1, n_tx=2, n_el=4)
        schedule = np.array([0, 1, 0, 1], dtype=np.int32)

        spec = FileSpec(
            tracks=[
                {"data": {"raw_data": raw}, "scan": scan, "label": "track_a"},
                {"data": {"raw_data": raw}, "scan": scan, "label": "track_b"},
            ],
            track_schedule=schedule,
        )
        path = tmp_path / "valid_schedule.hdf5"
        spec.save(str(path))  # should not raise

    # ------------------------------------------------------------------
    # Track.timestamps
    # ------------------------------------------------------------------

    def test_track_timestamps_none_without_schedule(self, tmp_path):
        """timestamps is None when the file has no track_schedule."""
        path, *_ = _make_two_track_spec(tmp_path)
        with File(path) as f:
            assert f.tracks[0].timestamps is None

    def test_track_timestamps_none_without_time_to_next_transmit(self, tmp_path):
        """timestamps is None when a track's scan has no time_to_next_transmit."""
        n_frames, n_tx, n_el, n_ax = 1, 2, 4, 8
        raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)

        # Build a scan dict without time_to_next_transmit
        scan_no_t2nt = _scan_minimal(n_frames=n_frames, n_tx=n_tx, n_el=n_el)
        del scan_no_t2nt["time_to_next_transmit"]

        spec = FileSpec(
            tracks=[
                {"data": {"raw_data": raw}, "scan": scan_no_t2nt, "label": "track_a"},
                {"data": {"raw_data": raw}, "scan": scan_no_t2nt, "label": "track_b"},
            ],
            track_schedule=np.array([0, 1, 0, 1], dtype=np.int32),
        )
        path = tmp_path / "no_t2nt.hdf5"
        spec.save(str(path))

        with File(path) as f:
            assert f.tracks[0].timestamps is None

    def test_track_timestamps_shape(self, tmp_path):
        """timestamps has shape (n_frames, n_tx_for_that_track)."""
        path, schedule, *_ = self._make_scheduled_file(tmp_path, n_frames=2, n_tx_a=3, n_tx_b=2)
        with File(path) as f:
            ts_a = f.tracks[0].timestamps
            ts_b = f.tracks[1].timestamps

        # schedule covers n_frames full cycles: 2*(3+2)=10 events total
        assert ts_a.shape == (2, 3)
        assert ts_b.shape == (2, 2)

    def test_track_timestamps_values_correct(self, tmp_path):
        """Timestamps equal cumulative sums of time_to_next_transmit across all tracks.

        Schedule [0,1,0,1,0] with dt_a=0.1, dt_b=0.05:
          global events:  0      1      2      3      4
          track index:    0      1      0      1      0
          cumtime:        0   +0.1  +0.05  +0.1  +0.05   → [0, 0.1, 0.15, 0.25, 0.30]

          track 0 fires at positions 0, 2, 4 → timestamps [0, 0.15, 0.30]
          track 1 fires at positions 1, 3   → timestamps [0.1, 0.25]
        """
        path, _, dt_a, dt_b = self._make_scheduled_file(tmp_path, n_frames=1, n_tx_a=3, n_tx_b=2)
        with File(path) as f:
            ts_a = f.tracks[0].timestamps  # (1, 3)
            ts_b = f.tracks[1].timestamps  # (1, 2)

        expected_a = np.array([[0.0, 0.15, 0.30]])
        expected_b = np.array([[0.1, 0.25]])

        np.testing.assert_allclose(ts_a, expected_a, atol=1e-6)
        np.testing.assert_allclose(ts_b, expected_b, atol=1e-6)

    def test_track_timestamps_monotonically_increasing(self, tmp_path):
        """Each track's timestamps are strictly increasing across frames."""
        path, *_ = self._make_scheduled_file(tmp_path, n_frames=3, n_tx_a=3, n_tx_b=2)
        with File(path) as f:
            for track in f.tracks:
                ts = track.timestamps
                assert ts is not None
                assert np.all(np.diff(ts.ravel()) > 0), f"Non-monotonic timestamps: {ts}"

    def test_track_timestamps_frame_invariant(self, tmp_path):
        """When t2nt is identical across frames, frame-to-frame increments are constant."""
        path, *_ = self._make_scheduled_file(tmp_path, n_frames=4, n_tx_a=3, n_tx_b=2)
        with File(path) as f:
            ts = f.tracks[0].timestamps  # (4, 3)
        # Each frame starts exactly one frame-period later than the previous;
        # with constant dt the increment is the same for every row.
        frame_diffs = np.diff(ts, axis=0)  # (3, 3)
        # All frame-to-frame increments are equal when dt is constant
        np.testing.assert_allclose(frame_diffs[1:], frame_diffs[:-1], atol=1e-5)

    def test_track_timestamps_unequal_frame_counts(self, tmp_path):
        """Timestamps are computed correctly when tracks have different n_frames.

        Track A has 3 frames (n_tx=3), track B has 2 frames (n_tx=2).
                Schedule [0,0,1,0,0,1,0,0,1,0,0,1,0] with dt_a=0.1, dt_b=0.05:
          - Track A result shape: (3, 3)
          - Track B result shape: (2, 2)
                Values are compared against a fixed expected matrix for each track.
        """
        n_el, n_ax = 4, 8
        n_frames_a, n_tx_a = 3, 3
        n_frames_b, n_tx_b = 2, 2

        raw_a = np.zeros((n_frames_a, n_tx_a, n_ax, n_el, 1), dtype=np.float32)
        raw_b = np.ones((n_frames_b, n_tx_b, n_ax, n_el, 1), dtype=np.float32)

        dt_a = np.full((n_frames_a, n_tx_a), 0.1, dtype=np.float32)
        dt_b = np.full((n_frames_b, n_tx_b), 0.05, dtype=np.float32)

        scan_a = _scan_minimal(n_frames=n_frames_a, n_tx=n_tx_a, n_el=n_el)
        scan_a["time_to_next_transmit"] = dt_a
        scan_b = _scan_minimal(n_frames=n_frames_b, n_tx=n_tx_b, n_el=n_el)
        scan_b["time_to_next_transmit"] = dt_b

        schedule = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.int32)

        path = tmp_path / "unequal_frames.hdf5"
        File.create(
            path,
            tracks=[
                {"data": {"raw_data": raw_a}, "scan": scan_a, "label": "track_a"},
                {"data": {"raw_data": raw_b}, "scan": scan_b, "label": "track_b"},
            ],
            track_schedule=schedule,
        ).close()

        with File(path) as f:
            ts_a = f.tracks[0].timestamps
            ts_b = f.tracks[1].timestamps

        assert ts_a.shape == (n_frames_a, n_tx_a), (
            f"Expected ({n_frames_a}, {n_tx_a}), got {ts_a.shape}"
        )
        assert ts_b.shape == (n_frames_b, n_tx_b), (
            f"Expected ({n_frames_b}, {n_tx_b}), got {ts_b.shape}"
        )

        expected_a = np.array(
            [
                [0.0, 0.1, 0.25],
                [0.35, 0.5, 0.6],
                [0.75000006, 0.8500001, 1.0000001],
            ],
            dtype=np.float32,
        )
        expected_b = np.array(
            [
                [0.2, 0.45],
                [0.70000005, 0.9500001],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(ts_a, expected_a, atol=1e-6)
        np.testing.assert_allclose(ts_b, expected_b, atol=1e-6)


class TestLegacyFileLoading:
    """Integration tests: legacy files written with generate_zea_dataset load correctly."""

    @pytest.fixture()
    def legacy_file(self, tmp_path):
        """Create a minimal legacy file using the deprecated writer."""
        n_frames, n_tx, n_el, n_ax = 2, 4, 16, 64
        raw = np.random.randn(n_frames, n_tx, n_ax, n_el, 1).astype(np.float32)
        # uint8 is the typical legacy scan-converted image format
        image_sc = np.random.randint(0, 256, (n_frames, 128, 96), dtype=np.uint8)
        path = tmp_path / "legacy.hdf5"
        generate_zea_dataset(
            path=str(path),
            raw_data=raw,
            image_sc=image_sc,
            probe_geometry=np.zeros((n_el, 3), dtype=np.float32),
            sampling_frequency=np.float32(40e6),
            center_frequency=np.float32(7e6),
            demodulation_frequency=np.float32(7e6),
            initial_times=np.zeros(n_tx, dtype=np.float32),
            t0_delays=np.zeros((n_tx, n_el), dtype=np.float32),
            tx_apodizations=np.ones((n_tx, n_el), dtype=np.float32),
            focus_distances=np.zeros(n_tx, dtype=np.float32),
            transmit_origins=np.zeros((n_tx, 3), dtype=np.float32),
            polar_angles=np.zeros(n_tx, dtype=np.float32),
            probe_name="legacy_probe",
            cast_to_float=False,
        )
        return path, raw, image_sc

    def test_legacy_warning_fires(self, legacy_file):
        """Opening a legacy file emits the version warning."""
        path, *_ = legacy_file
        zea.log._warned_locations.clear()
        with patch("zea.data.file.log.warning") as mock_warn:
            with File(path):
                pass
        mock_warn.assert_called_once()
        assert "legacy" in mock_warn.call_args.args[0].lower()

    def test_probe_name_mapped(self, legacy_file):
        """probe_name is resolved from the legacy 'probe' root attribute."""
        path, *_ = legacy_file
        with File(path) as f:
            assert f.probe.name == "legacy_probe"
            spec = f.validate_spec()
        assert spec.probe.name == "legacy_probe"

    def test_raw_data_loaded(self, legacy_file):
        """raw_data array is loaded with the correct shape."""
        path, raw, _ = legacy_file
        with File(path) as f:
            spec = f.validate_spec()
        assert spec.data.raw_data.shape == raw.shape

    def test_flat_image_sc_wrapped_as_values(self, legacy_file):
        """Flat legacy image_sc is loaded as an extra Map; coordinates is None."""
        path, _, image_sc = legacy_file
        with patch("zea.data.spec.log.warning"):
            with File(path) as f:
                spec = f.validate_spec()
        assert "image_sc" in spec.data._extra_map_keys
        assert spec.data.image_sc is not None
        np.testing.assert_array_equal(spec.data.image_sc.values, image_sc)
        assert spec.data.image_sc.coordinates is None

    def test_scalar_scan_fields_ignored(self, legacy_file):
        """Redundant scalar scan fields (n_frames, n_tx, etc.) are silently filtered."""
        path, *_ = legacy_file
        with File(path) as f:
            spec = f.validate_spec()  # would raise if scalars caused unexpected-kwarg errors
        assert spec.scan is not None
