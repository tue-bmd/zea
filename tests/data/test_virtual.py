"""Tests for the virtual (Zarr) read path: ``zea.data.virtual`` and ``Dataset(lazy='virtual')``."""

import json

import numpy as np
import pytest

from zea.data.datasets import Dataset
from zea.data.file import File

from . import generate_dummy_scan

pytest.importorskip("virtualizarr", reason="needs the 'zea[virtual]' extra")

from zea.data import virtual  # noqa: E402  (after importorskip)
from zea.data.virtual import (  # noqa: E402
    VirtualFile,
    build_virtual_reference,
    open_virtual_file,
    open_virtual_reference,
)

N_TX, N_AX, N_EL = 3, 64, 16


def _compressible_raw(n_frames, seed=0):
    """RF-like data that Blosc can actually shrink.

    Incompressible chunks are stored raw by HDF5 and deliberately left out of a
    reference (see :func:`test_incompressible_array_is_skipped`), so data used to test
    the happy path must compress.
    """
    rng = np.random.default_rng(seed)
    depth = np.linspace(0, 1, N_AX, dtype=np.float32)[None, None, :, None, None]
    noise = rng.standard_normal((n_frames, N_TX, 1, N_EL, 1)).astype(np.float32)
    return (np.exp(-2.5 * depth) * np.sin(40 * depth) * (1 + 0.1 * noise)).astype(np.float32)


def _write_file(path, raw, image=None, **kwargs):
    data = {"raw_data": raw}
    if image is not None:
        n_frames, height, width = image.shape
        data["image"] = {
            "values": image,
            "coordinates": np.zeros((n_frames, height, width, 3), dtype=np.float32),
        }
    File.create(
        path,
        data=data,
        scan=generate_dummy_scan(n_tx=N_TX, n_el=N_EL),
        probe={"name": "generic", "probe_geometry": np.zeros((N_EL, 3), dtype=np.float32)},
        overwrite=True,
        ignore_warnings=True,
        **kwargs,
    )
    return path


@pytest.fixture
def dataset_dir(tmp_path):
    """Three files: two of the same shape, one with a differing number of frames."""
    for i, n_frames in enumerate([2, 2, 3]):
        _write_file(tmp_path / f"file_{i}.hdf5", _compressible_raw(n_frames, seed=i))
    return tmp_path


@pytest.fixture
def reference(dataset_dir):
    path = build_virtual_reference(dataset_dir, dataset_dir / "virtual" / "index.json")
    return open_virtual_reference(path)


def test_read_matches_h5py(reference):
    """Every file reads back bit-identically through the reference."""
    for index, path in enumerate(reference.file_paths):
        with File(path) as file:
            expected = file.data.raw_data[:]
        assert np.array_equal(reference["raw_data"][index], expected)


def test_partial_and_cross_file_read(reference):
    """A single index expression can span files (within a shape group) and axes."""
    paths = reference.file_paths
    expected = np.stack(
        [File(path).data.raw_data[0:1, 0] for path in paths[:2]]  # same-shape group
    )
    assert np.array_equal(reference["raw_data"][[0, 1], 0:1, 0], expected)

    with File(paths[0]) as file:
        assert np.array_equal(reference["raw_data"][0, ..., 0], file.data.raw_data[..., 0])


def test_default_key_indexing(reference):
    """Indexing the reference directly reads raw_data."""
    assert np.array_equal(reference[0, 0], reference["raw_data"][0, 0])
    assert reference.default_key == "raw_data"


def test_differing_shapes_form_separate_groups(reference):
    """Files that cannot be stacked (differing n_frames) land in separate shape groups."""
    groups = reference.groups()
    assert len(groups) == 2
    assert [len(group["files"]) for group in groups] == [2, 1]

    # Their arrays have different shapes, so one selection cannot cover both.
    with pytest.raises(IndexError, match="spans 2 shape groups"):
        reference["raw_data"][[0, 2]]
    with pytest.raises(AttributeError, match="shape groups"):
        _ = reference["raw_data"].shape


def test_image_is_virtualized(tmp_path):
    """Sub-groups of the data group (e.g. image/) are virtualized too."""
    image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (2, 32, 1))
    _write_file(tmp_path / "file.hdf5", _compressible_raw(2), image=image)

    path = build_virtual_reference(tmp_path, tmp_path / "index.json")
    reference = open_virtual_reference(path)

    assert set(reference.keys()) == {"raw_data", "image/values", "image/coordinates"}
    assert np.array_equal(reference["image/values"][0], image)


def test_incompressible_array_is_skipped(tmp_path, monkeypatch):
    """Arrays whose chunks HDF5 stored raw are left out (Zarr cannot decode those)."""
    noise = np.random.default_rng(0).integers(0, 255, (2, 32, 32), dtype=np.uint8)
    _write_file(tmp_path / "file.hdf5", _compressible_raw(2), image=noise)

    warnings = []
    monkeypatch.setattr(virtual.log, "warning", warnings.append)

    path = build_virtual_reference(tmp_path, tmp_path / "index.json")
    reference = open_virtual_reference(path)

    assert "raw_data" in reference.keys()
    assert "image/values" not in reference.keys()
    assert any("Not virtualizing image/values" in warning for warning in warnings)


def test_lzf_file_is_rejected(tmp_path):
    """lzf (zea's pre-0.1.3 default) has no Zarr codec: fail with resave guidance."""
    _write_file(tmp_path / "file.hdf5", _compressible_raw(2), compression="lzf")

    with pytest.raises(ValueError, match="lzf-compressed.*zea data resave"):
        build_virtual_reference(tmp_path, tmp_path / "index.json")


def test_parameters_sidecar_matches_file(reference):
    """Parameters are not virtualized, but the sidecar reconstructs them exactly."""
    for index, path in enumerate(reference.file_paths):
        with File(path) as file:
            expected = file.load_parameters()
        assert reference.parameters(index) == expected


def test_parameters_are_deduplicated(dataset_dir):
    """Identical parameter sets are stored once (they carry per-transmit arrays)."""
    build_virtual_reference(dataset_dir, dataset_dir / "virtual" / "index.json")
    sidecar = json.loads((dataset_dir / "virtual" / "params.json").read_text())

    assert len(sidecar["files"]) == 3  # the files share one scan/probe setup...
    assert len(sidecar["parameters"]) == 1  # ...so it is written once


def test_parameters_without_sidecar(dataset_dir):
    """A reference without its sidecar names the tool that writes one."""
    path = build_virtual_reference(dataset_dir, dataset_dir / "virtual" / "index.json")
    (dataset_dir / "virtual" / "params.json").unlink()

    with pytest.raises(FileNotFoundError, match="zea data virtualize"):
        open_virtual_reference(path).parameters(0)


def test_dataset_virtual(dataset_dir, reference):
    """Dataset(lazy='virtual') finds the published reference and reads through it."""
    with Dataset(dataset_dir, lazy="virtual") as dataset:
        virtual = dataset.virtual
        assert len(virtual) == len(dataset) == 3
        assert np.array_equal(virtual["raw_data"][0], reference["raw_data"][0])
        assert dataset.total_frames == 2 + 2 + 3  # from the reference: nothing is opened


def test_dataset_getitem_is_file_like(dataset_dir, reference):
    """dataset[i] reads with the File API — no download, no HDF5 open."""
    with Dataset(dataset_dir, lazy="virtual") as dataset:
        for index, path in enumerate(dataset.file_paths):
            file = dataset[index]
            assert isinstance(file, VirtualFile)
            # dataset[i] and the reference must agree on *which* file this is: the
            # reference orders files by shape group, the dataset by discovery order.
            assert file.path == path

            with File(path) as h5py_file:
                assert np.array_equal(file.data.raw_data[0], h5py_file.data.raw_data[0])
                assert file.n_frames == h5py_file.n_frames
                assert file.load_parameters() == h5py_file.load_parameters()


def test_open_virtual_file_without_a_reference(dataset_dir):
    """A single file needs no published reference: its manifest is built on open.

    It goes through the same machinery as a dataset reference (a lone file is just a
    reference with one file in it), so it yields the same VirtualFile.
    """
    path = dataset_dir / "file_0.hdf5"
    file = open_virtual_file(path)

    assert isinstance(file, VirtualFile)
    with File(path) as h5py_file:
        assert np.array_equal(file.data.raw_data[:], h5py_file.data.raw_data[:])
        assert file.load_parameters() == h5py_file.load_parameters()


def test_virtual_file_data_proxy(tmp_path):
    """The proxy mirrors the File API: nested groups, shape/dtype, array conversion."""
    image = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (2, 32, 1))
    _write_file(tmp_path / "file.hdf5", _compressible_raw(2), image=image)

    file = open_virtual_file(tmp_path / "file.hdf5")

    assert np.array_equal(file.data.image.values[:], image)  # nested group
    assert file.data.raw_data.shape == (2, N_TX, N_AX, N_EL, 1)  # no leading file axis
    assert file.data.raw_data.dtype == np.float32
    assert len(file.data.raw_data) == 2
    assert np.array_equal(np.asarray(file.data.image.values), image)
    assert "raw_data" in file.data and "image" in file.data

    with pytest.raises(AttributeError, match="No key 'nope'"):
        _ = file.data.nope


def test_dataset_virtual_requires_lazy_virtual(dataset_dir):
    with Dataset(dataset_dir, lazy=True) as dataset:
        with pytest.raises(AttributeError, match="lazy='virtual'"):
            _ = dataset.virtual

    with pytest.raises(ValueError, match="lazy must be a bool"):
        Dataset(dataset_dir, lazy="eager")


def test_dataset_virtual_without_reference(dataset_dir):
    """Without a published reference, point the user at the tool that makes one."""
    with Dataset(dataset_dir, lazy="virtual") as dataset:
        with pytest.raises(FileNotFoundError, match="zea data virtualize"):
            _ = dataset.virtual
