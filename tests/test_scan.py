"""Tests for the Scan class."""

import numpy as np
import pytest

from zea.scan import Scan

scan_args = {
    "n_tx": 10,
    "n_el": 10,
    "n_ch": 1,
    "xlims": (-0.019, 0.019),
    "ylims": (0, 0),
    "zlims": (0, 0.04),
    "center_frequency": 7e6,
    "sampling_frequency": 28e6,
    "demodulation_frequency": 0.0,
    "sound_speed": 1540.0,
    "n_ax": 3328,
    "grid_size_x": 64,
    "grid_size_z": 128,
    "pixels_per_wavelength": 4,
    "polar_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "azimuth_angles": np.linspace(-np.pi / 2, np.pi / 2, 10),
    "t0_delays": np.repeat(np.linspace(0, 1e-6, 10)[..., None], 10, axis=-1),
    "tx_apodizations": np.ones((10, 10)),
    "focus_distances": np.ones(10) * 0.04,
    "initial_times": np.zeros((10,)),
    "tx_waveform_indices": np.zeros(10, dtype=int),
    "waveforms_one_way": np.zeros((2, 64)),
    "waveforms_two_way": np.zeros((2, 64)),
    "tgc_gain_curve": np.ones((3328,)),
}


def test_scan_compare():
    """Test comparison of Scan objects."""
    scan = Scan(**scan_args)
    scan2 = Scan(**scan_args)
    scan3 = Scan(**scan_args)
    scan3.sound_speed = 1000

    assert scan == scan2
    assert scan != scan3


def test_scan_copy():
    """Test copying of Scan objects."""
    scan = Scan(**scan_args)
    scan_copy = scan.copy()

    assert scan == scan_copy
    scan.n_tx = 20
    assert scan != scan_copy


def test_initialization():
    """Test initialization of Scan class."""
    scan = Scan(**scan_args)

    assert scan.n_tx == scan_args["n_tx"]
    assert scan.n_el == scan_args["n_el"]
    assert scan.n_ch == scan_args["n_ch"]
    assert scan.xlims == scan_args["xlims"]
    assert scan.ylims == scan_args["ylims"]
    assert scan.zlims == scan_args["zlims"]
    assert scan.center_frequency == scan_args["center_frequency"]
    assert scan.sampling_frequency == scan_args["sampling_frequency"]
    assert scan.demodulation_frequency == scan_args["demodulation_frequency"]
    assert scan.sound_speed == scan_args["sound_speed"]
    assert scan.n_ax == scan_args["n_ax"]
    assert scan.grid_size_x == scan_args["grid_size_x"]
    assert scan.grid_size_z == scan_args["grid_size_z"]
    assert np.all(scan.polar_angles == scan_args["polar_angles"])
    assert np.all(scan.azimuth_angles == scan_args["azimuth_angles"])
    assert np.all(scan.t0_delays == scan_args["t0_delays"])
    assert np.all(scan.tx_apodizations == scan_args["tx_apodizations"])
    assert np.all(scan.focus_distances == scan_args["focus_distances"])
    assert np.all(scan.initial_times == scan_args["initial_times"])
    assert scan.pixels_per_wavelength == scan_args["pixels_per_wavelength"]


@pytest.mark.parametrize(
    "attr, expected_shape",
    [
        ("polar_angles", (10,)),
        ("azimuth_angles", (10,)),
        ("t0_delays", (10, 10)),
        ("tx_apodizations", (10, 10)),
        ("focus_distances", (10,)),
        ("initial_times", (10,)),
        ("tx_waveform_indices", (10,)),
    ],
)
def test_selected_transmits_affects_shape(attr, expected_shape):
    scan = Scan(**scan_args)
    # Check initial shape
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape == val_tensor.shape == expected_shape

    # Select 3 transmits
    scan.set_transmits(3)
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]

    # For 2D arrays, first dimension is always n_tx
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select center transmit
    scan.set_transmits("center")
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 1

    # Select all again
    scan.set_transmits("all")
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == expected_shape[0]

    # Select with some numpy array
    scan.set_transmits(np.arange(3))
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select with a list
    scan.set_transmits([1, 2, 3])
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3

    # Select with a slice
    scan.set_transmits(slice(0, 5, 2))
    val = getattr(scan, attr)
    val_tensor = scan.to_tensor(include=[attr])[attr]
    assert val.shape[0] == val_tensor.shape[0] == 3


def test_set_attributes():
    """Test setting attributes of Scan class."""
    scan = Scan(**scan_args)

    scan.selected_transmits = [0]

    with pytest.raises(ValueError):
        scan.grid = np.zeros((10, 10))


def test_accessing_valid_but_unset_attributes():
    """Test accessing valid but unset attributes of Scan class."""

    scan = Scan(n_tx=5)
    scan.focus_distances


def test_scan_pickle():
    """Test pickling and unpickling of Scan class."""
    import pickle

    scan = Scan(**scan_args)
    scan_pickled = pickle.dumps(scan)
    scan_unpickled = pickle.loads(scan_pickled)

    assert scan == scan_unpickled, "Unpickled Scan object does not match the original"
    assert scan is not scan_unpickled, "Unpickled Scan object is the same instance as the original"


def test_valid_params_default():
    """Test that modifying pfield_kwargs in one Scan instance does not affect another.

    The origin of this test is a bug where in VALID_PARAMS, the default value for pfield_kwargs
    was a mutable dictionary, leading to shared state across instances.
    """
    from zea.internal.dummy_scan import get_scan

    scan1 = get_scan()
    scan1.pfield_kwargs["norm"] = False

    scan2 = get_scan()
    assert scan2.pfield_kwargs == {}, (
        "scan2.pfield_kwargs seems to be affected by scan1 modification"
    )
    assert scan1 != scan2, "scan1 and scan2 should be different after modification of scan1"


def test_inplace_modification():
    """Test that modifying pfield_kwargs in-place, will update the pfield."""
    from zea.internal.dummy_scan import get_scan

    def edit1(scan):
        """edit direct dependency (dict) in-place"""
        scan.pfield_kwargs["norm"] = False
        return scan

    def edit2(scan):
        """edit another indirect dependency (np.ndarray) in-place"""
        scan.probe_geometry[:, 0] += 0.001
        return scan

    def edit3(scan):
        """edit indirect dependency (list) in-place
        pfield -> grid -> zlims"""
        # convert to list to allow in-place edit
        # this will invalidate pfield
        scan.zlims = list(scan.zlims)
        # therefore we need to force a computation of pfield to cache it
        _ = scan.pfield.copy()
        # and then edit in-place
        scan.zlims[1] += 0.01
        return scan

    for edit_fn in (edit1, edit2, edit3):
        scan = get_scan(pfield_kwargs={"norm": True})
        original_pfield = scan.pfield.copy()
        assert "pfield" in scan._cache, "pfield should be cached after first access"

        # Modify something in-place
        scan = edit_fn(scan)

        # Check that the grid has been updated
        assert not np.array_equal(original_pfield, scan.pfield), (
            f"scan.pfield seems to be unaffected by in-place modification in {edit_fn.__name__}"
        )


def test_inplace_modification_tensor_cache():
    """Test that modifying pfield_kwargs in-place, will update the pfield_tensor."""
    from zea.internal.dummy_scan import get_scan

    scan = get_scan(pfield_kwargs={"norm": True})
    tensor_dict = scan.to_tensor(include=["pfield"])
    scan.pfield_kwargs["norm"] = False  # in-place modification
    tensor_dict2 = scan.to_tensor(include=["pfield"])

    assert not np.array_equal(tensor_dict["pfield"], tensor_dict2["pfield"]), (
        "_tensor_cache['pfield'] seems to be unaffected by in-place modification"
    )
