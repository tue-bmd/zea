"""zea H5 file functionality."""

import enum
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
from keras.utils import pad_sequences

from zea import log
from zea.data.spec import DataSpec, FileSpec, MetadataSpec, MetricsSpec, ScanSpec
from zea.internal.checks import _DATA_TYPES, _NON_IMAGE_DATA_TYPES
from zea.internal.core import DataTypes
from zea.internal.preset_utils import HF_PREFIX, _hf_resolve_path
from zea.internal.utils import deprecated, reduce_to_signature
from zea.probes import Probe
from zea.scan import Scan


class GroupProxy:
    """Lazy proxy for an h5py.Group that exposes children as attributes.

    Datasets are returned as-is (h5py.Dataset supports slicing without
    loading everything into RAM).  Sub-groups are wrapped in another
    ``GroupProxy`` so the dot-access pattern works recursively::

        with File(path) as f:
            # returns h5py.Dataset – no data loaded yet
            f.data.raw_data
            # slicing triggers the actual read, just like plain h5py
            f.data.raw_data[:, :n_tx]
            # nested groups work too
            f.data.image.values[0]
    """

    __slots__ = ("_group",)

    def __init__(self, group: h5py.Group):
        self._group = group

    def __getattr__(self, name: str):
        try:
            child = self._group[name]
        except KeyError:
            raise AttributeError(
                f"No key '{name}' in group '{self._group.name}'. "
                f"Available keys: {list(self._group.keys())}"
            )
        if isinstance(child, h5py.Group):
            return GroupProxy(child)
        return child  # h5py.Dataset – supports slicing natively

    def __dir__(self):
        return list(self._group.keys())

    def __repr__(self):
        return f"<GroupProxy '{self._group.name}' keys={list(self._group.keys())}>"

    def keys(self):
        """Return the keys of the underlying group."""
        return self._group.keys()

    def __contains__(self, key):
        return key in self._group

    def __iter__(self):
        return iter(self._group)


def assert_key(file: h5py.File, key: str):
    """Asserts key is in a h5py.File."""
    if key not in file.keys():
        raise KeyError(f"{key} not found in file")


def _warn_custom_keys(data: dict, metadata: dict):
    """Warn about custom keys in data/metadata dicts when saving."""
    custom_maps = [k for k in data if k not in DataSpec.SCHEMA]
    if custom_maps:
        supported = ", ".join(k for k, v in DataSpec.SCHEMA.items() if "spec" in v)
        log.warning(
            f"Custom spatial map key(s) added to 'data': {', '.join(sorted(custom_maps))}. "
            "These are validated as generic Map specs. "
            "If your data matches an existing type, prefer one of the supported "
            f"spatial maps: {supported}."
        )
    custom_signals = [k for k in metadata if k not in MetadataSpec.SCHEMA]
    if custom_signals:
        supported = ", ".join(k for k, v in MetadataSpec.SCHEMA.items() if "spec" in v)
        log.warning(
            f"Custom signal key(s) added to 'metadata': {', '.join(sorted(custom_signals))}. "
            "These are validated as generic SignalND specs. "
            "If your signal matches an existing type, prefer one of the supported "
            f"signal fields: {supported}."
        )


class File(h5py.File):
    """h5py.File in zea format."""

    def __init__(self, name, mode="r", *args, **kwargs):
        """Initialize the file.

        Args:
            name (str, Path, HFPath): The path to the file.
                Can be a string or a Path object. Additionally can be a string with
                the prefix 'hf://', in which case it will be resolved to a
                huggingface path.
            mode (str, optional): The mode to open the file in. Defaults to "r".
            *args: Additional arguments to pass to h5py.File.
            **kwargs: Additional keyword arguments to pass to h5py.File.
        """

        # Resolve huggingface path
        if str(name).startswith(HF_PREFIX):
            name = _hf_resolve_path(str(name))

        # Disable locking for read mode by default
        if "locking" not in kwargs and mode == "r":
            # If the file is opened in read mode, disable locking
            kwargs["locking"] = False

        # Initialize the h5py.File
        super().__init__(name, mode, *args, **kwargs)

    @property
    def path(self):
        """Return the path of the file."""
        return Path(self.filename)

    @property
    def zea_version(self) -> str | None:
        """Return the zea version that wrote this file, or ``None`` for legacy files.

        Files created with zea v0.1.0 and later store a ``zea_version``
        root attribute.  Files written before zea v0.1.0 return ``None``.
        """
        return self.attrs.get("zea_version", None)

    @classmethod
    def create(
        cls,
        path,
        data: dict,
        scan: dict | None = None,
        metadata: dict | None = None,
        metrics: dict | None = None,
        probe_name: str | None = None,
        us_machine: str | None = None,
        description: str | None = None,
        compression: str = "gzip",
        overwrite: bool = False,
    ) -> "File":
        """Create a new zea HDF5 file from data, scan, and optional metadata.

        All inputs are validated against the :class:`~zea.data.spec.FileSpec`
        schema (dtypes, shapes, dimension consistency) **before** anything is
        written to disk.

        Args:
            path: Destination file path.
            data: Data dict accepted by :class:`~zea.data.spec.DataSpec`.
            scan: Scan-parameter dict accepted by :class:`~zea.data.spec.ScanSpec`.
            metadata: Optional metadata dict accepted by
                :class:`~zea.data.spec.MetadataSpec`.
            metrics: Optional metrics dict accepted by
                :class:`~zea.data.spec.MetricsSpec`.
            probe_name: Name of the probe.
            us_machine: Name of the ultrasound machine.
            description: Free-text description of the acquisition.
            compression: HDF5 compression filter (default ``"gzip"``).
            overwrite: If *False* (default), raise if the file exists.

        Returns:
            File: The closed :class:`File` handle (re-open with
            ``File(path)`` to read).

        .. doctest::

            >>> import os, tempfile
            >>> import numpy as np
            >>> from zea import File

            >>> n_frames, n_tx, n_el, n_ax = 2, 4, 8, 64
            >>> raw = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
            >>> geom = np.zeros((n_el, 3), dtype=np.float32)
            >>> scan = {
            ...     "probe_geometry": geom,
            ...     "sampling_frequency": np.float32(40e6),
            ...     "center_frequency": np.float32(5e6),
            ...     "demodulation_frequency": np.float32(5e6),
            ...     "initial_times": np.zeros(n_tx, dtype=np.float32),
            ...     "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
            ...     "tx_apodizations": np.ones((n_tx, n_el), dtype=np.float32),
            ...     "focus_distances": np.full(n_tx, np.inf, dtype=np.float32),
            ...     "transmit_origins": np.zeros((n_tx, 3), dtype=np.float32),
            ...     "polar_angles": np.zeros(n_tx, dtype=np.float32),
            ...     "time_to_next_transmit": np.ones((n_frames, n_tx), dtype=np.float32) * 1e-4,
            ... }

            >>> _, path = tempfile.mkstemp(suffix=".hdf5")
            >>> f = File.create(
            ...     path, data={"raw_data": raw}, scan=scan, probe_name="L11-4v", overwrite=True
            ... )
            >>> f.probe_name
            'L11-4v'
            >>> f.close()
            >>> os.unlink(path)
        """
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")

        kwargs: dict = {"data": data}
        if scan:
            kwargs["scan"] = scan
        if metadata is not None:
            kwargs["metadata"] = metadata
        if metrics is not None:
            kwargs["metrics"] = metrics
        if probe_name is not None:
            kwargs["probe_name"] = probe_name
        if us_machine is not None:
            kwargs["us_machine"] = us_machine
        if description is not None:
            kwargs["description"] = description

        _warn_custom_keys(kwargs.get("data", {}), kwargs.get("metadata", {}))
        spec = FileSpec(**kwargs)
        spec.save(str(path), compression=compression)

        return cls(str(path), mode="r")

    @property
    def data(self) -> GroupProxy:
        """Lazy proxy for the ``data`` group.

        Returns a :class:`GroupProxy` so individual datasets can be accessed
        as attributes without loading everything into RAM::

            with File(path) as f:
                f.data.raw_data[:, :n_tx]  # read a slice
                f.data.image.values[0]  # nested group access
        """
        if "data" not in self:
            raise KeyError("No 'data' group in this file.")
        return GroupProxy(self["data"])

    @property
    def name(self):
        """Return the name of the file."""
        return self.path.name

    @property
    def stem(self):
        """Return the stem of the file."""
        return self.path.stem

    @property
    def n_frames(self):
        """Return number of frames in a file."""
        return int(self.file["scan"]["n_frames"][()])

    def shape(self, key) -> tuple:
        """Return shape of some key."""
        key = self.format_key(key)
        return self[key].shape

    def load_scan(self):
        """Alias for get_scan_parameters."""
        return self.get_scan_parameters()

    def format_key(self, key):
        """Format the key to match the data type."""
        if isinstance(key, enum.Enum):
            key = key.value

        assert isinstance(key, str), f"Key must be a string, got {type(key)}. "

        # Return the key if it is in the file
        if key in self.keys():
            return key

        # Add 'data/' prefix if not present
        if "data/" not in key:
            key = "data/" + key

        assert key in self.keys(), (
            f"Key {key} not found in file. Available keys: {list(self['data'].keys())}"
        )

        return key

    def to_iterator(self, key):
        """Convert the data to an iterator over all frames."""
        key = self.format_key(key)
        for frame_idx in range(self.n_frames):
            yield self[key][frame_idx]

    @staticmethod
    def key_to_data_type(key):
        """Convert the key to a data type."""
        data_type = key.split("/")[-1]
        return data_type

    def load_transmits(self, key, selected_transmits):
        """Load raw_data or aligned_data for a given list of transmits.
        Args:
            key (str): The type of data to load. Options are 'raw_data' and 'aligned_data'.
            selected_transmits (list, np.ndarray): The transmits to load.
        """
        key = self.format_key(key)
        data_type = self.key_to_data_type(key)
        assert data_type in ["raw_data", "aligned_data"], (
            f"Cannot load transmits for {data_type}. Only raw_data and aligned_data are supported."
        )
        # First axis: all frames, second axis: selected transmits
        indices = (slice(None), np.array(selected_transmits))
        return self[key][indices]

    @deprecated(replacement="File.data.<key> with h5py slice indexing")
    def load_data(
        self,
        data_type,
        indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    ) -> np.ndarray:
        """Load data from the file.

        .. deprecated::
           Use ``file.data.<key>`` with standard h5py slice indexing instead::

               with File(path) as f:
                   raw = f.data.raw_data[:]  # all frames
                   raw = f.data.raw_data[0]  # first frame
                   raw = f.data.raw_data[0, [0, 2]]  # frame 0, transmits 0 and 2

        .. include:: ../common/file_indexing.rst

        Args:
            data_type (str): The type of data to load. Options are 'raw_data', 'aligned_data',
                'beamformed_data', 'envelope_data', 'image' and 'image_sc'.
            indices (optional): The indices to load. Defaults to ``None`` in
                which case all data is loaded.
        """
        key = self.format_key(data_type)
        if indices is None or (isinstance(indices, str) and indices == "all"):
            indices = slice(None)

        data = self[key]
        try:
            data = data[indices]
        except (OSError, IndexError) as exc:
            raise ValueError(
                f"Invalid indices {indices} for key {key}. {key} has shape {data.shape}."
            ) from exc

        return data

    @property
    def probe_name(self):
        """Reads the probe name from the data file and returns it."""
        # Support both 'probe_name' (new spec) and 'probe' (legacy files)
        for attr_key in ("probe_name", "probe"):
            if attr_key in self.attrs:
                return self.attrs[attr_key]
        raise AttributeError(
            "Probe name not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )

    @property
    def us_machine(self):
        """Reads the ultrasound machine name from the data file and returns it."""
        assert "us_machine" in self.attrs, (
            "Ultrasound machine name not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        us_machine = self.attrs["us_machine"]
        return us_machine

    @property
    def description(self):
        """Reads the description from the data file and returns it."""
        assert "description" in self.attrs, (
            "Description not found in file attributes. "
            "Make sure you are using a zea file. "
            f"Found attributes: {list(self.attrs)}"
        )
        description = self.attrs["description"]
        return description

    def get_parameters(self):
        """Returns a dictionary of parameters to initialize a scan
        object that comes with the file (stored inside datafile).

        If there are no scan parameters in the hdf5 file, returns
        an empty dictionary.

        Returns:
            dict: The scan parameters.
        """
        if "scan" not in self:
            log.warning("Could not find scan parameters in file.")
            return {}

        scan_parameters = self.recursively_load_dict_contents_from_group("scan")
        scan_parameters = self._check_focus_distances(scan_parameters)

        return scan_parameters

    def _check_focus_distances(self, scan_parameters):
        if "focus_distances" in scan_parameters:
            focus_distances = scan_parameters["focus_distances"]
            # check if focus distances are in wavelengths
            if np.any(np.logical_and(focus_distances >= 1, focus_distances != np.inf)):
                log.warning(
                    f"We have detected that focus distances in '{self.path}' are "
                    "(probably) stored wavelengths. Please update your file! "
                    "Converting to meters automatically for now, but this assumes that "
                    "`center_frequency` is the probe center frequency which is not always "
                    "the case!"
                )
                assert "sound_speed" in scan_parameters, (
                    "Cannot convert focus distances from wavelengths to meters "
                    "because sound_speed is not defined in the scan parameters."
                )
                assert "center_frequency" in scan_parameters, (
                    "Cannot convert focus distances from wavelengths to meters "
                    "because center_frequency is not defined in the scan parameters."
                )
                wavelength = scan_parameters["sound_speed"] / scan_parameters["center_frequency"]
                focus_distances = focus_distances * wavelength
                scan_parameters["focus_distances"] = focus_distances
        return scan_parameters

    def get_scan_parameters(self) -> dict:
        """Returns a dictionary of scan parameters stored in the file."""
        return self.get_parameters()

    @property
    def n_ax(self) -> int:
        """Number of axial samples."""
        assert "data" in self, "Cannot determine n_ax because there is no data group in the file."
        assert "raw_data" in self["data"], (
            "Cannot determine n_ax because there is no raw_data in the data group."
        )
        return self["data"]["raw_data"].shape[2]

    def scan(self, safe=True, **kwargs) -> Scan:
        """Returns a Scan object initialized with the parameters from the file.

        Args:
            safe (bool, optional): If True, will only use parameters that are
                defined in the Scan class. If False, will use all parameters
                from the file. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the Scan object.
                These will override the parameters from the file if they are
                present in the file.

        Returns:
            Scan: The scan object.

        .. doctest::

            >>> from zea import File
            >>> path = (
            ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
            ... )
            >>> with File(path) as f:
            ...     scan = f.scan()
            >>> type(scan).__name__
            'Scan'
        """
        scan_dict = self.get_scan_parameters()

        # Try spec-based validation; fall back gracefully for legacy files
        # that may be missing fields the spec now requires.
        scan_spec_keys = set(ScanSpec.SCHEMA.keys())
        filtered = {k: v for k, v in scan_dict.items() if k in scan_spec_keys}

        try:
            scan_spec = ScanSpec(**filtered)
            scan_dict = scan_spec.to_dict()
            scan_dict["n_el"] = scan_spec.n_el
            scan_dict["n_tx"] = scan_spec.n_tx
            # Derive n_ax from the spec when possible (avoids requiring raw_data).
            # tgc_gain_curve has shape (n_ax,) and is the spec's authoritative source.
            if scan_spec.tgc_gain_curve is not None:
                scan_dict["n_ax"] = len(scan_spec.tgc_gain_curve)
            elif "data" in self and "raw_data" in self["data"]:
                scan_dict["n_ax"] = self.n_ax
        except (TypeError, ValueError) as exc:
            log.debug(
                f"ScanSpec validation skipped for '{self.path}': {exc}. "
                "Using raw scan parameters from file."
            )

        return Scan.merge(_reformat_waveforms(scan_dict), kwargs, safe=safe)

    def get_probe_parameters(self) -> dict:
        """Returns a dictionary of probe parameters to initialize a probe
        object that comes with the file (stored inside datafile).

        Returns:
            dict: The probe parameters.
        """
        file_scan_parameters = self.get_parameters()

        probe_parameters = reduce_to_signature(Probe.__init__, file_scan_parameters)
        return probe_parameters

    def probe(self) -> Probe:
        """Returns a Probe object initialized with the parameters from the file.

        Returns:
            Probe: The probe object.

        .. doctest::

            >>> from zea import File
            >>> path = (
            ...     "hf://zeahub/picmus/database/experiments/contrast_speckle/"
            ...     "contrast_speckle_expe_dataset_iq/contrast_speckle_expe_dataset_iq.hdf5"
            ... )
            >>> with File(path) as f:
            ...     probe = f.probe()
            >>> type(probe).__name__
            'Verasonics_l11_4v'
        """
        probe_parameters_file = self.get_probe_parameters()
        return Probe.from_parameters(self.probe_name, probe_parameters_file)

    def metadata(self) -> MetadataSpec:
        """Return a validated :class:`~zea.data.spec.MetadataSpec` object from the file.

        Returns:
            MetadataSpec: The validated metadata spec.

        Raises:
            KeyError: If the file has no ``metadata`` group.

        Example::

            >>> with File("my_file.hdf5") as f:  # doctest: +SKIP
            ...     meta = f.metadata()
            ...     print(meta.subject.id)
        """
        if "metadata" not in self:
            raise KeyError("No 'metadata' group in this file.")
        raw = self.recursively_load_dict_contents_from_group("metadata")
        return MetadataSpec(**raw)

    def metrics(self) -> MetricsSpec:
        """Return a validated :class:`~zea.data.spec.MetricsSpec` object from the file.

        Returns:
            MetricsSpec: The validated metrics spec.

        Raises:
            KeyError: If the file has no ``metrics`` group.

        Example::

            >>> with File("my_file.hdf5") as f:  # doctest: +SKIP
            ...     met = f.metrics()
            ...     print(met.coherence_factor.shape)
        """
        if "metrics" not in self:
            raise KeyError("No 'metrics' group in this file.")
        raw = self.recursively_load_dict_contents_from_group("metrics")
        return MetricsSpec(**raw)

    def recursively_load_dict_contents_from_group(self, path: str) -> dict:
        """Load dict from contents of group

        Values inside the group are converted to numpy arrays
        or primitive types (int, float, str).

        Args:
            path (str): path to group
        Returns:
            dict: dictionary with contents of group
        """
        ans = {}
        for key, item in self[path].items():
            if isinstance(item, h5py.Dataset):
                if h5py.check_string_dtype(item.dtype) is not None:
                    val = item.asstr()[()]
                    # h5py returns object-dtype arrays for strings;
                    # convert back to np.str_ so spec dtype checks pass.
                    if isinstance(val, np.ndarray) and val.dtype == object:
                        val = val.astype(np.str_)
                    ans[key] = val
                else:
                    ans[key] = item[()]
            elif isinstance(item, h5py.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(path + "/" + key + "/")
        return ans

    def has_key(self, key: str) -> bool:
        """Check if the file has a specific key.

        Args:
            key (str): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        try:
            key = self.format_key(key)
        except AssertionError:
            return False
        return True

    @classmethod
    def get_shape(cls, path: str, key: str) -> tuple:
        """Get the shape of a key in a file.

        Args:
            path (str): The path to the file.
            key (str): The key to get the shape of.

        Returns:
            tuple: The shape of the key.
        """
        with cls(path, mode="r") as file:
            return file.shape(key)

    def validate(self):
        """Lightweight structural validation — no array data is loaded into RAM.

        Checks that the file has a ``data`` group and that all keys within it
        are recognised zea data types.  For legacy files (before zea v0.1.0)
        a minimal key-name check is performed.  For files created with
        zea v0.1.0 and later (via :meth:`File.create`) the keys are checked
        against the :class:`~zea.data.spec.DataSpec` schema.

        Use :meth:`validate_spec` for a **full** validation that loads all data
        and checks dtypes, shapes, and cross-field dimension consistency.

        Returns:
            dict: ``{"status": "success"}`` on success.

        Raises:
            AssertionError: If the file is missing required groups or contains
                unrecognised data keys.
        """
        try:
            return validate_file(file=self)
        except Exception as e:
            log.error(f"File {self.path} is not a valid zea file.\n{e}\n")
            raise

    def validate_spec(self) -> FileSpec:
        """Full schema validation — loads all data into RAM.

        Reads every dataset in the file and runs dtype, shape, and
        cross-dimension consistency checks as defined by :class:`~zea.data.spec.FileSpec`.
        Use this to confirm a file is fully spec-compliant before sharing or
        processing it.

        For a fast, zero-IO structural check use :meth:`validate` instead.

        .. note::
            This method only works on files created with zea v0.1.0 and later.
            Files written before zea v0.1.0 should be re-saved through
            :meth:`File.create`.

        Returns:
            FileSpec: The fully validated spec object, with all data accessible
            as typed attributes (e.g. ``spec.data.raw_data``, ``spec.scan.n_tx``).

        Raises:
            TypeError, ValueError: If the file does not conform to the spec.

        .. doctest::

            >>> with File("my_file.hdf5") as f:  # doctest: +SKIP
            ...     spec = f.validate_spec()
            ...     print(spec.scan.n_tx)
        """
        return FileSpec.from_hdf5(self)

    def __repr__(self):
        return (
            f"<zea.data.file.File at 0x{id(self):x} "
            f'("{Path(self.filename).name}" mode={self.mode})>'
        )

    def __str__(self):
        return f"zea HDF5 File: '{self.path.name}' (mode={self.mode})"

    def copy_key(self, key: str, dst: "File"):
        """Copy a specific key to another file.

        Will always copy the attributes and the scan data if it exists. Will warn if the key is
        not in this file or if the key already exists in the destination file.

        Args:
            key (str): The key to copy.
            dst (File): The destination file to copy the key to.
        """
        key = self.format_key(key)

        # Copy the key if it does not already exist in the destination file
        if key in dst:
            log.warning(f"Skipping key '{key}' because it already exists in dst file {dst.path}.")
        elif key in self:
            self.copy(key, dst, name=key)
        else:
            log.warning(f"Key '{key}' not found in src file {self.path}. Skipping copy.")

        # Copy attributes from src to dst
        for attr_key, attr_value in self.attrs.items():
            dst[key].attrs[attr_key] = attr_value

        # Copy scan data if requested
        if "scan" in self and "scan" not in dst:
            # Copy the scan data if it exists
            self.copy("scan", dst)

    def summary(self):
        """Print the contents of the file."""
        _print_hdf5_attrs(self)


def load_file_all_data_types(
    path,
    indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    scan_kwargs: dict = None,
):
    """Loads a zea data files (h5py file).

    Returns all data types together with a scan object containing the parameters
    of the acquisition and a probe object containing the parameters of the probe.

    Additionally, it can load a specific subset of frames / transmits.

    .. include:: ../common/file_indexing.rst

    Args:
        path (str, pathlike): The path to the hdf5 file.
        indices (optional): The indices to load. Defaults to None in
            which case all frames are loaded.
        scan_kwargs (Config, dict, optional): Additional keyword arguments
            to pass to the Scan object. These will override the parameters from the file
            if they are present in the file. Defaults to None.

    Returns:
        (dict): A dictionary with all data types as keys and the corresponding data as values.
        (Scan): A scan object containing the parameters of the acquisition.
        (Probe): A probe object containing the parameters of the probe.
    """
    # Define the additional keyword parameters from the scan object
    if scan_kwargs is None:
        scan_kwargs = {}

    data_dict = {}

    # Data types stored as HDF5 groups (Map-based specs with values/extent)
    _GROUP_DATA_TYPES = {"beamformed_data", "envelope_data", "image_sc", "image"}

    with File(path, mode="r") as file:
        # Load the probe object from the file
        probe = file.probe()

        for data_type in DataTypes:
            if not file.has_key(data_type.value):
                data_dict[data_type.value] = None
                continue

            # Load the desired frames from the file
            _key = file.format_key(data_type.value)
            _indices = indices if indices is not None else slice(None)
            item = file[_key]

            if isinstance(item, h5py.Group) and data_type.value in _GROUP_DATA_TYPES:
                # Map-based group: load all sub-datasets as a dict
                group_dict = {}
                for sub_key in item.keys():
                    ds = item[sub_key]
                    if isinstance(ds, h5py.Dataset):
                        if sub_key == "values":
                            group_dict[sub_key] = ds[_indices]
                        elif h5py.check_string_dtype(ds.dtype) is not None:
                            val = ds.asstr()[()]
                            if isinstance(val, np.ndarray) and val.dtype == object:
                                val = val.astype(np.str_)
                            group_dict[sub_key] = val
                        else:
                            group_dict[sub_key] = ds[()]
                data_dict[data_type.value] = group_dict
            else:
                data_dict[data_type.value] = item[_indices]

        # extract transmits from indices
        # we only have to do this when the data has a n_tx dimension
        # in that case we also have update scan parameters to match
        # the number of selected transmits
        if isinstance(indices, tuple) and len(indices) > 1:
            scan_kwargs["selected_transmits"] = indices[1]

        scan = file.scan(**scan_kwargs)

        return data_dict, scan, probe


def load_file(
    path,
    data_type="raw_data",
    indices: Tuple[Union[list, slice, int], ...] | List[int] | int | None = None,
    scan_kwargs: dict = None,
) -> Tuple[np.ndarray, Scan, Probe]:
    """Loads a zea data files (h5py file).

    Returns the data together with a scan object containing the parameters
    of the acquisition and a probe object containing the parameters of the probe.

    Additionally, it can load a specific subset of frames / transmits.

    .. include:: ../common/file_indexing.rst

    Args:
        path (str, pathlike): The path to the hdf5 file.
        data_type (str, optional): The type of data to load. Defaults to
            'raw_data'. Other options are 'aligned_data', 'beamformed_data',
            'envelope_data', 'image' and 'image_sc'.
        indices (optional): The indices to load. Defaults to None in
            which case all frames are loaded.
        scan_kwargs (Config, dict, optional): Additional keyword arguments
            to pass to the Scan object. These will override the parameters from the file
            if they are present in the file. Defaults to None.

    Returns:
        (np.ndarray): The raw data of shape (n_frames, n_tx, n_ax, n_el, n_ch).
        (Scan): A scan object containing the parameters of the acquisition.
        (Probe): A probe object containing the parameters of the probe.
    """
    # Define the additional keyword parameters from the scan object
    if scan_kwargs is None:
        scan_kwargs = {}

    with File(path, mode="r") as file:
        # Load the probe object from the file
        probe = file.probe()

        # Load the desired frames from the file
        _key = file.format_key(data_type)
        _indices = indices if indices is not None else slice(None)
        item = file[_key]
        if isinstance(item, h5py.Group):
            data = item["values"][_indices]
        else:
            data = item[_indices]

        # extract transmits from indices
        # we only have to do this when the data has a n_tx dimension
        # in that case we also have update scan parameters to match
        # the number of selected transmits
        if data_type in ["raw_data", "aligned_data"]:
            if isinstance(indices, tuple) and len(indices) > 1:
                scan_kwargs["selected_transmits"] = indices[1]

        scan = file.scan(**scan_kwargs)

        return data, scan, probe


def _print_hdf5_attrs(hdf5_obj, prefix=""):
    """Recursively prints all keys, attributes, and shapes in an HDF5 file.

    Args:
        hdf5_obj (h5py.File, h5py.Group, h5py.Dataset): HDF5 object to print.
        prefix (str, optional): Prefix to print before each line. This
            parameter is used in internal recursion and should not be supplied
            by the user.
    """
    assert isinstance(hdf5_obj, (h5py.File, h5py.Group, h5py.Dataset)), (
        "ERROR: hdf5_obj must be a File, Group, or Dataset object"
    )

    if isinstance(hdf5_obj, h5py.File):
        name = "root" if hdf5_obj.name == "/" else hdf5_obj.name
        print(prefix + name + "/")
        prefix += "    "
    elif isinstance(hdf5_obj, h5py.Dataset):
        shape_str = str(hdf5_obj.shape).replace(",)", ")")
        print(prefix + "├── " + hdf5_obj.name + " (shape=" + shape_str + ")")
        prefix += "│   "

    # Print all attributes
    for key, val in hdf5_obj.attrs.items():
        print(prefix + "├── " + key + ": " + str(val))

    # Recursively print all keys, attributes, and shapes in groups
    if isinstance(hdf5_obj, h5py.Group):
        for i, key in enumerate(hdf5_obj.keys()):
            is_last = i == len(hdf5_obj.keys()) - 1
            if is_last:
                marker = "└── "
                new_prefix = prefix + "    "
            else:
                marker = "├── "
                new_prefix = prefix + "│   "
            print(prefix + marker + key + "/")
            _print_hdf5_attrs(hdf5_obj[key], new_prefix)


def validate_file(path: str = None, file: File = None):
    """Validate the structure and data of a zea HDF5 file.

    For files created with zea v0.1.0 and later this runs the full
    :class:`~zea.data.spec.FileSpec` schema validation (dtypes, shapes, and
    dimension consistency).  Legacy files (before zea v0.1.0) are detected by the
    presence of scalar dataset ``scan/n_frames``; for those only a lightweight
    structural ``data`` group check is performed.

    Provide either *path* or *file*, but not both.

    Args:
        path (str | pathlike): Path to the HDF5 file.
        file (File): An already-open :class:`File` instance.

    Returns:
        dict: ``{"status": "success"}`` on success.

    Raises:
        AssertionError: If the file is missing the ``data`` group.
        TypeError, ValueError: If spec validation fails on files created with zea v0.1.0 and later.
    """
    assert (path is not None) ^ (file is not None), (
        "Provide either the path or the file, but not both."
    )

    if path is not None:
        with File(path, "r") as _file:
            _validate_file_impl(_file)
    else:
        _validate_file_impl(file)

    return {"status": "success"}


def _is_legacy_file(file: File) -> bool:
    """Return ``True`` when *file* pre-dates the dataspec format.

    Files created with zea v0.1.0 and later always store a
    ``zea_version`` root attribute.  Files that lack it were produced by
    the legacy data format path and are treated as legacy.
    """
    return "zea_version" not in file.attrs


def _validate_file_impl(file: File) -> None:
    """Lightweight structural validation — no array data is loaded.

    Checks that:
    - a ``data`` group is present at root OR one or more ``event_*`` groups each
      containing a ``data`` group (event-structured files)
    - for legacy files, every key in ``data`` is a recognised zea data type
    - for files created with zea v0.1.0 and later, every key in ``data``
    is in :class:`~zea.data.spec.DataSpec`\'s schema
    """
    # Collect all data groups to validate: either root /data or per-event /event_*/data
    data_groups: list[tuple[str, h5py.Group]] = []

    if "data" in file:
        assert isinstance(file["data"], h5py.Group), (
            "'data' is not a group - this may not be a zea file."
        )
        data_groups.append(("data", file["data"]))
    else:
        event_keys = [
            k for k in file.keys() if k.startswith("event_") and k[len("event_") :].isdigit()
        ]
        for event_key in event_keys:
            assert "data" in file[event_key], (
                f"Event group '{event_key}' is missing a 'data' subgroup."
            )
            assert isinstance(file[event_key]["data"], h5py.Group), (
                f"'{event_key}/data' is not a group - this may not be a zea file."
            )
            data_groups.append((f"{event_key}/data", file[event_key]["data"]))

    assert data_groups, (
        "'data' group not found in file. "
        "Expected either a root 'data' group or event groups named 'event_*'."
    )

    for group_path, data_group in data_groups:
        if _is_legacy_file(file):
            # For legacy files: accepted keys are the flat _DATA_TYPES list.
            has_raw = any(k in data_group for k in _NON_IMAGE_DATA_TYPES)
            if has_raw:
                assert "scan" in file, "Legacy file is missing the 'scan' group."
            for key in data_group.keys():
                assert key in _DATA_TYPES, (
                    f"'{group_path}/{key}' is not a recognised zea data type."
                )
        else:
            # For new-format files: flat datasets must be known DataSpec keys.
            # HDF5 Groups are Map specs (either a named type or a custom map)
            # and are always accepted; validate() is a structural check only.
            known = set(DataSpec.SCHEMA.keys())
            known_flat = {k for k, v in DataSpec.SCHEMA.items() if "spec" not in v}
            for key in data_group.keys():
                if isinstance(data_group[key], h5py.Group):
                    # Named map or custom map — accepted without further checks here.
                    continue
                assert key in known_flat, (
                    f"'{group_path}/{key}' is not in the DataSpec schema. "
                    f"Known keys: {sorted(known)}"
                )


def _reformat_waveforms(scan_kwargs: dict) -> dict:
    """Reformat waveforms from dict to array if needed. This is for backwards compatibility and will
    be removed in a future version of zea.

    Args:
        scan_kwargs (dict): The scan parameters.

    Returns:
        scan_kwargs (dict): The scan parameters with the keys waveforms_one_way and
            waveforms_two_way reformatted to arrays if they were stored as dicts.
    """

    # TODO: remove this in a future version of zea
    if "waveforms_one_way" in scan_kwargs and isinstance(scan_kwargs["waveforms_one_way"], dict):
        log.warning(
            "The waveforms_one_way parameter is stored as a dictionary in the file. "
            "Converting to array. This will be deprecated in future versions of zea. "
            "Please update your files to store waveforms as arrays of shape `(n_tx, n_samples)`."
        )
        scan_kwargs["waveforms_one_way"] = _waveforms_dict_to_array(
            scan_kwargs["waveforms_one_way"]
        )

    if "waveforms_two_way" in scan_kwargs and isinstance(scan_kwargs["waveforms_two_way"], dict):
        log.warning(
            "The waveforms_two_way parameter is stored as a dictionary in the file. "
            "Converting to array. This will be deprecated in future versions of zea. "
            "Please update your files to store waveforms as arrays of shape `(n_tx, n_samples)`."
        )
        scan_kwargs["waveforms_two_way"] = _waveforms_dict_to_array(
            scan_kwargs["waveforms_two_way"]
        )
    return scan_kwargs


def _waveforms_dict_to_array(waveforms_dict: dict):
    """Convert waveforms stored as a dictionary to a padded numpy array."""
    waveforms = dict_to_sorted_list(waveforms_dict)
    return pad_sequences(waveforms, dtype=np.float32, padding="post")


def dict_to_sorted_list(dictionary: dict):
    """Convert a dictionary with sortable keys to a sorted list of values.

    .. note::

        This function operates on the top level of the dictionary only.
        If the dictionary contains nested dictionaries, those will not be sorted.

    Example:
        .. doctest::

            >>> from zea.data.file import dict_to_sorted_list
            >>> input_dict = {"number_000": 5, "number_001": 1, "number_002": 23}
            >>> dict_to_sorted_list(input_dict)
            [5, 1, 23]

    Args:
        dictionary (dict): The dictionary to convert. The keys must be sortable.

    Returns:
        list: The sorted list of values.
    """
    return [value for _, value in sorted(dictionary.items())]
