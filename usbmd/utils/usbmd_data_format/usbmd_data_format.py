"""
The USBMD data format has the following structure:
- The dataset always contains a `data` and a `settings` group.
- The `data` group contains the actual data.
- The `settings` group contains all the settings that are necessary to
    interpret the data.
- The `data` group contains one or more of the following keys:
    - `raw_data`: The raw data of the ultrasound measurement. This can be
    either rf-data or iq-data.
    - `aligned_data`: The time-of-flight corrected data. This can be either
    rf-data or iq-data.
    - `envelope_data`: The envelope of the aligned data.
    - `beamformed_data`: The data after beamforming.
    - `image`: The processed image.
    - `image_sc`: The scan converted image.
- The `settings` group must contain the following keys:
    - `probe_geometry`: The geometry of the probe.
    - `t0_delays`: The time delays of the transducer elements.
    - `n_frames`: The number of frames in the dataset.
    - `n_tx`: The number of transmits per frame.
    - `n_el`: The number of elements in the transducer.
    - `n_ax`: The number of axial samples per transmit.
    - `n_ch`: The number of rf/iq channels in the data (either 1 or 2).
    - `sampling_frequency`: The sampling frequency of the data.
    - `center_frequency`: The center frequency of the transducer.

"""
import logging
import numpy as np
import h5py

from usbmd.utils.utils import print_hdf5_attrs


def generate_example_dataset(path, add_optional_fields=False):
    """Generates an example dataset that contains all the necessary fields.
    Note: This dataset does not contain actual data, but is filled with random
    values.

    Args:
        path (str): The path to write the dataset to.
        add_optional_fields (bool, optional): Whether to add optional fields to
            the dataset. Defaults to False.

    Returns:
        (h5py.File): The example dataset.
    """

    n_ax = 2048
    n_el = 128
    n_tx = 8
    n_ch = 1
    n_frames = 2

    raw_data = np.zeros((n_frames, n_tx, n_el, n_ax, n_ch))

    t0_delays = np.zeros((n_tx, n_el))
    tx_apodizations = np.zeros((n_tx, n_el))
    probe_geometry = np.zeros((n_el, 3))

    if add_optional_fields:
        focus_distances = np.zeros((n_tx,))
        tx_apodizations = np.zeros((n_tx, n_el))
        polar_angles = np.zeros((n_tx,))
        azimuth_angles = np.zeros((n_tx,))
    else:
        focus_distances = None
        tx_apodizations = None
        polar_angles = None
        azimuth_angles = None

    generate_usbmd_dataset(path,
                           raw_data=raw_data,
                           probe_geometry=probe_geometry,
                           sampling_frequency=40e6,
                           center_frequency=7e6,
                           initial_times=np.zeros((n_tx,)),
                           t0_delays=t0_delays,
                           sound_speed=1540,
                           tx_apodizations=tx_apodizations,
                           probe_name='example_probe',
                           focus_distances=focus_distances,
                           polar_angles=polar_angles,
                           azimuth_angles=azimuth_angles)


def generate_usbmd_dataset(path, raw_data, probe_geometry, sampling_frequency,
                           center_frequency, initial_times, t0_delays,
                           sound_speed, probe_name,
                           description='No description was supplied',
                           focus_distances=None, polar_angles=None,
                           azimuth_angles=None, tx_apodizations=None):
    """Generates a dataset in the USBMD format.

    Args:
        path (str): The path to write the dataset to.
        raw_data (np.ndarray): The raw data of the ultrasound measurement of
            shape (n_frames, n_tx, n_el, n_ax, n_ch).
        add_optional_fields (bool, optional): Whether to add optional fields to
            the dataset. Defaults to False.

    Returns:
        (h5py.File): The example dataset.
    """

    dataset = h5py.File(path, 'w')

    dataset.attrs['probe'] = probe_name
    dataset.attrs['description'] = description

    def add_dataset(group, name, data, description, unit):
        """Adds a dataset to the given group with a description and unit."""
        dataset = group.create_dataset(name, data=data)
        dataset.attrs['description'] = description
        dataset.attrs['unit'] = unit

    n_frames = raw_data.shape[0]
    n_tx = raw_data.shape[1]
    n_el = raw_data.shape[2]
    n_ax = raw_data.shape[3]
    n_ch = raw_data.shape[4]

    # Write data group
    data_group = dataset.create_group('data')
    data_shape = (n_frames, n_tx, n_el, n_ax, n_ch)
    assert raw_data.shape == data_shape, \
        f'The raw_data has the wrong shape. Expected {data_shape}, ' \
        f'got {raw_data.shape}.'

    add_dataset(group=data_group,
                name='raw_data',
                data=raw_data.astype(np.float32),
                description='The raw_data of shape (n_frames, n_tx, n_el, n_ax, n_ch).',
                unit='unitless')

    # Write settings group
    settings_group = dataset.create_group('settings')

    add_dataset(group=settings_group,
                name='n_ax',
                data=n_ax,
                description='The number of axial samples.',
                unit='unitless')

    add_dataset(group=settings_group,
                name='n_el',
                data=n_el,
                description='The number of elements in the probe.',
                unit='unitless')

    add_dataset(group=settings_group,
                name='n_tx',
                data=n_tx,
                description='The number of transmits per frame.',
                unit='unitless')

    add_dataset(group=settings_group,
                name='n_frames',
                data=n_tx,
                description='The number of frames.',
                unit='unitless')

    add_dataset(group=settings_group,
                name='sound_speed',
                data=sound_speed,
                description='The speed of sound in m/s',
                unit='m/s')

    add_dataset(group=settings_group,
                name='probe_geometry',
                data=probe_geometry,
                description='The probe geometry of shape (n_el, 3).',
                unit='m')

    add_dataset(group=settings_group,
                name='sampling_frequency',
                data=sampling_frequency,
                description='The sampling frequency in Hz.',
                unit='Hz')

    add_dataset(group=settings_group,
                name='center_frequency',
                data=center_frequency,
                description='The center frequency in Hz.',
                unit='Hz')

    add_dataset(group=settings_group,
                name='initial_times',
                data=initial_times,
                description='The times when the A/D converter starts sampling '
                'in seconds of shape (n_tx,). This is the time between the '
                'first element firing and the first recorded sample.',
                unit='s')

    add_dataset(group=settings_group,
                name='t0_delays',
                data=t0_delays,
                description='The t0_delays of shape (n_tx, n_el).',
                unit='s')

    if tx_apodizations is not None:
        add_dataset(group=settings_group,
                    name='tx_apodizations',
                    data=tx_apodizations,
                    description='The transmit delays for each element defining the'
                    ' wavefront in seconds of shape (n_tx, n_el). This is the'
                    ' time at which each element fires shifted such that the'
                    ' first element fires at t=0.',
                    unit='unitless')

    if focus_distances is not None:
        add_dataset(group=settings_group,
                    name='focus_distances',
                    data=focus_distances,
                    description='The transmit focus distances in meters of '
                    'shape (n_tx,). For planewaves this is set to 0.',
                    unit='m')

    if polar_angles is not None:
        add_dataset(group=settings_group,
                    name='polar_angles',
                    data=polar_angles,
                    description='The polar angles of the transmit beams in '
                    'radians of shape (n_tx,).',
                    unit='rad')

    if azimuth_angles is not None:
        add_dataset(group=settings_group,
                    name='azimuth_angles',
                    data=azimuth_angles,
                    description='The azimuthal angles of the transmit beams in '
                    'radians of shape (n_tx,).',
                    unit='rad')

    dataset.close()
    validate_dataset(path)

def validate_dataset(path):
    """Reads the hdf5 dataset at the given path and validates its structure.

    Args:
        path (str, pathlike): The path to the hdf5 dataset.


    """
    dataset = h5py.File(path, 'r')

    def check_key(dataset, key):
        assert key in dataset.keys(), \
            f'The dataset does not contain the key {key}.'

    # Validate the root group
    check_key(dataset, 'data')
    check_key(dataset, 'settings')

    # validate the settings group
    check_key(dataset['settings'], 'n_ax')
    check_key(dataset['settings'], 'n_el')
    check_key(dataset['settings'], 'n_tx')
    check_key(dataset['settings'], 'probe_geometry')
    check_key(dataset['settings'], 'sampling_frequency')
    check_key(dataset['settings'], 'center_frequency')
    check_key(dataset['settings'], 't0_delays')

    # validate the data group
    allowed_data_keys = [
        'raw_data',
        'aligned_data',
        'beamformed_data',
        'envelope_data',
        'image',
        'image_sc']
    for key in dataset['data'].keys():
        assert key in allowed_data_keys, \
            'The data group contains an unexpected key.'

        # Validate data shape
        data_shape = dataset['data'][key].shape
        if key == 'raw_data':
            assert len(data_shape) == 5, \
                'The raw_data group does not have a shape of length 5.'
            assert data_shape[1] == dataset['settings']['n_tx'][()], \
                'n_tx does not match the second dimension of raw_data.'
            assert data_shape[2] == dataset['settings']['n_el'][()], \
                'n_el does not match the third dimension of raw_data.'
            assert data_shape[3] == dataset['settings']['n_ax'][()], \
                'n_ax does not match the fourth dimension of raw_data.'
            assert data_shape[4] in (1, 2), \
                'The fifth dimension of raw_data is not 1 or 2.'

        elif key == 'aligned_data':
            logging.warning('No validation has been defined for aligned data.')
        elif key == 'beamformed_data':
            logging.warning('No validation has been defined for beamformed data.')
        elif key == 'envelope_data':
            logging.warning('No validation has been defined for envelope data.')
        elif key == 'image':
            logging.warning('No validation has been defined for image data.')
        elif key == 'image_sc':
            logging.warning('No validation has been defined for image_sc data.')

    required_settings_keys = [
        'n_ax',
        'n_el',
        'n_tx',
        'n_frames',
        'probe_geometry',
        'sampling_frequency',
        'center_frequency']

    # Ensure that all required keys are present
    for required_key in required_settings_keys:
        assert required_key in dataset['settings'].keys(), \
            ('The settings group does not contain the required key '
             f'{required_key}.')

    # Ensure that all keys have the correct shape
    for key in dataset['settings'].keys():
        if key == 'probe_geometry':
            assert dataset['settings'][key].shape == (dataset['settings']['n_el'][()], 3), \
                'The probe_geometry does not have the correct shape.'
        elif key == 't0_delays':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()], dataset['settings']['n_el'][()]), \
                'The t0_delays does not have the correct shape.'
        elif key == 'tx_apodizations':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()], dataset['settings']['n_el'][()]), \
                'The tx_apodizations does not have the correct shape.'
        elif key == 'focus_distances':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()],), \
                'The focus_distances does not have the correct shape.'
        elif key == 'polar_angles':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()],), \
                'The polar_angles does not have the correct shape.'
        elif key == 'azimuth_angles':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()],), \
                'The azimuthal_angles does not have the correct shape.'
        elif key == 'initial_times':
            assert dataset['settings'][key].shape == (dataset['settings']['n_tx'][()],), \
                'The initial_times does not have the correct shape.'
        elif key in ('sampling_frequency', 'center_frequency', 'n_frames',
                     'n_tx', 'n_el', 'n_ax', 'sound_speed'):
            assert dataset['settings'][key].size == 1, \
                f'{key} does not have the correct shape.'
        else:
            logging.warning('No validation has been defined for %s.', key)

    assert_unit_and_description_present(dataset)

def assert_unit_and_description_present(hdf5_file, _prefix=''):
    """Checks that all datasets have a unit and description attribute.

    Args:
        hdf5_file (h5py.File): The hdf5 file to check.

    Raises:
        AssertionError: If a dataset does not have a unit or description
            attribute.
    """
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Group):
            assert_unit_and_description_present(hdf5_file[key],
                                               _prefix=_prefix+key+'/')
        else:
            assert 'unit' in hdf5_file[key].attrs.keys(), \
                f'The dataset {_prefix}/{key} does not have a unit attribute.'
            assert 'description' in hdf5_file[key].attrs.keys(), \
                f'The dataset {_prefix}/{key} does not have a description attribute.'

if __name__ == '__main__':
    path = 'example_dataset.hdf5'
    generate_example_dataset(path)
    file = h5py.File(path, 'r')
    print_hdf5_attrs(file)
    validate_dataset(path)