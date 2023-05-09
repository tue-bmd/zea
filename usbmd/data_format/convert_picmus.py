"""Functionality to convert the PICMUS dataset to the USBMD format."""

import os
import logging
import h5py
import numpy as np

from usbmd.data_format.usbmd_data_format import generate_usbmd_dataset
from usbmd.scan import compute_t0_delays_planewave

def convert_picmus(source_path, output_path, overwrite=False):
    """Converts the PICMUS database to the USBMD format.

    Args:
        source_path (str, pathlike): The path to the original PICMUS file.
        output_path (str, pathlike): The path to the output file.
        overwrite (bool, optional): Set to True to overwrite existing file.
            Defaults to False.
    """

    # Check if output file already exists and remove
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            logging.warning('Output file already exists. Skipping conversion.')
            return

    # Open the file
    file = h5py.File(source_path, 'r')

    # Get the group containing the dataset
    file = file['US']['US_DATASET0000']

    # Extract I- and Q-data (shape (tx, el, ax))
    i_data = np.array(file['data']['real'], dtype='float32')
    q_data = np.array(file['data']['imag'], dtype='float32')

    if np.sum(q_data) < 0.1:
        # Use only the I-data and add dummy dimension (shape (tx, el, ax, ch=1))
        raw_data = i_data[..., None]
    else:
        # Stack I- and Q-data (shape (tx, el, ax, 2))
        raw_data = np.stack([i_data, q_data], axis=-1)

    # Add dummy frame dimension (shape (frame=1, tx, el, ax, ch=1))
    raw_data = raw_data[None]

    # raw_data = np.transpose(raw_data, (0, 1, 3, 2, 4))


    n_frames, n_tx, n_el, n_ax, n_ch = raw_data.shape

    center_frequency = int(file['modulation_frequency'][()])
    # Fix a mistake in one of the PICMUS files
    if center_frequency == 0:
        center_frequency = 5.208e6
    sampling_frequency = int(file['sampling_frequency'][()])
    probe_geomtry = np.transpose(file['probe_geometry'][()], (1, 0))

    sound_speed = float(file['sound_speed'][()])
    focus_distances = np.zeros((n_tx,), dtype=np.float32)
    polar_angles = file['angles'][()]
    azimuth_angles = np.zeros((n_tx,), dtype=np.float32)
    t0_delays = np.zeros((n_tx, n_el), dtype=np.float32)
    tx_apodizations = np.ones((n_tx, n_el), dtype=np.float32)

    initial_times = np.zeros((n_tx,))
    for n in range(n_tx):
        v = np.array([np.sin(polar_angles[n]), 0, np.cos(0)])
        initial_times[n] = -np.min(np.sum(probe_geomtry* v[None], axis=1))/sound_speed

        t0_delays[n] = compute_t0_delays_planewave(
            ele_pos=probe_geomtry,
            polar_angle=polar_angles[n],
            c=sound_speed)
        # This line changes the data format to work with the old beamformer,
        # which is not in accordance with the new USBMD format
        # TODO: Remove this line when updating the beamformer.
        t0_delays[n] -= np.max(t0_delays[n])*0.5



    generate_usbmd_dataset(path=output_path,
                           raw_data=raw_data,
                           center_frequency=center_frequency,
                           sampling_frequency=sampling_frequency,
                           probe_geometry=probe_geomtry,
                           initial_times=initial_times,
                           sound_speed=sound_speed,
                           t0_delays=t0_delays,
                           focus_distances=focus_distances,
                           polar_angles=polar_angles,
                           azimuth_angles=azimuth_angles,
                           tx_apodizations=tx_apodizations,
                           probe_name='verasonics_l11_4v',
                           description='PICMUS dataset converted to USBMD format')
