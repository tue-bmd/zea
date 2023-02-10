""" Video processing functions
Author(s): Ben Luijten
"""

from time import perf_counter

import cv2
import numpy as np


class FPS_counter():
    """ An FPS counter class that overlays a frames-per-second count on an image stream"""

    def __init__(self, buffer_size = 30):
        """_summary_

        Args:
            buffer_size (int, optional): Size of the MA window (in frames). Defaults to 30.
        """

        self.buffer_size = buffer_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.time_buffer = list(np.zeros((buffer_size,)))

    def overlay(self, img):
        """Function that overlays the FPS count on a provided image"""
        self.time_buffer.append(perf_counter())
        self.time_buffer = self.time_buffer[-self.buffer_size::]
        fps = 1/np.mean(np.diff(self.time_buffer))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, f'{fps:.1f}', (7, 70), self.font, 1, (100, 255, 0), 3, cv2.LINE_AA)
        return img

#TODO: This class is currently used during streaming, but will be replaced with USBMD functions.
class Scan_converter():
    """Class that handles visualization of ultrasound images"""
    def __init__(self, norm_mode='max', env_mode = 'abs', buffer_size = 30, norm_factor=2**14):
        """_summary_

        Args:
            norm_mode (str, optional): Normalization mode ('max', 'smoothnormal', 'fixed').
            Defaults to 'max'.
            env_mode (str, optional): Envelope detection settings. Defaults to 'abs'.
            buffer_size (int, optional): Size of the frame buffer. Defaults to 30.
            norm_factor (_type_, optional): Normalization value for the fixed setting .
            Defaults to 2**14.
        """
        self.norm_mode = norm_mode
        self.env_mode = env_mode
        self.buffer_size = buffer_size
        self.norm_factor = norm_factor
        self.img_buffer = []
        self.max_buffer = list(np.zeros((buffer_size,)))

    def convert(self, img):
        """Conversion function that applies all transformations"""
        img = self.normalize(img)
        img = self.envelope(img)
        img = self.compression(img)
        img = np.clip(img, -60, 0)
        img = ((img + 60)*(255/60)).astype('uint8')
        return img

    def normalize(self, img):
        "Normalization function"
        max_val = np.maximum(input.max(), 1e-9)
        self.max_buffer.append(input.max())
        self.max_buffer = self.max_buffer[-self.buffer_size::]

        if self.norm_mode == 'normal':
            img = input/max_val
        elif self.norm_mode == 'smoothnormal':
            img = input/np.mean(self.max_buffer)
        elif self.norm_mode == 'fixed':
            img = input/self.norm_factor
        else:
            img = input/max_val

        return img

    @staticmethod
    def envelope(data):
        """Envelope detection"""
        env = np.abs(data)
        return env

    @staticmethod
    def compression(data):
        """Logarithmic compression"""
        return 20*np.log10(data)

    def set_value(self, key, val):
        """Function for setting parameters"""
        setattr(self, key, val)
