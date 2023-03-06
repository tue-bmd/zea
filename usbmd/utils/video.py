""" Video processing functions
Author(s): Ben Luijten
"""

from collections import deque
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
    def __init__(
            self,
            norm_mode='max',
            env_mode = 'abs',
            img_buffer_size = 1,
            max_buffer_size = 10,
            norm_factor=2**14):
        """_summary_

        Args:
            norm_mode (str, optional): Normalization mode ('max', 'smoothnormal', 'fixed').
            Defaults to 'max'.
            env_mode (str, optional): Envelope detection settings. Defaults to 'abs'.
            img_buffer_size (int, optional): Size of the image buffer. Defaults to 1.
            max_buffer_size (int, optional): Size of the max (normalization) buffer. Defaults to 10.
            norm_factor (_type_, optional): Normalization value for the fixed setting .
            Defaults to 2**14.
            n_persistance (int, optional): Number of frames to average over. Defaults to 1.
        """
        self.norm_mode = norm_mode
        self.env_mode = env_mode
        self.norm_factor = norm_factor
        self.img_buffer = deque(maxlen=img_buffer_size)
        self.max_buffer = deque(maxlen=max_buffer_size)

        self.alpha = 0.5

    def convert(self, img):
        """Conversion function that applies all transformations"""
        img = self.envelope(img)
        img = self.normalize(img)
        img = self.compression(img)

        self.img_buffer.append(img)
        img = self.persistance_MA(img)
 
        img = np.clip(img, -60, 0)
        img = ((img + 60)*(255/60)).astype('uint8')

        return img

    def persistance_MA(self):
        """Function that applies moving average persistance to the image"""
        img = np.mean(self.img_buffer, axis=0)
        return img
    
    def persistance_AR(self, img):
        """Function that applies autoregressive persistance to the image"""
        img = self.alpha*img + (1-self.alpha)*self.img_buffer[-1]
        return img

    def normalize(self, img):
        "Normalization function"
        max_val = np.maximum(img.max(), 1e-9)
        self.max_buffer.append(img.max())
        #self.max_buffer = self.max_buffer[-self.buffer_size::]

        if self.norm_mode == 'normal':
            img = img/max_val
        elif self.norm_mode == 'smoothnormal':
            img = img/np.mean(self.max_buffer)
        elif self.norm_mode == 'fixed':
            img = img/self.norm_factor
        else:
            img = img/max_val

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

    def apply_contrast_curve(self, img, curve):
        """Function for applying a contrast curve"""
        img = np.interp(img, (0, 255), curve)
        return img

    def apply_gamma(self, img, gamma):
        """Function for applying gamma correction"""
        img = np.power(img/255, gamma)*255
        return img

    def apply_color_map(self, img, cmap):
        """Function for applying a color map"""
        img = cv2.applyColorMap(img, cmap)
        return img
    
    def resize_deque_buffer(self, old_buffer_name, new_buffer_size):
        """Function for resizing a deque buffer"""
        old_buffer = getattr(self, old_buffer_name)
        old_buffer_size = len(old_buffer)
        new_buffer = deque(maxlen=new_buffer_size)
        for i in range(np.minimum(new_buffer_size, old_buffer_size)):
            new_buffer.append(old_buffer[i])
        setattr(self, old_buffer_name, new_buffer)