""" Video processing functions
Author(s): Ben Luijten
"""

from collections import deque
from time import perf_counter

import cv2
import numpy as np

import tensorflow as tf


class FPS_counter():
    """ An FPS counter class that overlays a frames-per-second count on an image stream"""

    def __init__(self, buffer_size = 5):
        """_summary_

        Args:
            buffer_size (int, optional): Size of the MA window (in frames). Defaults to 30.
        """

        self.buffer_size = buffer_size
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.time_buffer = list(np.zeros((10,)))
        self.fps_value = 0
        self.update_interval = 0.1 # Update interval in seconds
        self.last_update_time = perf_counter()

    def overlay(self, img):
        """Function that overlays the FPS count on a provided image"""
        curr_time = perf_counter()
        elapsed_time = curr_time - self.last_update_time

        self.time_buffer.append(curr_time)
        self.time_buffer = self.time_buffer[-self.buffer_size::]

        if elapsed_time > self.update_interval:
            fps = 1/np.mean(np.diff(self.time_buffer))
            self.fps_value = fps
            self.last_update_time = curr_time

        cv2.putText(img, f'{self.fps_value:.1f}', (7, 50), self.font, 0.8, (50, 255, 0, 255), 2, cv2.LINE_AA)
        return img

class ScanConverter():
    """Class that handles visualization of ultrasound images"""
    def __init__(
            self,
            grid,
            norm_mode='max',
            env_mode = 'abs',
            img_buffer_size = 30,
            max_buffer_size = 10,
            norm_factor=2**14,
            dtype='iq'):
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
        self.dynamic_range = 60

        self.persistence_mode = 'MA'
        self.n_persistence = 1 # Number of frames to average over for MA persistence
        self.alpha = 0.5 # AR persistance parameter

        self.dtype = dtype # Currently only supports iq data

        # Scaling settings
        self.grid = grid
        self.Nx = self.grid.shape[1]
        self.Nz = self.grid.shape[0]
        self.width = self.grid[:,:,0].max()-self.grid[:,:,0].min()
        self.height = self.grid[:,:,2].max()-self.grid[:,:,2].min()
        self.aspect_ratio = self.width/self.height
        self.aspect_scaling_x = self.aspect_ratio/(self.Nx/self.Nz)

        # viewport width divided by number of horizontal pixels after aspect scaling
        self.scale = 500/(self.Nx*self.aspect_scaling_x)


    def convert(self, img):
        """Conversion function that applies all transformations"""
        img = self.envelope(img)
        img = self.normalize(img)
        img = self.compression(img)
        img = self.remove_nan_and_inf(img)
        #img = self.resize(img)
        #img = self.persistence(img)
        img = np.clip(img, -60, 0)
        img = ((img + 60)*(255/60)).astype('uint8')

        return img


    def resize(self, img):
        """Function that resizes the image"""
        img = cv2.resize(
            img,
            dsize=(
                int(self.scale * self.Nx * self.aspect_scaling_x),
                int(self.scale * self.Nz)
            )
        )
        return img

    def persistence(self, img):
        """Function that applies persistance to the image"""
        if self.persistence_mode == 'MA':
            self.img_buffer.append(img)
            img = self.persistence_MA()
        elif self.persistence_mode == 'AR':
            img = self.persistence_AR(img)
            self.img_buffer.append(img)
        return img

    def persistence_MA(self):
        """Function that applies moving average persistence to the image"""
        if self.n_persistence > 1:
            max_index = np.minimum(len(self.img_buffer), self.n_persistence)
            img = np.mean(np.array([self.img_buffer[i] for i in range(max_index)]), axis=0)
        else:
            img = self.img_buffer[0]
        return img

    def persistence_AR(self, img):
        """Function that applies autoregressive persistence to the image"""
        img = (1-self.alpha)*img + (self.alpha)*self.img_buffer[0]
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

    def envelope(self, img):
        """Envelope detection"""
        if self.dtype == 'iq':
            return np.linalg.norm(img, axis=-1)
        else:
            raise NotImplementedError

    @staticmethod
    def compression(img):
        """Logarithmic compression"""
        return 20*np.log10(img)

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

    def remove_nan_and_inf(self, img):
        """Function for removing nan and inf values"""
        img[np.isnan(img)] = -self.dynamic_range
        img[np.isinf(img)] = -self.dynamic_range
        return img


class ScanConverterTF(ScanConverter):
    """ScanConverter class for converting raw data to images using tensorflow"""

    #@tf.function(jit_compile=True)
    def convert(self, img):
        """Conversion function that applies all transformations"""
        img = self.envelope(img)
        img = self.normalize(img)
        img = self.compression(img)
        img = self.remove_nan_and_inf(img)
        img = self.resize(img)
        #img = self.persistence(img)
        img = tf.clip_by_value(img, -self.dynamic_range, 0)
        img = tf.cast((img + self.dynamic_range)*(255./self.dynamic_range), tf.uint8)

        return img

    def resize(self, img):
        """Function that resizes the image"""
        img = tf.expand_dims(img, axis=-1)
        img = tf.image.resize(
            img,
            size=(
                int(self.scale * self.Nz),
                int(self.scale * self.Nx  * self.aspect_scaling_x)
            )
        )
        img = tf.squeeze(img, axis=-1)
        return img

    @staticmethod
    def remove_nan_and_inf(img):
        """Function for removing nan and inf values"""
        img = tf.where(tf.math.is_nan(img), -60., img)
        img = tf.where(tf.math.is_inf(img), -60., img)
        return img

    @staticmethod
    def compression(img):
        """Logarithmic compression"""
        return 20*tf.math.log(img)/tf.math.log(10.)

    def envelope(self, img):
        """Envelope detection"""
        if self.dtype == 'iq':
            return tf.norm(img, axis=-1)
        else:
            raise ValueError('Envelope detection only supported for IQ data')


    def normalize(self, img):
        "Normalization function"
        max_val = tf.reduce_max(img)
        self.max_buffer.append(max_val)

        if self.norm_mode == 'normal':
            img = img/max_val
        elif self.norm_mode == 'smoothnormal':
            img = img/tf.reduce_mean(list(self.max_buffer))
        elif self.norm_mode == 'fixed':
            img = img/self.norm_factor
        else:
            img = img/max_val

        return img