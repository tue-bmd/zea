"""Extent numpy ops a bit for main ops module."""

import numpy as np

np.cast = lambda x, dtype: x.astype(dtype)
np.convert_to_tensor = lambda x: x
np.complex = lambda x, y: x + 1j * y
np.permute = np.transpose
