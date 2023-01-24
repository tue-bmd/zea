"""Basic testing
"""
import sys
from pathlib import Path

import numpy as np
import pytest

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd.processing import companding
from usbmd.tensorflow_ultrasound.processing import companding_tf


@pytest.mark.parametrize('comp_type, size, lib', [
    ('a', (2, 1, 128, 32), 'numpy'),
    ('a', (512, 512), 'numpy'),
    ('mu', (2, 1, 128, 32), 'numpy'),
    ('mu', (512, 512), 'numpy'),
    ('a', (2, 1, 128, 32), 'tensorflow'),
    ('a', (512, 512), 'tensorflow'),
    ('mu', (2, 1, 128, 32), 'tensorflow'),
    ('mu', (512, 512), 'tensorflow'),
])
def test_companding(comp_type, size, lib):
    """Test companding function"""
    signal = np.clip((np.random.random(size) - 0.5) *2, -1, 1)
    signal = signal.astype(np.float32)

    if lib == 'tensorflow':
        compand_func = companding_tf
    elif lib == 'numpy':
        compand_func = companding

    signal_out = compand_func(signal, expand=False, comp_type=comp_type)
    signal_out = compand_func(signal_out, expand=True, comp_type=comp_type)

    return np.testing.assert_almost_equal(signal, signal_out, decimal=6)
