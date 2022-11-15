"""Basic testing
"""
import sys
from pathlib import Path

import numpy as np
import pytest

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd.processing import companding

@pytest.mark.parametrize('comp_type, size', [
    ('a', (2, 1, 128, 32)),
    ('a', (512, 512)),
    ('mu', (2, 1, 128, 32)),
    ('mu', (512, 512)),
])
def test_companding(comp_type, size):
    """Test companding function"""
    signal = np.clip((np.random.random(size) - 0.5) *2, -1, 1)

    signal_out = companding(signal, expand=False, comp_type=comp_type)
    signal_out = companding(signal_out, expand=True, comp_type=comp_type)

    return np.testing.assert_almost_equal(signal, signal_out)
