"""Basic testing
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd import ui


def test_ui_initialization():
    """Test ui initialization function"""

    config = ui.setup('./configs/config_picmus.yaml')
    _ui = ui.DataLoaderUI(config)
    image = _ui.run()
    plt.close()
    return image
