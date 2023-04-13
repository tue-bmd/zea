"""User settings testing
"""
import sys
from pathlib import Path

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from usbmd.common import set_data_paths

def test_set_data_paths():
    """Test set data paths"""
    set_data_paths()
    set_data_paths(local=False, user_config_path='users.yaml')
    set_data_paths(local=True, user_config_path='users.yaml')
