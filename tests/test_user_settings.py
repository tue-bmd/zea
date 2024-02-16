"""User settings testing
"""

import pytest

from usbmd.common import set_data_paths


def test_set_data_paths(monkeypatch):
    """Test set data paths"""
    # Will raise OSError because input() cannot be called in test
    with pytest.raises(OSError):
        set_data_paths()
    set_data_paths(user_config="users.yaml", local=False)
    set_data_paths(user_config="users.yaml", local=True)
    user_config = {
        "myusername": {"hostname": {"data_root": "C:/path_to_my_data_root/"}}
    }
    set_data_paths(user_config)
