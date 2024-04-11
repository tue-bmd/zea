"""User settings testing
"""

import getpass
import socket
from pathlib import Path

import pytest
import yaml

from usbmd.common import DEFAULT_USER, DEFAULT_USERS_CONFIG_PATH, set_data_paths


def test_set_data_paths():
    """Test set data paths"""

    # Create default users.yaml because test cannot handle stdin in create_new_user()
    if not Path(DEFAULT_USERS_CONFIG_PATH).is_file():
        with open(DEFAULT_USERS_CONFIG_PATH, "w", encoding="utf-8") as file:
            yaml.dump(DEFAULT_USER, file, default_flow_style=False)

    # Test set_data_paths
    set_data_paths(local=True)
    set_data_paths(local=False)

    # Test with custom user_config
    user_config = {
        getpass.getuser(): {
            socket.gethostname(): {
                "data_root": "C:/path_to_my_data_root/",
                "output": {
                    "local": "C:/path_to_my_output/",
                    "remote": "Z:/path_to_my_output/",
                },
            }
        }
    }
    set_data_paths(user_config)
    data_paths = set_data_paths(user_config, local=False)
    assert "data_root" in data_paths
    assert "output" in data_paths


if __name__ == "__main__":
    pytest.main(["-v", __file__])
