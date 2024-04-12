"""User settings testing
"""

import getpass
import socket
from pathlib import Path

import pytest

from usbmd.common import set_data_paths


def test_set_data_paths():
    """Test set data paths"""

    user_config_path = "users.test.yaml"

    # Test set_data_paths with default user_config
    set_data_paths(user_config_path, local=True)
    set_data_paths(user_config_path, local=False)

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

    # Test with custom user_config that does not have the optional output key
    user_config = {
        getpass.getuser(): {
            socket.gethostname(): {
                "data_root": {
                    "local": "C:/path_to_my_output/",
                    "remote": "Z:/path_to_my_output/",
                },
            }
        }
    }
    data_paths = set_data_paths(user_config)
    assert "data_root" in data_paths
    data_paths = set_data_paths(user_config, local=False)
    assert "data_root" in data_paths

    # clean up
    Path(user_config_path).unlink()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
