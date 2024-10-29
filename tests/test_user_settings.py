"""User settings testing
"""

import getpass
import socket

import pytest

from usbmd.datapaths import set_data_paths

user_config0 = {
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

user_config1 = {
    getpass.getuser(): {
        socket.gethostname(): {
            "data_root": {
                "local": "C:/path_to_my_output/",
                "remote": "Z:/path_to_my_output/",
            },
        }
    }
}

user_config2 = {
    "data_root": {
        "local": "C:/path_to_my_data_root/",
        "remote": "Z:/path_to_my_data_root/",
    },
    "output": {
        "local": "C:/path_to_my_output/",
        "remote": "Z:/path_to_my_output/",
    },
}


@pytest.mark.parametrize(
    "user_config", [user_config0, user_config1, user_config2, "users.test.yaml"]
)
def test_set_data_paths(tmp_path, user_config):
    """Test set data paths"""

    if isinstance(user_config, str):
        # Add temp path and set as string
        user_config = str(tmp_path / user_config)

    for local in [True, False]:
        data_paths = set_data_paths(user_config, local=local)
        assert (
            "data_root" in data_paths
        ), f"data_root not in data_paths for local={local}"
        assert "output" in data_paths, f"output not in data_paths for local={local}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
