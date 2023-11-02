"""Camus example"""
import os

# Change the working directory to the root of the project by moving up in the
# directory tree until the file 'setup.py' is found.
while not os.path.exists("setup.py"):
    os.chdir("..")

from usbmd.common import set_data_paths
from usbmd.setup_usbmd import setup_config
from usbmd.ui import DataLoaderUI

## load a config fileq
config = setup_config(file="./configs/config_camus.yaml")

## setup data_root through users settings
config.data.user = set_data_paths("./users.yaml", local=config.data.local)

ui = DataLoaderUI(config)
image = ui.run()
