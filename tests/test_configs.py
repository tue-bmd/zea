"""Test configs
"""
import sys
from pathlib import Path

import yaml
from schema import SchemaError

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from configs.config_validation import config_schema


def test_configs():
    """Test if configs are valide according to schema"""
    files = Path('./configs').glob('*.yaml')
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            configuration = yaml.load(f, Loader=yaml.FullLoader)
        try:
            print('hoi')
            config_schema.validate(dict(configuration))
        except SchemaError as se:
            raise ValueError(f'Error in config {f}') from se
