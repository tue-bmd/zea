"""Test configs
"""
import sys
from pathlib import Path

import yaml
from schema import SchemaError
import pytest

wd = Path(__file__).parent.parent
sys.path.append(str(wd))

from configs.config_validation import config_schema

@pytest.mark.parametrize('file', list(Path('./configs').glob('*.yaml')))
def test_config(file):
    """Test if configs are valide according to schema"""
    with open(file, 'r', encoding='utf-8') as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    try:
        config_schema.validate(dict(configuration))
    except SchemaError as se:
        raise ValueError(f'Error in config {f}') from se
