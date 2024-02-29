from pathlib import Path

import schema
import yaml

from usbmd.utils.config_validation import check_config, config_schema
from usbmd.utils.io_lib import green, red
from usbmd.utils.utils import get_date_string
from usbmd.utils.yaml_comments import DESCRIPTIONS, add_comments_to_yaml


def create_config():
    def ask_user_input(config, schema_obj, descriptions):
        for key, value in schema_obj.schema.items():
            if isinstance(value, schema.Schema):
                if not isinstance(key, str):
                    _key = key.key
                else:
                    _key = key
                if isinstance(key, schema.Optional):
                    # skip optional keys
                    continue
                config[_key] = ask_user_input(
                    config.setdefault(_key, {}), value, descriptions[_key]
                )
            elif not isinstance(key, schema.Optional):
                while True:
                    input_val = input(f"Enter a value for {key}: ")
                    if not isinstance(key, str):
                        _key = key.key
                    else:
                        _key = key
                    if input_val == "help":
                        if _key not in descriptions:
                            print(red(f"No description available for {_key}"))
                            continue
                        print("\t" + green(descriptions[_key]))
                        continue
                    try:
                        config[_key] = input_val
                        if isinstance(value, schema.And):
                            value.validate(config[_key])
                        else:
                            value(config[_key])
                        break
                    except Exception as e:
                        print(f"Invalid input: {red(e)}")
        return config

    config = {}
    ask_user_input(config, config_schema, DESCRIPTIONS)

    return config


print(
    f"Creating a new config file, always use {green('help')} to get a description of the parameter."
)
config = create_config()
# Run the config through check_config
config = check_config(config)

# Save the config to a YAML file
name = input("Enter a name for the config: ")
timestamp = get_date_string()

custom_configs_folder = Path("custom_configs")
custom_configs_folder.mkdir(exist_ok=True)
filename = custom_configs_folder / f"{name}_{timestamp}.yaml"

with open(filename, "w", encoding="utf-8") as file:
    yaml.dump(config, file)

add_comments_to_yaml(filename, DESCRIPTIONS)

print(f"Find your config at {str(filename)}")
