"""Create a new config file by asking the user for input.

- **Author(s)**     : Tristan Stevens
- **Date**          : 29/02/2024
"""

import sys
from pathlib import Path

import schema

from usbmd.config import Config
from usbmd.config.comments import DESCRIPTIONS, add_comments_to_yaml
from usbmd.config.validation import check_config, config_schema
from usbmd.utils import get_date_string, strtobool
from usbmd.utils.log import green, red


def _get_input_value(config, schema_key, schema_value, descriptions):
    while True:
        input_val = input(f"Enter a value for {schema_key}: ")
        if not isinstance(schema_key, str):
            _key = schema_key.key
        else:
            _key = schema_key
        if input_val == "help":
            if _key not in descriptions:
                print(red(f"No description available for {_key}"))
                continue
            print("\t" + green(descriptions[_key]))
            continue
        try:
            config[_key] = input_val
            if isinstance(schema_value, schema.And):
                for _type in schema_value.args:
                    try:
                        config[_key] = _type(config[_key])
                        break
                    except Exception:
                        pass
                schema_value.validate(config[_key])
            else:
                schema_value(config[_key])
            break
        except Exception as e:
            print(f"Invalid input: {red(e)}")
    return config


def create_config():
    """Create a new config file by asking the user for input."""

    def _ask_user_input(config, schema_obj, descriptions):
        for key, value in schema_obj.schema.items():
            if isinstance(value, schema.Schema):
                if not isinstance(key, str):
                    _key = key.key
                else:
                    _key = key
                if isinstance(key, schema.Optional):
                    # skip optional keys
                    continue
                config[_key] = _ask_user_input(
                    config.setdefault(_key, {}), value, descriptions[_key]
                )
            elif not isinstance(key, schema.Optional):
                config = _get_input_value(config, key, value, descriptions)

        return config

    config = {}
    _ask_user_input(config, config_schema, DESCRIPTIONS)

    # Ask user if they want to change any optional keys
    while True:
        try:
            key = None
            input_val = input("Do you want to change any optional keys? (yes/no): ")
            change_optional = strtobool(input_val)

            if change_optional:
                key = input("Enter the key name (e.g., 'model/beamformer/param'): ")
                keys = key.split("/")
                base_schemas = [
                    "data",
                    "plot",
                    "model",
                    "preprocess",
                    "postprocess",
                    "scan",
                ]

                if len(keys) > 1:
                    if keys[0] not in base_schemas:
                        print(red(f"Invalid key {key}, please try again."))
                        continue

                if len(keys) == 1:
                    if keys[0] in base_schemas:
                        print(
                            red(
                                f"Invalid key, cannot be part of base keys {base_schemas} "
                                "please try again."
                            )
                        )
                        continue

                nested_dict = config
                for k in keys[:-1]:
                    nested_dict = nested_dict.setdefault(k, {})

                # retrieve schema value from the nested key
                schema_obj = config_schema
                for k in keys:
                    sub_keys = [
                        s.key if not isinstance(s, str) else s
                        for s in schema_obj.schema.keys()
                    ]

                    schema_key = list(schema_obj.schema.keys())[sub_keys.index(k)]

                    schema_obj = schema_obj.schema[schema_key]

                descriptions = DESCRIPTIONS
                for k in keys[:-1]:
                    descriptions = descriptions[k]

                nested_dict = _get_input_value(
                    nested_dict, keys[-1], schema_obj, descriptions
                )
            else:
                print("No optional keys will be changed.")
                break
        except KeyboardInterrupt:
            print(red("KeyboardInterrupt, exiting."))
            sys.exit()
        except:
            if key is None:
                print(red("Invalid input, please try again."))
            else:
                print(red(f"Invalid key: {key}, please try again."))
            continue

    return config


if __name__ == "__main__":
    print(
        f"Let's create a new config file 🪄\n"
        f"You can always type {green('help')} "
        "to get a description of the parameter."
    )
    config = create_config()
    print(config)

    config = check_config(config)

    # Save the config to a YAML file
    name = input("Enter a name for the config: ")
    timestamp = get_date_string()

    custom_configs_folder = Path("custom_configs")
    custom_configs_folder.mkdir(exist_ok=True)
    filename = custom_configs_folder / f"{timestamp}_{name}.yaml"

    Config(config).save_to_yaml(filename)

    add_comments_to_yaml(filename, DESCRIPTIONS)

    print(f"Find your config at {str(filename)}")
