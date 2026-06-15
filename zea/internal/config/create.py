"""Create a new config file by asking the user for input."""

import sys
from pathlib import Path

import yaml

from zea.config import Config, check_config
from zea.internal.config.parameters import PARAMETER_DESCRIPTIONS
from zea.internal.config.validation import ConfigSchema
from zea.log import green, red
from zea.utils import get_date_string, strtobool


def _get_input_value(config, key, validator, descriptions):
    """Prompt for a value for ``key``, parse it as YAML, and validate it."""
    while True:
        input_val = input(f"Enter a value for {key}: ")
        if input_val == "help":
            desc = descriptions.get(key) if isinstance(descriptions, dict) else None
            if not desc:
                print(red(f"No description available for {key}"))
            else:
                print("\t" + green(desc))
            continue
        try:
            # YAML parsing mirrors how config files are actually loaded, so e.g.
            # "5" -> int, "true" -> bool, "[1, 2]" -> list, "all" -> str.
            parsed = yaml.safe_load(input_val)
            if validator is not None:
                validator(parsed)
            config[key] = parsed
            break
        except Exception as e:  # noqa: BLE001 - report any parse/validation error and retry
            print(f"Invalid input: {red(e)}")
    return config


def _resolve_spec_field(keys):
    """Resolve a (slash separated) key path to its validator and descriptions.

    Unknown sections/keys resolve to ``None`` validator (extra keys are allowed).
    """
    spec_cls = ConfigSchema
    descriptions = PARAMETER_DESCRIPTIONS
    for k in keys[:-1]:
        spec_cls = spec_cls.NESTED.get(k) if spec_cls is not None else None
        if isinstance(descriptions, dict):
            descriptions = descriptions.get(k, {})
    validator = spec_cls.VALIDATORS.get(keys[-1]) if spec_cls is not None else None
    return validator, descriptions


def create_config():
    """Create a new config file by asking the user for input."""

    def _ask_user_input(config, spec_cls, descriptions):
        for name in spec_cls.required_fields():
            nested = spec_cls.NESTED.get(name)
            if nested is not None:
                sub_desc = descriptions.get(name, {}) if isinstance(descriptions, dict) else {}
                config[name] = _ask_user_input(config.setdefault(name, {}), nested, sub_desc)
            else:
                validator = spec_cls.VALIDATORS.get(name)
                config = _get_input_value(config, name, validator, descriptions)
        return config

    config = {}
    _ask_user_input(config, ConfigSchema, PARAMETER_DESCRIPTIONS)

    # Sections that are validated nested specs (cannot be set as a single value).
    base_schemas = list(ConfigSchema.NESTED)

    # Ask user if they want to change any optional keys
    while True:
        try:
            key = None
            input_val = input("Do you want to change any optional keys? (yes/no): ")
            change_optional = strtobool(input_val)

            if change_optional:
                key = input("Enter the key name (e.g., 'parameters/grid_size_x'): ")
                keys = key.split("/")

                if len(keys) > 1 and keys[0] not in base_schemas:
                    print(red(f"Invalid key {key}, please try again."))
                    continue
                if len(keys) == 1 and keys[0] in base_schemas:
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

                validator, descriptions = _resolve_spec_field(keys)
                nested_dict = _get_input_value(nested_dict, keys[-1], validator, descriptions)
            else:
                print("No optional keys will be changed.")
                break
        except KeyboardInterrupt:
            print(red("KeyboardInterrupt, exiting."))
            sys.exit()
        except Exception:
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

    Config(config).to_yaml(filename)

    print(f"Find your config at {str(filename)}")
