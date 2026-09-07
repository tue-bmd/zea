"""Validate ``users.yaml`` files.

A ``users.yaml`` file maps a user and a machine to the data paths to use on it,
see :mod:`zea.datapaths`. This module gives that file a schema, in the same
dataclass-Spec style used for the zea config in
:mod:`zea.internal.config.validation` (and for the array specs in
:mod:`zea.data.spec`): a :class:`~zea.internal.config.validation.ConfigSpec`
subclass declaring its fields, their defaults and their validators.

The file is *recursive*: the same three keys (``system``, ``data_root``,
``output``) may appear at the top level, inside a username block, and inside a
hostname block within that. Every other key is therefore read as a nested user
or machine section, validated with the same spec::

    alice:                      # username section
      workstation:              # hostname section
        system: linux
        data_root:
          local: /mnt/data/alice
          remote: /mnt/remote/alice
        output: /mnt/data/alice/output
      data_root: /mnt/data/alice        # fallback for alice's other machines
    data_root: /mnt/shared/data         # fallback for everybody else

Unlike the zea config schema, defaults are **not** filled in when serializing
back to a dict: an unset key here is meaningful, it means "fall back to the level
above". An explicit ``null`` says the same thing and is treated identically, so
neither is carried into the validated output.
"""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional

from zea.internal.config.validation import (
    ConfigSpec,
    optional,
    string,
    string_or_path,
)

#: The sub keys a ``data_root`` / ``output`` mapping may define.
LOCAL_REMOTE_KEYS = ("local", "remote")


def local_remote_paths(value: Any) -> dict:
    """Validate a ``{local: ..., remote: ...}`` mapping of paths."""
    if not isinstance(value, dict):
        raise ValueError(f"must be a mapping, got {type(value).__name__}")
    if not value:
        raise ValueError(f"must define at least one of {list(LOCAL_REMOTE_KEYS)}")
    unexpected = sorted(str(key) for key in value if key not in LOCAL_REMOTE_KEYS)
    if unexpected:
        raise ValueError(f"unexpected keys {unexpected}, expected {list(LOCAL_REMOTE_KEYS)}")
    for key, path in value.items():
        try:
            string_or_path(path)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
    return value


def path_or_local_remote(value: Any) -> Any:
    """Validate a path, or a mapping with ``local`` and / or ``remote`` paths."""
    if isinstance(value, dict):
        return local_remote_paths(value)
    return string_or_path(value)


@dataclass
class UserProfileSpec(ConfigSpec):
    """One section of a ``users.yaml`` file.

    The same shape applies at every level of the file — the top level, a username
    block, and a hostname block inside it — so any key that is not one of the
    fields below is validated as a nested user or machine section.
    """

    system: Any = None
    data_root: Any = None
    output: Any = None

    ALLOW_EXTRA: ClassVar[bool] = True
    VALIDATORS: ClassVar[dict[str, Callable[[Any], Any]]] = {
        "system": optional(string),
        "data_root": optional(path_or_local_remote),
        "output": optional(path_or_local_remote),
    }

    @classmethod
    def from_dict(cls, dictionary: Optional[dict]) -> "UserProfileSpec":
        """Validate ``dictionary``, recursing into the user / machine sections."""
        obj = super().from_dict(dictionary)
        for name, value in list(obj._extra.items()):
            if not isinstance(value, dict):
                raise ValueError(
                    f"{cls.__name__}.{name}: expected a mapping for a user or machine "
                    f"section, got {type(value).__name__}. Only {list(cls.field_names())} "
                    "may be set to a value directly."
                )
            try:
                obj._extra[name] = cls.from_dict(value)
            except ValueError as exc:
                raise ValueError(f"{name}: {exc}") from exc
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict, keeping only the keys this section actually sets.

        Defaults are deliberately not filled in: an unset ``data_root`` means "fall
        back to the level above", and :func:`zea.datapaths.set_data_paths` resolves it
        against the enclosing sections. An explicit ``null`` expresses the same intent,
        so it is dropped here rather than kept as a value that overrides the fallback.
        """
        result = super().to_dict()
        declared = set(self.field_names())
        return {
            key: value for key, value in result.items() if value is not None or key not in declared
        }


def validate_users_config(users_config: Optional[dict]) -> dict:
    """Validate a ``users.yaml`` dict and return it as a plain dict.

    The ``users.yaml`` counterpart of
    :func:`zea.internal.config.validation.validate_config`.

    Args:
        users_config (dict, optional): The contents of a ``users.yaml`` file.
            None is treated as an empty file.

    Returns:
        dict: The validated config. Keys that the file does not set are left out.

    Raises:
        ValueError: If the file does not follow the ``users.yaml`` schema.
    """
    return UserProfileSpec.from_dict(users_config).to_dict()
