"""Fixture module for testing paths B and C inside get_ops.

Neither class is decorated with ``@ops_registry``, so
``ops_registry.get_name(cls)`` raises ``KeyError`` for both (path B).

* ``UnregisteredOp`` — unique name, not in registry → path B caught, path C
  check also fails → ``ValueError`` raised.
* ``Identity`` — lower-cased name ``"identity"`` IS in the registry (the
  built-in :class:`~zea.ops.base.Identity`) → path B caught, path C shortname
  fallback succeeds and returns the built-in class.
"""

from zea.ops.base import Operation


class UnregisteredOp(Operation):
    """Importable but has no registry entry — exercises path B + ValueError."""

    def call(self, data, **kwargs):
        return {"data": data}


class Identity(Operation):
    """Importable, no registry entry, but ``"identity"`` IS a registry key.

    Exercises path B (``KeyError`` caught) followed by path C (shortname
    fallback returns the built-in :class:`~zea.ops.base.Identity`).
    """

    def call(self, data, **kwargs):
        return {"data": data}
