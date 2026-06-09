"""Fixture module for testing path A inside get_ops.

``DottedKeyOp`` is registered under its full module path as the key.  When
``get_ops("tests.fixtures.dotted_key_ops.DottedKeyOp")`` is called for the
first time (before this module has been imported), the module is imported, the
decorator fires, and the second ``ops_name in ops_registry`` check (path A)
inside the dotted-path branch succeeds.

This file must **not** be imported by any other fixture or test file, so that
the module is fresh when the path-A test runs.
"""

from zea.internal.registry import ops_registry
from zea.ops.base import Operation


@ops_registry("tests.fixtures.dotted_key_ops.DottedKeyOp")
class DottedKeyOp(Operation):
    """Registered under its full dotted module path as the registry key."""

    def call(self, data, **kwargs):
        return {"data": data}
