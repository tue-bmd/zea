"""Custom operation classes used as test fixtures for get_ops / pipeline-config tests.

These classes are intentionally registered under names that differ from their
class names, so tests can verify that :func:`~zea.ops.base.get_ops` resolves
operations by class-object identity rather than by class name.
"""

from zea.internal.registry import ops_registry
from zea.ops.base import Operation


@ops_registry("fixture_scale_op")
class ScaleByFactorOp(Operation):
    """Multiply ``data`` by a scalar *factor*.

    Registry key ``"fixture_scale_op"`` intentionally differs from the class
    name ``ScaleByFactorOp`` to exercise the identity-based lookup path in
    :func:`~zea.ops.base.get_ops`.
    """

    def __init__(self, factor: float = 3.0, **kwargs):
        """
        :param factor: Scalar multiplier applied to *data*.
        :type factor: float
        """
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, data, **kwargs):
        return {"data": data * self.factor}


@ops_registry("fixture_passthrough")
class Fixturepassthrough(Operation):
    """Return *data* unchanged.

    Registry key ``"fixture_passthrough"`` matches ``Fixturepassthrough``
    when lower-cased, so this class exercises the shortname-fallback path that
    was already present before the identity-based fix.
    """

    def call(self, data, **kwargs):
        return {"data": data}
