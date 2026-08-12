"""Tests for zea.internal.core."""

import numpy as np
import pytest

from zea.internal.core import dict_to_tensor, serialize_elements


def _plain_function(x):
    """Module-level (thus picklable) function for serialization tests."""
    return x + 1


class _Unpicklable:
    """Callable that always fails pickling."""

    def __reduce__(self):
        raise RuntimeError("cannot pickle")

    def __call__(self):
        return None


class _UnpicklableWrapped(_Unpicklable):
    """Non-JIT callable that merely carries a ``__wrapped__`` attribute."""

    __wrapped__ = _plain_function

    def __call__(self, x):
        return self.__wrapped__(x)


class _JitObj:
    """Object holding a ``torch.compile``d bound method (like ``DDS._call``)."""

    def __init__(self, k):
        import torch

        self.k = k
        self._call = torch.compile(self.call)

    def call(self, x):
        return x * self.k


def test_dict_to_tensor_ignores_framework_to_tensor_methods():
    """Objects with a `to_tensor` method whose signature doesn't accept `keep_as_is`
    (e.g. `tf.RaggedTensor`) must not be dispatched through zea's conversion protocol.
    """
    tf = pytest.importorskip("tensorflow")

    ragged = tf.ragged.constant([[1, 2], [3]])
    result = dict_to_tensor({"ragged": ragged, "value": 1.0})

    assert result["ragged"] is ragged
    assert np.asarray(result["value"]) == 1.0


def test_serialize_torch_compiled_function():
    """A ``torch.compile``d function serializes like the function it wraps."""
    torch = pytest.importorskip("torch")

    compiled = torch.compile(_plain_function)
    serialized = serialize_elements([compiled])
    assert serialized == "compiled:" + serialize_elements([_plain_function])
    # Stable across separate compilations of the same target
    assert serialize_elements([torch.compile(_plain_function)]) == serialized


def test_serialize_compiled_bound_method_of_unpicklable_instance():
    """A ``torch.compile``d bound method still serializes, via its module and qualname.

    The instance cannot be pickled (it holds the compiled method), so the target
    itself is unpicklable too and the key falls back to the unbound function's
    module and qualname.
    """
    pytest.importorskip("torch")

    assert serialize_elements([_JitObj(2)._call]) == f"compiled:{__name__}._JitObj.call"

    # That constant fallback must not swallow the rest of the object's state.
    obj2, obj3 = _JitObj(2), _JitObj(3)
    assert serialize_elements([obj2.k, obj2._call]) != serialize_elements([obj3.k, obj3._call])


def test_serialize_unpicklable_object_raises_type_error():
    """Unpicklable, non-compiled elements raise TypeError (no silent fallback)."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_Unpicklable()])


def test_serialize_non_jit_wrapped_raises_type_error():
    """Regression: ``__wrapped__`` is not JIT-only; a generic wrapped callable
    must raise TypeError rather than collapse to the inner function's key."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_UnpicklableWrapped()])
