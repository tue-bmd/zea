"""Tests for the core Object class."""

import timeit

import numpy as np
import pytest

from zea.internal.core import Object, serialize_elements

from . import DEFAULT_TEST_SEED


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


class _JitObj(Object):
    """Object holding a ``torch.compile``d bound method (like ``DDS._call``)."""

    def __init__(self, k):
        super().__init__()
        import torch

        self.k = k
        self._call = torch.compile(self.call)

    def call(self, x):
        return x * self.k


class SomeObj(Object):
    """Test object with random data"""

    def __init__(self):
        super().__init__()
        self.data = np.random.rand(2**16)
        self.vars1 = np.random.rand(2**10)
        self.vars2 = 0


def test_equality():
    """Test the equality of the Object class"""
    # Create 3 objects, 2 of which are equal
    np.random.seed(DEFAULT_TEST_SEED)
    obj1 = SomeObj()
    np.random.seed(DEFAULT_TEST_SEED)
    obj2 = SomeObj()
    obj3 = SomeObj()

    assert obj1 == obj2
    assert obj1 != obj3


def test_timing():
    """Test the timing of the equality comparison"""
    # TODO: this test only prints, no assertions

    # Create 3 objects, 2 of which are equal
    np.random.seed(DEFAULT_TEST_SEED)
    obj1 = SomeObj()
    np.random.seed(DEFAULT_TEST_SEED)
    obj2 = SomeObj()
    obj3 = SomeObj()

    print(f"obj1 == obj2: {obj1 == obj2}")
    print(f"obj1 == obj3: {obj1 == obj3}")

    print("timing the comparison:")

    N = 1

    # compare without changing the object
    time_cached = timeit.timeit(lambda: obj1 == obj2, number=N)
    print(f"obj1 == obj2: {time_cached:.2f}, or: {time_cached / N * 1000:.2f}(ms) per comparison")

    # compare while changing the object in between
    def _time_with_change(obj1, obj2):
        obj1.vars2 += 1
        return obj1 == obj2

    time_non_cached = timeit.timeit(lambda: _time_with_change(obj1, obj2), number=N)
    print(
        f"obj1 == obj2: {time_non_cached:.2f}, or: "
        f"{time_non_cached / N * 1000:.2f}(ms) per comparison"
    )


def test_serialize_torch_compiled_function():
    """A ``torch.compile``d function serializes like the function it wraps."""
    torch = pytest.importorskip("torch")

    compiled = torch.compile(_plain_function)
    serialized = serialize_elements([compiled])
    assert serialized == "compiled:" + serialize_elements([_plain_function])
    # Stable across separate compilations of the same target
    assert serialize_elements([torch.compile(_plain_function)]) == serialized


def test_serialize_object_with_compiled_method():
    """An Object with a ``torch.compile``d bound method still serializes."""
    pytest.importorskip("torch")

    assert f"compiled:{__name__}._JitObj.call" in _JitObj(2).serialized
    assert _JitObj(2).serialized != _JitObj(3).serialized


def test_serialize_unpicklable_object_raises_type_error():
    """Unpicklable, non-compiled elements raise TypeError (no silent fallback)."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_Unpicklable()])


def test_serialize_non_jit_wrapped_raises_type_error():
    """Regression: ``__wrapped__`` is not JIT-only; a generic wrapped callable
    must raise TypeError rather than collapse to the inner function's key."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_UnpicklableWrapped()])
