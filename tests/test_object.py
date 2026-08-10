"""Tests for the core Object class."""

import numpy as np

from zea.internal.core import Object

from . import DEFAULT_TEST_SEED


class SomeObj(Object):
    """Test object with random data"""

    def __init__(self):
        super().__init__()
        self.data = np.random.rand(2**16)
        self.vars1 = np.random.rand(2**10)
        self.vars2 = 0


def test_serialized_is_content_based():
    """`serialized` is the content checksum the caching machinery relies on."""
    # Create 3 objects, 2 of which hold the same data
    np.random.seed(DEFAULT_TEST_SEED)
    obj1 = SomeObj()
    np.random.seed(DEFAULT_TEST_SEED)
    obj2 = SomeObj()
    obj3 = SomeObj()

    assert obj1.serialized == obj2.serialized
    assert obj1.serialized != obj3.serialized


def test_serialized_invalidated_on_mutation():
    """Mutating an object invalidates the cached checksum."""
    np.random.seed(DEFAULT_TEST_SEED)
    obj = SomeObj()

    before = obj.serialized
    obj.vars2 += 1

    assert obj.serialized != before
