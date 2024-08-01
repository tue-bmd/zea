import timeit

import numpy as np

from usbmd.core import Object


class TestObj(Object):
    """Test object with random data"""

    def __init__(self):
        super().__init__()
        self.data = np.random.rand(2**16)
        self.vars1 = np.random.rand(2**10)
        self.vars2 = 0


def test_equality():
    # Create 3 objects, 2 of which are equal
    np.random.seed(0)
    obj1 = TestObj()
    np.random.seed(0)
    obj2 = TestObj()
    obj3 = TestObj()

    assert obj1 == obj2
    assert obj1 != obj3


def test_timing():
    # Create 3 objects, 2 of which are equal
    np.random.seed(0)
    obj1 = TestObj()
    np.random.seed(0)
    obj2 = TestObj()
    obj3 = TestObj()

    print(f"obj1 == obj2: {obj1 == obj2}")
    print(f"obj1 == obj3: {obj1 == obj3}")

    print("timing the comparison:")

    N = 1

    # compare without changing the object
    time_cached = timeit.timeit(lambda: obj1 == obj2, number=N)
    print(
        f"obj1 == obj2: {time_cached:.2f}, or: {time_cached/N*1000:.2f}(ms) per comparison"
    )

    # compare while changing the object in between
    def _time_with_change(obj1, obj2):
        obj1.vars2 += 1
        return obj1 == obj2

    time_non_cached = timeit.timeit(lambda: _time_with_change(obj1, obj2), number=N)
    print(
        f"obj1 == obj2: {time_non_cached:.2f}, or: {time_non_cached/N*1000:.2f}(ms) per comparison"
    )
