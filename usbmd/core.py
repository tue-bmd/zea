import pickle
import timeit
import numpy as np


class Object:
    """Base class for all data objects in the toolbox"""

    def __init__(self):
        self._serialized = None

    @property
    def serialized(self):
        """Compute the checksum of the object only if not already done"""
        if self._serialized is None:
            attributes = self.__dict__.copy()
            attributes.pop(
                "_serialized", None
            )  # Remove the cached serialized attribute to avoid recursion
            self._serialized = pickle.dumps(attributes)
        return self._serialized

    def __setattr__(self, name: str, value):
        """Reset the serialized data if the object is modified"""
        if name != "_serialized":  # Avoid resetting when setting _serialized itself
            self._serialized = None
        super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.serialized == other.serialized

if __name__ == "__main__":

    class TestObj(Object):
        """Test object with random data"""
        def __init__(self):
            super().__init__()
            self.data = np.random.rand(2**16)
            self.vars1 = np.random.rand(2**10)
            self.vars2 = 0

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
    time_equal = timeit.timeit(lambda: obj1 == obj2, number=N)
    time_not_equal = timeit.timeit(lambda: obj1 == obj3, number=N)


    # compare while changing the object in between
    def _time_with_change(obj1, obj2):
        obj1.vars2 += 1
        return obj1 == obj2

    time_non_cached = timeit.timeit(lambda: _time_with_change(obj1, obj2), number=N)
    print(
        f"obj1 == obj2: {time_non_cached:.2f}, or: {time_non_cached/N*1000:.2f}(ms) per comparison"
    )
