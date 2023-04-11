import pytest
from usbmd.utils.registry import RegisterDecorator

def test_names():
    """Test the register decorator"""
    registry = RegisterDecorator()

    @registry('A')
    class ClassA:
        pass

    @registry(name='B')
    class ClassB:
        pass

    assert registry['A'] == ClassA, 'ClassA should be registered as A'
    assert registry['B'] == ClassB, 'ClassB should be registered as B'
    assert registry['a'] == ClassA, 'Keys must be case insensitive'


def test_getitem():
    """Test the getitem method of the register decorator"""
    registry = RegisterDecorator()

    @registry('A')
    class ClassA:
        pass


    assert registry['A'] == ClassA, 'Key should have linked to class'
    assert registry['A'] == registry.registry['a'], ('getitem should be'\
        ' equivalent to indexing self.registry.')


def test_duplicate_name():
    """Test if the decorator raises an error when called with a name that is
    already registered."""
    registry = RegisterDecorator()

    @registry('test')
    class TestClass:
        pass

    with pytest.raises(AssertionError):
        @registry('test')
        class TestClass2:
            pass

def test_additional_parameters():
    """Test if the decorator can register additional parameters"""
    registry = RegisterDecorator(['lucky_number'])

    @registry('A', lucky_number=8)
    class ClassA:
        pass

    assert registry.get_parameter('A', 'lucky_number') == 8, ('Failed to '\
        'retrieve additional parameter')

def test_requesting_nonexistent_parameter():
    """Test if the decorator raises an error when a parameter is requested
    that was not registered."""
    registry = RegisterDecorator()

    @registry('A')
    class ClassA:
        pass

    with pytest.raises(KeyError):
        registry.get_parameter('A', 'lucky_number')
