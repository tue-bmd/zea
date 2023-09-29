"""Test the registration decorator"""
import pytest

from usbmd.utils.registration import RegisterDecorator


def test_names():
    """Test the register decorator"""
    registry = RegisterDecorator()

    @registry("A")
    class ClassA:
        """Test class A."""

    @registry(name="B")
    class ClassB:
        """Test class B."""

    assert registry["A"] == ClassA, "ClassA should be registered as A"
    assert registry["B"] == ClassB, "ClassB should be registered as B"
    assert registry["a"] == ClassA, "Keys must be case insensitive"


def test_getitem():
    """Test the getitem method of the register decorator"""
    registry = RegisterDecorator()

    @registry("A")
    class ClassA:
        """Test class A."""

    assert registry["A"] == ClassA, "Key should have linked to class"
    assert (
        registry["A"] == registry.registry["a"]
    ), "getitem should be equivalent to indexing self.registry."


def test_duplicate_name():
    """Test if the decorator raises an error when called with a name that is
    already registered."""
    registry = RegisterDecorator()

    # pylint: disable=unused-variable
    @registry("test")
    class TestClass:
        """Test class."""

    # pylint: disable=unused-variable
    with pytest.raises(AssertionError):

        @registry("test")
        class TestClass2:
            """Test class 2."""


def test_additional_parameters():
    """Test if the decorator can register additional parameters"""
    registry = RegisterDecorator(["lucky_number"])

    # pylint: disable=unused-variable
    @registry("A", lucky_number=8)
    class ClassA:
        """Test class A."""

    assert (
        registry.get_parameter("A", "lucky_number") == 8
    ), "Failed to retrieve additional parameter"


def test_requesting_nonexistent_parameter():
    """Test if the decorator raises an error when a parameter is requested
    that was not registered."""
    registry = RegisterDecorator()

    # pylint: disable=unused-variable
    @registry("A")
    class ClassA:
        """Test class A."""

    with pytest.raises(KeyError):
        registry.get_parameter("A", "lucky_number")
