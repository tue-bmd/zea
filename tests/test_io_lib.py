"""Test the IO library functionality."""

from unittest.mock import Mock

import pytest

from usbmd.utils.io_lib import retry_on_io_error


def test_retry_on_io_error_succeeds():
    """Test that the function retries and eventually succeeds."""
    mock_func = Mock(side_effect=[IOError(), IOError(), "success"])

    @retry_on_io_error(max_retries=3, initial_delay=0.1)
    def test_func():
        return mock_func()

    result = test_func()

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_on_io_error_fails():
    """Test that the function fails after max retries."""
    mock_func = Mock(side_effect=IOError("test error"))

    @retry_on_io_error(max_retries=3, initial_delay=0.1)
    def test_func():
        return mock_func()

    with pytest.raises(ValueError) as exc_info:
        test_func()

    assert "Failed to complete operation after 3 attempts" in str(exc_info.value)
    assert mock_func.call_count == 3


def test_retry_action_callback():
    """Test that the retry action callback is called correctly."""
    mock_func = Mock(side_effect=[IOError(), IOError(), "success"])
    retry_action = Mock()

    @retry_on_io_error(max_retries=3, initial_delay=0.1, retry_action=retry_action)
    def test_func():
        return mock_func()

    result = test_func()

    assert result == "success"
    assert retry_action.call_count == 2  # Called for first two failures

    # callback is passed both the exception and the retry count
    assert isinstance(retry_action.call_args_list[0][0][0], IOError)
    for i in range(retry_action.call_count):
        assert retry_action.call_args_list[i][0][1] == i
