"""Test the IO library functionality."""

from unittest.mock import Mock

import pytest

from usbmd.io_lib import retry_on_io_error

MAX_RETRIES = 3
INITIAL_DELAY = 0.01


def test_retry_on_io_error_succeeds():
    """Test that the function retries and eventually succeeds."""
    mock_func = Mock(side_effect=[IOError(), IOError(), "success"])

    @retry_on_io_error(max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)
    def test_func():
        return mock_func()

    result = test_func()

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_on_io_error_fails():
    """Test that the function fails after max retries."""
    mock_func = Mock(side_effect=IOError("test error"))

    @retry_on_io_error(max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)
    def test_func():
        return mock_func()

    with pytest.raises(ValueError) as exc_info:
        test_func()

    assert "Failed to complete operation after 3 attempts" in str(exc_info.value)
    assert mock_func.call_count == MAX_RETRIES


def test_retry_action_callback():
    """Test that the retry action callback is called correctly."""
    mock_func = Mock(side_effect=[IOError(), IOError(), "success"])
    retry_action = Mock()

    @retry_on_io_error(
        max_retries=MAX_RETRIES,
        initial_delay=INITIAL_DELAY,
        retry_action=retry_action,
    )
    def test_func():
        return mock_func()

    result = test_func()

    assert result == "success"
    assert retry_action.call_count == MAX_RETRIES - 1  # Called for first two failures

    # callback is passed both the exception and the retry count
    for i in range(retry_action.call_count):
        kwargs = retry_action.call_args_list[i][1]
        assert isinstance(kwargs["exception"], IOError)
        assert kwargs["retry_count"] == i
