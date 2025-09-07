"""Test the IO library functionality."""

from unittest.mock import Mock

import imageio
import numpy as np
import pytest
from PIL import Image

from zea.io_lib import load_image, load_video, retry_on_io_error

MAX_RETRIES = 3
INITIAL_DELAY = 0.01


@pytest.fixture
def temp_image(tmp_path):
    arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "test_img.png"
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def temp_gif(tmp_path):
    arrs = [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(5)]
    gif_path = tmp_path / "test_anim.gif"
    Image.fromarray(arrs[0]).save(
        gif_path,
        save_all=True,
        append_images=[Image.fromarray(a) for a in arrs[1:]],
        loop=0,
    )
    return gif_path


@pytest.fixture
def temp_mp4(tmp_path):
    arrs = [np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(5)]
    mp4_path = tmp_path / "test_vid.mp4"
    writer = imageio.get_writer(str(mp4_path), fps=2)
    for arr in arrs:
        writer.append_data(arr)
    writer.close()
    return mp4_path


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


def test_load_image_basic(temp_image):
    arr = load_image(temp_image, mode="L")
    assert arr.shape == (32, 32)
    arr_rgb = load_image(temp_image, mode="RGB")
    assert arr_rgb.shape == (32, 32, 3)


def test_load_video_gif(temp_gif):
    arr = load_video(temp_gif, mode="L")
    assert arr.shape[0] == 5
    assert arr.shape[1:] == (16, 16)
    arr_rgb = load_video(temp_gif, mode="RGB")
    assert arr_rgb.shape == (5, 16, 16, 3)


def test_load_video_mp4(temp_mp4):
    arr = load_video(temp_mp4, mode="L")
    assert arr.shape[0] == 5
    assert arr.shape[1:] == (16, 16)
    arr_rgb = load_video(temp_mp4, mode="RGB")
    assert arr_rgb.shape == (5, 16, 16, 3)
