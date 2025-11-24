"""Test the IO library functionality."""

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from zea.io_lib import load_image, load_video, retry_on_io_error, save_video
from . import DEFAULT_TEST_SEED

MAX_RETRIES = 3
INITIAL_DELAY = 0.01


@pytest.fixture
def temp_image(tmp_path):
    """Create a test image file using the save_video function."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    img_path = tmp_path / "test_img.png"
    Image.fromarray(arr).save(img_path)
    return img_path


@pytest.fixture
def temp_gif(tmp_path):
    """Create a test GIF file using the save_video function."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    arrs = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(5)]
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
    """Create a test MP4 file using the save_video function."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    arrs = rng.integers(0, 255, (5, 16, 16, 3), dtype=np.uint8)
    mp4_path = tmp_path / "test_vid.mp4"
    save_video(arrs, mp4_path, fps=2)
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


def test_save_and_load_video_mp4(tmp_path):
    """Test that we can save and load MP4 videos with correct colors."""
    from zea.io_lib import save_video

    # Create a simple test pattern with distinct colors
    n_frames = 3
    height, width = 16, 16
    frames = []

    # Frame 1: Red
    frame1 = np.zeros((height, width, 3), dtype=np.uint8)
    frame1[:, :, 0] = 255  # Red channel
    frames.append(frame1)

    # Frame 2: Green
    frame2 = np.zeros((height, width, 3), dtype=np.uint8)
    frame2[:, :, 1] = 255  # Green channel
    frames.append(frame2)

    # Frame 3: Blue
    frame3 = np.zeros((height, width, 3), dtype=np.uint8)
    frame3[:, :, 2] = 255  # Blue channel
    frames.append(frame3)

    images = np.array(frames)

    # Save to MP4
    mp4_path = tmp_path / "test_roundtrip.mp4"
    save_video(images, mp4_path, fps=10)

    # Load back
    loaded = load_video(mp4_path, mode="RGB")

    # Check shape
    assert loaded.shape == (n_frames, height, width, 3)

    # Check that colors are approximately correct (allow some compression artifacts)
    # Frame 1 should be predominantly red
    assert loaded[0, :, :, 0].mean() > 200  # Red channel should be high
    assert loaded[0, :, :, 1].mean() < 50  # Green should be low
    assert loaded[0, :, :, 2].mean() < 50  # Blue should be low

    # Frame 2 should be predominantly green
    assert loaded[1, :, :, 0].mean() < 50  # Red should be low
    assert loaded[1, :, :, 1].mean() > 200  # Green channel should be high
    assert loaded[1, :, :, 2].mean() < 50  # Blue should be low

    # Frame 3 should be predominantly blue
    assert loaded[2, :, :, 0].mean() < 50  # Red should be low
    assert loaded[2, :, :, 1].mean() < 50  # Green should be low
    assert loaded[2, :, :, 2].mean() > 200  # Blue channel should be high


def test_save_and_load_video_gif(tmp_path):
    """Test that we can save and load GIF videos."""
    from zea.io_lib import save_video

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # Create simple test frames
    n_frames = 3
    height, width = 16, 16
    frames = rng.integers(0, 255, (n_frames, height, width, 3), dtype=np.uint8)

    # Save to GIF
    gif_path = tmp_path / "test_roundtrip.gif"
    save_video(frames, gif_path, fps=10)

    # Load back
    loaded = load_video(gif_path, mode="RGB")

    # Check shape
    assert loaded.shape == (n_frames, height, width, 3)


def test_save_video_grayscale_to_rgb(tmp_path):
    """Test that grayscale videos are properly converted to RGB."""
    from zea.io_lib import save_video

    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    # Create grayscale frames
    n_frames = 2
    height, width = 16, 16
    frames = rng.integers(0, 255, (n_frames, height, width), dtype=np.uint8)

    # Save to MP4
    mp4_path = tmp_path / "test_grayscale.mp4"
    save_video(frames, mp4_path, fps=10)

    # Load back
    loaded = load_video(mp4_path, mode="RGB")

    # Should be RGB format
    assert loaded.shape == (n_frames, height, width, 3)


def test_color_palette(tmp_path):
    """Test saving videos to both MP4 and GIF formats with and without shared color palette."""
    rng = np.random.default_rng(DEFAULT_TEST_SEED)
    n_frames = 4
    height, width = 16, 16
    frames = rng.integers(0, 255, (n_frames, height, width, 1), dtype=np.uint8)

    # Save to MP4 without shared palette
    mp4_path_no_palette = tmp_path / "test_no_palette.mp4"
    save_video(frames, mp4_path_no_palette, fps=10, shared_color_palette=False)
    loaded_no_palette = load_video(mp4_path_no_palette, mode="L")
    assert loaded_no_palette.shape == (n_frames, height, width)

    # Save to MP4 with shared palette
    mp4_path_with_palette = tmp_path / "test_with_palette.mp4"
    save_video(frames, mp4_path_with_palette, fps=10, shared_color_palette=True)
    loaded_with_palette = load_video(mp4_path_with_palette, mode="L")
    assert loaded_with_palette.shape == (n_frames, height, width)

    # Save to GIF without shared palette
    gif_path_no_palette = tmp_path / "test_no_palette.gif"
    save_video(frames, gif_path_no_palette, fps=10, shared_color_palette=False)
    loaded_no_palette_gif = load_video(gif_path_no_palette, mode="L")
    assert loaded_no_palette_gif.shape == (n_frames, height, width)

    # Save to GIF with shared palette
    gif_path_with_palette = tmp_path / "test_with_palette.gif"
    save_video(frames, gif_path_with_palette, fps=10, shared_color_palette=True)
    loaded_with_palette_gif = load_video(gif_path_with_palette, mode="L")
    assert loaded_with_palette_gif.shape == (n_frames, height, width)
