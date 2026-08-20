"""Tests for the zea.utils module.

Contains both tests for zea.utils and zea.internal.utils.
"""

import gc
import re

import numpy as np
import pytest
from keras import ops

from zea import log
from zea.backend import jit
from zea.internal.utils import (
    atomic_write,
    calculate_file_hash,
    find_first_nonzero_index,
    find_key,
    first_not_none_item,
)
from zea.utils import (
    FunctionTimer,
    ProgressBar,
    block_until_ready,
    canonicalize_axis,
    date_string_to_readable,
    deep_compare,
    get_date_string,
    map_negative_indices,
    print_clear_line,
    strtobool,
    update_dictionary,
)


def test_calculate_file_hash_omit_line(tmp_path):
    """Test that calculate_file_hash correctly omits lines containing a string."""

    # Create a temporary file
    file_content = [
        "Dataset: test_folder\n",
        "Validated on: 2025_10_14_120000\n",
        "hash: should_be_ignored\n",
    ]
    file_path = tmp_path / "validation_file.txt"
    file_path.write_text("".join(file_content), encoding="utf-8")

    # Calculate hash ignoring the 'hash' line
    hash_without_hash_line = calculate_file_hash(file_path, omit_line_str="hash")

    expected_hash = "02d7d3d3f7731f715cc3c886752196c67893267b12a880455f0aeca0ad4d7da9"

    assert hash_without_hash_line == expected_hash

    hash_with_hash_line = calculate_file_hash(file_path, omit_line_str=None)

    assert hash_with_hash_line != expected_hash


@pytest.mark.parametrize(
    "dict1, dict2, keep_none, expected_result",
    [
        (
            {1: "one", 2: "two"},
            {2: "new_two", 3: "three"},
            False,
            {1: "one", 2: "new_two", 3: "three"},
        ),
        (
            {1: "one", 2: "two"},
            {2: None, 3: "three"},
            False,
            {1: "one", 2: "two", 3: "three"},
        ),
        ({}, {1: "one"}, False, {1: "one"}),
        ({1: "one"}, {}, False, {1: "one"}),
        (
            {1: "one", 2: "two"},
            {2: None, 3: "three"},
            True,
            {1: "one", 2: None, 3: "three"},
        ),
        ({}, {}, False, {}),
    ],
)
def test_update_dictionary(dict1, dict2, keep_none, expected_result):
    """Tests the update_dictionary function using simple equality check."""
    result = update_dictionary(dict1, dict2, keep_none)
    assert result == expected_result


@pytest.mark.parametrize(
    "contains, case_sensitive",
    [["apple", False], ["apple", True], ["pie", True]],
)
def test_find_key(contains, case_sensitive):
    """Tests the find_key function by providing a test dictionary and checking the
    number of keys found."""
    dictionary = {
        "APPLES": 1,
        "apple pie": 2,
        "cherry pie": 3,
        "what apple": 4,
        "rainbow": 5,
    }

    result = find_key(dictionary, contains, case_sensitive)

    # Check that the result is a string
    assert isinstance(result, str), "Result is not a list"
    # Check that the result is actually in the dictionary
    assert result in dictionary.keys(), "Key not found in dictionary"

    # Check that the result contains the search string
    if not case_sensitive:
        result = result.lower()
        contains = contains.lower()

    assert contains in result, "Key does not contain the search string"


def test_nonexistent_key_raises_keyerror():
    """Tests that a KeyError is raised if the key is not found."""
    dictionary = {"APPLES": 1, "apple pie": 2, "cherry pie": 3, "rainbow": 5}

    with pytest.raises(KeyError):
        find_key(dictionary, "banana", case_sensitive=True)


def test_strtobool():
    """ "Test strtobool function with multiple user inputs."""
    # 1. Non string input raises assertion error
    with pytest.raises(AssertionError, match="Input value must be a string"):
        strtobool(1)

    # 2. strtobool is case insensitive
    assert strtobool("TRUE") is True
    assert strtobool("TruE") is True
    assert strtobool("true") is True

    # 3. valid 'true' values get mapped to True
    valid_true_values = ["y", "yes", "t", "true", "on", "1"]
    assert np.all([strtobool(v) for v in valid_true_values])

    # 4. valid 'false' values get mapped to False
    valid_false_values = ["n", "no", "f", "false", "off", "0"]
    assert not np.any([strtobool(v) for v in valid_false_values])

    # 5. any other value raises a ValueError
    sample_invalid_values = ["🤔", "invalid_value", "hello!"]
    for invalid_value in sample_invalid_values:
        with pytest.raises(ValueError, match=f"invalid truth value {invalid_value}"):
            strtobool(invalid_value)


def test_get_date_string():
    """Tests the get_date_string function."""

    # Test default date format
    date_string = get_date_string()
    assert isinstance(date_string, str), "Result is not a string"
    date_string = get_date_string()

    # Check if date string matches pattern YYYY_MM_DD_HHMMSS
    regex_pattern = r"^\d{4}_\d{2}_\d{2}_\d{6}$"
    assert re.match(regex_pattern, date_string), "Date string does not match pattern"

    # Test alternative date format
    date_string = get_date_string(string="%d-%m-%Y")
    assert isinstance(date_string, str), "Result is not a string"
    regex_pattern = r"^\d{2}-\d{2}-\d{4}$"
    assert re.match(regex_pattern, date_string), "Date string does not match pattern"

    # Test if the function raises an error at invalid input
    with pytest.raises(TypeError):
        get_date_string(string=1)

    with pytest.raises(TypeError):
        get_date_string(string=lambda x: x)


@pytest.mark.parametrize(
    "arr, axis, invalid_val, expected",
    [
        ((0, 0, 0, 5, 0, 3, 0), 0, -1, 3),
        ([[0, 0, 0], [4, 0, 0], [0, 0, 7]], 1, None, [None, 0, 2]),
    ],
)
def test_find_first_nonzero_index(arr, axis, invalid_val, expected):
    """Tests the find_first_nonzero_index function."""
    arr = np.array(arr)
    result = find_first_nonzero_index(arr, axis, invalid_val=invalid_val)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([None, None], None),
        ([None, False, 0, 1, 2.0], False),
        ([2.0, None], 2.0),
    ],
)
def test_first_not_none_item(arr, expected):
    """Tests the find_first_nonzero_index function."""
    result = first_not_none_item(arr)
    np.testing.assert_equal(result, expected)


def test_block_until_ready_timing():
    """Tests that block_until_ready calls the correct backend-specific function."""
    from unittest.mock import patch

    import keras

    @jit
    def slow_computation(x):
        # Vectorized heavy computation (no Python for-loop):
        # - Build an outer-product matrix (n x n)
        # - Apply elementwise trig operations
        # - Reduce to a scalar
        y = ops.matmul(x[:, None], x[None, :])  # shape (n, n)
        z = ops.sin(y) + ops.cos(y)
        return ops.sum(z, axis=1)

    x = ops.ones(1000)

    # Compile first
    _ = slow_computation(x)

    backend_name = keras.backend.backend()

    if backend_name == "jax":
        # Test that jax.block_until_ready is called for JAX backend
        with patch("jax.block_until_ready") as mock_jax_block:
            # Make mock return the input unchanged
            mock_jax_block.side_effect = lambda x: x

            result = block_until_ready(slow_computation)(x)

            # Verify jax.block_until_ready was called
            mock_jax_block.assert_called_once()
            assert result is not None
    else:
        # Test that keras.ops.convert_to_numpy is called for other backends
        with patch("keras.ops.convert_to_numpy") as mock_convert:
            # Make mock return the input unchanged
            mock_convert.side_effect = lambda x: x

            result = block_until_ready(slow_computation)(x)

            # Verify keras.ops.convert_to_numpy was called
            mock_convert.assert_called_once()
            assert result is not None

    print(f"Backend: {backend_name}")
    print("block_until_ready backend-specific function test completed!")


def test_progressbar_registers_with_log_when_verbose():
    bar = ProgressBar(5, verbose=1)
    assert bar._dynamic_display, "test assumes a dynamic-display-capable environment"
    assert bar in log._active_progress


def test_progressbar_does_not_register_when_verbose_is_zero():
    bar = ProgressBar(5, verbose=0)
    assert bar not in log._active_progress


def test_progressbar_unregisters_once_target_is_reached():
    bar = ProgressBar(3, verbose=1)
    bar.update(1)
    assert bar in log._active_progress
    bar.update(3)  # reaches target -> finalizes
    assert bar not in log._active_progress


def test_progressbar_is_garbage_collected_if_loop_breaks_before_target():
    bar = ProgressBar(50, verbose=1)
    bar.update(1)  # far from target, never finalized
    assert bar in log._active_progress

    del bar
    gc.collect()
    assert len(log._active_progress) == 0 or all(
        not isinstance(b, ProgressBar) for b in log._active_progress
    )


def test_progressbar_redraw_bypasses_update_throttle(capsys):
    """``redraw`` re-renders the bar immediately, without advancing progress."""
    bar = ProgressBar(10, verbose=1)
    bar.update(4)
    capsys.readouterr()  # discard the output of the update above

    bar.redraw()

    assert "4/10" in capsys.readouterr().out
    assert bar._seen_so_far == 4


@pytest.mark.parametrize(
    "axis, num_dims, expected",
    [(0, 3, 0), (2, 3, 2), (-1, 3, 2), (-3, 3, 0)],
)
def test_canonicalize_axis(axis, num_dims, expected):
    """Negative axes are mapped into ``[0, num_dims)``, positive ones pass through."""
    assert canonicalize_axis(axis, num_dims) == expected


@pytest.mark.parametrize("axis, num_dims", [(3, 3), (-4, 3), (10, 2)])
def test_canonicalize_axis_out_of_bounds(axis, num_dims):
    """Axes outside ``[-num_dims, num_dims)`` raise a ValueError."""
    with pytest.raises(ValueError, match="out of bounds"):
        canonicalize_axis(axis, num_dims)


def test_map_negative_indices():
    """Maps a list of (possibly negative) indices to positive ones."""
    assert map_negative_indices([-1, -2], 5) == [4, 3]
    assert map_negative_indices([0, 3], 4) == [0, 3]

    with pytest.raises(ValueError, match="out of bounds"):
        map_negative_indices([-6], 5)


def test_print_clear_line(capsys):
    """Prints the ANSI escape codes that move up one line and clear it."""
    print_clear_line()
    out = capsys.readouterr().out
    assert "\033[1A" in out
    assert "\x1b[2K" in out


@pytest.mark.parametrize(
    "include_time, expected",
    [(False, "March 05, 2024"), (True, "March 05, 2024 01:02 PM")],
)
def test_date_string_to_readable(include_time, expected):
    """Converts the zea date string format into a human readable one."""
    assert date_string_to_readable("2024_03_05_130215", include_time=include_time) == expected


def test_date_string_to_readable_invalid_format():
    """Date strings that do not match the zea format raise a ValueError."""
    with pytest.raises(ValueError):
        date_string_to_readable("05-03-2024")


def test_date_string_roundtrip():
    """A generated date string can be parsed back into a readable date."""
    assert isinstance(date_string_to_readable(get_date_string()), str)


@pytest.mark.parametrize(
    "obj1, obj2, expected",
    [
        ({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}}, True),
        ({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 3}}, False),
        ({"a": 1}, {"a": 1, "b": 2}, False),  # different keys
        ({"a": [1, 2, 3]}, {"a": [1, 2, 3]}, True),
        ({"a": [1, 2, 3]}, {"a": [1, 2, 4]}, False),
        ("string", "string", True),
        ("string", "other", False),
        (1.0, 1, True),  # scalars fall back to ==
        ({"a": np.array([1, 2])}, {"a": np.array([1, 2])}, True),
        ({"a": np.array([1, 2])}, {"a": np.array([1, 3])}, False),
        ({"a": {"b": [{"c": 1}]}}, {"a": {"b": [{"c": 1}]}}, True),
        ({"a": {"b": [{"c": 1}]}}, {"a": {"b": [{"c": 2}]}}, False),
    ],
)
def test_deep_compare(obj1, obj2, expected):
    """Recursively compares nested dicts, iterables and scalars."""
    assert deep_compare(obj1, obj2) is expected


def test_block_until_ready_container_outputs():
    """Blocks every array in list, tuple and dict outputs, preserving the container type."""

    def multiple_outputs():
        return [ops.ones(3), ops.zeros(3)]

    def tuple_outputs():
        return ops.ones(3), "not-an-array"

    def dict_outputs():
        return {"array": ops.ones(3), "scalar": 5}

    listed = block_until_ready(multiple_outputs)()
    assert isinstance(listed, list) and len(listed) == 2

    tupled = block_until_ready(tuple_outputs)()
    assert isinstance(tupled, tuple)
    assert tupled[1] == "not-an-array"

    dicted = block_until_ready(dict_outputs)()
    assert isinstance(dicted, dict)
    assert dicted["scalar"] == 5
    np.testing.assert_allclose(ops.convert_to_numpy(dicted["array"]), np.ones(3))


@pytest.fixture
def timer():
    """A fresh FunctionTimer with one timed function that has been called twice."""
    _timer = FunctionTimer()
    timed = _timer(lambda x: x + 1, name="add_one")
    timed(1)
    timed(2)
    return _timer


def test_function_timer_get_stats(timer):
    """Statistics are computed over all recorded timings of a function."""
    stats = timer.get_stats("add_one")
    assert stats["count"] == 2
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std_dev"] >= 0
    assert set(stats) == {"mean", "median", "std_dev", "min", "max", "count"}


@pytest.mark.parametrize("drop_first, expected_count", [(False, 2), (True, 1), (0, 2), (1, 1)])
def test_function_timer_drop_first(timer, drop_first, expected_count):
    """``drop_first`` skips the first n timings, e.g. to ignore compilation time."""
    assert timer.get_stats("add_one", drop_first=drop_first)["count"] == expected_count


def test_function_timer_single_timing_has_zero_std_dev():
    """A single recorded timing yields a std_dev of 0 instead of a statistics error."""
    timer = FunctionTimer()
    timed = timer(lambda: None, name="once")
    timed()
    assert timer.get_stats("once")["std_dev"] == 0


def test_function_timer_invalid_drop_first(timer):
    """``drop_first`` must be a bool or an int."""
    with pytest.raises(ValueError, match="drop_first must be a boolean or an integer"):
        timer.get_stats("add_one", drop_first="1")


def test_function_timer_unknown_function(timer):
    """Requesting stats for a function that was never timed raises a ValueError."""
    with pytest.raises(ValueError, match="No timings recorded"):
        timer.get_stats("does_not_exist")


def test_function_timer_defaults_to_function_name():
    """Without an explicit name, the decorated function's own name is used."""
    timer = FunctionTimer()

    def my_function():
        return 1

    timed = timer(my_function)
    assert timed() == 1
    assert timer.get_stats("my_function")["count"] == 1


def test_function_timer_rejects_decorating_the_same_function_twice():
    """The same function instance cannot be timed under two names."""
    timer = FunctionTimer()

    def my_function():
        return 1

    timer(my_function, name="first")
    with pytest.raises(ValueError, match="has already been"):
        timer(my_function, name="second")


def test_function_timer_resolves_name_conflicts():
    """Two different functions timed under the same name get a numbered suffix."""
    timer = FunctionTimer()
    # Keep both functions alive: FunctionTimer identifies them by id().
    first, second = lambda: None, lambda: None
    timer(first, name="shared")()
    timer(second, name="shared")()

    assert "shared" in timer.timings
    assert "shared_1" in timer.timings


def test_function_timer_export_to_yaml(timer, tmp_path):
    """Timings are exported as a YAML mapping of name -> list of durations."""
    import yaml

    filename = tmp_path / "timings.yaml"
    timer.export_to_yaml(filename)

    exported = yaml.safe_load(filename.read_text(encoding="utf-8"))
    assert list(exported) == ["add_one"]
    assert len(exported["add_one"]) == 2


def test_function_timer_append_to_yaml(tmp_path):
    """Appending only writes the timings recorded since the previous append."""
    import yaml

    timer = FunctionTimer()
    timed = timer(lambda x: x + 1, name="add_one")
    timed(1)
    timed(2)

    filename = tmp_path / "timings.yaml"
    timer.append_to_yaml(filename, "add_one")
    assert len(yaml.safe_load(filename.read_text(encoding="utf-8"))) == 2

    # A third call, then append again: only the new timing is appended
    timed(3)
    timer.append_to_yaml(filename, "add_one")
    assert len(yaml.safe_load(filename.read_text(encoding="utf-8"))) == 3


def test_function_timer_print(timer, capsys):
    """Printing renders one row per timed function, optionally with the total time."""
    timer.print()
    out = log.remove_color_escape_codes(capsys.readouterr().out)
    assert "Function Timing Statistics" in out
    assert "add_one" in out
    assert "Mean Total Time" not in out

    timer.print(drop_first=True, total_time=True)
    out = log.remove_color_escape_codes(capsys.readouterr().out)
    assert "Mean Total Time" in out


def test_atomic_write_moves_result_into_place(tmp_path):
    """The destination appears only once the block completes, leaving no temp files."""
    destination = tmp_path / "result.txt"

    with atomic_write(destination) as tmp:
        tmp.write_text("done")
        assert tmp != destination
        assert tmp.parent == destination.parent, "temp file must share the filesystem"
        assert not destination.exists(), "destination must not appear before the block ends"

    assert destination.read_text() == "done"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_keeps_previous_file_on_failure(tmp_path):
    """A failed write leaves the old contents in place instead of a truncated file."""
    destination = tmp_path / "result.txt"
    destination.write_text("original")

    with pytest.raises(RuntimeError, match="write failed"):
        with atomic_write(destination) as tmp:
            tmp.write_text("partial")
            raise RuntimeError("write failed")

    assert destination.read_text() == "original"
    assert list(tmp_path.iterdir()) == [destination], "temp file must be cleaned up"


def test_atomic_write_suffix(tmp_path):
    """Writers that key off the extension can ask for a matching temp suffix."""
    destination = tmp_path / "result.hdf5"

    with atomic_write(destination) as default_suffix:
        assert default_suffix.suffix == ".hdf5"

    with atomic_write(destination, suffix=".part") as custom_suffix:
        assert custom_suffix.suffix == ".part"
