"""Helper functions for testing"""

import multiprocessing
import pickle

import decorator
import jax
import numpy as np

from usbmd.setup_usbmd import set_backend


def run_test_in_process(test_func, *args, seed=42, _keras_backend=None, **kwargs):
    """Run a test function in a separate process for a specific backend."""

    def func_wrapper(queue):
        try:
            set_backend(_keras_backend)
            import keras  # pylint: disable=import-outside-toplevel

            keras.utils.set_random_seed(seed)
            with jax.disable_jit():
                result = np.array(test_func(*args, **kwargs))
            queue.put(pickle.dumps(result))
        except Exception as e:
            queue.put(e)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func_wrapper, args=(queue,))
    process.start()
    process.join()
    output = queue.get()
    if isinstance(output, Exception):
        raise output
    return pickle.loads(output)


def equality_libs_processing(
    decimal=4, backends: list | None = None, verbose: bool = False
):
    """Test the processing functions of different libraries

    Check if numpy, tensorflow, torch and jax processing funcs produce equal output.

    > [!WARNING]
    > It requires you to reload the modules that use `keras` inside the test function.

    Example:
        ```python
            @pytest.mark.parametrize('some_keys', [some_values])
            @equality_libs_processing(decimal=4) # <-- add as inner most decorator
            def test_my_processing_func(some_arguments):
                from usbmd import my_processing_func # <-- reload the function(s)

                # Do some processing
                output = my_processing_func(some_arguments)
                return output # <-- return the output!
        ```
    """
    gt_backend = "numpy"
    if backends is None:
        backends = ["tensorflow", "torch", "jax"]
    assert gt_backend not in backends, "numpy is already tested."
    if verbose:
        print(f"Running tests with backends: {backends}")

    def wrapper(test_func, *args, **kwargs):
        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        output = {}
        for backend in [gt_backend, *backends]:
            if verbose:
                print(f"Running {func_name} in {backend}")

            # Use process-based isolation for test_func
            output[backend] = run_test_in_process(
                test_func, *args, **kwargs, _keras_backend=backend
            )

        # Check if the outputs from the individual test functions are equal
        for backend in backends:
            np.testing.assert_almost_equal(
                output[gt_backend],
                output[backend],
                decimal=decimal,
                err_msg=f"Function {func_name} failed with {backend} processing.",
            )
            if verbose:
                print(f"Function {func_name} passed with {backend} output.")

    return decorator.decorator(wrapper)
