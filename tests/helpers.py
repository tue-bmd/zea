"""Helper functions for testing"""

import functools
import multiprocessing
import os
import pickle
import time
import traceback

import decorator
import jax
import numpy as np

from usbmd.setup_usbmd import set_backend


def run_in_backend(backend, seed=42):
    """
    Decorator to run a test function in one specific backend.

    Args:
        backend (str): Backend to run the test in.
        seed (int): Seed to set for the backend. Defaults to 42.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            return run_test_in_process(
                test_func, *args, _seed=seed, _keras_backend=backend, **kwargs
            )

        return wrapper

    return decorator


def run_test_in_process(test_func, *args, _seed=42, _keras_backends=None, **kwargs):
    """Run a test function in a separate process for a specific backend."""

    def func_wrapper(queue, env, backend):
        print(f"Running {test_func.__name__} in {backend}")
        start_time = time.perf_counter()
        os.environ.update(env)
        try:
            set_backend(backend)
            import keras  # pylint: disable=import-outside-toplevel

            keras.utils.set_random_seed(_seed)
            with jax.disable_jit():
                result = test_func(*args, **kwargs)
            if result is not None:
                result = np.array(result)
            queue.put(pickle.dumps(result))
            func_time = time.perf_counter() - start_time
            print(f"{test_func.__name__} in {backend} took {func_time} seconds")
        except Exception as e:
            tb = traceback.format_exc()
            # Return both exception and traceback string
            queue.put((e, tb))

    if not _keras_backends:
        raise ValueError("No backends provided to run the test in.")

    processes = {}
    queues = {}
    outputs = {}
    for backend in _keras_backends:
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=func_wrapper,
            args=(queue, os.environ.copy(), backend),
        )
        processes[backend] = p
        queues[backend] = queue
        p.start()
    for backend, p in processes.items():
        output = queues[backend].get(timeout=120)
        p.join()
        if isinstance(output, tuple) and isinstance(output[0], Exception):
            exc, tb_str = output
            raise RuntimeError(
                f"Child process traceback for backend {backend}:\n" + tb_str + "\n"
            ) from exc
        outputs[backend] = pickle.loads(output)
    return outputs


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

        # Use process-based isolation for test_func
        output = run_test_in_process(
            test_func, *args, **kwargs, _keras_backends=[gt_backend, *backends]
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
