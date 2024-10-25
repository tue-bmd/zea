"""Helper functions for testing"""

import decorator
import jax
import numpy as np

from usbmd.setup_usbmd import set_backend


def equality_libs_processing(decimal=4, backends: list | None = None):
    """Test the processing functions of different libraries

    Check if numpy, tensorflow, torch and jax processing funcs produce equal output.
    Sometimes it requires you to reload the modules that use keras.ops inside the test function.

    Example:
        ```python
            @pytest.mark.parametrize('some_keys', [some_values])
            @equality_libs_processing(decimal=4) # <-- add as inner most decorator
            def test_my_processing_func(some_arguments):
                # Do some processing
                output = my_processing_func(some_arguments)
                return output
        ```
    """

    gt_backend = "numpy"
    if backends is None:
        backends = ["tensorflow", "torch", "jax"]
    print(f"Running tests with backends: {backends}")

    def wrapper(test_func, *args, **kwargs):
        # Set random seed
        seed = np.random.randint(0, 1000)

        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        output = {}
        for backend in [gt_backend, *backends]:
            print(f"Running {func_name} in {backend}")
            set_backend(backend)
            import keras  # pylint: disable=import-outside-toplevel

            keras.utils.set_random_seed(seed)
            with jax.disable_jit():
                output[backend] = np.array(test_func(*args, **kwargs))

        # Check if the outputs from the individual test functions are equal
        for backend in backends[1:]:
            np.testing.assert_almost_equal(
                output[gt_backend],
                output[backend],
                decimal=decimal,
                err_msg=f"Function {func_name} failed with {backend} processing.",
            )
            print(f"Function {func_name} passed with {backend} output.")

    return decorator.decorator(wrapper)
