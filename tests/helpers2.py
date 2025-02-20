import functools
import hashlib
import multiprocessing
import os

import cloudpickle as pickle
import decorator
import jax
import numpy as np

from usbmd.setup_usbmd import set_backend

result_queues = {}
processes = {}
job_queues = {}


def worker(job_queue, result_queues, env, backend, seed):
    # setup worker
    os.environ.update(env)
    set_backend(backend)
    import keras  # pylint: disable=import-outside-toplevel

    keras.utils.set_random_seed(seed)

    # start worker
    with jax.disable_jit():
        while True:
            job = job_queue.get()
            if job is None:  # Signal to exit
                break

            job_id, func_blob, args, kwargs = job
            func = pickle.loads(func_blob)
            result = func(*args, **kwargs)
            if result is not None:
                result = np.array(result)
            result_queues.put((job_id, result))


def start_workers(backends, seed=42):
    env = os.environ.copy()
    for backend in backends:
        result_queues[backend] = multiprocessing.Queue()
        job_queues[backend] = multiprocessing.Queue()
        processes[backend] = multiprocessing.Process(
            target=worker,
            args=(job_queues[backend], result_queues[backend], env, backend, seed),
        )
        processes[backend].start()


# backends = ["tensorflow", "torch", "jax", "numpy"]
# job_queues, result_queues, processes = start_workers(backends)


def start_func_in_backend(func, args, kwargs, backend):
    if backend not in job_queues:
        start_workers([backend])
    job_queue = job_queues[backend]
    job_id = hashlib.md5(str((func.__name__, args, kwargs)).encode()).hexdigest()
    job_queue.put((job_id, pickle.dumps(func), args, kwargs))


def collect_results(result_queues):
    results = {}
    job_ids = []
    for backend, result_queue in result_queues.items():
        job_id, result = result_queue.get(timeout=60)
        job_ids.append(job_id)
        results[backend] = result
    assert len(set(job_ids)) == 1, "Job IDs do not match across backends."
    return results


def stop_workers():
    for job_queue in job_queues.values():
        job_queue.put(None)
    for process in processes.values():
        process.join()


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
        for backend in [gt_backend, *backends]:
            start_func_in_backend(test_func, args, kwargs, backend)

        # Collect results before signaling the worker to stop
        output = collect_results(result_queues)

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


def run_in_backend(backend):
    """
    Decorator to run a test function in one specific backend.

    Args:
        backend (str): Backend to run the test in.
        seed (int): Seed to set for the backend. Defaults to 42.
    """

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            start_func_in_backend(test_func, args, kwargs, backend)
            result_queue = {backend: result_queues[backend]}
            return collect_results(result_queue)

        return wrapper

    return decorator
