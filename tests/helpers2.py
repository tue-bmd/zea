"""Helper functions for testing"""

import functools
import multiprocessing
import os
import traceback
from queue import Empty

import cloudpickle as pickle
import decorator
import jax
import numpy as np
import pytest

from usbmd.setup_usbmd import set_backend

result_queues = {}
processes = {}
job_queues = {}
job_ids = {}


def worker(job_queue, result_queue, env, backend, seed):
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

            try:
                job_id, func_blob, args_blob, kwargs_blob = job
                func = pickle.loads(func_blob)
                args = pickle.loads(args_blob)
                kwargs = pickle.loads(kwargs_blob)
                result = func(*args, **kwargs)
                if result is not None:
                    result = np.array(result)
                result_queue.put((job_id, result))
            except Exception as e:
                tb = traceback.format_exc()
                result_queue.put((job_id, (e, tb)))


def start_workers(backends, seed=42):
    global result_queues, processes, job_queues
    env = os.environ.copy()
    for backend in backends:
        result_queues[backend] = multiprocessing.Queue()
        job_queues[backend] = multiprocessing.Queue()
        processes[backend] = multiprocessing.Process(
            target=worker,
            args=(job_queues[backend], result_queues[backend], env, backend, seed),
        )
        processes[backend].start()


def start_func_in_backend(func, args, kwargs, backend, job_id):
    global job_queues
    if backend not in job_queues:
        start_workers([backend])
    job_queue = job_queues[backend]
    job_queue.put(
        (job_id, pickle.dumps(func), pickle.dumps(args), pickle.dumps(kwargs))
    )


def collect_results(result_queues, timeout: int = 30):
    results = {}
    job_ids = []
    for backend, result_queue in result_queues.items():
        try:
            job_id, result = result_queue.get(timeout=timeout)
            job_ids.append(job_id)
            results[backend] = result
        except Empty:
            raise TimeoutError(
                f"Timeout occurred while waiting for results from backend {backend}"
            )
    assert len(set(job_ids)) in [
        0,
        1,
    ], f"Job IDs do not match across backends: {job_ids}"
    for backend, result in results.items():
        if isinstance(result, tuple) and isinstance(result[0], Exception):
            raise Exception(
                f"Child process traceback for backend {backend}:\n" + result[1] + "\n"
            ) from result[0]
    return results


def stop_workers():
    global result_queues, processes, job_queues, job_ids
    for job_queue in job_queues.values():
        job_queue.put(None)
    for process in processes.values():
        process.join()
    result_queues = {}
    processes = {}
    job_queues = {}
    job_ids = {}


def get_job_id(name):
    name = str(name)
    if name not in job_ids:
        job_ids[name] = 0
    else:
        job_ids[name] += 1
    return name + "_" + str(job_ids[name])


def equality_libs_processing(
    decimal=4, backends: list | None = None, verbose: bool = False, timeout: int = 30
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
    global result_queues
    gt_backend = "numpy"
    if backends is None:
        backends = ["tensorflow", "torch", "jax"]
    assert gt_backend not in backends, "numpy is already tested."
    all_backends = [gt_backend, *backends]
    if verbose:
        print(f"Running tests with backends: {backends}")

    def wrapper(test_func, *args, **kwargs):
        # Extract function name from test function
        func_name = test_func.__name__.split("test_", 1)[-1]

        # Use process-based isolation for test_func
        job_id = get_job_id(test_func.__name__)
        for backend in all_backends:
            start_func_in_backend(test_func, args, kwargs, backend, job_id)

        # Collect results before signaling the worker to stop
        result_queues_local = {
            backend: result_queues[backend] for backend in all_backends
        }
        output = collect_results(result_queues_local, timeout=timeout)

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
    """
    global result_queues

    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            job_id = get_job_id(test_func.__name__)
            start_func_in_backend(test_func, args, kwargs, backend, job_id)
            result_queue = {backend: result_queues[backend]}
            return collect_results(result_queue)

        return wrapper

    return decorator


@pytest.fixture(scope="session", autouse=True)
def run_once_after_all_tests():
    yield
    print("Stopping workers")
    stop_workers()
