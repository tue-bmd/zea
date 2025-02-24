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


class EqualityLibsProcessing:
    """This class is used to run a test function in multiple backends and compare the results.
    It starts workers for each backend and runs the test function in each worker.
    The workers are generally started once per test session in the __init__ file."""

    def __init__(self):
        self.result_queues = {}
        self.processes = {}
        self.job_queues = {}
        self.job_ids = {}

    @staticmethod
    def worker(job_queue, result_queue, env, backend, seed):
        """Worker function to run the test function in a separate process."""
        # setup worker
        os.environ.update(env)
        os.environ["KERAS_BACKEND"] = backend
        import keras  # pylint: disable=import-outside-toplevel

        keras.utils.set_random_seed(seed)

        # start worker
        while True:
            job = job_queue.get()
            if job is None:  # Signal to exit
                break

            try:
                job_id, func_blob, args_blob, kwargs_blob = job
                func = pickle.loads(func_blob)
                args = pickle.loads(args_blob)
                kwargs = pickle.loads(kwargs_blob)
                with jax.disable_jit():
                    result = func(*args, **kwargs)
                if result is not None:
                    result = np.array(result)
                result_queue.put((job_id, result))
            except Exception as e:
                tb = traceback.format_exc()
                result_queue.put((job_id, (e, tb)))

    def start_workers(self, backends, seed=42):
        """Start workers for the specified backends."""
        env = os.environ.copy()
        ctx = multiprocessing.get_context("spawn")
        for backend in backends:
            job_queue = ctx.Queue(maxsize=1)
            result_queue = ctx.Queue(maxsize=1)
            self.result_queues[backend] = result_queue
            self.job_queues[backend] = job_queue
            self.processes[backend] = ctx.Process(
                target=self.worker,
                args=(job_queue, result_queue, env, backend, seed),
                daemon=True,
            )
            self.processes[backend].start()

    def start_func_in_backend(self, func, args, kwargs, backend, job_id):
        """Start the test function in the specified backend."""
        # If no worker is running for the backend, start one
        if backend not in self.job_queues:
            self.start_workers([backend])
        # Put the job in the job queue
        job_queue = self.job_queues[backend]
        job_queue.put(
            (job_id, pickle.dumps(func), pickle.dumps(args), pickle.dumps(kwargs))
        )

    @staticmethod
    def collect_results(result_queues, timeout: int = 30):
        """Collect results from the result queues of the workers.
        Will wait for all backends to return a result or raise a TimeoutError."""
        results = {}
        job_ids = []
        for backend, result_queue in result_queues.items():
            try:
                job_id, result = result_queue.get(timeout=timeout)
                job_ids.append(job_id)
                results[backend] = result
            except Empty as exc:
                raise TimeoutError(
                    f"Timeout occurred while waiting for results from backend {backend}"
                ) from exc
        assert len(set(job_ids)) in [
            0,
            1,
        ], f"Job IDs do not match across backends: {job_ids}"
        for backend, result in results.items():
            if isinstance(result, tuple) and isinstance(result[0], Exception):
                raise RuntimeError(
                    f"Child process traceback for backend {backend}:\n"
                    + result[1]
                    + "\n"
                ) from result[0]
        return results

    def stop_workers(self):
        """Stop all workers. This should be called at the end of the test session."""
        for job_queue in self.job_queues.values():
            job_queue.put(None)
        for process in self.processes.values():
            process.join()
        self.result_queues = {}
        self.processes = {}
        self.job_queues = {}
        self.job_ids = {}

    def get_job_id(self, name):
        """Get a unique job ID for the test function."""
        name = str(name)
        if name not in self.job_ids:
            self.job_ids[name] = 0
        else:
            self.job_ids[name] += 1
        return name + "_" + str(self.job_ids[name])

    def equality_libs_processing(
        self,
        decimal=4,
        backends: list | None = None,
        gt_backend: str = "numpy",
        verbose: bool = False,
        timeout: int = 30,
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
        if backends is None:
            backends = ["tensorflow", "torch", "jax"]
        assert (
            gt_backend not in backends
        ), f"gt_backend: {gt_backend} is already tested."
        all_backends = [gt_backend, *backends]
        if verbose:
            print(f"Running tests with backends: {backends}")

        def wrapper(test_func, *args, **kwargs):
            # Extract function name from test function
            func_name = test_func.__name__.split("test_", 1)[-1]

            # Use process-based isolation for test_func
            job_id = self.get_job_id(test_func.__name__)
            for backend in all_backends:
                self.start_func_in_backend(test_func, args, kwargs, backend, job_id)

            # Collect results before signaling the worker to stop
            result_queues_local = {
                backend: self.result_queues[backend] for backend in all_backends
            }
            output = self.collect_results(result_queues_local, timeout=timeout)

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

    def run_in_backend(self, backend):
        """
        Decorator to run a test function in one specific backend.

        Args:
            backend (str): Backend to run the test in.
        """

        def decorator(test_func):
            @functools.wraps(test_func)
            def wrapper(*args, **kwargs):
                job_id = self.get_job_id(test_func.__name__)
                self.start_func_in_backend(test_func, args, kwargs, backend, job_id)
                result_queue = {backend: self.result_queues[backend]}
                return self.collect_results(result_queue)

            return wrapper

        return decorator
