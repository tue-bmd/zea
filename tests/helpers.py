"""Helper functions for testing"""

import functools
import multiprocessing
import os
import sys
import traceback
from queue import Empty

import cloudpickle as pickle
import debugpy
import decorator
import jax
import numpy as np
import pytest

debugging = sys.gettrace() or debugpy.is_client_connected() is not None


def run_func(func):
    """Run a function from a blob."""
    pickle.loads(func)()  # run func


def run_in_subprocess(func):
    """Run a function in a subprocess, does not support outputs."""

    @functools.wraps(func)
    def wrapper():
        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(target=run_func, args=(pickle.dumps(func),))
        process.start()
        process.join()
        assert process.exitcode == 0, f"Process failed with exit code {process.exitcode}"

    return wrapper


class BackendEqualityCheck:
    """This class is used to run a test function in multiple backends and compare the results.
    It starts workers for each backend and runs the test function in each worker.
    The workers are generally started once per test session in the __init__ file.

    NOTE: the workers only run on CPU.
    """

    def __init__(self):
        self.result_queues = {}
        self.processes = {}
        self.job_queues = {}
        self.job_ids = {}

    @staticmethod
    def worker(job_queue, result_queue, env, backend, seed):
        """Worker function to run the test function in a separate process."""
        # setup worker (only cpu!)
        os.environ.update(env)
        os.environ["KERAS_BACKEND"] = backend
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["JAX_PLATFORMS"] = "cpu"  # only affects jaxs
        import keras

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
                    keras.utils.set_random_seed(seed)
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
        job_queue.put((job_id, pickle.dumps(func), pickle.dumps(args), pickle.dumps(kwargs)))

    def collect_results(self, result_queues, timeout: int = 30):
        """
        Collect results from the result queues of the workers.
        Will wait for all backends to return a result or raise a TimeoutError.

        Returns:
            dict: Results for each backend in `result_queues.keys()`.
        """
        timeout = timeout if not debugging else None
        results = {}
        job_ids = []
        for backend, result_queue in result_queues.items():
            try:
                job_id, result = result_queue.get(timeout=timeout)
                job_ids.append(job_id)
                results[backend] = result
            except Empty as exc:
                # stop all the workers
                # this can be done in a more elegant way, e.g. only stopping the backend that fails
                self.stop_workers()
                msg = (
                    f"Timeout occurred while waiting for results from backend {backend}, "
                    + "possibly also from other backends."
                )
                pytest.fail(msg)
                raise TimeoutError(msg) from exc
        assert len(set(job_ids)) in [
            0,
            1,
        ], f"Job IDs do not match across backends: {job_ids}"
        for backend, result in results.items():
            if isinstance(result, tuple) and isinstance(result[0], Exception):
                raise RuntimeError(
                    f"Child process traceback for backend {backend}:\n" + result[1] + "\n"
                ) from result[0]
        return results

    def stop_workers(self, force=True):
        """Stop all workers. This should be called at the end of the test session."""
        for job_queue in self.job_queues.values():
            job_queue.put(None)
        for process in self.processes.values():
            if force:
                process.terminate()
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

    def backend_equality_check(
        self,
        decimal: int | list = 4,
        backends: list | None = None,
        gt_backend: str = "numpy",
        verbose: bool = False,
        timeout: int = 30,
        allow_none: bool = False,
    ):
        """Test the processing functions of different libraries (on CPU).

        Check if numpy, tensorflow, torch and jax processing funcs produce equal output.

        > [!WARNING]
        > It requires you to reload the modules that use `keras` inside the test function.

        > [!TIP]
        > Will set the random seed before every function evaluation. But it is better to get a
        > random number generator inside the function,
        > e.g. `rng = np.random.default_rng(seed=42)`

        Example:
            ```python
                @pytest.mark.parametrize('some_keys', [some_values])
                @backend_equality_check(decimal=4) # <-- add as inner most decorator
                def test_my_processing_func(some_arguments):
                    from zea import my_processing_func # <-- reload the function(s)

                    # Do some processing
                    output = my_processing_func(some_arguments)
                    return output # <-- return the output!
            ```
        """
        if backends is None:
            backends = ["tensorflow", "torch", "jax"]
        if isinstance(decimal, int):
            decimal = [decimal] * len(backends)
        else:
            assert len(decimal) == len(backends), "decimal must be an integer or a list."
        assert gt_backend not in backends, f"gt_backend: {gt_backend} is already tested."
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
            result_queues_local = {backend: self.result_queues[backend] for backend in all_backends}
            output = self.collect_results(result_queues_local, timeout=timeout)

            # Check if the outputs from the individual test functions are equal
            errors = []
            for i, backend in enumerate(backends):
                # if both outputs are None, skip the check
                # i.e. we are just checking if it runs, not if produces the same output
                if output[gt_backend] is None and output[backend] is None:
                    if allow_none:
                        continue
                    raise ValueError(
                        "Both outputs are None. Set allow_none=True to allow None outputs."
                    )

                try:
                    np.testing.assert_almost_equal(
                        output[gt_backend],
                        output[backend],
                        decimal=decimal[i],
                        err_msg=f"Function {func_name} failed with {backend} processing.",
                    )
                    if verbose:
                        print(f"Function {func_name} passed with {backend} output.")
                except AssertionError as e:
                    errors.append(str(e))
            if errors:
                raise AssertionError("Errors occurred in backends:\n" + "\n".join(errors))

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
                return self.collect_results(result_queue)[backend]

            return wrapper

        return decorator
