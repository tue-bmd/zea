""" GPU utilities for selecting GPUs and checking memory usage. """

import os
import subprocess as sp
import pandas as pd


def check_nvidia_smi():
    """Checks whether nvidia-smi is available."""
    ret_code = sp.call(
        ["nvidia-smi"], stdout=sp.PIPE, stderr=sp.PIPE)

    if ret_code == 0:
        return True
    else:
        return False

def get_gpu_memory(verbose=True):
    """ Retrieve memory allocation information of all gpus.

    Args:
        verbose (bool): prints output if True.

    Returns:
        memory_free_values: list of available memory for each gpu in MiB.
    """
    if not check_nvidia_smi():
        raise RuntimeError('nvidia-smi is not available. Install nvidia-smi '
                           'to use this function.')

    def _output_to_list(x):
        return x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    except Exception as e:
        print(f"An error occurred: {e}")

    memory_free_values = [int(x.split()[0])
                          for i, x in enumerate(memory_free_info)]

    # only show enabled devices
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpus = os.environ['CUDA_VISIBLE_DEVICES']
        gpus = [int(gpu) for gpu in gpus.split(',')][:len(memory_free_values)]
        if verbose:
            # Report the number of disabled GPUs out of the total
            num_disabled_gpus = len(memory_free_values) - len(gpus)
            num_gpus = len(memory_free_values)

            print(f'{num_disabled_gpus/num_gpus} GPUs were disabled')

        memory_free_values = [memory_free_values[gpu] for gpu in gpus]

    if verbose:
        df = df = pd.DataFrame({'memory': memory_free_values})
        df.index.name = 'GPU'
        print(df)

    return memory_free_values

def select_gpus(available_gpu_ids, memory_free, device=None,
                verbose=True, hide_others=True):
    """ Select GPU based on the device argument and available GPU's. This
    function does not rely on pytorch or tensorflow, and is shared between both
    frameworks.

    Hides other GPUs from the system by default by setting the
    CUDA_VISIBLE_DEVICES environment variable. Use the hide_others argument to
    disable this behavior.

    Args:
        available_gpu_ids (list): list of available GPU ids.
        memory_free (list): list of available memory for each gpu in MiB.
        device (str/int/list): GPU device(s) to select.
            - If 'cpu', use CPU.
            - If 'gpu', select GPU based on available memory.
                Throw an error if no GPU is available.
            - If None, try to select GPU based on available memory.
                Fall back to CPU if no GPU is available.
            - If an integer or a list of integers, use the corresponding GPU(s).
                If the list contains None values (e.g. [0, None, 2]), a GPU
                will be selected based on available memory.
            - If formatted as 'cuda:xx' or 'gpu:xx', where xx is an integer,
                use the corresponding GPU(s).
            - If formatted as 'auto:xx', where xx is an integer, automatically
                select xx GPUs based on available memory. If xx is -1, use all
                available GPUs.
        verbose (bool): prints output if True.
        hide_others (bool): if True, hide other GPUs from the system by setting
            the CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        gpu_ids: list of selected GPU ids. If no GPU is selected, returns an
            empty list. If a CPU is selected, returns None.
    """

     # Check if GPU mode is forced or if GPU should be selected based on memory
    if device == 'cpu' or (device is None and not available_gpu_ids):
        print('Setting device to CPU')
        return None
    elif device == 'gpu' or device == 'cuda' or device is None:
        gpu_ids = [None]  # Use None to select GPU based on available memory later
    elif isinstance(device, int) or device is None:
        gpu_ids = [device]  # Use a specific GPU if an integer is provided
    elif isinstance(device, list):
        gpu_ids = device  # Use multiple specific GPUs if a list of integers is provided
    elif isinstance(device, str):
        device = device.lower()  # Parse the device string

        if device.startswith('cuda:') or device.startswith('gpu:'):
            # Parse and use a specific GPU or all GPUs
            device_id = int(device.split(':')[1])

            if not isinstance(device_id, int):
                raise ValueError(f'Invalid device format: {device}. '
                                 f'Expected "cuda:<gpu_id>".')
            gpu_ids = [device_id]

        elif device.startswith('auto:'):
            # Automatically select GPUs based on available memory
            num_gpus = int(device.split(':')[1])  # number of GPUs to use

            print(f'Selecting {num_gpus} GPUs based on available memory.')

            if not isinstance(num_gpus, int):
                raise ValueError(f'Invalid device format: {device}. '
                                 f'Expected "auto:<num_gpus>".')
            if num_gpus == -1:
                num_gpus = len(available_gpu_ids)  # use all available GPUs
            # Create list of N None values corresponding to unassigned GPUs
            gpu_ids = num_gpus * [None]

        else:
            raise ValueError(f'Invalid device format: {device}. ')

    # Auto-select GPUs based on available memory for None values
    if None in gpu_ids:
        # Automatically select GPUs based on available memory
        sorted_gpu_ids = [
            x for x, _ in sorted(enumerate(memory_free), key=lambda x: x[1],
                                 reverse=True)
            ]

        assert len(gpu_ids) <= len(sorted_gpu_ids), \
            f'Selected more GPUs ({len(gpu_ids)}) than available ({len(sorted_gpu_ids)})'

        for i, gpu in enumerate(gpu_ids):
            if gpu is None and sorted_gpu_ids[i] in available_gpu_ids:
                gpu_ids[i] = sorted_gpu_ids[i]
    else:
        bad_gpus = set(gpu_ids) - set(available_gpu_ids)
        if bad_gpus:
            raise ValueError(f'GPUs {bad_gpus} not available!!')

    if verbose:
        for gpu_id in gpu_ids:
            print(f'Selected GPU {gpu_id} with Free Memory: '
                  f'{memory_free[gpu_id]:.2f} MiB')

    # Hide other GPUs from the system
    if hide_others:
        # Set the CUDA_VISIBLE_DEVICES environment variable to the selected
        # GPU(s)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    return gpu_ids
