"""Input / output functions for reading and writing files.

Use to quickly read and write files or interact with file system.
"""

import functools
import multiprocessing
import os
import time
from io import BytesIO
from pathlib import Path

import imageio
import numpy as np
import pydicom
import tqdm
import yaml
from PIL import Image
from pydicom.pixel_data_handlers import convert_color_space

from zea import log
from zea.data.file import File

_SUPPORTED_VID_TYPES = [".avi", ".mp4", ".gif", ""]
_SUPPORTED_IMG_TYPES = [".jpg", ".png", ".JPEG", ".PNG", ".jpeg"]
_SUPPORTED_ZEA_TYPES = [".hdf5", ".h5"]


def load_video(filename):
    """Load a video file and return a numpy array of frames.

    Supported file types: avi, mp4, gif, dcm.

    Args:
        filename (str): The path to the video file.

    Returns:
        numpy.ndarray: A numpy array of frames.
    Raises:
        ValueError: If the file extension is not supported.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for video loading. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    filename = Path(filename)
    assert Path(filename).exists(), f"File {filename} does not exist"
    extension = filename.suffix
    assert extension in _SUPPORTED_VID_TYPES, f"File extension {extension} not supported"

    if extension in [".avi", ".mp4"]:
        cap = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif extension == ".gif":
        frames = imageio.mimread(filename)
    elif extension == "":
        ds = pydicom.dcmread(filename)
        frames = convert_color_space(ds.pixel_array, "YBR_FULL", "RGB")
    else:
        raise ValueError("Unsupported file extension")
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
    return np.array(frames)


def load_image(filename, grayscale=True, color_order="RGB"):
    """Load an image file and return a numpy array.

    Supported file types: jpg, png.

    Args:
        filename (str): The path to the image file.
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to True.
        color_order (str, optional): The desired color channel ordering. Defaults to 'RGB'.

    Returns:
        numpy.ndarray: A numpy array of the image.

    Raises:
        ValueError: If the file extension is not supported.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for image loading. "
            "Please install it with 'pip install opencv-python' or "
            "'pip install opencv-python-headless'."
        ) from exc

    filename = Path(filename)
    assert Path(filename).exists(), f"File {filename} does not exist"
    extension = filename.suffix
    assert extension in _SUPPORTED_IMG_TYPES, f"File extension {extension} not supported"

    image = cv2.imread(str(filename))

    if grayscale and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if color_order == "BGR":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_order == "RGB":
        pass
    else:
        raise ValueError(f"Unsupported color order: {color_order}")

    return image


def search_file_tree(
    directory,
    filetypes=None,
    write=True,
    dataset_info_filename="dataset_info.yaml",
    hdf5_key_for_length=None,
    redo=False,
    parallel=False,
    verbose=True,
):
    """Lists all files in directory and sub-directories.

    If dataset_info.yaml is detected in the directory, that file is read and used
    to deduce the file paths. If not, the file paths are searched for in the
    directory and written to a dataset_info.yaml file.

    Args:
        directory (str): Path to base directory to start file search.
        filetypes (str or list, optional): Filetypes to look for in directory.
            Defaults to image types (.png etc.). Make sure to include the dot.
        write (bool, optional): Whether to write to dataset_info.yaml file.
            Defaults to True. If False, the file paths are not written to file
            and simply returned.
        dataset_info_filename (str, optional): Name of dataset info file.
            Defaults to "dataset_info.yaml", but can be changed to any name.
        hdf5_key_for_length (str, optional): Key to use for getting length of hdf5 files.
            Defaults to None. If set, the number of frames in each hdf5 file is
            calculated and stored in the dataset_info.yaml file. This is extra
            functionality of ``search_file_tree`` and only works with hdf5 files.
        redo (bool, optional): Whether to redo the search and overwrite the dataset_info.yaml file.
        parallel (bool, optional): Whether to use multiprocessing for hdf5 shape reading.
        verbose (bool, optional): Whether to print progress and info.

    Returns:
        dict: Dictionary containing file paths and total number of files.
            Has the following structure:

            .. code-block:: python

                {
                    "file_paths": list of file paths,
                    "total_num_files": total number of files,
                    "file_lengths": list of number of frames in each hdf5 file,
                    "file_shapes": list of shapes of each image file,
                    "total_num_frames": total number of frames in all hdf5 files
                }

    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(
            log.error(f"Directory {directory} does not exist. Please provide a valid directory.")
        )
    assert Path(dataset_info_filename).suffix == ".yaml", (
        "Currently only YAML files are supported for dataset info file when "
        f"using `search_file_tree`, got {dataset_info_filename}"
    )

    if (directory / dataset_info_filename).is_file() and not redo:
        with open(directory / dataset_info_filename, "r", encoding="utf-8") as file:
            dataset_info = yaml.load(file, Loader=yaml.FullLoader)

        # Check if the file_shapes key is present in the dataset_info, otherwise redo the search
        if "file_shapes" in dataset_info:
            if verbose:
                log.info(
                    "Using pregenerated dataset info file: "
                    f"{log.yellow(directory / dataset_info_filename)} ..."
                )
                log.info(f"...for reading file paths in {log.yellow(directory)}")
            return dataset_info

    if redo and verbose:
        log.info(f"Overwriting dataset info file: {log.yellow(directory / dataset_info_filename)}")

    # set default file type
    if filetypes is None:
        filetypes = _SUPPORTED_IMG_TYPES + _SUPPORTED_VID_TYPES + _SUPPORTED_ZEA_TYPES

    file_paths = []

    if isinstance(filetypes, str):
        filetypes = [filetypes]

    if hdf5_key_for_length is not None:
        assert isinstance(hdf5_key_for_length, str), "hdf5_key_for_length must be a string"
        assert set(filetypes).issubset({".hdf5", ".h5"}), (
            "hdf5_key_for_length only works with when filetypes is set to "
            f"`.hdf5` or `.h5`, got {filetypes}"
        )

    # Traverse file tree to index all files from filetypes
    if verbose:
        log.info(f"Searching {log.yellow(directory)} for {filetypes} files...")
    for dirpath, _, filenames in os.walk(directory):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if Path(file).suffix in filetypes:
                file_path = Path(dirpath) / file
                file_path = file_path.relative_to(directory)
                file_paths.append(str(file_path))

    if hdf5_key_for_length is not None:
        # using multiprocessing to speed up reading hdf5 files
        # and getting the number of frames in each file
        if verbose:
            log.info("Getting number of frames in each hdf5 file...")

        get_shape_partial = functools.partial(File.get_shape, key=hdf5_key_for_length)
        # make sure to call search_file_tree from within a function
        # or use if __name__ == "__main__":
        # to avoid freezing the main process
        absolute_file_paths = [directory / file for file in file_paths]
        if parallel:
            with multiprocessing.Pool() as pool:
                file_shapes = list(
                    tqdm.tqdm(
                        pool.imap(
                            get_shape_partial,
                            absolute_file_paths,
                        ),
                        total=len(file_paths),
                        desc="Getting number of frames in each hdf5 file",
                        disable=not verbose,
                    )
                )
        else:
            file_shapes = []
            for file_path in tqdm.tqdm(
                absolute_file_paths,
                desc="Getting number of frames in each hdf5 file",
                disable=not verbose,
            ):
                file_shapes.append(File.get_shape(file_path, hdf5_key_for_length))

    assert len(file_paths) > 0, f"No image files were found in: {directory}"
    if verbose:
        log.info(f"Found {len(file_paths)} image files in {log.yellow(directory)}")
        log.info(f"Writing dataset info to {log.yellow(directory / dataset_info_filename)}")

    dataset_info = {"file_paths": file_paths, "total_num_files": len(file_paths)}
    if len(file_shapes) > 0:
        dataset_info["file_shapes"] = file_shapes
        file_lengths = [shape[0] for shape in file_shapes]
        dataset_info["file_lengths"] = file_lengths
        dataset_info["total_num_frames"] = sum(file_lengths)

    if write:
        with open(directory / dataset_info_filename, "w", encoding="utf-8") as file:
            yaml.dump(dataset_info, file)

    return dataset_info


def matplotlib_figure_to_numpy(fig):
    """Convert matplotlib figure to numpy array.

    Args:
        fig (matplotlib.figure.Figure): figure to convert.

    Returns:
        np.ndarray: numpy array of figure.

    """
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)[..., :3]
    buf.close()
    return image


def retry_on_io_error(max_retries=3, initial_delay=0.5, retry_action=None):
    """Decorator to retry functions on I/O errors with exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (float): Initial delay between retries in seconds.
        retry_action (callable, optional): Optional function to call before each retry attempt.
            If decorating a method: ``retry_action(self, exception, attempt, *args, **kwargs)``
            If decorating a function: ``retry_action(exception, attempt, *args, **kwargs)``

    Returns:
        callable: Decorated function with retry logic.

    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError) as e:
                    last_exception = e

                    # if args exist and first arg is a class, update retry count of that method
                    if args and hasattr(args[0], "retry_count"):
                        args[0].retry_count = attempt + 1

                    if attempt < max_retries - 1:
                        # Execute custom retry action if provided
                        if retry_action:
                            # Pass all original arguments to retry_action
                            retry_action(
                                *args,
                                exception=e,
                                retry_count=attempt,
                                **kwargs,
                            )

                        time.sleep(delay)

                    else:
                        # Last attempt failed
                        log.error(f"Failed after {max_retries} attempts: {e}")

            # If we've exhausted all retries
            raise ValueError(
                f"Failed to complete operation after {max_retries} attempts. "
                f"Last error: {last_exception}"
            )

        return wrapper

    return decorator
