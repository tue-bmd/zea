"""
Mask generation utilities.

These masks are used as a measurement operator for focused scan-line subsampling.
"""

from typing import List

import keras
from keras import ops

from zea import tensor_ops
from zea.agent.gumbel import hard_straight_through

_DEFAULT_DTYPE = "bool"


def indices_to_k_hot(
    indices: List[int],
    n_possible_actions: int,
    dtype=_DEFAULT_DTYPE,
):
    """Convert a list of indices to a k-hot encoded vector.

    A k-hot encoded vector is suitable during tracing when the number of actions can change.
    This is the default represenation for actions in zea.

    Args:
        indices (List[int]): List of indices to set to 1.
        n_possible_actions (int): Total number of possible actions.
        dtype (str, optional): Data type of the mask. Defaults to _DEFAULT_DTYPE.

    Returns:
        Tensor: k-hot-encoded vector of shape (n_possible_actions).
    """
    mask = ops.zeros(n_possible_actions, dtype=dtype)
    return ops.scatter_update(
        mask, ops.expand_dims(indices, axis=1), ops.ones(len(indices), dtype=dtype)
    )


def k_hot_to_indices(selected_lines, n_actions: int, fill_value=-1):
    """Convert k-hot encoded lines to indices of selected actions.

    Args:
        selected_lines (Tensor): k-hot encoded lines of shape (batch_size, n_possible_actions).
        n_actions (int): Number of lines selected.
        fill_value (int, optional): Value to fill in case there are not enough selected actions.
            Defaults to -1.

    Returns:
        Tensor: Indices of selected actions of shape (batch_size, n_actions).
            If there are fewer than `n_actions` selected, the remaining indices will be
            filled with `fill_value`.
    """

    # Find nonzero indices for each frame
    def get_nonzero(row):
        return tensor_ops.nonzero(row > 0, size=n_actions, fill_value=fill_value)[0]

    indices = ops.vectorized_map(get_nonzero, selected_lines)
    return indices


def random_uniform_lines(
    n_actions: int,
    n_possible_actions: int,
    n_masks: int,
    seed: int | keras.random.SeedGenerator = None,
    dtype=_DEFAULT_DTYPE,
):
    """Will generate a mask with random lines.

    Guarantees precisely n_actions.

    Args:
        n_actions (int): Number of actions to be selected.
        n_possible_actions (int): Number of possible actions.
        n_masks (int): Number of masks to generate.
        seed (int | SeedGenerator | jax.random.key, optional): Seed for random number generation.
            Defaults to None.

    Returns:
        Tensor: k-hot-encoded line vectors of shape (n_masks, n_possible_actions).
                Needs to be converted to image size.
    """
    masks = keras.random.uniform([n_masks, n_possible_actions], seed=seed, dtype="float32")
    masks = hard_straight_through(masks, n_actions)
    return ops.cast(masks, dtype=dtype)


def _assert_equal_spacing(n_actions, n_possible_actions):
    assert n_possible_actions % n_actions == 0, (
        "Number of actions must divide evenly into possible actions to use equispaced sampling."
    )


def initial_equispaced_lines(
    n_actions, n_possible_actions, dtype=_DEFAULT_DTYPE, assert_equal_spacing=True
):
    """Generate an initial equispaced k-hot line mask.

    For example, if ``n_actions=2`` and ``n_possible_actions=6``,
    then ``initial_mask=[1, 0, 0, 1, 0, 0]``.

    Args:
        n_actions (int): Number of actions to be selected.
        n_possible_actions (int): Number of possible actions.
        dtype (str, optional): Data type of the mask. Defaults to _DEFAULT_DTYPE.
        assert_equal_spacing (bool, optional): If True, asserts that
            `n_possible_actions` is divisible by `n_actions`, this means that every
            line will have the exact same spacing. Otherwise, there might be
            some spacing differences. Defaults to True.

    Returns:
        Tensor: k-hot-encoded line vector of shape (n_possible_actions).
            Needs to be converted to image size.
    """
    if assert_equal_spacing:
        _assert_equal_spacing(n_actions, n_possible_actions)
        selected_indices = ops.arange(0, n_possible_actions, n_possible_actions // n_actions)
    else:
        selected_indices = ops.linspace(0, n_possible_actions - 1, n_actions, dtype="int32")

    return indices_to_k_hot(selected_indices, n_possible_actions, dtype=dtype)


def next_equispaced_lines(previous_lines, shift=1):
    """
    Rolls the previous equispaced mask of shape (..., n_possible_actions) to the right by
    `shift` which is 1 by default.
    """
    return ops.roll(previous_lines, shift=shift, axis=-1)


def lines_to_im_size(lines, img_size: tuple):
    """
    Convert k-hot-encoded line vectors to image size.

    Args:
        lines (Tensor): shape is (n_masks, n_possible_actions)
        img_size (tuple): (height, width)

    Returns:
        Tensor: Masks of shape (n_masks, img_size, img_size)
    """
    height, width = img_size

    remainder = width % ops.shape(lines)[1]
    assert remainder == 0, (
        f"Width must be divisible by number of actions. Got remainder: {remainder}."
    )

    # Repeat till width of image
    masks = ops.repeat(lines, width // ops.shape(lines)[1], axis=1)

    # Repeat till height of image
    masks = ops.repeat(masks[:, None], height, axis=1)

    return masks


def make_line_mask(
    line_indices: List[int],
    image_shape: List[int],
    line_width: int = 1,
    dtype=_DEFAULT_DTYPE,
):
    """
    Creates a mask with vertical (i.e. second axis) lines at specified indices.

    Args:
        line_indices (List[int]): A list of indices where the lines should be drawn.
        image_shape (List[int]): The shape of the image as [height, width, channels].
        line_width (int, optional): The width of each line. Defaults to 1.
        dtype (str, optional): The data type of the mask. Defaults to "float32".

    Returns:
        mask (Tensor): A tensor of the same shape as `image_shape` with lines drawn
            at the specified indices.
    """
    height, width, channels = image_shape

    # Create k-hot vector for the line indices
    k_hot = indices_to_k_hot(line_indices, width // line_width, dtype=dtype)
    # Expand to (1, n_possible_actions) for lines_to_im_size
    k_hot = ops.expand_dims(k_hot, axis=0)
    # Use lines_to_im_size to create the mask of shape (1, height, width)
    mask_2d = lines_to_im_size(k_hot, (height, width))[0]

    # Expand to (height, width, channels)
    return ops.broadcast_to(mask_2d[..., None], (height, width, channels))
