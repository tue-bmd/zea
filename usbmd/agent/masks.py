"""Masks generation utilities."""

from typing import List

import keras
from keras import ops

from usbmd.agent.gumbel import hard_straight_through


def random_uniform_lines(
    n_actions: int,
    n_possible_actions: int,
    n_masks: int,
    seed: int | keras.random.SeedGenerator = None,
):
    """
    Will generate a mask with random lines. Guarantees precisely n_actions.

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
    masks = keras.random.uniform(
        [n_masks, n_possible_actions], seed=seed, dtype="float32"
    )
    masks = hard_straight_through(masks, n_actions)
    return masks


def equispaced_lines(
    n_actions: int,
    n_possible_actions: int,
    previous_mask=None,
):
    """
    Generates equispaced k-hot line mask.
    If a previous mask is provided, will shift the mask by one.

    Args:
        n_actions (int): Number of actions to be selected.
        n_possible_actions (int): Number of possible actions.
        previous_actions (Tensor, optional): Previous actions. Defaults to None.

    Returns:
        Tensor: k-hot-encoded line vector of shape (n_possible_actions).
                Needs to be converted to image size.
    """
    assert (
        n_possible_actions % n_actions == 0
    ), "Number of actions must divide evenly into possible actions to use equispaced sampling."
    if previous_mask is None:
        selected_indices = ops.arange(
            0, n_possible_actions - 1, n_possible_actions // n_actions
        )
        masks = ops.zeros(n_possible_actions)
        return ops.scatter_update(
            masks, ops.expand_dims(selected_indices, axis=1), ops.ones(n_actions)
        )
    else:
        return ops.roll(previous_mask, shift=1)


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
    assert (
        remainder == 0
    ), f"Width must be divisible by number of actions. Got remainder: {remainder}."

    # Repeat till width of image
    masks = ops.repeat(lines, width // ops.shape(lines)[1], axis=1)

    # Repeat till height of image
    masks = ops.repeat(masks[:, None], height, axis=1)

    return masks


def make_line_mask(
    line_indices: List[int],
    image_shape: List[int],
    line_width: int = 1,
    dtype="float32",
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

    height, _, channels = image_shape
    mask = ops.zeros(image_shape, dtype=dtype)

    line_indices = ops.expand_dims(line_indices, axis=1)
    base_range = ops.arange(line_width, dtype=line_indices.dtype)
    selected_columns = line_indices * line_width + base_range
    selected_columns = ops.reshape(selected_columns, (-1,))
    num_columns = selected_columns.shape[0]

    rows = ops.arange(height)
    rows = ops.reshape(rows, (height, 1, 1))
    rows = ops.broadcast_to(rows, (height, num_columns, channels))
    columns = ops.broadcast_to(
        ops.reshape(selected_columns, (1, num_columns, 1)),
        (height, num_columns, channels),
    )
    channel_indices = ops.arange(channels)
    channel_indices = ops.reshape(channel_indices, (1, 1, channels))
    channel_indices = ops.broadcast_to(channel_indices, (height, num_columns, channels))

    indices = ops.stack([rows, columns, channel_indices], axis=-1)
    indices = ops.reshape(indices, (-1, 3))
    updates = ops.ones((height * num_columns * channels,), dtype="float32")

    mask = ops.scatter_update(mask, indices, updates)
    return mask
