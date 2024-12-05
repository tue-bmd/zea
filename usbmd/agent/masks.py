"""Masks generation utilities."""

import keras
from keras import ops

from usbmd.agent.gumbel import hard_straight_through


def generate_random_lines(
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
        seed (int | SeedGenerator, optional): Seed for random number generation. Defaults to None.

    Returns:
        Tensor: Line vectors of shape (n_masks, n_possible_actions).
                Need to be converted to image size.
    """
    masks = keras.random.uniform(
        [n_masks, n_possible_actions], seed=seed, dtype="float32"
    )
    masks = hard_straight_through(masks, n_actions)
    return masks


def lines_to_im_size(lines, img_size: tuple):
    """
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
