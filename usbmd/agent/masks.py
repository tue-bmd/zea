import keras
from keras import ops

from usbmd import tensor_ops


def generate_random_lines(
    n_actions: int,
    n_possible_actions: int,
    n_masks: int,
    seed: int | keras.random.SeedGenerator = None,
):
    """Will generate a mask with random lines."""
    fraction = n_actions / n_possible_actions
    masks = keras.random.uniform([n_masks, n_possible_actions], seed=seed) < fraction
    masks = ops.cast(masks, "float32")
    masks = tensor_ops.hard_straight_through(masks, n_actions)
    return masks


def generate_random_lines_uni(
    n_actions: int,
    n_possible_actions: int,
    n_masks: int,
    seed: int | keras.random.SeedGenerator = None,
):
    """
    Quicker version of generate_random_lines.
    But does not guarantee precisely n_actions.
    """
    fraction = n_actions / n_possible_actions
    masks = keras.random.uniform([n_masks, n_possible_actions], seed=seed) < fraction
    return ops.cast(masks, "float32")


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
