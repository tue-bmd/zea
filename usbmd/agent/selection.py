"""
Module for action selection strategies.
"""

import keras
from keras import ops

from usbmd import tensor_ops
from usbmd.agent import masks


class MaskActionModel:
    """
    Base class for any action selection method that does masking.
    """

    def apply(self, action, observation):
        """
        Args:
            action (Tensor): The mask to be applied.
            observation (Tensor): The observation to which the action is applied.

        Returns:
            Tensor: The masked tensor
        """
        return observation * action


class MaxEntropySamplingLines(MaskActionModel):
    """
    This class models the line-to-line correlation to select the mask with the highest entropy.
    """

    def __init__(
        self,
        img_size: int,
        n_actions: int,
        n_possible_actions: int,
        decoder: keras.layers.Layer = None,
        seed: int = 42,
        n_masks: int = 200,
    ):
        """
        Args:
            img_size (int): The size of the input image.
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            decoder (keras.layers.Layer, optional): The decoder layer that brings the particles to
                the image space. Defaults to None.
            seed (int, optional): The seed for random number generation. Defaults to 42.
            n_masks (int, optional): The number of masks. Defaults to 200.

        Raises:
            AssertionError: If img_size is not divisible by n_possible_actions.
        """
        self.img_size = img_size
        self.n_actions = n_actions
        self.n_possible_actions = n_possible_actions
        if decoder is None:
            self.decoder = keras.layers.Identity()
        else:
            self.decoder = decoder

        self.seed = keras.random.SeedGenerator(seed)
        self.n_masks = n_masks

        stack_n_cols = self.img_size / self.n_possible_actions
        assert (
            stack_n_cols.is_integer()
        ), "Image size must be divisible by n_possible_actions."
        self.stack_n_cols = int(stack_n_cols)

    def random_uniform_lines(self, batch_size):
        """Wrapper around `random_uniform_lines` function to use attributes from class."""
        lines = masks.random_uniform_lines(
            self.n_actions,
            self.n_possible_actions,
            batch_size * self.n_masks,
            seed=self.seed,
        )
        return ops.reshape(lines, [self.n_masks, batch_size, self.n_possible_actions])

    def sample(self, particles):
        """
        Args:
            particles (Tensor): Particles of shape (n_particles, batch_size, *features)

        Returns:
            Tensor: The mask of shape (batch_size, img_size, img_size)
        """
        particles_img_space = self.decoder(particles)

        # [n_particles, batch_size, rows, cols]
        n_particles, batch_size, rows, _ = ops.shape(particles_img_space)

        # [batch_size, rows, cols, n_particles]
        particles_img_space = ops.transpose(particles_img_space, (1, 2, 3, 0))

        # [batch_size, rows * stack_n_cols, n_possible_actions, n_particles]
        shape = [
            batch_size,
            rows * self.stack_n_cols,
            self.n_possible_actions,
            n_particles,
        ]
        particles_img_space = ops.reshape(particles_img_space, shape)

        # [batch_size, rows, n_possible_actions, n_possible_actions]
        cov_matrix = tensor_ops.batch_cov(particles_img_space)

        # Sum over the row dimension [batch_size, n_possible_actions, n_possible_actions]
        cov_matrix = ops.sum(cov_matrix, axis=1)

        # Generate random lines [n_masks, batch_size, n_possible_actions]
        lines = self.random_uniform_lines(batch_size)
        bool_lines = ops.cast(lines, "bool")

        # Make matrix masks [n_masks, batch_size, n_possible_actions, n_possible_actions]
        bool_lines = ops.repeat(bool_lines[..., None], self.n_possible_actions, axis=-1)
        bool_masks = ops.logical_and(bool_lines, ops.swapaxes(bool_lines, -1, -2))

        # Subsample the covariance matrix with random lines
        subsampled_cov_matrices = []
        for mask in bool_masks:
            subsampled_cov_matrix = tensor_ops.boolean_mask(cov_matrix, mask)
            subsampled_cov_matrix = ops.reshape(
                subsampled_cov_matrix, [batch_size, self.n_actions, self.n_actions]
            )
            subsampled_cov_matrices.append(subsampled_cov_matrix)

        # [n_masks, batch_size, cols, cols]
        subsampled_cov_matrices = ops.stack(subsampled_cov_matrices)

        # [n_masks, batch_size, 1]
        entropies = tensor_ops.logdet(subsampled_cov_matrices)[..., None]

        # [1, batch_size, 1]
        best_mask_index = ops.argmax(entropies, axis=0, keepdims=True)

        # [batch_size, n_possible_actions]
        best_mask = ops.take_along_axis(lines, best_mask_index, axis=0)
        best_mask = ops.squeeze(best_mask, axis=0)

        # [batch_size, h, w]
        return masks.lines_to_im_size(best_mask, (self.img_size, self.img_size))
