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


class LinesActionModel(MaskActionModel):
    """Base class for action selection methods that select lines."""

    def __init__(
        self, n_actions: int, n_possible_actions: int, img_width: int, img_height: int
    ):
        """
        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.

        Attributes:
            stack_n_cols (int): The number of columns in the image that correspond
                to a single action.
        """
        super().__init__()
        self.n_actions = n_actions
        self.n_possible_actions = n_possible_actions
        self.img_width = img_width
        self.img_height = img_height

        stack_n_cols = self.img_width / self.n_possible_actions
        assert (
            stack_n_cols.is_integer()
        ), "Image size must be divisible by n_possible_actions."
        self.stack_n_cols = int(stack_n_cols)


class GreedyEntropy(LinesActionModel):
    """
    Selects the max entropy line and reweights the entropy values around it,
    approximating the decrease in entropy that would occur from observing that line.

    The neighbouring values are decreased by a Gaussian function centered at the selected line.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        mean: float = 0,
        std_dev: float = 1,
        num_lines_to_update: int = 5,
    ):
        """
        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.
            mean (float, optional): The mean of the RBF. Defaults to 0.
            std_dev (float, optional): The standard deviation of the RBF. Defaults to 1.
            num_lines_to_update (int, optional): The number of lines around the selected line
                to update. Must be odd.
        """
        super().__init__(n_actions, n_possible_actions, img_width, img_height)

        # Number of samples must be odd so that the entropy
        # of the selected line is set to 0 once it's been selected.
        assert num_lines_to_update % 2 == 1, "num_samples must be odd."
        self.num_lines_to_update = num_lines_to_update

        # see here what I mean by upside_down_gaussian:
        # https://colab.research.google.com/drive/1CQp_Z6nADzOFsybdiH5Cag0vtVZjjioU?usp=sharing
        upside_down_gaussian = lambda x: 1 - ops.exp(-0.5 * ((x - mean) / std_dev) ** 2)
        # Sample `num_lines_to_update` points symmetrically around the mean.
        # This can be tuned to determine how the entropy for neighbouring lines is updated
        # TODO: learn this function from training data
        points_to_evaluate = ops.linspace(
            mean - 2 * std_dev,
            mean + 2 * std_dev,
            self.num_lines_to_update,
        )
        self.upside_down_gaussian = upside_down_gaussian(points_to_evaluate)
        self.entropy_sigma = 1

    def compute_gmm_entropy_per_line(self, particles):
        """
        This function computes the entropy for each line using a Gaussian Mixture Model
        approximation of the posterior distribution.
        For more details see Section 4 here: https://arxiv.org/abs/2406.14388

        Args:
            particles (Tensor): Particles of shape (n_particles, batch_size, height, width)

        Returns:
            Tensor: batch of entropies per line, of shape (batch, n_possible_actions)
        """
        assert (
            particles.shape[0] > 1
        ), "The entropy cannot be approximated using a single sample."

        # TODO: I think we only need to compute the lower triangular
        # of this matrix, since it's symmetric
        squared_l2_error_matrices = (
            particles[:, None, ...] - particles[None, :, ...]
        ) ** 2
        gaussian_error_per_pixel_i_j = ops.exp(
            (squared_l2_error_matrices) / (2 * self.entropy_sigma**2)
        )
        # Vertically stack all columns corresponding with the same line
        # This way we can just sum across the height axis and get the entropy
        # for each pixel in a given line
        _, n_particles, batch_size, height, _ = gaussian_error_per_pixel_i_j.shape
        gaussian_error_per_pixel_stacked = ops.reshape(
            gaussian_error_per_pixel_i_j,
            [
                n_particles,
                n_particles,
                batch_size,
                height * self.stack_n_cols,
                self.n_possible_actions,
            ],
        )
        gaussian_error_per_line = ops.sum(gaussian_error_per_pixel_stacked, axis=3)
        # sum out first dimension of (n_particles x n_particles) error matrix
        # [n_particles, batch, n_possible_actions]
        entropy_per_line_i = ops.sum(gaussian_error_per_line, axis=0)
        # sum out second dimension of (n_particles x n_particles) error matrix
        # [batch, n_possible_actions]
        entropy_per_line = ops.sum(entropy_per_line_i, axis=0)
        return entropy_per_line

    def sample(self, particles):
        """
        Args:
            particles (Tensor): Particles of shape (n_particles, batch_size, height, width)

        Returns:
            Tensor: Batch of masks of shape (batch_size, height, width)
        """
        entropy_per_line = self.compute_gmm_entropy_per_line(particles)

        def select_line_and_reweight_entropy(entropy_per_line):
            """
            Selected the max entropy line and reweights the entropy values around it,
            approximating the decrease in entropy that would occur from observing that line.

            Args:
                entropy_per_line (Tensor): Entropy per line of shape
                    (batch_size, n_possible_actions)

            Returns:
                Tuple: The selected line index and the updated entropies per line
            """

            # Find the line with maximum entropy
            max_entropy_line = ops.argmax(entropy_per_line)

            ## The rest of this function updates the entropy values around max_entropy_line
            ## by multiplying them with an upside-down Gaussian function centered at
            ## max_entropy_line, setting the entropy of the selected line to 0, and decreasing
            ## the entropies of neighbouring lines.

            # Pad the entropy per line to allow for re-weighting with fixed
            # size RBF, which is necessary for tracing.
            padded_entropy_per_line = ops.pad(
                entropy_per_line,
                (self.num_lines_to_update // 2, self.num_lines_to_update // 2),
            )
            # because the entropy per line has now been padded, the start index
            # of the set of lines to update is simply the index of the max_entropy_line
            start_index = max_entropy_line

            # Create the re-weighting vector
            reweighting = ops.ones_like(padded_entropy_per_line)
            reweighting = ops.slice_update(
                reweighting, (start_index,), self.upside_down_gaussian
            )

            # Apply re-weighting to entropy values
            updated_entropy_per_line_padded = padded_entropy_per_line * reweighting
            updated_entropy_per_line = ops.slice(
                updated_entropy_per_line_padded,
                (self.num_lines_to_update // 2,),
                (self.n_possible_actions,),
            )
            return max_entropy_line, updated_entropy_per_line

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_entropy_line, entropy_per_line = ops.vectorized_map(
                select_line_and_reweight_entropy, entropy_per_line
            )
            all_selected_lines.append(max_entropy_line)

        def selected_lines_to_line_mask(selected_lines):
            mask = masks.make_line_mask(
                selected_lines, (self.img_height, self.img_width, 1)
            )
            return ops.squeeze(mask, axis=-1)

        return ops.vectorized_map(selected_lines_to_line_mask, all_selected_lines)


class CovarianceSamplingLines(LinesActionModel):
    """
    This class models the line-to-line correlation to select the mask with the highest entropy.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        seed: int = 42,
        n_masks: int = 200,
    ):
        """
        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.
            seed (int, optional): The seed for random number generation. Defaults to 42.
            n_masks (int, optional): The number of masks. Defaults to 200.

        Raises:
            AssertionError: If image width is not divisible by n_possible_actions.
        """
        super().__init__(n_actions, n_possible_actions, img_width, img_height)
        self.seed = keras.random.SeedGenerator(seed)
        self.n_masks = n_masks

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
            particles (Tensor): Particles of shape (n_particles, batch_size, h, w)

        Returns:
            Tensor: The mask of shape (batch_size, img_size, img_size)
        """
        # [n_particles, batch_size, rows, cols]
        n_particles, batch_size, rows, _ = ops.shape(particles)

        # [batch_size, rows, cols, n_particles]
        particles = ops.transpose(particles, (1, 2, 3, 0))

        # [batch_size, rows * stack_n_cols, n_possible_actions, n_particles]
        shape = [
            batch_size,
            rows * self.stack_n_cols,
            self.n_possible_actions,
            n_particles,
        ]
        particles = ops.reshape(particles, shape)

        # [batch_size, rows, n_possible_actions, n_possible_actions]
        cov_matrix = tensor_ops.batch_cov(particles)

        # Sum over the row dimension [batch_size, n_possible_actions, n_possible_actions]
        cov_matrix = ops.sum(cov_matrix, axis=1)

        # Generate random lines [n_masks, batch_size, n_possible_actions]
        lines = self.random_uniform_lines(batch_size)
        bool_lines = ops.cast(lines, "bool")

        # Make matrix masks [n_masks, batch_size, n_possible_actions, n_possible_actions]
        bool_lines = ops.repeat(bool_lines[..., None], self.n_possible_actions, axis=-1)
        bool_masks = ops.logical_and(bool_lines, ops.swapaxes(bool_lines, -1, -2))

        # Subsample the covariance matrix with random lines
        def subsample_with_mask(mask):
            """
            Subsample the covariance matrix with a single mask
            """
            subsampled_cov_matrix = tensor_ops.boolean_mask(
                cov_matrix, mask, size=self.n_actions**2
            )
            return ops.reshape(
                subsampled_cov_matrix, [batch_size, self.n_actions, self.n_actions]
            )

        subsampled_cov_matrices = ops.vectorized_map(subsample_with_mask, bool_masks)

        # [n_masks, batch_size, cols, cols]
        subsampled_cov_matrices = ops.stack(subsampled_cov_matrices)

        # [n_masks, batch_size, 1]
        entropies = ops.logdet(subsampled_cov_matrices)[..., None]

        # [1, batch_size, 1]
        best_mask_index = ops.argmax(entropies, axis=0, keepdims=True)

        # [batch_size, n_possible_actions]
        best_mask = ops.take_along_axis(lines, best_mask_index, axis=0)
        best_mask = ops.squeeze(best_mask, axis=0)

        # [batch_size, h, w]
        return masks.lines_to_im_size(best_mask, (self.img_height, self.img_width))
