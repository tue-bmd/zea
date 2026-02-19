"""
Action selection strategies
===========================

These selection strategies implement a variety of policies for choosing which focused
transmit to fire next, potentially given some beliefs about the state of tissue,
as represented by ``particles``.

For a comprehensive example usage, see: :doc:`../notebooks/agent/agent_example`

All strategies are stateless, meaning that they do not maintain any internal state.
"""

from typing import Callable

import keras
from keras import ops

from zea.agent import masks
from zea.backend.autograd import AutoGrad
from zea.func import tensor
from zea.internal.registry import action_selection_registry


class MaskActionModel:
    """Base class for any action selection method that does masking."""

    def apply(self, action, observation):
        """Apply the action to the observation.

        Args:
            action (Tensor): The mask to be applied.
            observation (Tensor): The observation to which the action is applied.

        Returns:
            Tensor: The masked tensor
        """
        return observation * action


class LinesActionModel(MaskActionModel):
    """Base class for action selection methods that select lines."""

    def __init__(self, n_actions: int, n_possible_actions: int, img_width: int, img_height: int):
        """Initialize the LinesActionModel.

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
        assert stack_n_cols.is_integer(), "Image width must be divisible by n_possible_actions."
        self.stack_n_cols = int(stack_n_cols)

    def lines_to_im_size(self, lines):
        """Convert k-hot-encoded line vectors to image size.

        Args:
            lines (Tensor): shape is (n_masks, n_possible_actions)

        Returns:
            Tensor: Masks of shape (n_masks, img_height, img_width)
        """
        return masks.lines_to_im_size(lines, (self.img_height, self.img_width))


@action_selection_registry(name="greedy_entropy")
class GreedyEntropy(LinesActionModel):
    """Greedy entropy action selection.

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
        entropy_sigma: float = 1.0,
        average_entropy_across_batch: bool = False,
    ):
        """Initialize the GreedyEntropy action selection model.

        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions.
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.
            mean (float, optional): The mean of the RBF. Defaults to 0.
            std_dev (float, optional): The standard deviation of the RBF. Defaults to 1.
            num_lines_to_update (int, optional): The number of lines around the selected line
                to update. Must be odd.
            entropy_sigma (float, optional): The standard deviation of the Gaussian
                Mixture components used to approximate the posterior.
            average_entropy_across_batch (bool, optional): Whether to average entropy
                across the batch when selecting lines. This can be useful when
                selecting planes in 3D imaging, where the batch dimension represents
                a third spatial dimension. Defaults to False.
        """
        super().__init__(n_actions, n_possible_actions, img_width, img_height)

        # Number of samples must be odd so that the entropy
        # of the selected line is set to 0 once it's been selected.
        assert num_lines_to_update % 2 == 1, "num_samples must be odd."
        self.num_lines_to_update = num_lines_to_update
        self.average_entropy_across_batch = average_entropy_across_batch

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
        self.entropy_sigma = entropy_sigma
        self.select_line_and_reweight_entropy_vmap = tensor.vmap(
            self.select_line_and_reweight_entropy
        )

    @staticmethod
    def compute_pairwise_pixel_gaussian_error(
        particles, stack_n_cols=1, n_possible_actions=None, entropy_sigma=1
    ):
        """Compute the pairwise pixelwise Gaussian error.

        This function computes the Gaussian error between each pair of pixels in the
        set of particles provided. This can be used to approximate the entropy of
        a Gaussian mixture model, where the particles are the means of the Gaussians.
        For more details see Section 4 here: https://arxiv.org/abs/2406.14388

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tensor: batch of pixelwise pairwise Gaussian errors,
            of shape (n_particles, n_particles, batch, height, width)
        """
        assert particles.shape[1] > 1, "The entropy cannot be approximated using a single particle."

        if n_possible_actions is None:
            n_possible_actions = ops.shape(particles)[-1]

        # TODO: I think we only need to compute the lower triangular
        # of this matrix, since it's symmetric
        squared_l2_error_matrices = (particles[:, :, None, ...] - particles[:, None, :, ...]) ** 2
        gaussian_error_per_pixel_i_j = ops.exp(
            -(squared_l2_error_matrices) / (2 * entropy_sigma**2)
        )
        # Vertically stack all columns corresponding with the same line
        # This way we can just sum across the height axis and get the entropy
        # for each pixel in a given line
        batch_size, n_particles, _, height, _ = ops.shape(gaussian_error_per_pixel_i_j)

        gaussian_error_per_pixel_stacked = ops.transpose(
            ops.reshape(
                ops.transpose(gaussian_error_per_pixel_i_j, (0, 1, 2, 4, 3)),
                [
                    batch_size,
                    n_particles,
                    n_particles,
                    n_possible_actions,
                    height * stack_n_cols,
                ],
            ),
            (0, 1, 2, 4, 3),
        )
        # [n_particles, n_particles, batch, height, width]
        return gaussian_error_per_pixel_stacked

    def compute_pixelwise_entropy(self, particles):
        """
        This function computes the entropy for each line using a Gaussian Mixture Model
        approximation of the posterior distribution.
        For more details see Section VI. B here: https://arxiv.org/pdf/2410.13310

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tensor: batch of entropies per pixel, of shape (batch, height, width)
        """
        n_particles = ops.shape(particles)[1]
        gaussian_error_per_pixel_stacked = self.compute_pairwise_pixel_gaussian_error(
            particles,
            self.stack_n_cols,
            self.n_possible_actions,
            self.entropy_sigma,
        )
        # sum out first dimension of (n_particles x n_particles) error matrix
        # [n_particles, batch, height, width]
        pixelwise_entropy_sum_j = ops.sum(
            (1 / n_particles) * gaussian_error_per_pixel_stacked, axis=1
        )
        log_pixelwise_entropy_sum_j = ops.log(pixelwise_entropy_sum_j)
        # sum out second dimension of (n_particles x n_particles) error matrix
        # [batch, height, width]
        pixelwise_entropy = -ops.sum((1 / n_particles) * log_pixelwise_entropy_sum_j, axis=1)
        return pixelwise_entropy

    def select_line_and_reweight_entropy(self, entropy_per_line, max_entropy_line):
        """Select the line with maximum entropy and reweight the entropies.

        Selected the max entropy line and reweights the entropy values around it,
        approximating the decrease in entropy that would occur from observing that line.

        Args:
            entropy_per_line (Tensor): Entropy per line of shape
                (batch_size, n_possible_actions)

        Returns:
            Tuple: The selected line index and the updated entropies per line
        """

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
            reweighting,
            (start_index,),
            ops.cast(self.upside_down_gaussian, dtype=reweighting.dtype),
        )

        # Apply re-weighting to entropy values
        updated_entropy_per_line_padded = padded_entropy_per_line * reweighting
        updated_entropy_per_line = ops.slice(
            updated_entropy_per_line_padded,
            (self.num_lines_to_update // 2,),
            (self.n_possible_actions,),
        )
        return updated_entropy_per_line

    def sample(self, particles):
        """Sample the action using the greedy entropy method.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
            - Masks of shape (batch_size, img_height, img_width)
        """

        pixelwise_entropy = self.compute_pixelwise_entropy(particles)
        linewise_entropy = ops.sum(pixelwise_entropy, axis=1)
        if self.average_entropy_across_batch:
            linewise_entropy = ops.expand_dims(ops.mean(linewise_entropy, axis=0), axis=0)

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_entropy_line = ops.argmax(linewise_entropy, axis=1)
            linewise_entropy = self.select_line_and_reweight_entropy_vmap(
                linewise_entropy, max_entropy_line
            )
            all_selected_lines.append(max_entropy_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE),
            axis=0,
        )
        return selected_lines_k_hot, self.lines_to_im_size(selected_lines_k_hot)


@action_selection_registry(name="uniform_random")
class UniformRandomLines(LinesActionModel):
    """Uniform random lines action selection.

    Creates masks with uniformly randomly sampled lines.
    """

    def sample(self, batch_size=1, seed=None):
        """Sample the action using the uniform random method.

        Generates or updates an equispaced mask to sweep rightwards by one step across the image.

        Args:
            seed (int | SeedGenerator | jax.random.key, optional): Seed for random
                number generation. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        selected_lines_batched = masks.random_uniform_lines(
            n_actions=self.n_actions,
            n_possible_actions=self.n_possible_actions,
            n_masks=batch_size,
            seed=seed,
        )
        return selected_lines_batched, self.lines_to_im_size(selected_lines_batched)


@action_selection_registry(name="equispaced")
class EquispacedLines(LinesActionModel):
    """Equispaced lines action selection.

    Creates masks with equispaced lines that sweep across
    the image.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        assert_equal_spacing: bool = True,
    ):
        super().__init__(n_actions, n_possible_actions, img_width, img_height)

        self.assert_equal_spacing = assert_equal_spacing
        if self.assert_equal_spacing:
            masks._assert_equal_spacing(n_actions, n_possible_actions)

    def sample(self, current_lines=None, batch_size=1):
        """Sample the action using the equispaced method.

        Generates or updates an equispaced mask to sweep rightwards by one step across the image.

        Returns:
            Tensor: The mask of shape (batch_size, img_size, img_size)
        """
        if current_lines is None:
            return self.initial_sample_stateless(batch_size)
        else:
            return self.sample_stateless(current_lines)

    def initial_sample_stateless(self, batch_size=1):
        """Initial sample stateless.

        Generates a batch of initial equispaced line masks.

        Returns:
            Tuple[Tensor, Tensor]:
                - Selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        initial_lines = masks.initial_equispaced_lines(
            self.n_actions,
            self.n_possible_actions,
            assert_equal_spacing=self.assert_equal_spacing,
        )
        initial_lines = ops.tile(initial_lines, (batch_size, 1))  # (batch_size, n_actions)
        return initial_lines, self.lines_to_im_size(initial_lines)

    def sample_stateless(self, current_lines):
        """Sample stateless.

        Updates an existing equispaced mask to sweep rightwards by one step across the image.

        Args:
            current_lines: Currently selected lines as k-hot vectors,
                shaped (batch_size, n_possible_actions)

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        new_lines = masks.next_equispaced_lines(current_lines)
        return new_lines, self.lines_to_im_size(new_lines)


@action_selection_registry(name="covariance")
class CovarianceSamplingLines(LinesActionModel):
    """Covariance sampling action selection.

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
        """Initialize the CovarianceSamplingLines action selection model.

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

    def random_uniform_lines(self, batch_size, seed=None):
        """Wrapper around `random_uniform_lines` function to use attributes from class."""
        lines = masks.random_uniform_lines(
            self.n_actions,
            self.n_possible_actions,
            batch_size * self.n_masks,
            seed=self.seed if seed is None else seed,
        )
        return ops.reshape(lines, [self.n_masks, batch_size, self.n_possible_actions])

    def sample(self, particles, seed=None):
        """Sample the action using the covariance sampling method.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, h, w)
            seed (int | SeedGenerator | jax.random.key, optional): Seed for random number
                generation. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]:
                - Newly selected lines as k-hot vectors, shaped (batch_size, n_possible_actions)
                - Masks of shape (batch_size, img_height, img_width)
        """
        batch_size, n_particles, rows, _ = ops.shape(particles)

        # [batch_size, rows, cols, n_particles]
        particles = ops.transpose(particles, (0, 2, 3, 1))

        # [batch_size, rows * stack_n_cols, n_possible_actions, n_particles]
        shape = [
            batch_size,
            rows * self.stack_n_cols,
            self.n_possible_actions,
            n_particles,
        ]
        particles = ops.reshape(particles, shape)

        # [batch_size, rows * stack_n_cols, n_possible_actions, n_possible_actions]
        cov_matrix = tensor.batch_cov(particles)

        # Sum over the row dimension [batch_size, n_possible_actions, n_possible_actions]
        cov_matrix = ops.sum(cov_matrix, axis=1)

        # Generate random lines [n_masks, batch_size, n_possible_actions]
        lines = self.random_uniform_lines(batch_size, seed=seed)

        # Make matrix masks [n_masks, batch_size, n_possible_actions, n_possible_actions]
        reshaped_lines = ops.repeat(lines[..., None], self.n_possible_actions, axis=-1)
        bool_masks = ops.logical_and(reshaped_lines, ops.swapaxes(reshaped_lines, -1, -2))

        # Subsample the covariance matrix with random lines
        def subsample_with_mask(mask):
            """Subsample the covariance matrix with a single mask."""
            subsampled_cov_matrix = tensor.boolean_mask(
                cov_matrix, mask, size=batch_size * self.n_actions**2
            )
            return ops.reshape(subsampled_cov_matrix, [batch_size, self.n_actions, self.n_actions])

        # [n_masks, batch_size, cols, cols]
        subsampled_cov_matrices = ops.vectorized_map(subsample_with_mask, bool_masks)

        # [n_masks, batch_size, 1]
        entropies = ops.logdet(subsampled_cov_matrices)[..., None]

        # [1, batch_size, 1]
        best_mask_index = ops.argmax(entropies, axis=0, keepdims=True)

        # [batch_size, n_possible_actions]
        best_mask = ops.take_along_axis(lines, best_mask_index, axis=0)
        best_mask = ops.squeeze(best_mask, axis=0)

        return best_mask, self.lines_to_im_size(best_mask)


class TaskBasedLines(GreedyEntropy):
    """Task-based line selection for maximizing information gain.

    This action selection strategy chooses lines to maximize information gain with respect
    to a downstream task outcome. It uses gradient-based saliency to identify which image
    regions contribute most to task uncertainty, then selects lines accordingly.
    """

    def __init__(
        self,
        n_actions: int,
        n_possible_actions: int,
        img_width: int,
        img_height: int,
        downstream_task_function: Callable,
        mean: float = 0,
        std_dev: float = 1,
        num_lines_to_update: int = 5,
        **kwargs,
    ):
        """Initialize the TaskBasedLines action selection model.

        Args:
            n_actions (int): The number of actions the agent can take.
            n_possible_actions (int): The number of possible actions (line positions).
            img_width (int): The width of the input image.
            img_height (int): The height of the input image.
            downstream_task_function (Callable): A differentiable function that takes a
                batch of inputs and produces scalar outputs. This represents the downstream
                task for which information gain should be maximized.
            mean (float, optional): The mean of the RBF used for reweighting. Defaults to 0.
            std_dev (float, optional): The standard deviation of the RBF used for reweighting.
                Defaults to 1.
            num_lines_to_update (int, optional): The number of lines around the selected line
                to update during reweighting. Must be odd. Defaults to 5.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            n_actions,
            n_possible_actions,
            img_width,
            img_height,
            mean,
            std_dev,
            num_lines_to_update,
        )
        self.downstream_task_function = downstream_task_function

    def compute_output_and_saliency_propagation(self, particles):
        """Compute saliency-weighted posterior variance for task-based selection.

        This method computes how much each pixel contributes to the variance of the
        downstream task output. It uses automatic differentiation to compute gradients
        of the task function with respect to each particle, then weights the posterior
        variance by the squared mean gradient.

        Args:
            particles (Tensor): Particles of shape (batch_size, n_particles, height, width)
                representing the posterior distribution over images.

        Returns:
            Tensor: Pixelwise contribution to downstream task variance,
                of shape (batch_size, height, width). Higher values indicate pixels
                that contribute more to task uncertainty.
        """
        autograd = AutoGrad()

        autograd.set_function(self.downstream_task_function)
        downstream_grad_and_value_fn = autograd.get_gradient_and_value_jit_fn()
        jacobian, _ = ops.vectorized_map(
            lambda p: ops.vectorized_map(
                downstream_grad_and_value_fn,
                p,
            ),
            particles,
        )

        posterior_variance = ops.var(particles, axis=1)
        mean_jacobian = ops.mean(jacobian, axis=1)
        return posterior_variance * (mean_jacobian**2)

    def sum_neighbouring_columns_into_n_possible_actions(self, full_linewise_salience):
        """Aggregate column-wise saliency into line-wise saliency scores.

        This method groups neighboring columns together to create saliency scores
        for each possible line action. Since each line action may correspond to
        multiple image columns, this aggregation is necessary to match the action space.

        Args:
            full_linewise_salience (Tensor): Saliency values for each column,
                of shape (batch_size, full_image_width).

        Returns:
            Tensor: Aggregated saliency scores for each possible action,
                of shape (batch_size, n_possible_actions).

        Raises:
            AssertionError: If the image width is not evenly divisible by n_possible_actions.
        """
        batch_size = ops.shape(full_linewise_salience)[0]
        full_image_width = ops.shape(full_linewise_salience)[1]
        assert full_image_width % self.n_possible_actions == 0, (
            "n_possible_actions must divide evenly into image width"
        )
        cols_per_action = full_image_width // self.n_possible_actions
        stacked_linewise_salience = ops.reshape(
            full_linewise_salience,
            (batch_size, self.n_possible_actions, cols_per_action),
        )
        return ops.sum(stacked_linewise_salience, axis=2)

    def sample(self, particles):
        """Sample actions using task-based information gain maximization.

        This method computes which lines would provide the most information about
        the downstream task by:
        1. Computing pixelwise contribution to task variance using gradients
        2. Aggregating contributions into line-wise scores
        3. Greedily selecting lines with highest contribution scores
        4. Reweighting scores around selected lines (inherited from GreedyEntropy)

        Args:
            particles (Tensor): Particles representing the posterior distribution,
                of shape (batch_size, n_particles, height, width).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - selected_lines_k_hot: Selected lines as k-hot vectors,
                  shaped (batch_size, n_possible_actions)
                - masks: Binary masks of shape (batch_size, img_height, img_width)
                - pixelwise_contribution_to_var_dst: Pixelwise contribution to downstream
                  task variance, of shape (batch_size, height, width)

        Note:
            Unlike the parent GreedyEntropy class, this method returns an additional
            tensor containing the pixelwise contribution scores for analysis.
        """
        pixelwise_contribution_to_var_dst = self.compute_output_and_saliency_propagation(particles)
        linewise_contribution_to_var_dst = ops.sum(pixelwise_contribution_to_var_dst, axis=1)
        actionwise_contribution_to_var_dst = self.sum_neighbouring_columns_into_n_possible_actions(
            linewise_contribution_to_var_dst
        )

        # Greedily select best line, reweight entropies, and repeat
        all_selected_lines = []
        for _ in range(self.n_actions):
            max_contribution_line = ops.argmax(actionwise_contribution_to_var_dst, axis=1)
            actionwise_contribution_to_var_dst = self.select_line_and_reweight_entropy_vmap(
                actionwise_contribution_to_var_dst, max_contribution_line
            )
            all_selected_lines.append(max_contribution_line)

        selected_lines_k_hot = ops.any(
            ops.one_hot(all_selected_lines, self.n_possible_actions, dtype=masks._DEFAULT_DTYPE),
            axis=0,
        )
        return (
            selected_lines_k_hot,
            self.lines_to_im_size(selected_lines_k_hot),
            pixelwise_contribution_to_var_dst,
        )
