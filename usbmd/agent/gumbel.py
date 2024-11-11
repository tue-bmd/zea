"""
Gumbel-Softmax trick implemented with the multi-backend `keras.ops`.
TODO: implement tests for these functions.
"""

import keras
import numpy as np
from keras import ops


class SubsetOperator:
    """
    SubsetOperator applies the Gumbel-Softmax trick to perform continuous top-k selection.

    Args:
        k (int): The number of elements to select.
        tau (float, optional): The temperature parameter for Gumbel-Softmax. Defaults to 1.0.
    hard (bool, optional): Whether to use straight-through Gumbel-Softmax. Defaults to False.

    Sources:
        - [Reparameterizable Subset Sampling via Continuous Relaxations](https://github.com/ermongroup/subsets)   # pylint: disable=line-too-long
        - [Sampling Subsets with Gumbel-Top Relaxations](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/subsets.html)  # pylint: disable=line-too-long
    """

    def __init__(self, k, tau=1.0, hard=False, n_value_dims=1):
        self.k = k
        self.tau = tau
        self.hard = hard
        self.EPSILON = np.finfo(np.float32).tiny
        self.n_value_dims = n_value_dims  # for a image mask: n_value_dims=2

    def gumbel_sample(self, shape):
        """Samples from Gumbel(0,1) distribution"""
        uniform = keras.random.uniform(shape, minval=0, maxval=1)
        return -ops.log(-ops.log(uniform + self.EPSILON) + self.EPSILON)

    def __call__(self, scores):
        # Gumbel-Softmax trick to make the sampling differentiable
        gumbel_noise = self.gumbel_sample(ops.shape(scores))
        scores = scores + gumbel_noise

        # Continuous top-k selection
        khot = ops.zeros_like(scores)
        onehot_approx = ops.zeros_like(scores)

        for _ in range(self.k):
            khot_mask = ops.max(1.0 - onehot_approx, self.EPSILON)
            scores = scores + ops.log(khot_mask)
            onehot_approx = ops.softmax(scores / self.tau, axis=1)
            khot = khot + onehot_approx

        # Optionally convert soft selection to hard selection using straight-through estimator
        if self.hard:
            res = hard_straight_through(khot, self.k, self.n_value_dims)
        else:
            res = khot

        return res


def hard_straight_through(khot_orig, k, n_value_dims=1):
    """
    Applies the hard straight-through estimator to the given k-hot encoded tensor.
    Args:
        khot_orig (Tensor): The original k-hot encoded tensor.
        k (int): The number of top elements to select.
        n_value_dims (int, optional): The number of value dimensions in the input tensor.
            Defaults to 1. E.g. for a 2D image mask, `n_value_dims=2`.
    Returns:
        Tensor: The tensor after applying the hard straight-through estimator,
            with the same shape as `khot_orig`.
    """

    # Extract the batch dimensions and the value dimensions
    original_shape = ops.shape(khot_orig)
    value_dims = original_shape[-n_value_dims:]

    # Flatten the input tensor along the value dimensions
    khot = ops.reshape(khot_orig, (-1, ops.prod(value_dims)))

    # Get the top-k indices
    indices = ops.top_k(khot, k)[1]

    # Reshape the indices for use with ops.scatter
    scatter_indices = ops.stack(
        [
            ops.repeat(ops.arange(ops.shape(khot)[0]), k),
            ops.reshape(indices, (-1,)),
        ],
        axis=-1,
    )

    # Create the hard k-hot tensor
    khot_hard = ops.scatter(
        scatter_indices,
        ops.ones(ops.prod(ops.shape(indices)), "float32"),
        ops.shape(khot),
    )

    # Straight-through estimator
    out = khot_hard - ops.stop_gradient(khot) + khot

    # Reshape to the original shape
    return ops.reshape(out, original_shape)
