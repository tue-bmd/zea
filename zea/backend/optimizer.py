"""Simple implementation of optimizers that support multi-backend autodiff."""

import keras


def adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for Adam.

    Implementation adapted from `JAX's example <https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adam>`_
    See example usage: `JAX's example usage <https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html#jax.example_libraries.optimizers.adam>`_

    Args:
        step_size: positive scalar
        b1: optional, a positive scalar value for beta_1, the exponential decay rate
            for the first moment estimates (default 0.9).
        b2: optional, a positive scalar value for beta_2, the exponential decay rate
            for the second moment estimates (default 0.999).
        eps: optional, a positive scalar value for epsilon, a small constant for
            numerical stability (default 1e-8).

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """

    def init(x0):
        m0 = keras.ops.zeros_like(x0)
        v0 = keras.ops.zeros_like(x0)
        i = 0
        return x0, m0, v0, i

    def update(g, state):
        """Update rule for Adam optimizer.

        Args:
            g (array): gradient
            state (tuple): state of the optimizer
                (x, m, v, i) = (parameter, first moment estimate,
                second moment estimate, iteration count)

        Returns:
            state: updated state
        """
        x, m, v, i = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * keras.ops.square(g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))
        x = x - step_size * mhat / (keras.ops.sqrt(vhat) + eps)
        i = i + 1
        return x, m, v, i

    def get_params(state):
        """Returns just the parameter from the state."""
        x, *_ = state
        return x

    return init, update, get_params
