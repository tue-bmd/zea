"""Matrix-free linear-algebra utilities for inverse ultrasound problems.

This module provides the solver primitives used by :mod:`zea.inverse`:

* :func:`linear_adjoint` — build the adjoint (transpose) of a linear operator
  with backend-agnostic automatic differentiation.
* :func:`cgls` — conjugate gradient least squares for matrix-free linear
  operators.

Operators are plain callables on tensors of any shape; inner products are
taken over all elements, so no explicit flattening or matrix assembly is
required.
"""

from keras import ops

from zea import log
from zea.backend.autograd import AutoGrad


def linear_adjoint(matvec, input_template):
    r"""Construct the adjoint of a linear operator via automatic differentiation.

    For a linear operator :math:`A` the adjoint satisfies
    :math:`\langle A x, y \rangle = \langle x, A^T y \rangle`. It is obtained
    here as the gradient of :math:`x \mapsto \langle A x,\, y \rangle`
    evaluated at :math:`x = 0`, which equals :math:`A^T y` exactly (no
    linearization error) because the map is linear in :math:`x`.

    Args:
        matvec (callable): Linear function mapping an input tensor to an
            output tensor. Must be built from differentiable Keras ops.
        input_template (Tensor): Tensor with the shape and dtype of the
            operator input. Only shape and dtype are used.

    Returns:
        callable: Function mapping an output-shaped tensor ``y`` to
        :math:`A^T y` with the shape of ``input_template``.

    Example:
        .. doctest::

            >>> import numpy as np
            >>> from keras import ops
            >>> from zea.inverse import linear_adjoint

            >>> matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
            >>> matvec = lambda x: ops.matmul(matrix, x)
            >>> rmatvec = linear_adjoint(matvec, ops.zeros(5))
            >>> y = np.ones(3, dtype=np.float32)
            >>> bool(np.allclose(rmatvec(y), matrix.T @ y, atol=1e-5))  # doctest: +SKIP
            True
    """
    zeros = ops.zeros_like(input_template)

    autograd = AutoGrad()
    autograd.set_function(lambda x, cotangent: ops.sum(matvec(x) * cotangent))

    def rmatvec(y):
        return autograd.gradient(zeros, cotangent=ops.convert_to_tensor(y))

    return rmatvec


def cgls(matvec, rmatvec, b, x0, n_iter=50, verbose=False):
    r"""Conjugate gradient least squares (CGLS).

    Iteratively minimizes :math:`\|A x - b\|^2` for a matrix-free linear
    operator :math:`A` given by ``matvec`` and its adjoint ``rmatvec``.
    Started from zero on an underdetermined system, CGLS converges to the
    minimum-norm least-squares (Moore-Penrose pseudo-inverse) solution.

    Args:
        matvec (callable): The linear operator :math:`A`.
        rmatvec (callable): The adjoint operator :math:`A^T` (see
            :func:`linear_adjoint`).
        b (Tensor): Measurement, shaped like the output of ``matvec``.
        x0 (Tensor): Initial iterate, shaped like the input of ``matvec``.
            Use zeros for the minimum-norm solution.
        n_iter (int, optional): Number of iterations. Defaults to ``50``.
        verbose (bool, optional): Log the relative residual periodically.
            Defaults to ``False``.

    Returns:
        Tensor: The solution estimate with the shape of ``x0``.
    """
    eps = 1e-30
    x = ops.convert_to_tensor(x0)
    b = ops.convert_to_tensor(b)
    residual = b - matvec(x)
    s = rmatvec(residual)
    direction = s
    gamma = ops.sum(s * s)
    b_norm = ops.sqrt(ops.sum(b * b))

    log_every = max(1, n_iter // 8)
    for iteration in range(1, n_iter + 1):
        q = matvec(direction)
        alpha = gamma / (ops.sum(q * q) + eps)
        x = x + alpha * direction
        residual = residual - alpha * q
        s = rmatvec(residual)
        gamma_new = ops.sum(s * s)
        direction = s + (gamma_new / (gamma + eps)) * direction
        gamma = gamma_new
        if verbose and (iteration % log_every == 0 or iteration == n_iter):
            relative_residual = ops.sqrt(ops.sum(residual * residual)) / (b_norm + eps)
            log.info(
                f"cgls iteration {iteration:3d} | "
                f"relative residual {float(ops.convert_to_numpy(relative_residual)):.4e}"
            )
    return x
