"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc

from keras import ops

from usbmd.core import Object
from usbmd.registry import operator_registry


class LinearOperator(abc.ABC, Object):
    """Linear operator class y = Ax + n."""

    sigma = 0.0

    @abc.abstractmethod
    def forward(self, data):
        """Implements the forward operator A: x -> y."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        """String repentation of the operator."""
        raise NotImplementedError

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children)

    def _tree_flatten(self):
        return (), ()


@operator_registry(name="inpainting")
class InpaintingOperator(LinearOperator):
    """Inpainting operator A = I * M."""

    def __init__(self, min_val=0.0, **kwargs):
        """Initialize the inpainting operator.

        Args:
            min_val: Minimum value for the mask.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.min_val = min_val

    def forward(self, data, mask):
        # return self.mask * data
        return ops.where(mask, data, self.min_val)

    def __str__(self):
        return "y = Ax + n, where A = I * M"
