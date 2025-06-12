"""Measurement operators.

Handles task-dependent operations (A) and noises (n) to simulate a measurement y = Ax + n.

"""

import abc

from keras import ops

from zea.internal.core import Object
from zea.internal.registry import operator_registry
from zea.utils import translate


class Operator(abc.ABC, Object):
    """Operator base class.

    Used to define a generatic operator for a specific task / forward model.

    Examples are denoising, inpainting, deblurring, etc.

    One can derive linear and non-linear operators from this class.

    - Linear operators: y = Ax + n
    - Non-linear operators: y = f(x) + n

    """

    sigma = 0.0

    @abc.abstractmethod
    def forward(self, data, *args, **kwargs):
        """Implements the forward operator A: x -> y."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        """String representation of the operator."""
        raise NotImplementedError


@operator_registry(name="inpainting")
class InpaintingOperator(Operator):
    """Inpainting operator class.

    Inpainting task is a linear operator that masks the data with a mask.

    Formally defined as:
        y = Ax + n, where A = I * M

    Note that this generally only is the case for min_val = 0.0.
    Since we implement the operator using `ops.where`.

    where I is the identity operator, M is the mask, and n is the noise.
    """

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


@operator_registry(name="soft_inpainting")
class SoftInpaintingOperator(Operator):
    """Soft inpainting operator class.

    Soft inpainting uses a soft grayscale mask for a smooth transition between
    the inpainted and generated regions of the image.
    """

    def __init__(self, image_range, mask_range=None):
        self.image_range = tuple(image_range)
        assert len(self.image_range) == 2

        if mask_range is None:
            mask_range = (0.0, 1.0)
        self.mask_range = tuple(mask_range)
        assert len(self.mask_range) == 2
        assert self.mask_range[0] == 0.0, "mask_range[0] must be 0.0"

    def forward(self, data, mask):
        data1 = translate(data, self.image_range, self.mask_range)
        data2 = mask * data1
        data3 = translate(data2, self.mask_range, self.image_range)
        return data3

    def __str__(self):
        return "SoftInpaintingOperator"
