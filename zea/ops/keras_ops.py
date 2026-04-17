"""Auto-generated :class:`zea.Operation` for all unary :mod:`keras.ops`
and :mod:`keras.ops.image` functions.

They can be used in zea pipelines like any other :class:`zea.Operation`, for example:

.. doctest::

    >>> from zea.ops.keras_ops import Squeeze

    >>> op = Squeeze(axis=1)

This file is generated automatically. Do not edit manually.
Generated with Keras 3.12.0
"""

import keras

from zea.internal.registry import ops_registry
from zea.ops.base import Lambda

class MissingKerasOps(ValueError):
    def __init__(self, class_name: str, func: str):
        super().__init__(
            f"Failed to create {class_name} with {func}. " +
            "This may be due to an incompatible version of `keras`. " +
            "Please try to upgrade `keras` to the latest version by running " +
            "`pip install --upgrade keras`."
        )


@ops_registry("keras.ops.abs")
class Abs(Lambda):
    """Operation wrapping keras.ops.abs."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.abs, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Abs", "keras.ops.abs") from e

@ops_registry("keras.ops.absolute")
class Absolute(Lambda):
    """Operation wrapping keras.ops.absolute."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.absolute, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Absolute", "keras.ops.absolute") from e

@ops_registry("keras.ops.all")
class All(Lambda):
    """Operation wrapping keras.ops.all."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.all, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("All", "keras.ops.all") from e

@ops_registry("keras.ops.amax")
class Amax(Lambda):
    """Operation wrapping keras.ops.amax."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.amax, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Amax", "keras.ops.amax") from e

@ops_registry("keras.ops.amin")
class Amin(Lambda):
    """Operation wrapping keras.ops.amin."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.amin, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Amin", "keras.ops.amin") from e

@ops_registry("keras.ops.angle")
class Angle(Lambda):
    """Operation wrapping keras.ops.angle."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.angle, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Angle", "keras.ops.angle") from e

@ops_registry("keras.ops.any")
class Any(Lambda):
    """Operation wrapping keras.ops.any."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.any, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Any", "keras.ops.any") from e

@ops_registry("keras.ops.arccos")
class Arccos(Lambda):
    """Operation wrapping keras.ops.arccos."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arccos, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arccos", "keras.ops.arccos") from e

@ops_registry("keras.ops.arccosh")
class Arccosh(Lambda):
    """Operation wrapping keras.ops.arccosh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arccosh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arccosh", "keras.ops.arccosh") from e

@ops_registry("keras.ops.arcsin")
class Arcsin(Lambda):
    """Operation wrapping keras.ops.arcsin."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arcsin, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arcsin", "keras.ops.arcsin") from e

@ops_registry("keras.ops.arcsinh")
class Arcsinh(Lambda):
    """Operation wrapping keras.ops.arcsinh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arcsinh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arcsinh", "keras.ops.arcsinh") from e

@ops_registry("keras.ops.arctan")
class Arctan(Lambda):
    """Operation wrapping keras.ops.arctan."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arctan, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arctan", "keras.ops.arctan") from e

@ops_registry("keras.ops.arctanh")
class Arctanh(Lambda):
    """Operation wrapping keras.ops.arctanh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.arctanh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Arctanh", "keras.ops.arctanh") from e

@ops_registry("keras.ops.argmax")
class Argmax(Lambda):
    """Operation wrapping keras.ops.argmax."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.argmax, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Argmax", "keras.ops.argmax") from e

@ops_registry("keras.ops.argmin")
class Argmin(Lambda):
    """Operation wrapping keras.ops.argmin."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.argmin, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Argmin", "keras.ops.argmin") from e

@ops_registry("keras.ops.argpartition")
class Argpartition(Lambda):
    """Operation wrapping keras.ops.argpartition."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.argpartition, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Argpartition", "keras.ops.argpartition") from e

@ops_registry("keras.ops.argsort")
class Argsort(Lambda):
    """Operation wrapping keras.ops.argsort."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.argsort, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Argsort", "keras.ops.argsort") from e

@ops_registry("keras.ops.array")
class Array(Lambda):
    """Operation wrapping keras.ops.array."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.array, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Array", "keras.ops.array") from e

@ops_registry("keras.ops.average")
class Average(Lambda):
    """Operation wrapping keras.ops.average."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.average, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Average", "keras.ops.average") from e

@ops_registry("keras.ops.bartlett")
class Bartlett(Lambda):
    """Operation wrapping keras.ops.bartlett."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bartlett, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Bartlett", "keras.ops.bartlett") from e

@ops_registry("keras.ops.batch_normalization")
class BatchNormalization(Lambda):
    """Operation wrapping keras.ops.batch_normalization."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.batch_normalization, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BatchNormalization", "keras.ops.batch_normalization") from e

@ops_registry("keras.ops.bincount")
class Bincount(Lambda):
    """Operation wrapping keras.ops.bincount."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bincount, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Bincount", "keras.ops.bincount") from e

@ops_registry("keras.ops.bitwise_and")
class BitwiseAnd(Lambda):
    """Operation wrapping keras.ops.bitwise_and."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_and, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseAnd", "keras.ops.bitwise_and") from e

@ops_registry("keras.ops.bitwise_invert")
class BitwiseInvert(Lambda):
    """Operation wrapping keras.ops.bitwise_invert."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_invert, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseInvert", "keras.ops.bitwise_invert") from e

@ops_registry("keras.ops.bitwise_left_shift")
class BitwiseLeftShift(Lambda):
    """Operation wrapping keras.ops.bitwise_left_shift."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_left_shift, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseLeftShift", "keras.ops.bitwise_left_shift") from e

@ops_registry("keras.ops.bitwise_not")
class BitwiseNot(Lambda):
    """Operation wrapping keras.ops.bitwise_not."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_not, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseNot", "keras.ops.bitwise_not") from e

@ops_registry("keras.ops.bitwise_or")
class BitwiseOr(Lambda):
    """Operation wrapping keras.ops.bitwise_or."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_or, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseOr", "keras.ops.bitwise_or") from e

@ops_registry("keras.ops.bitwise_right_shift")
class BitwiseRightShift(Lambda):
    """Operation wrapping keras.ops.bitwise_right_shift."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_right_shift, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseRightShift", "keras.ops.bitwise_right_shift") from e

@ops_registry("keras.ops.bitwise_xor")
class BitwiseXor(Lambda):
    """Operation wrapping keras.ops.bitwise_xor."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.bitwise_xor, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BitwiseXor", "keras.ops.bitwise_xor") from e

@ops_registry("keras.ops.blackman")
class Blackman(Lambda):
    """Operation wrapping keras.ops.blackman."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.blackman, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Blackman", "keras.ops.blackman") from e

@ops_registry("keras.ops.broadcast_to")
class BroadcastTo(Lambda):
    """Operation wrapping keras.ops.broadcast_to."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.broadcast_to, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("BroadcastTo", "keras.ops.broadcast_to") from e

@ops_registry("keras.ops.cast")
class Cast(Lambda):
    """Operation wrapping keras.ops.cast."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cast, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cast", "keras.ops.cast") from e

@ops_registry("keras.ops.cbrt")
class Cbrt(Lambda):
    """Operation wrapping keras.ops.cbrt."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cbrt, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cbrt", "keras.ops.cbrt") from e

@ops_registry("keras.ops.ceil")
class Ceil(Lambda):
    """Operation wrapping keras.ops.ceil."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.ceil, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Ceil", "keras.ops.ceil") from e

@ops_registry("keras.ops.celu")
class Celu(Lambda):
    """Operation wrapping keras.ops.celu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.celu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Celu", "keras.ops.celu") from e

@ops_registry("keras.ops.cholesky")
class Cholesky(Lambda):
    """Operation wrapping keras.ops.cholesky."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cholesky, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cholesky", "keras.ops.cholesky") from e

@ops_registry("keras.ops.cholesky_inverse")
class CholeskyInverse(Lambda):
    """Operation wrapping keras.ops.cholesky_inverse."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cholesky_inverse, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("CholeskyInverse", "keras.ops.cholesky_inverse") from e

@ops_registry("keras.ops.clip")
class Clip(Lambda):
    """Operation wrapping keras.ops.clip."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.clip, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Clip", "keras.ops.clip") from e

@ops_registry("keras.ops.conj")
class Conj(Lambda):
    """Operation wrapping keras.ops.conj."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.conj, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Conj", "keras.ops.conj") from e

@ops_registry("keras.ops.conjugate")
class Conjugate(Lambda):
    """Operation wrapping keras.ops.conjugate."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.conjugate, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Conjugate", "keras.ops.conjugate") from e

@ops_registry("keras.ops.convert_to_numpy")
class ConvertToNumpy(Lambda):
    """Operation wrapping keras.ops.convert_to_numpy."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.convert_to_numpy, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ConvertToNumpy", "keras.ops.convert_to_numpy") from e

@ops_registry("keras.ops.convert_to_tensor")
class ConvertToTensor(Lambda):
    """Operation wrapping keras.ops.convert_to_tensor."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.convert_to_tensor, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ConvertToTensor", "keras.ops.convert_to_tensor") from e

@ops_registry("keras.ops.copy")
class Copy(Lambda):
    """Operation wrapping keras.ops.copy."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.copy, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Copy", "keras.ops.copy") from e

@ops_registry("keras.ops.corrcoef")
class Corrcoef(Lambda):
    """Operation wrapping keras.ops.corrcoef."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.corrcoef, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Corrcoef", "keras.ops.corrcoef") from e

@ops_registry("keras.ops.cos")
class Cos(Lambda):
    """Operation wrapping keras.ops.cos."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cos, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cos", "keras.ops.cos") from e

@ops_registry("keras.ops.cosh")
class Cosh(Lambda):
    """Operation wrapping keras.ops.cosh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cosh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cosh", "keras.ops.cosh") from e

@ops_registry("keras.ops.count_nonzero")
class CountNonzero(Lambda):
    """Operation wrapping keras.ops.count_nonzero."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.count_nonzero, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("CountNonzero", "keras.ops.count_nonzero") from e

@ops_registry("keras.ops.cumprod")
class Cumprod(Lambda):
    """Operation wrapping keras.ops.cumprod."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cumprod, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cumprod", "keras.ops.cumprod") from e

@ops_registry("keras.ops.cumsum")
class Cumsum(Lambda):
    """Operation wrapping keras.ops.cumsum."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.cumsum, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Cumsum", "keras.ops.cumsum") from e

@ops_registry("keras.ops.deg2rad")
class Deg2rad(Lambda):
    """Operation wrapping keras.ops.deg2rad."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.deg2rad, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Deg2rad", "keras.ops.deg2rad") from e

@ops_registry("keras.ops.det")
class Det(Lambda):
    """Operation wrapping keras.ops.det."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.det, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Det", "keras.ops.det") from e

@ops_registry("keras.ops.diag")
class Diag(Lambda):
    """Operation wrapping keras.ops.diag."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.diag, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Diag", "keras.ops.diag") from e

@ops_registry("keras.ops.diagflat")
class Diagflat(Lambda):
    """Operation wrapping keras.ops.diagflat."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.diagflat, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Diagflat", "keras.ops.diagflat") from e

@ops_registry("keras.ops.diagonal")
class Diagonal(Lambda):
    """Operation wrapping keras.ops.diagonal."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.diagonal, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Diagonal", "keras.ops.diagonal") from e

@ops_registry("keras.ops.digitize")
class Digitize(Lambda):
    """Operation wrapping keras.ops.digitize."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.digitize, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Digitize", "keras.ops.digitize") from e

@ops_registry("keras.ops.dtype")
class Dtype(Lambda):
    """Operation wrapping keras.ops.dtype."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.dtype, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Dtype", "keras.ops.dtype") from e

@ops_registry("keras.ops.eig")
class Eig(Lambda):
    """Operation wrapping keras.ops.eig."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.eig, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Eig", "keras.ops.eig") from e

@ops_registry("keras.ops.eigh")
class Eigh(Lambda):
    """Operation wrapping keras.ops.eigh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.eigh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Eigh", "keras.ops.eigh") from e

@ops_registry("keras.ops.elu")
class Elu(Lambda):
    """Operation wrapping keras.ops.elu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.elu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Elu", "keras.ops.elu") from e

@ops_registry("keras.ops.erf")
class Erf(Lambda):
    """Operation wrapping keras.ops.erf."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.erf, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Erf", "keras.ops.erf") from e

@ops_registry("keras.ops.erfinv")
class Erfinv(Lambda):
    """Operation wrapping keras.ops.erfinv."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.erfinv, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Erfinv", "keras.ops.erfinv") from e

@ops_registry("keras.ops.exp")
class Exp(Lambda):
    """Operation wrapping keras.ops.exp."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.exp, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Exp", "keras.ops.exp") from e

@ops_registry("keras.ops.exp2")
class Exp2(Lambda):
    """Operation wrapping keras.ops.exp2."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.exp2, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Exp2", "keras.ops.exp2") from e

@ops_registry("keras.ops.expand_dims")
class ExpandDims(Lambda):
    """Operation wrapping keras.ops.expand_dims."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.expand_dims, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ExpandDims", "keras.ops.expand_dims") from e

@ops_registry("keras.ops.expm1")
class Expm1(Lambda):
    """Operation wrapping keras.ops.expm1."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.expm1, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Expm1", "keras.ops.expm1") from e

@ops_registry("keras.ops.extract_sequences")
class ExtractSequences(Lambda):
    """Operation wrapping keras.ops.extract_sequences."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.extract_sequences, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ExtractSequences", "keras.ops.extract_sequences") from e

@ops_registry("keras.ops.fft")
class Fft(Lambda):
    """Operation wrapping keras.ops.fft."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.fft, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Fft", "keras.ops.fft") from e

@ops_registry("keras.ops.fft2")
class Fft2(Lambda):
    """Operation wrapping keras.ops.fft2."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.fft2, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Fft2", "keras.ops.fft2") from e

@ops_registry("keras.ops.flip")
class Flip(Lambda):
    """Operation wrapping keras.ops.flip."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.flip, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Flip", "keras.ops.flip") from e

@ops_registry("keras.ops.floor")
class Floor(Lambda):
    """Operation wrapping keras.ops.floor."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.floor, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Floor", "keras.ops.floor") from e

@ops_registry("keras.ops.full_like")
class FullLike(Lambda):
    """Operation wrapping keras.ops.full_like."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.full_like, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("FullLike", "keras.ops.full_like") from e

@ops_registry("keras.ops.gelu")
class Gelu(Lambda):
    """Operation wrapping keras.ops.gelu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.gelu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Gelu", "keras.ops.gelu") from e

@ops_registry("keras.ops.get_item")
class GetItem(Lambda):
    """Operation wrapping keras.ops.get_item."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.get_item, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("GetItem", "keras.ops.get_item") from e

@ops_registry("keras.ops.glu")
class Glu(Lambda):
    """Operation wrapping keras.ops.glu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.glu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Glu", "keras.ops.glu") from e

@ops_registry("keras.ops.hamming")
class Hamming(Lambda):
    """Operation wrapping keras.ops.hamming."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hamming, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Hamming", "keras.ops.hamming") from e

@ops_registry("keras.ops.hanning")
class Hanning(Lambda):
    """Operation wrapping keras.ops.hanning."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hanning, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Hanning", "keras.ops.hanning") from e

@ops_registry("keras.ops.hard_shrink")
class HardShrink(Lambda):
    """Operation wrapping keras.ops.hard_shrink."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hard_shrink, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HardShrink", "keras.ops.hard_shrink") from e

@ops_registry("keras.ops.hard_sigmoid")
class HardSigmoid(Lambda):
    """Operation wrapping keras.ops.hard_sigmoid."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hard_sigmoid, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HardSigmoid", "keras.ops.hard_sigmoid") from e

@ops_registry("keras.ops.hard_silu")
class HardSilu(Lambda):
    """Operation wrapping keras.ops.hard_silu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hard_silu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HardSilu", "keras.ops.hard_silu") from e

@ops_registry("keras.ops.hard_swish")
class HardSwish(Lambda):
    """Operation wrapping keras.ops.hard_swish."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hard_swish, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HardSwish", "keras.ops.hard_swish") from e

@ops_registry("keras.ops.hard_tanh")
class HardTanh(Lambda):
    """Operation wrapping keras.ops.hard_tanh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.hard_tanh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HardTanh", "keras.ops.hard_tanh") from e

@ops_registry("keras.ops.histogram")
class Histogram(Lambda):
    """Operation wrapping keras.ops.histogram."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.histogram, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Histogram", "keras.ops.histogram") from e

@ops_registry("keras.ops.ifft2")
class Ifft2(Lambda):
    """Operation wrapping keras.ops.ifft2."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.ifft2, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Ifft2", "keras.ops.ifft2") from e

@ops_registry("keras.ops.imag")
class Imag(Lambda):
    """Operation wrapping keras.ops.imag."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.imag, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Imag", "keras.ops.imag") from e

@ops_registry("keras.ops.inv")
class Inv(Lambda):
    """Operation wrapping keras.ops.inv."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.inv, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Inv", "keras.ops.inv") from e

@ops_registry("keras.ops.irfft")
class Irfft(Lambda):
    """Operation wrapping keras.ops.irfft."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.irfft, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Irfft", "keras.ops.irfft") from e

@ops_registry("keras.ops.is_tensor")
class IsTensor(Lambda):
    """Operation wrapping keras.ops.is_tensor."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.is_tensor, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("IsTensor", "keras.ops.is_tensor") from e

@ops_registry("keras.ops.isfinite")
class Isfinite(Lambda):
    """Operation wrapping keras.ops.isfinite."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isfinite, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isfinite", "keras.ops.isfinite") from e

@ops_registry("keras.ops.isinf")
class Isinf(Lambda):
    """Operation wrapping keras.ops.isinf."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isinf, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isinf", "keras.ops.isinf") from e

@ops_registry("keras.ops.isnan")
class Isnan(Lambda):
    """Operation wrapping keras.ops.isnan."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isnan, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isnan", "keras.ops.isnan") from e

@ops_registry("keras.ops.isneginf")
class Isneginf(Lambda):
    """Operation wrapping keras.ops.isneginf."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isneginf, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isneginf", "keras.ops.isneginf") from e

@ops_registry("keras.ops.isposinf")
class Isposinf(Lambda):
    """Operation wrapping keras.ops.isposinf."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isposinf, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isposinf", "keras.ops.isposinf") from e

@ops_registry("keras.ops.isreal")
class Isreal(Lambda):
    """Operation wrapping keras.ops.isreal."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.isreal, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Isreal", "keras.ops.isreal") from e

@ops_registry("keras.ops.istft")
class Istft(Lambda):
    """Operation wrapping keras.ops.istft."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.istft, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Istft", "keras.ops.istft") from e

@ops_registry("keras.ops.kaiser")
class Kaiser(Lambda):
    """Operation wrapping keras.ops.kaiser."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.kaiser, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Kaiser", "keras.ops.kaiser") from e

@ops_registry("keras.ops.layer_normalization")
class LayerNormalization(Lambda):
    """Operation wrapping keras.ops.layer_normalization."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.layer_normalization, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LayerNormalization", "keras.ops.layer_normalization") from e

@ops_registry("keras.ops.leaky_relu")
class LeakyRelu(Lambda):
    """Operation wrapping keras.ops.leaky_relu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.leaky_relu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LeakyRelu", "keras.ops.leaky_relu") from e

@ops_registry("keras.ops.left_shift")
class LeftShift(Lambda):
    """Operation wrapping keras.ops.left_shift."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.left_shift, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LeftShift", "keras.ops.left_shift") from e

@ops_registry("keras.ops.log")
class Log(Lambda):
    """Operation wrapping keras.ops.log."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Log", "keras.ops.log") from e

@ops_registry("keras.ops.log10")
class Log10(Lambda):
    """Operation wrapping keras.ops.log10."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log10, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Log10", "keras.ops.log10") from e

@ops_registry("keras.ops.log1p")
class Log1p(Lambda):
    """Operation wrapping keras.ops.log1p."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log1p, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Log1p", "keras.ops.log1p") from e

@ops_registry("keras.ops.log2")
class Log2(Lambda):
    """Operation wrapping keras.ops.log2."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log2, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Log2", "keras.ops.log2") from e

@ops_registry("keras.ops.log_sigmoid")
class LogSigmoid(Lambda):
    """Operation wrapping keras.ops.log_sigmoid."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log_sigmoid, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LogSigmoid", "keras.ops.log_sigmoid") from e

@ops_registry("keras.ops.log_softmax")
class LogSoftmax(Lambda):
    """Operation wrapping keras.ops.log_softmax."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.log_softmax, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LogSoftmax", "keras.ops.log_softmax") from e

@ops_registry("keras.ops.logdet")
class Logdet(Lambda):
    """Operation wrapping keras.ops.logdet."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.logdet, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Logdet", "keras.ops.logdet") from e

@ops_registry("keras.ops.logical_not")
class LogicalNot(Lambda):
    """Operation wrapping keras.ops.logical_not."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.logical_not, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LogicalNot", "keras.ops.logical_not") from e

@ops_registry("keras.ops.logsumexp")
class Logsumexp(Lambda):
    """Operation wrapping keras.ops.logsumexp."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.logsumexp, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Logsumexp", "keras.ops.logsumexp") from e

@ops_registry("keras.ops.lu_factor")
class LuFactor(Lambda):
    """Operation wrapping keras.ops.lu_factor."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.lu_factor, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("LuFactor", "keras.ops.lu_factor") from e

@ops_registry("keras.ops.max")
class Max(Lambda):
    """Operation wrapping keras.ops.max."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.max, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Max", "keras.ops.max") from e

@ops_registry("keras.ops.mean")
class Mean(Lambda):
    """Operation wrapping keras.ops.mean."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.mean, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Mean", "keras.ops.mean") from e

@ops_registry("keras.ops.median")
class Median(Lambda):
    """Operation wrapping keras.ops.median."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.median, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Median", "keras.ops.median") from e

@ops_registry("keras.ops.meshgrid")
class Meshgrid(Lambda):
    """Operation wrapping keras.ops.meshgrid."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.meshgrid, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Meshgrid", "keras.ops.meshgrid") from e

@ops_registry("keras.ops.min")
class Min(Lambda):
    """Operation wrapping keras.ops.min."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.min, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Min", "keras.ops.min") from e

@ops_registry("keras.ops.moments")
class Moments(Lambda):
    """Operation wrapping keras.ops.moments."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.moments, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Moments", "keras.ops.moments") from e

@ops_registry("keras.ops.moveaxis")
class Moveaxis(Lambda):
    """Operation wrapping keras.ops.moveaxis."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.moveaxis, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Moveaxis", "keras.ops.moveaxis") from e

@ops_registry("keras.ops.nan_to_num")
class NanToNum(Lambda):
    """Operation wrapping keras.ops.nan_to_num."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.nan_to_num, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("NanToNum", "keras.ops.nan_to_num") from e

@ops_registry("keras.ops.ndim")
class Ndim(Lambda):
    """Operation wrapping keras.ops.ndim."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.ndim, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Ndim", "keras.ops.ndim") from e

@ops_registry("keras.ops.negative")
class Negative(Lambda):
    """Operation wrapping keras.ops.negative."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.negative, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Negative", "keras.ops.negative") from e

@ops_registry("keras.ops.nonzero")
class Nonzero(Lambda):
    """Operation wrapping keras.ops.nonzero."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.nonzero, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Nonzero", "keras.ops.nonzero") from e

@ops_registry("keras.ops.norm")
class Norm(Lambda):
    """Operation wrapping keras.ops.norm."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.norm, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Norm", "keras.ops.norm") from e

@ops_registry("keras.ops.normalize")
class Normalize(Lambda):
    """Operation wrapping keras.ops.normalize."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.normalize, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Normalize", "keras.ops.normalize") from e

@ops_registry("keras.ops.one_hot")
class OneHot(Lambda):
    """Operation wrapping keras.ops.one_hot."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.one_hot, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("OneHot", "keras.ops.one_hot") from e

@ops_registry("keras.ops.ones_like")
class OnesLike(Lambda):
    """Operation wrapping keras.ops.ones_like."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.ones_like, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("OnesLike", "keras.ops.ones_like") from e

@ops_registry("keras.ops.pad")
class Pad(Lambda):
    """Operation wrapping keras.ops.pad."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.pad, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Pad", "keras.ops.pad") from e

@ops_registry("keras.ops.prod")
class Prod(Lambda):
    """Operation wrapping keras.ops.prod."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.prod, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Prod", "keras.ops.prod") from e

@ops_registry("keras.ops.qr")
class Qr(Lambda):
    """Operation wrapping keras.ops.qr."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.qr, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Qr", "keras.ops.qr") from e

@ops_registry("keras.ops.quantile")
class Quantile(Lambda):
    """Operation wrapping keras.ops.quantile."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.quantile, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Quantile", "keras.ops.quantile") from e

@ops_registry("keras.ops.ravel")
class Ravel(Lambda):
    """Operation wrapping keras.ops.ravel."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.ravel, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Ravel", "keras.ops.ravel") from e

@ops_registry("keras.ops.real")
class Real(Lambda):
    """Operation wrapping keras.ops.real."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.real, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Real", "keras.ops.real") from e

@ops_registry("keras.ops.reciprocal")
class Reciprocal(Lambda):
    """Operation wrapping keras.ops.reciprocal."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.reciprocal, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Reciprocal", "keras.ops.reciprocal") from e

@ops_registry("keras.ops.relu")
class Relu(Lambda):
    """Operation wrapping keras.ops.relu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.relu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Relu", "keras.ops.relu") from e

@ops_registry("keras.ops.relu6")
class Relu6(Lambda):
    """Operation wrapping keras.ops.relu6."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.relu6, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Relu6", "keras.ops.relu6") from e

@ops_registry("keras.ops.repeat")
class Repeat(Lambda):
    """Operation wrapping keras.ops.repeat."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.repeat, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Repeat", "keras.ops.repeat") from e

@ops_registry("keras.ops.reshape")
class Reshape(Lambda):
    """Operation wrapping keras.ops.reshape."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.reshape, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Reshape", "keras.ops.reshape") from e

@ops_registry("keras.ops.rfft")
class Rfft(Lambda):
    """Operation wrapping keras.ops.rfft."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.rfft, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Rfft", "keras.ops.rfft") from e

@ops_registry("keras.ops.right_shift")
class RightShift(Lambda):
    """Operation wrapping keras.ops.right_shift."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.right_shift, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("RightShift", "keras.ops.right_shift") from e

@ops_registry("keras.ops.rms_normalization")
class RmsNormalization(Lambda):
    """Operation wrapping keras.ops.rms_normalization."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.rms_normalization, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("RmsNormalization", "keras.ops.rms_normalization") from e

@ops_registry("keras.ops.roll")
class Roll(Lambda):
    """Operation wrapping keras.ops.roll."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.roll, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Roll", "keras.ops.roll") from e

@ops_registry("keras.ops.round")
class Round(Lambda):
    """Operation wrapping keras.ops.round."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.round, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Round", "keras.ops.round") from e

@ops_registry("keras.ops.rsqrt")
class Rsqrt(Lambda):
    """Operation wrapping keras.ops.rsqrt."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.rsqrt, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Rsqrt", "keras.ops.rsqrt") from e

@ops_registry("keras.ops.saturate_cast")
class SaturateCast(Lambda):
    """Operation wrapping keras.ops.saturate_cast."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.saturate_cast, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("SaturateCast", "keras.ops.saturate_cast") from e

@ops_registry("keras.ops.selu")
class Selu(Lambda):
    """Operation wrapping keras.ops.selu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.selu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Selu", "keras.ops.selu") from e

@ops_registry("keras.ops.shape")
class Shape(Lambda):
    """Operation wrapping keras.ops.shape."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.shape, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Shape", "keras.ops.shape") from e

@ops_registry("keras.ops.sigmoid")
class Sigmoid(Lambda):
    """Operation wrapping keras.ops.sigmoid."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sigmoid, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sigmoid", "keras.ops.sigmoid") from e

@ops_registry("keras.ops.sign")
class Sign(Lambda):
    """Operation wrapping keras.ops.sign."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sign, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sign", "keras.ops.sign") from e

@ops_registry("keras.ops.signbit")
class Signbit(Lambda):
    """Operation wrapping keras.ops.signbit."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.signbit, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Signbit", "keras.ops.signbit") from e

@ops_registry("keras.ops.silu")
class Silu(Lambda):
    """Operation wrapping keras.ops.silu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.silu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Silu", "keras.ops.silu") from e

@ops_registry("keras.ops.sin")
class Sin(Lambda):
    """Operation wrapping keras.ops.sin."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sin, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sin", "keras.ops.sin") from e

@ops_registry("keras.ops.sinh")
class Sinh(Lambda):
    """Operation wrapping keras.ops.sinh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sinh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sinh", "keras.ops.sinh") from e

@ops_registry("keras.ops.size")
class Size(Lambda):
    """Operation wrapping keras.ops.size."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.size, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Size", "keras.ops.size") from e

@ops_registry("keras.ops.slogdet")
class Slogdet(Lambda):
    """Operation wrapping keras.ops.slogdet."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.slogdet, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Slogdet", "keras.ops.slogdet") from e

@ops_registry("keras.ops.soft_shrink")
class SoftShrink(Lambda):
    """Operation wrapping keras.ops.soft_shrink."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.soft_shrink, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("SoftShrink", "keras.ops.soft_shrink") from e

@ops_registry("keras.ops.softmax")
class Softmax(Lambda):
    """Operation wrapping keras.ops.softmax."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.softmax, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Softmax", "keras.ops.softmax") from e

@ops_registry("keras.ops.softplus")
class Softplus(Lambda):
    """Operation wrapping keras.ops.softplus."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.softplus, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Softplus", "keras.ops.softplus") from e

@ops_registry("keras.ops.softsign")
class Softsign(Lambda):
    """Operation wrapping keras.ops.softsign."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.softsign, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Softsign", "keras.ops.softsign") from e

@ops_registry("keras.ops.sort")
class Sort(Lambda):
    """Operation wrapping keras.ops.sort."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sort, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sort", "keras.ops.sort") from e

@ops_registry("keras.ops.sparse_plus")
class SparsePlus(Lambda):
    """Operation wrapping keras.ops.sparse_plus."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sparse_plus, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("SparsePlus", "keras.ops.sparse_plus") from e

@ops_registry("keras.ops.sparse_sigmoid")
class SparseSigmoid(Lambda):
    """Operation wrapping keras.ops.sparse_sigmoid."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sparse_sigmoid, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("SparseSigmoid", "keras.ops.sparse_sigmoid") from e

@ops_registry("keras.ops.sparsemax")
class Sparsemax(Lambda):
    """Operation wrapping keras.ops.sparsemax."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sparsemax, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sparsemax", "keras.ops.sparsemax") from e

@ops_registry("keras.ops.split")
class Split(Lambda):
    """Operation wrapping keras.ops.split."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.split, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Split", "keras.ops.split") from e

@ops_registry("keras.ops.sqrt")
class Sqrt(Lambda):
    """Operation wrapping keras.ops.sqrt."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sqrt, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sqrt", "keras.ops.sqrt") from e

@ops_registry("keras.ops.square")
class Square(Lambda):
    """Operation wrapping keras.ops.square."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.square, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Square", "keras.ops.square") from e

@ops_registry("keras.ops.squareplus")
class Squareplus(Lambda):
    """Operation wrapping keras.ops.squareplus."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.squareplus, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Squareplus", "keras.ops.squareplus") from e

@ops_registry("keras.ops.squeeze")
class Squeeze(Lambda):
    """Operation wrapping keras.ops.squeeze."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.squeeze, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Squeeze", "keras.ops.squeeze") from e

@ops_registry("keras.ops.stack")
class Stack(Lambda):
    """Operation wrapping keras.ops.stack."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.stack, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Stack", "keras.ops.stack") from e

@ops_registry("keras.ops.std")
class Std(Lambda):
    """Operation wrapping keras.ops.std."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.std, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Std", "keras.ops.std") from e

@ops_registry("keras.ops.stft")
class Stft(Lambda):
    """Operation wrapping keras.ops.stft."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.stft, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Stft", "keras.ops.stft") from e

@ops_registry("keras.ops.sum")
class Sum(Lambda):
    """Operation wrapping keras.ops.sum."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.sum, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Sum", "keras.ops.sum") from e

@ops_registry("keras.ops.svd")
class Svd(Lambda):
    """Operation wrapping keras.ops.svd."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.svd, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Svd", "keras.ops.svd") from e

@ops_registry("keras.ops.swapaxes")
class Swapaxes(Lambda):
    """Operation wrapping keras.ops.swapaxes."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.swapaxes, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Swapaxes", "keras.ops.swapaxes") from e

@ops_registry("keras.ops.swish")
class Swish(Lambda):
    """Operation wrapping keras.ops.swish."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.swish, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Swish", "keras.ops.swish") from e

@ops_registry("keras.ops.take")
class Take(Lambda):
    """Operation wrapping keras.ops.take."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.take, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Take", "keras.ops.take") from e

@ops_registry("keras.ops.take_along_axis")
class TakeAlongAxis(Lambda):
    """Operation wrapping keras.ops.take_along_axis."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.take_along_axis, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("TakeAlongAxis", "keras.ops.take_along_axis") from e

@ops_registry("keras.ops.tan")
class Tan(Lambda):
    """Operation wrapping keras.ops.tan."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.tan, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Tan", "keras.ops.tan") from e

@ops_registry("keras.ops.tanh")
class Tanh(Lambda):
    """Operation wrapping keras.ops.tanh."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.tanh, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Tanh", "keras.ops.tanh") from e

@ops_registry("keras.ops.tanh_shrink")
class TanhShrink(Lambda):
    """Operation wrapping keras.ops.tanh_shrink."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.tanh_shrink, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("TanhShrink", "keras.ops.tanh_shrink") from e

@ops_registry("keras.ops.threshold")
class Threshold(Lambda):
    """Operation wrapping keras.ops.threshold."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.threshold, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Threshold", "keras.ops.threshold") from e

@ops_registry("keras.ops.tile")
class Tile(Lambda):
    """Operation wrapping keras.ops.tile."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.tile, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Tile", "keras.ops.tile") from e

@ops_registry("keras.ops.top_k")
class TopK(Lambda):
    """Operation wrapping keras.ops.top_k."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.top_k, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("TopK", "keras.ops.top_k") from e

@ops_registry("keras.ops.trace")
class Trace(Lambda):
    """Operation wrapping keras.ops.trace."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.trace, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Trace", "keras.ops.trace") from e

@ops_registry("keras.ops.transpose")
class Transpose(Lambda):
    """Operation wrapping keras.ops.transpose."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.transpose, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Transpose", "keras.ops.transpose") from e

@ops_registry("keras.ops.tril")
class Tril(Lambda):
    """Operation wrapping keras.ops.tril."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.tril, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Tril", "keras.ops.tril") from e

@ops_registry("keras.ops.triu")
class Triu(Lambda):
    """Operation wrapping keras.ops.triu."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.triu, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Triu", "keras.ops.triu") from e

@ops_registry("keras.ops.trunc")
class Trunc(Lambda):
    """Operation wrapping keras.ops.trunc."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.trunc, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Trunc", "keras.ops.trunc") from e

@ops_registry("keras.ops.unfold")
class Unfold(Lambda):
    """Operation wrapping keras.ops.unfold."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.unfold, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Unfold", "keras.ops.unfold") from e

@ops_registry("keras.ops.unstack")
class Unstack(Lambda):
    """Operation wrapping keras.ops.unstack."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.unstack, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Unstack", "keras.ops.unstack") from e

@ops_registry("keras.ops.var")
class Var(Lambda):
    """Operation wrapping keras.ops.var."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.var, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Var", "keras.ops.var") from e

@ops_registry("keras.ops.view")
class View(Lambda):
    """Operation wrapping keras.ops.view."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.view, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("View", "keras.ops.view") from e

@ops_registry("keras.ops.view_as_complex")
class ViewAsComplex(Lambda):
    """Operation wrapping keras.ops.view_as_complex."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.view_as_complex, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ViewAsComplex", "keras.ops.view_as_complex") from e

@ops_registry("keras.ops.view_as_real")
class ViewAsReal(Lambda):
    """Operation wrapping keras.ops.view_as_real."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.view_as_real, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ViewAsReal", "keras.ops.view_as_real") from e

@ops_registry("keras.ops.zeros_like")
class ZerosLike(Lambda):
    """Operation wrapping keras.ops.zeros_like."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.zeros_like, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ZerosLike", "keras.ops.zeros_like") from e

@ops_registry("keras.ops.image.affine_transform")
class AffineTransform(Lambda):
    """Operation wrapping keras.ops.image.affine_transform."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.affine_transform, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("AffineTransform", "keras.ops.image.affine_transform") from e

@ops_registry("keras.ops.image.crop_images")
class CropImages(Lambda):
    """Operation wrapping keras.ops.image.crop_images."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.crop_images, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("CropImages", "keras.ops.image.crop_images") from e

@ops_registry("keras.ops.image.elastic_transform")
class ElasticTransform(Lambda):
    """Operation wrapping keras.ops.image.elastic_transform."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.elastic_transform, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ElasticTransform", "keras.ops.image.elastic_transform") from e

@ops_registry("keras.ops.image.extract_patches")
class ExtractPatches(Lambda):
    """Operation wrapping keras.ops.image.extract_patches."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.extract_patches, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ExtractPatches", "keras.ops.image.extract_patches") from e

@ops_registry("keras.ops.image.gaussian_blur")
class GaussianBlur(Lambda):
    """Operation wrapping keras.ops.image.gaussian_blur."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.gaussian_blur, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("GaussianBlur", "keras.ops.image.gaussian_blur") from e

@ops_registry("keras.ops.image.hsv_to_rgb")
class HsvToRgb(Lambda):
    """Operation wrapping keras.ops.image.hsv_to_rgb."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.hsv_to_rgb, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("HsvToRgb", "keras.ops.image.hsv_to_rgb") from e

@ops_registry("keras.ops.image.pad_images")
class PadImages(Lambda):
    """Operation wrapping keras.ops.image.pad_images."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.pad_images, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("PadImages", "keras.ops.image.pad_images") from e

@ops_registry("keras.ops.image.perspective_transform")
class PerspectiveTransform(Lambda):
    """Operation wrapping keras.ops.image.perspective_transform."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.perspective_transform, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("PerspectiveTransform", "keras.ops.image.perspective_transform") from e

@ops_registry("keras.ops.image.resize")
class Resize(Lambda):
    """Operation wrapping keras.ops.image.resize."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.resize, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("Resize", "keras.ops.image.resize") from e

@ops_registry("keras.ops.image.rgb_to_grayscale")
class RgbToGrayscale(Lambda):
    """Operation wrapping keras.ops.image.rgb_to_grayscale."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.rgb_to_grayscale, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("RgbToGrayscale", "keras.ops.image.rgb_to_grayscale") from e

@ops_registry("keras.ops.image.rgb_to_hsv")
class RgbToHsv(Lambda):
    """Operation wrapping keras.ops.image.rgb_to_hsv."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.rgb_to_hsv, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("RgbToHsv", "keras.ops.image.rgb_to_hsv") from e

@ops_registry("keras.ops.image.scale_and_translate")
class ScaleAndTranslate(Lambda):
    """Operation wrapping keras.ops.image.scale_and_translate."""

    def __init__(self, **kwargs):
        try:
            super().__init__(func=keras.ops.image.scale_and_translate, **kwargs)
        except AttributeError as e:
            raise MissingKerasOps("ScaleAndTranslate", "keras.ops.image.scale_and_translate") from e
