"""Tests for zea.internal.core."""

import functools
import json

import keras
import numpy as np
import pytest

from zea.internal.core import (
    DataTypes,
    ModTypes,
    ZEADecoderJSON,
    ZEAEncoderJSON,
    _unwrap_compiled_callable,
    classproperty,
    dict_to_tensor,
    serialize_elements,
)

from . import run_in_backend


def _plain_function(x):
    """Module-level (thus picklable) function for serialization tests."""
    return x + 1


class _Unpicklable:
    """Callable that always fails pickling."""

    def __reduce__(self):
        raise RuntimeError("cannot pickle")

    def __call__(self):
        return None


class _UnpicklableWrapped(_Unpicklable):
    """Non-JIT callable that merely carries a ``__wrapped__`` attribute."""

    __wrapped__ = _plain_function

    def __call__(self, x):
        return self.__wrapped__(x)


class _JitObj:
    """Object holding a ``torch.compile``d bound method (like ``DDS._call``)."""

    def __init__(self, k):
        import torch

        self.k = k
        self._call = torch.compile(self.call)

    def call(self, x):
        return x * self.k


def test_dict_to_tensor_ignores_framework_to_tensor_methods():
    """Objects with a `to_tensor` method whose signature doesn't accept `keep_as_is`
    (e.g. `tf.RaggedTensor`) must not be dispatched through zea's conversion protocol.
    """
    tf = pytest.importorskip("tensorflow")

    ragged = tf.ragged.constant([[1, 2], [3]])
    result = dict_to_tensor({"ragged": ragged, "value": 1.0})

    assert result["ragged"] is ragged
    assert np.asarray(result["value"]) == 1.0


# The torch tests run in a worker process: `triton`, which `torch.compile` pulls in on
# first use, segfaults when loaded into a process that already imported TensorFlow --
# as the test above does.
@run_in_backend("torch")
def test_serialize_torch_compiled_function():
    """A ``torch.compile``d function serializes like the function it wraps."""
    import torch

    compiled = torch.compile(_plain_function)
    serialized = serialize_elements([compiled])
    assert serialized == "compiled:" + serialize_elements([_plain_function])
    # Stable across separate compilations of the same target
    assert serialize_elements([torch.compile(_plain_function)]) == serialized


@run_in_backend("torch")
def test_serialize_compiled_bound_method_of_unpicklable_instance():
    """A ``torch.compile``d bound method still serializes, via its module and qualname.

    The instance cannot be pickled (it holds the compiled method), so the target
    itself is unpicklable too and the key falls back to the unbound function's
    module and qualname.
    """
    assert serialize_elements([_JitObj(2)._call]) == f"compiled:{__name__}._JitObj.call"

    # That constant fallback must not swallow the rest of the object's state.
    obj2, obj3 = _JitObj(2), _JitObj(3)
    assert serialize_elements([obj2.k, obj2._call]) != serialize_elements([obj3.k, obj3._call])


def test_serialize_unpicklable_object_raises_type_error():
    """Unpicklable, non-compiled elements raise TypeError (no silent fallback)."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_Unpicklable()])


def test_serialize_non_jit_wrapped_raises_type_error():
    """Regression: ``__wrapped__`` is not JIT-only; a generic wrapped callable
    must raise TypeError rather than collapse to the inner function's key."""
    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_UnpicklableWrapped()])


class _JaxJitLike(_Unpicklable):
    """Stand-in for a ``jax.jit`` wrapper: unpicklable and self-identifying."""

    _is_jax_jit_wrapper = True

    def __init__(self, target):
        # Instance attribute: a function stored on the class would bind as a method.
        self.__wrapped__ = target


class _CompiledTrampoline(_Unpicklable):
    """Unpicklable wrapper around a ``functools.wraps`` trampoline.

    Mirrors what a JIT compiler builds around a non-function callable: the
    outer object points at a local trampoline, which in turn wraps the target.
    ``functools.wraps`` leaves the trampoline's own ``<locals>`` qualname in
    place here, because a ``functools.partial`` has no qualname to copy over.
    """

    def __init__(self, target):
        def _trampoline(*args, **kwargs):  # qualname carries "<locals>"
            return target(*args, **kwargs)

        self._torchdynamo_orig_callable = functools.wraps(target)(_trampoline)


def test_classproperty_reads_from_the_class():
    """``classproperty`` resolves against the owning class, not an instance."""

    class Owner:
        NAME = "owner"

        @classproperty
        def name(cls):
            return cls.NAME

    class Child(Owner):
        NAME = "child"

    assert Owner.name == "owner"
    assert Child.name == "child"
    assert Owner().name == "owner"


def test_dict_to_tensor_skips_private_keys_and_unconvertible_values():
    """Dunder/private keys and str/bytes/callable values are left out entirely."""
    result = dict_to_tensor(
        {
            "_private": 1.0,
            "text": "not a tensor",
            "raw": b"bytes",
            "func": _plain_function,
            "value": 2.0,
        }
    )
    assert set(result) == {"value"}


def test_dict_to_tensor_dispatches_zea_objects():
    """Objects implementing zea's protocol convert through their own to_tensor."""
    from zea.config import Config

    result = dict_to_tensor({"cfg": Config(alpha=1.0), "value": 2.0}, keep_as_is=["value"])

    assert isinstance(result["cfg"], dict)
    assert keras.ops.convert_to_numpy(result["cfg"]["alpha"]) == 1.0
    # keep_as_is propagates into the nested object and applies at the top level
    assert result["value"] == 2.0


def test_dict_to_tensor_uses_bool_dtype_for_booleans():
    """Python bools and boolean arrays keep a boolean dtype, not int32."""
    result = dict_to_tensor({"flag": True, "mask": np.array([True, False])})

    assert keras.ops.dtype(result["flag"]) == "bool"
    assert keras.ops.dtype(result["mask"]) == "bool"


def test_json_encoder_converts_enums_and_numpy():
    """The encoder flattens zea enums and numpy scalars/arrays to JSON natives."""
    payload = {
        "dtype": DataTypes.IMAGE,
        "modtype": ModTypes.IQ,
        "count": np.int32(3),
        "ratio": np.float32(0.5),
        "arr": np.arange(3),
    }
    loaded = json.loads(json.dumps(payload, cls=ZEAEncoderJSON))

    assert loaded == {
        "dtype": "image",
        "modtype": "iq",
        "count": 3,
        "ratio": 0.5,
        "arr": [0, 1, 2],
    }


def test_json_encoder_rejects_unsupported_types():
    """Anything the encoder does not know about still raises, as JSON does."""
    with pytest.raises(TypeError):
        json.dumps({"obj": object()}, cls=ZEAEncoderJSON)


def test_json_decoder_restores_enums_and_arrays():
    """The decoder is the inverse of the encoder for enums, arrays and None modtype."""
    decoded = json.loads(
        json.dumps({"dtype": "image", "modtype": "iq", "arr": [1, 2, 3]}),
        cls=ZEADecoderJSON,
    )

    assert decoded["dtype"] is DataTypes.IMAGE
    assert decoded["modtype"] is ModTypes.IQ
    np.testing.assert_array_equal(decoded["arr"], np.array([1, 2, 3]))

    # `modtype: None` is a valid modulation type and must survive the round trip
    assert json.loads('{"modtype": null}', cls=ZEADecoderJSON)["modtype"] is None


def test_unwrap_returns_none_for_plain_callable():
    """A callable that is not a JIT wrapper has no target to unwrap."""
    assert _unwrap_compiled_callable(_plain_function) is None
    assert _unwrap_compiled_callable(None) is None


def test_unwrap_terminates_on_self_referential_wrapper():
    """A wrapper pointing at itself must not send the unwrap loop spinning."""

    class _SelfWrapper:
        def __call__(self):
            return None

    wrapper = _SelfWrapper()
    wrapper._torchdynamo_orig_callable = wrapper

    assert _unwrap_compiled_callable(wrapper) is wrapper


def test_serialize_jax_jit_wrapper():
    """Wrappers marked as jax.jit serialize through their ``__wrapped__`` target."""
    wrapper = _JaxJitLike(_plain_function)

    assert _unwrap_compiled_callable(wrapper) is _plain_function
    assert serialize_elements([wrapper]) == "compiled:" + serialize_elements([_plain_function])


def test_serialize_follows_functools_wraps_trampoline():
    """A JIT trampoline around a partial resolves to the partial it wraps."""
    target = functools.partial(_plain_function)
    wrapper = _CompiledTrampoline(target)

    assert _unwrap_compiled_callable(wrapper) == target
    assert serialize_elements([wrapper]) == "compiled:" + serialize_elements([target])


def test_serialize_compiled_local_target_raises_type_error():
    """An unpicklable compiled target without a stable qualname has no usable key."""

    class _CompiledLocal(_Unpicklable):
        def __init__(self, target):
            self._torchdynamo_orig_callable = target

    def _local_target(x):  # unpicklable: qualname contains "<locals>"
        return x

    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_CompiledLocal(_local_target)])

    with pytest.raises(TypeError, match="not a recognized compiled callable"):
        serialize_elements([_CompiledLocal(lambda x: x)])
