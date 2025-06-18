"""Wrapper module to load tf2jax if available."""

try:
    from tf2jax import *  # noqa: F401, F403
except ImportError as exc:
    raise ImportError(
        "tf2jax is not installed. "
        "Suggested installation: `pip install tf2jax==0.3.6 && pip install keras -U`"
        "Note that this may install a newer version of Keras than the one you have installed! "
    ) from exc
