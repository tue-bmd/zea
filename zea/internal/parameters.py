"""
Parameter management system for ultrasound imaging.

This module provides the :class:`Parameters` base class, which implements
dependency-tracked, type-checked, and cacheable parameter logic for scientific
applications, primarily to support :class:`zea.Scan`.

See the Parameters class docstring for details on features and usage.
"""

import functools
import hashlib

import numpy as np

from zea import log
from zea.internal.core import Object as ZeaObject
from zea.internal.core import _to_tensor


def cache_with_dependencies(*deps):
    """Decorator to mark a method as a computed property with dependencies."""

    def decorator(func):
        func._dependencies = deps

        @functools.wraps(func)
        def wrapper(self):
            failed = set()
            if not self._resolve_dependency_tree(func.__name__, failed):
                raise AttributeError(
                    f"Cannot access '{func.__name__}' due to missing base dependencies: "
                    f"{sorted(failed)}"
                )

            if func.__name__ in self._cache:
                # Check if dependencies changed
                current_hash = self._current_dependency_hash(deps)
                if current_hash == self._dependency_versions.get(func.__name__):
                    return self._cache[func.__name__]

            result = func(self)
            self._computed.add(func.__name__)
            self._cache[func.__name__] = result
            self._dependency_versions[func.__name__] = self._current_dependency_hash(deps)
            return result

        return property(wrapper)

    return decorator


class Parameters(ZeaObject):
    """Base class for parameters with dependencies.

    This class provides a robust parameter management system,
    supporting dependency tracking, lazy evaluation, and type validation.

    **Features:**

    - **Type Validation:** All parameters must be validated against their
      expected types as specified in the `VALID_PARAMS` dictionary.
      Setting a parameter to an invalid type raises a `TypeError`.

    - **Dependency Tracking:** Computed properties can declare dependencies on
      other parameters or properties using the `@cache_with_dependencies`
      decorator. The system automatically tracks and resolves these dependencies.

    - **Lazy Computation:** Computed properties are evaluated only when accessed,
      and their results are cached for efficiency.

    - **Cache Invalidation:** When a parameter changes, all dependent computed
      properties are invalidated and recomputed on next access.

    - **Leaf Parameter Enforcement:** Only leaf parameters
      (those directly listed in `VALID_PARAMS`) can be set. Attempting to set a computed
      property raises an informative `AttributeError` listing the leaf parameters
      that must be changed instead.

    - **Optional Dependency Parameters:** Parameters can be both set directly (as a leaf)
      or computed from dependencies if not set. If a parameter is present in `VALID_PARAMS`
      and also decorated with `@cache_with_dependencies`, it will use the explicitly set
      value if provided, or fall back to the computed value if not set or set to `None`.
      If you set such a parameter after it has been computed, the explicitly set value
      will override the computed value and remain in effect until you set it back to `None`,
      at which point it will again be computed from its dependencies. This pattern is useful
      for parameters that are usually derived from other values, but can also be overridden
      directly when needed, and thus don't have a forced relationship with the dependencies.

    - **Tensor Conversion:** The `to_tensor` method converts all parameters and optionally all
      computed properties to tensors for machine learning workflows.

    - **Error Reporting:** If a computed property cannot be resolved due to missing dependencies,
      an informative `AttributeError` is raised, listing the missing parameters.

    **Usage Example:**

    .. code-block:: python

        class MyParams(Parameters):
            VALID_PARAMS = {
                "a": {"type": int, "default": 1},
                "b": {"type": float, "default": 2.0},
                "d": {"type": float, "default": None},  # optional dependency
            }

            @cache_with_dependencies("a", "b")
            def c(self):
                return self.a + self.b

            @cache_with_dependencies("a", "b")
            def d(self):
                if self._params.get("d") is not None:
                    return self._params["d"]
                return self.a * self.b


        p = MyParams(a=3)
        print(p.c)  # Computes and caches c
        print(p.c)  # Returns cached value

        # Changing a parameter invalidates the cache
        p.a = 4
        print(p.c)  # Recomputes c

        # You are not allowed to set computed properties
        # p.c = 5  # Raises AttributeError

        # Now check out the optional dependency, this can be either
        # set directly during initialization or computed from dependencies (default)
        print(p.d)  # Returns 6 (=3 * 2.0)
        p = MyParams(a=3, d=9.99)
        print(p.d)  # Returns 9.99

    """

    VALID_PARAMS = None

    def __init__(self, **kwargs):
        super().__init__()

        if self.VALID_PARAMS is None:
            raise NotImplementedError("VALID_PARAMS must be defined in subclasses of Parameters.")

        for param, config in self.VALID_PARAMS.items():
            if param not in kwargs and config["default"] is not None:
                kwargs[param] = config["default"]

        # Validate parameter types
        for param, value in kwargs.items():
            if param not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {param}. "
                    f"Valid parameters are: {list(self.VALID_PARAMS.keys())}"
                )
            expected_type = self.VALID_PARAMS[param]["type"]
            if expected_type is not None and value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        allowed = ", ".join([t.__name__ for t in expected_type])
                        raise TypeError(
                            f"Parameter '{param}' expected type {allowed}, "
                            f"got {type(value).__name__}"
                        )
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{param}' expected type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

        self._params = {}
        self._computed = set()
        self._cache = {}
        self._dependency_versions = {}
        for k, v in kwargs.items():
            self._params[k] = v

    def __getattr__(self, item):
        # First check regular params
        if item in self._params:
            return self._params[item]

        # Then check if it's a known property on the class with dependencies
        cls_attr = getattr(type(self), item, None)
        if isinstance(cls_attr, property) and hasattr(cls_attr.fget, "_dependencies"):
            # Try to resolve dependencies
            failed = set()
            if self._resolve_dependency_tree(item, failed):
                # Use descriptor protocol directly
                try:
                    return cls_attr.__get__(self, self.__class__)
                except Exception as e:
                    raise AttributeError(f"Error computing '{item}': {str(e)}")
            else:
                raise AttributeError(
                    f"Cannot access '{item}' due to missing base dependencies: {sorted(failed)}"
                )

        # Otherwise raise normal attribute error
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{item}'")

    def __setattr__(self, key, value):
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            cls_attr = getattr(self.__class__, key, None)
            # Allow setting if it's a valid parameter, even if it's also a computed property
            if (
                isinstance(cls_attr, property)
                and hasattr(cls_attr.fget, "_dependencies")
                and key not in self.VALID_PARAMS
            ):
                # Only block if not a leaf parameter
                def find_leaf_params(name, seen=None):
                    if seen is None:
                        seen = set()
                    attr = getattr(self.__class__, name, None)
                    if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                        leaves = set()
                        for dep in attr.fget._dependencies:
                            leaves |= find_leaf_params(dep, seen)
                        return leaves
                    else:
                        if name in self.VALID_PARAMS:
                            return {name}
                        return set()

                leaf_params = sorted(find_leaf_params(key))
                raise AttributeError(
                    f"Cannot set computed property '{key}'. Only leaf parameters can be set. "
                    f"To change '{key}', set one or more of its leaf parameters: {leaf_params}"
                )

            # Validate that parameter is in VALID_PARAMS
            if key not in self.VALID_PARAMS:
                raise ValueError(
                    f"Invalid parameter: {key}. "
                    f"Valid parameters are: {list(self.VALID_PARAMS.keys())}"
                )

            # Validate parameter type
            expected_type = self.VALID_PARAMS[key]["type"]
            if expected_type is not None and value is not None:
                if isinstance(expected_type, tuple):
                    if not isinstance(value, expected_type):
                        allowed = ", ".join([t.__name__ for t in expected_type])
                        raise TypeError(
                            f"Parameter '{key}' expected type {allowed}, got {type(value).__name__}"
                        )
                else:
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"Parameter '{key}' expected type {expected_type.__name__}, "
                            f"got {type(value).__name__}"
                        )

            # Set the parameter and invalidate dependencies
            self._params[key] = value

            # Invalidate cache for this parameter if it is also a computed property
            self._cache.pop(key, None)
            self._computed.discard(key)
            self._dependency_versions.pop(key, None)

            self._invalidate_dependents(key)

    def _invalidate_dependents(self, changed_key):
        """
        Invalidate all cached computed properties that (directly or indirectly)
        depend on the changed_key.
        """

        # Find all computed properties that depend (directly or indirectly) on changed_key
        def find_all_dependents(target):
            dependents = set()
            for name in self.__class__.__dict__:
                attr = getattr(self.__class__, name, None)
                if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                    deps = attr.fget._dependencies
                    if target in deps:
                        dependents.add(name)
                        # Recursively add dependents of this property
                        dependents |= find_all_dependents(name)
            return dependents

        to_invalidate = find_all_dependents(changed_key)
        for key in to_invalidate:
            self._cache.pop(key, None)
            self._computed.discard(key)
            self._dependency_versions.pop(key, None)

    def _current_dependency_hash(self, deps):
        relevant = [str(self._params.get(dep, None)) for dep in deps]
        return hashlib.sha1("".join(relevant).encode()).hexdigest()

    def _resolve_dependency_tree(self, name, failed=None):
        if failed is None:
            failed = set()

        # Return immediately if already in params or cache
        if name in self._params:
            return True
        if name in self._cache:
            return True

        cls_attr = getattr(self.__class__, name, None)
        if isinstance(cls_attr, property):
            func = cls_attr.fget
            if hasattr(func, "_dependencies"):
                all_ok = True
                for dep in func._dependencies:
                    if not self._resolve_dependency_tree(dep, failed):
                        all_ok = False
                if all_ok:
                    # Don't actually access the property here
                    # Just mark that all dependencies are met
                    return True
                else:
                    return False
        else:
            failed.add(name)
            return False

    def to_tensor(self, compute_missing=False, compute_keys=None):
        """
        Convert all parameters (and optionally computed properties) to tensors.

        Args:
            compute_missing (bool): If True, compute missing computed properties.
            compute_keys (list or None): If not None, only compute these
                computed properties (by name).
        """
        tensor_dict = {k: _to_tensor(k, v) for k, v in self._params.items()}

        # Compute missing properties if requested
        if compute_missing:
            for name in dir(self.__class__):
                if compute_keys is not None and name not in compute_keys:
                    continue
                attr = getattr(self.__class__, name)
                if isinstance(attr, property) and hasattr(attr.fget, "_dependencies"):
                    failed = set()
                    if self._resolve_dependency_tree(name, failed):
                        try:
                            val = getattr(self, name)
                            if val is not None:
                                pass
                        except Exception as e:
                            log.warning(f"Could not compute '{name}': {str(e)}")

        # Always include all already computed properties
        for key in self._computed:
            val = getattr(self, key)
            tensor_dict[key] = _to_tensor(key, val)

        return tensor_dict

    def __repr__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"{k}=array(shape={v.shape})")
            else:
                param_lines.append(f"{k}={repr(v)}")

        param_str = ", ".join(param_lines)
        return f"{self.__class__.__name__}({param_str})"

    def __str__(self):
        param_lines = []
        for k, v in self._params.items():
            if v is None:
                continue

            # Handle arrays by showing their shape instead of content
            if isinstance(v, np.ndarray):
                param_lines.append(f"    {k}=array(shape={v.shape})")
            else:
                param_lines.append(f"    {k}={v}")

        param_str = ",\n".join(param_lines)
        return f"{self.__class__.__name__}(\n{param_str}\n)"

    @classmethod
    def safe_initialize(cls, **kwargs):
        """Overwrite safe initialize from zea.core.Object.

        We do not want safe initialization here.
        """
        return cls(**kwargs)
