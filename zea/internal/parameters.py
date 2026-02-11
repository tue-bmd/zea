"""
Parameter management system for ultrasound imaging.

This module provides the :class:`Parameters` base class, which implements
dependency-tracked, type-checked, and cacheable parameter logic for scientific
applications, primarily to support :class:`zea.Scan`.

See the Parameters class docstring for details on features and usage.
"""

import functools
import inspect
from copy import deepcopy

import numpy as np

from zea import log
from zea.internal.core import Object as ZeaObject
from zea.internal.core import _to_tensor, hash_elements, serialize_elements


def cache_with_dependencies(*deps):
    """Decorator to mark a method as a computed property with dependencies."""

    def decorator(func):
        func._dependencies = deps

        @functools.wraps(func)
        def wrapper(self: Parameters):
            self._assert_dependencies_met(func.__name__)

            if func.__name__ in self._cache:
                # Check if dependencies changed for mutable parameters
                current_hash = self._current_dependency_hash(func.__name__)
                if current_hash == self._dependency_versions.get(func.__name__):
                    return self._cache[func.__name__]
                else:
                    self._invalidate(func.__name__)

            result = func(self)
            self._cache[func.__name__] = result
            self._dependency_versions[func.__name__] = self._current_dependency_hash(func.__name__)
            return result

        return property(wrapper)

    return decorator


class MissingDependencyError(ValueError):
    """Exception indicating that a dependency of an attribute was not met."""

    def __init__(self, attribute: str, missing_dependencies: set):
        super().__init__(
            f"Cannot access '{attribute}' due to missing dependencies: "
            + f"{sorted(missing_dependencies)}"
        )


class NoDependencyError(ValueError):
    """Exception indicating that an attribute has no dependencies defined."""

    def __init__(self, name: str):
        super().__init__(f"'{name}' is not a computed property with dependencies.")


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
      property raises an informative `ValueError` listing the leaf parameters
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
      an informative `MissingDependencyError` is raised, listing the missing parameters.

    **Usage Example:**

    .. doctest::

        >>> class MyParams(Parameters):
        ...     VALID_PARAMS = {
        ...         "a": {"type": int, "default": 1},
        ...         "b": {"type": float, "default": 2.0},
        ...         "d": {"type": float},  # optional dependency
        ...     }

        ...     @cache_with_dependencies("a", "b")
        ...     def c(self):
        ...        return self.a + self.b

        ...     @cache_with_dependencies("a", "b")
        ...     def d(self):
        ...         if self._params.get("d") is not None:
        ...             return self._params["d"]
        ...         return self.a * self.b

        >>> p = MyParams(a=3)
        >>> print(p.c)  # Computes and caches c
        5.0
        >>> print(p.c)  # Returns cached value
        5.0

        # Changing a parameter invalidates the cache
        >>> p.a = 4
        >>> print(p.c)  # Recomputes c, now 4 + 2.0 = 6.0
        6.0

        >>> # You are not allowed to set computed properties
        >>> # p.c = 5  # Raises ValueError

        >>> # Now check out the optional dependency, this can be either
        >>> # set directly during initialization or computed from dependencies (default)
        >>> print(p.d)  # Returns 8 (=4 * 2.0)
        8.0
        >>> p = MyParams(a=3, d=9.99)
        >>> print(p.d)
        9.99

    """

    VALID_PARAMS = None

    def __init__(self, **kwargs):
        super().__init__()

        # Check if VALID_PARAMS is defined
        if self.VALID_PARAMS is None:
            raise NotImplementedError("VALID_PARAMS must be defined in subclasses of Parameters.")

        # Check if the definition of the class has circular dependencies
        for name in self.__class__.__dict__:
            self._check_for_circular_dependencies(name)

        # Check if all dependencies are valid parameters or computed properties
        # Will automatically catch typo's etc.
        self._check_validity_of_dependencies()

        # Internal state
        self._params = {}
        self._properties = self.get_properties()
        self._cache = {}
        self._dependency_versions = {}

        # Tensor cache stores converted tensors for parameters and computed properties
        # to avoid converting them multiple times if there are no changes.
        self._tensor_cache = {}

        # Initialize parameters with defaults
        for param, config in self.VALID_PARAMS.items():
            if param not in kwargs and "default" in config:
                # need to deepcopy in case default is mutable
                kwargs[param] = deepcopy(config["default"])

        # Set provided parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def _validate_parameter(cls, key, value):
        # Check if the parameter is valid
        if key not in cls.VALID_PARAMS:
            raise ValueError(
                f"Invalid parameter: {key}. Valid parameters are: {list(cls.VALID_PARAMS.keys())}"
            )

        # Cast the value if needed and possible
        expected_type = cls.VALID_PARAMS[key]["type"]
        if expected_type is not None and value is not None and not isinstance(value, expected_type):
            value = cls._cast(key, value)

        # Check again
        if expected_type is not None and value is not None and not isinstance(value, expected_type):
            allowed = cls._human_readable_type(expected_type)
            raise TypeError(
                f"Parameter '{key}' expected type {allowed}, got {type(value).__name__}"
            )

        return value

    @classmethod
    def _cast(cls, key, value):
        """Cast parameter to the expected type if 'cast_from' is specified.

        Additionally, int to float conversion is allowed implicitly."""
        # If the value is a single-element array, convert it to a scalar
        # If it's a numpy scalar, convert it to a native Python type
        if (isinstance(value, np.ndarray) and value.size == 1) or isinstance(value, np.generic):
            value = value.item()

        # Assume the key exists in VALID_PARAMS
        config = cls.VALID_PARAMS[key]

        if value is None:
            return value

        cast_to = config["type"]
        if isinstance(cast_to, tuple):
            raise ValueError(f"Casting to multiple types is not supported for parameter '{key}'.")

        if "cast_from" not in config:
            if isinstance(value, int) and cast_to is float:
                # Allow implicit conversion from int to float
                return float(value)
            return value

        cast_types = config["cast_from"]
        if not isinstance(cast_types, tuple):
            cast_types = (cast_types,)

        if any(isinstance(value, t) for t in cast_types):
            value = cast_to(value)

        return value

    @staticmethod
    def _human_readable_type(type):
        """Convert a type or tuple of types to a human-readable string."""
        return (
            type.__name__ if not isinstance(type, tuple) else ", ".join([t.__name__ for t in type])
        )

    def copy(self):
        """Return a deep copy of the Parameters object."""
        return self.__class__(**deepcopy(self._params))

    @property
    def serialized(self):
        """Compute the checksum of the object only if not already done"""
        if self._serialized is None:
            self._serialized = serialize_elements([self._params])
        return self._serialized

    @classmethod
    def _is_property_with_dependencies(cls, name):
        """Check if a class attribute is a property with dependencies."""
        attr = getattr(cls, name, None)
        return isinstance(attr, property) and hasattr(attr.fget, "_dependencies")

    @classmethod
    def _get_dependencies(cls, name):
        """Get the dependencies of a computed property."""
        if not cls._is_property_with_dependencies(name):
            raise NoDependencyError(name)
        return getattr(cls, name).fget._dependencies

    def _find_leaf_params(self, name, seen=None):
        """Recursively find all leaf parameters that a property depends on.

        If it is an optional dependency parameter, it will be included as a leaf. Not the ones it
        depends on.
        """
        if seen is None:
            seen = set()
        if name in seen:
            return set()
        seen.add(name)

        # If the name is already a leaf parameter, return it
        if name in self._params or name in self.VALID_PARAMS:
            return {name}

        # If the name is a property with dependencies, find its leaf parameters
        if self._is_property_with_dependencies(name):
            leaves = set()
            for dep in self._get_dependencies(name):
                leaves |= self._find_leaf_params(dep, seen)  # union
            return leaves

        raise AttributeError(f"'{name}' is not a valid parameter or computed property.")

    def _has_param(self, name):
        """Check if a parameter is set (i.e., exists in _params)."""
        # Check for existence of _params to avoid issues during unpickling
        return "_params" in self.__dict__ and name in self._params

    def __getattr__(self, item):
        """Handle attribute access for parameters only.

        Properties with dependencies are handled by cache_with_dependencies decorator.
        Regular properties are handled by normal Python descriptor protocol.
        """
        # Return parameter value if it exists
        if self._has_param(item):
            return self._params[item]

        # Attribute not found
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        # Give clear error message on assignment to methods
        class_attr = getattr(self.__class__, key, None)
        if callable(class_attr):
            raise AttributeError(
                f"Cannot assign to method '{key}'. "
                f"'{key}' is a method, not an attribute. "
                f"To use it, call it as a function, e.g.: '{self.__class__.__name__}.{key}(...)'"
            )

        # Allow setting private attributes
        if key.startswith("_"):
            return super().__setattr__(key, value)

        # Give clear error message on assignment to computed properties
        if self._is_property_with_dependencies(key) and key not in self.VALID_PARAMS:
            leaf_params = sorted(self._find_leaf_params(key))
            raise ValueError(
                f"Cannot set computed property '{key}'. Only leaf parameters can be set. "
                f"To change '{key}', set one or more of its leaf parameters: {leaf_params}"
            )

        # Validate new value
        value = self._validate_parameter(key, value)

        # Set the parameter
        self._params[key] = value

        # Invalidate cache for this parameter if it is also a computed property
        self._invalidate(key)

    def __delattr__(self, name):
        # Allow deletion of parameters, but not properties
        if name in self._params:
            del self._params[name]
            self._invalidate(name)
        elif name in self.VALID_PARAMS:
            raise ValueError(f"Cannot delete parameter '{name}' because it is not set.")
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @classmethod
    def _check_for_circular_dependencies(cls, name, seen=None):
        """Check for circular dependencies in the dependency tree with a depth-first search."""
        if seen is None:
            seen = set()
        if name in seen:
            raise RuntimeError(f"Circular dependency detected for '{name}'")
        seen = seen.copy()
        seen.add(name)

        if cls._is_property_with_dependencies(name):
            for dep in cls._get_dependencies(name):
                cls._check_for_circular_dependencies(dep, seen)

    @classmethod
    def _check_validity_of_dependencies(cls):
        for dep in cls.get_properties_with_dependencies():
            for d in cls._get_dependencies(dep):
                if not (d in cls.VALID_PARAMS or cls._is_property_with_dependencies(d)):
                    raise ValueError(
                        f"Dependency '{d}' of computed property '{dep}' is not a valid "
                        f"parameter or computed property. This has to be fixed in the "
                        f"{cls.__name__} class definition."
                    )

    def _find_all_dependents(self, target, seen=None):
        """
        Find all computed properties that depend (directly or indirectly) on the target parameter
        with a global search. Returns a set of property names that depend on the target.
        """
        dependents = set()
        if seen is None:
            seen = set()
        if target in seen:
            return dependents
        seen.add(target)
        for name in self.__class__.__dict__:
            if self._is_property_with_dependencies(name):
                if target in self._get_dependencies(name):
                    dependents.add(name)
                    # Recursively add dependents of this property
                    dependents |= self._find_all_dependents(name, seen)
        return dependents

    def _invalidate(self, key):
        """Invalidate a specific cached computed property and its dependencies."""
        self._cache.pop(key, None)
        self._dependency_versions.pop(key, None)
        self._tensor_cache.pop(key, None)
        self._serialized = None  # see core object
        self._invalidate_dependents(key)

    def _invalidate_dependents(self, changed_key):
        """
        Invalidate all cached computed properties that (directly or indirectly)
        depend on the changed_key.
        """
        for key in self._find_all_dependents(changed_key):
            self._invalidate(key)

    def _current_dependency_hash(self, key) -> str:
        """Compute a hash representing the current state of the dependencies of key.

        Mainly needed to track changes in mutable parameters.
        """
        if not self._is_property_with_dependencies(key):
            raise NoDependencyError(key)
        deps = self._find_leaf_params(key)
        values = [self._params.get(dep) for dep in sorted(deps)]
        return hash_elements(values)

    def _assert_dependencies_met(self, name):
        """Assert that all dependencies for a computed property are met."""
        missing_set = self._find_missing_dependencies(name)
        if missing_set:
            raise MissingDependencyError(name, missing_set)

    def _find_missing_dependencies(self, name) -> set:
        missing_set = set()

        # Return immediately if already in params or cache
        if name in self._params or name in self._cache:
            return missing_set

        if self._is_property_with_dependencies(name):
            for dep in self._get_dependencies(name):
                missing_set |= self._find_missing_dependencies(dep)  # union
        else:
            missing_set.add(name)

        return missing_set

    @classmethod
    def get_properties(cls):
        """Get all properties of the class, including those with dependencies."""
        return [name for name, value in inspect.getmembers(cls) if isinstance(value, property)]

    @classmethod
    def get_properties_with_dependencies(cls):
        """Get all properties of the class that have dependencies."""
        return [name for name in cls.get_properties() if cls._is_property_with_dependencies(name)]

    def to_tensor(self, include=None, exclude=None, keep_as_is: list = None):
        """
        Convert parameters and computed properties to tensors.

        Only one of `include` or `exclude` can be set.

        Args:
            include ("all", or list): Only include these parameter/property names.
                If "all", include all available parameters (i.e. their dependencies are met).
                If specified, will take the intersection with possible parameters, so non-existing
                keys will be ignored. Default is "all".
            exclude (None or list): Exclude these parameter/property names.
                If provided, these keys will be excluded from the output.
            keep_as_is (list): List of parameter/property names that should not be converted to
                tensors, but included as-is in the output.
        """
        if include is None and exclude is None:
            include = "all"

        if include is not None and exclude is not None:
            raise ValueError("Only one of 'include' or 'exclude' can be set.")

        # Determine which keys to include
        param_keys = set(self._params.keys())
        property_keys = set(self._properties)
        all_keys = param_keys | property_keys

        if include == "all":
            keys = all_keys
        elif include is not None:
            # Filter include list to only existing keys
            keys = set(include).intersection(all_keys)
        elif exclude is not None:
            # Take all keys except those in exclude
            keys = all_keys - set(exclude)

        tensor_dict = {}
        # Convert parameters and computed properties to tensors
        for key in keys:
            # Get the value from params or computed properties
            # This is essential to trigger dependency checks
            try:
                val = getattr(self, key)
            except MissingDependencyError as exc:
                if include == "all" or exclude is not None:
                    # If we are including all, we can skip this key
                    continue
                else:
                    raise exc

            if key in self._tensor_cache:
                tensor_dict[key] = self._tensor_cache[key]
            else:
                tensor_val = _to_tensor(key, val, keep_as_is=keep_as_is)
                tensor_dict[key] = tensor_val
                self._tensor_cache[key] = tensor_val

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
    def standardize_params(cls, **kwargs) -> dict:
        """Return a dict with only valid parameters set and cast to the right type."""
        params = {}
        for parameter, value in kwargs.items():
            if parameter in cls.VALID_PARAMS:
                params[parameter] = value
            else:
                log.debug(f"Skipping invalid parameter '{parameter}'.")
        return params

    @classmethod
    def safe_initialize(cls, **kwargs):
        """Reduce kwargs to only valid parameters and convert types as needed."""
        params = cls.standardize_params(**kwargs)

        if len(params) == 0:
            log.info(f"Could not find proper scan parameters in {kwargs}.")
        return cls(**params)
