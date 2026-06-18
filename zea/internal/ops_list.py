"""OperationList: a list subclass with name-based indexing for pipeline operations."""

import difflib


class OperationList(list):
    """A list of operations that supports both integer and name-based string indexing.

    Works for both:

    - **Pipeline operations** (:class:`~zea.ops.Operation` /
      :class:`~zea.ops.Pipeline` instances): names are resolved via the
      :data:`~zea.internal.registry.ops_registry`.
    - **Config operations** (``str`` / ``dict`` elements produced by
      :func:`~zea.config._compact_operation`): the string itself or the
      ``"name"`` key of the dict is used directly.

    Duplicate operations are disambiguated with a ``_N`` numeric suffix.
    If a pipeline contains two ``Normalize`` ops, use ``"normalize_0"``
    and ``"normalize_1"``.  Using the bare name ``"normalize"`` when
    duplicates exist raises a :exc:`KeyError` with a helpful hint.

    Call :meth:`keys` to see all available string keys.

    Examples:
        .. code-block:: python

            pipeline = zea.Pipeline.from_default()
            pipeline.operations["beamform"]
            pipeline.operations["beamform"].operations["tof_correction"]
            # or using the shorthand on Pipeline directly:
            pipeline["beamform"]["tof_correction"]

            pipeline.operations.keys()
            # ['cast', 'apply_window', 'demodulate', 'beamform', ...]

            config = zea.Config.from_path("config.yaml")
            config.pipeline.operations["beamform"].params.enable_pfield = True
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._get_by_name(key)
        return super().__getitem__(key)

    @staticmethod
    def _get_op_name(op):
        """Return the user-facing lookup name for one element of the list."""
        if isinstance(op, str):
            return op
        if isinstance(op, dict):
            return op.get("name")
        # Operation/Pipeline instances — lazy import avoids circular deps with config.py
        from zea.internal.registry import ops_registry

        try:
            reg_name = ops_registry.get_name(type(op))
        except KeyError:
            return getattr(op, "name", None)

        # Dotted registry names (e.g. "keras.ops.cast") → expose only the last component
        # so users can write operations["cast"] instead of operations["keras.ops.cast"].
        return reg_name.rsplit(".", 1)[-1] if "." in reg_name else reg_name

    def _get_by_name(self, name):
        # Exact match first: ops whose registered name ends in _N are reachable
        # without being confused for a disambiguated duplicate.
        exact = [op for op in self if self._get_op_name(op) == name]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            numbered = [f"'{name}_{i}'" for i in range(len(exact))]
            raise KeyError(
                f"Ambiguous: {len(exact)} operations named '{name}'. "
                f"Use a numbered form to be specific: {', '.join(numbered)}."
            )

        # No exact match — resolve optional numeric suffix:
        # "normalize_1" -> base="normalize", index=1
        base, sep, suffix = name.rpartition("_")
        if sep and suffix.isdigit():
            base_name, index = base, int(suffix)
        else:
            base_name, index = name, None

        matches = [op for op in self if self._get_op_name(op) == base_name]

        if not matches:
            available = self.keys()
            msg = f"No operation named '{name}' found."
            closest = difflib.get_close_matches(name, available, n=1, cutoff=0.6)
            if closest:
                msg += f" Did you mean '{closest[0]}'?"
            msg += f" Available operations: {available}"
            raise KeyError(msg)

        if index is not None:
            if index >= len(matches):
                raise KeyError(
                    f"Index {index} out of range for '{base_name}' "
                    f"(found {len(matches)} match{'es' if len(matches) != 1 else ''})."
                )
            return matches[index]

        if len(matches) > 1:
            numbered = [f"'{base_name}_{i}'" for i in range(len(matches))]
            raise KeyError(
                f"Ambiguous: {len(matches)} operations named '{base_name}'. "
                f"Use a numbered form to be specific: {', '.join(numbered)}."
            )

        return matches[0]

    def keys(self):
        """Return the list of string keys that can be used for name-based indexing.

        Duplicate names are disambiguated with a ``_N`` suffix so that every
        returned key is unambiguously usable::

            pipeline.keys()  # ['cast', 'normalize_0', 'normalize_1', ...]
        """
        raw = [self._get_op_name(op) for op in self]
        counts = {}
        for name in raw:
            if name is not None:
                counts[name] = counts.get(name, 0) + 1

        seen: dict = {}
        result = []
        for name in raw:
            if name is None:
                continue
            if counts[name] > 1:
                idx = seen.get(name, 0)
                result.append(f"{name}_{idx}")
                seen[name] = idx + 1
            else:
                result.append(name)
        return result
