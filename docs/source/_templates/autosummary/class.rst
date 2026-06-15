{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

.. minimal template: member rendering is controlled by ``autodoc_default_options``
   in conf.py (members, undoc-members, show-inheritance, special-members), matching
   how classes are rendered inline on the module pages. This avoids pulling in every
   inherited method (e.g. ``dict``/``h5py.File`` internals) via ``:inherited-members:``.
