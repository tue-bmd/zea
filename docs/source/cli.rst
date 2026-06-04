Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` also provides a command line interface (CLI).

.. note::
   The ``zea`` CLI is currently a placeholder. Extended visualization and data
   inspection commands will be added in a future release. In the meantime,
   use ``python -m zea.data.convert`` and ``python -m zea.data`` below.

-------------------------------
Convert datasets
-------------------------------

.. autoprogram:: zea.data.convert.__main__:get_parser()
   :prog: python -m zea.data.convert

-------------------------------
Data copying
-------------------------------

.. autoprogram:: zea.data.__main__:get_parser()
   :prog: python -m zea.data

.. _cli-file-operations:

-------------------------------
Data file manipulation
-------------------------------

.. autoprogram:: zea.data.file_operations:get_parser()
   :prog: python -m zea.data.file_operations

