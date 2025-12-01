Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` also provides a command line interface (CLI).

-------------------------------
File reading and visualization
-------------------------------

.. argparse::
   :module: zea.__main__
   :func: get_parser
   :prog: zea

-------------------------------
Convert datasets
-------------------------------

.. argparse::
   :module: zea.data.convert.__main__
   :func: get_parser
   :prog: python -m zea.data.convert
   :nodefaultconst:

-------------------------------
Data copying
-------------------------------

.. argparse::
   :module: zea.data.__main__
   :func: get_parser
   :prog: python -m zea.data
   :nodefaultconst:

-------------------------------
Data file manipulation
-------------------------------

.. argparse::
   :module: zea.data.file_operations
   :func: get_parser
   :prog: python -m zea.data.file_operations

