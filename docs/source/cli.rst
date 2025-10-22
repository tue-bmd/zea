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
Data copying
-------------------------------

.. argparse::
   :module: zea.data.__main__
   :func: get_parser
   :prog: python -m zea.data
   :nodefaultconst:
