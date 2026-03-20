Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` also provides a command line interface (CLI).

-------------------------------
File reading and visualization
-------------------------------

.. autoprogram:: zea.__main__:get_parser()
   :prog: zea

-------------------------------
Process dataset to videos
-------------------------------

.. autoprogram:: zea.data.process:get_parser()
   :prog: python -m zea.data.process

-------------------------------
Convert datasets
-------------------------------

.. autoprogram:: zea.data.convert.__main__:get_parser()
   :prog: python -m zea.data.convert

-------------------------------
Data copying
-------------------------------

.. autoprogram:: zea.data.copy:get_parser()
   :prog: python -m zea.data.copy

-------------------------------
Data file manipulation
-------------------------------

.. autoprogram:: zea.data.file_operations:get_parser()
   :prog: python -m zea.data.file_operations

