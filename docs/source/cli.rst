Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` provides a
command line interface (CLI) with two primary subcommands.

-------------------------------
zea — main entry point
-------------------------------

The ``zea`` command exposes two subcommands:

.. code-block:: text

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform
    zea app [--share] [--server_port PORT]                         # Gradio visualizer

.. autoprogram:: zea.__main__:get_parser()
   :prog: zea

--------------------------------
Process dataset (standalone CLI)
--------------------------------

The beamformer can also be invoked directly as a module (equivalent to ``zea process``).
Both ``--dataset`` / ``-d`` and ``--config`` / ``-c`` are required; ``--save-dir`` is optional
(defaults to ``output/``):

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

.. autoprogram:: zea.data.__main__:get_parser()
   :prog: python -m zea.data

.. _cli-file-operations:

-------------------------------
Data file manipulation
-------------------------------

.. autoprogram:: zea.data.file_operations:get_parser()
   :prog: python -m zea.data.file_operations
