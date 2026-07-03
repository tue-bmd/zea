Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` provides a
command line interface (CLI) built with `tyro <https://brentyi.github.io/tyro/>`_. The
reference below is generated directly from the CLI definitions in
:mod:`zea.cli_args`, so it always matches the installed version.

.. code-block:: text

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer
    zea data <operation> [options]                                 # manipulate zea data files

.. tyroprogram:: zea.__main__:SubCmd
   :prog: zea
