Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` provides a
command line interface (CLI) with four primary subcommands (process, app, data, convert).

Note that is very new functionality, and might change in future releases. Please report any issues you encounter.

.. code-block:: text

    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer
    zea data <operation> [options]                                 # manipulate zea data files
    zea convert <dataset> <src> <dst> [options]                    # convert raw datasets to zea

The ``convert`` subcommand (documented below) is also available as a standalone
module with an identical interface: ``python -m zea.data.convert <dataset> <src> <dst>``.

.. tyroprogram:: zea.__main__:CLI
   :prog: zea