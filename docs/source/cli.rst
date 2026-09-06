Command line interface
================================

Besides the main :doc:`zea API documentation <_autosummary/zea>`, ``zea`` provides a
command line interface (CLI) with six primary subcommands (app, convert, data, datapaths,
process, tools).

Note that is very new functionality, and might change in future releases. Please report any issues you encounter.

.. code-block:: text

    zea app [--share] [--server-port PORT]                         # launch the Gradio visualizer
    zea convert <dataset> <src> <dst> [options]                    # convert raw datasets to zea
    zea data <operation> [options]                                 # manipulate zea data files
    zea datapaths [--user-config users.yaml]                       # set up your local data paths
    zea process --dataset <path> --config <config.yaml> [options]  # batch beamform a dataset
    zea tools select [files] [options]                             # annotate regions of interest

.. tyroprogram:: zea.__main__:CLI
   :prog: zea