"""Standalone tools built on top of ``zea``.

Unlike the rest of the package, the modules here are self-contained utilities rather
than building blocks for a pipeline. Some of them double as a command line tool, in
which case they are reachable through the ``zea`` CLI (see :doc:`../cli`).

Command line tools
------------------

- :mod:`zea.tools.selection_tool` (``zea tools select``) — interactively select regions
  of interest in images or a video, compare them with a metric, and interpolate masks
  across the frames of a sequence.

Python-only tools
-----------------

These are used as libraries; they are not exposed as ``zea`` subcommands.

- :mod:`zea.tools.hf` — helpers for the Hugging Face Hub: downloading models and
  uploading folders / datasets.
- :mod:`zea.tools.fit_scan_cone` — detect the scan cone of an ultrasound image and crop
  it so the apex sits at the top center, ready for scan conversion. Also ships a small
  visualization demo, runnable with
  ``python -m zea.tools.fit_scan_cone --input_file <clip.avi>``.

"""
