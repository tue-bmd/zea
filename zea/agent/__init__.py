"""Agent subpackage for closing action-perception loop in ultrasound imaging.

The ``zea.agent`` subpackage provides tools and utilities for implementing **cognitive ultrasound**
imaging via active perception. This module enables intelligent, adaptive transmit design, where
acquisition decisions are informed by the current belief state about the imaged tissue.

Overview
========

Active perception in ultrasound involves iteratively:

1. **Perceiving** the current state of tissue from acquired measurements
2. **Selecting transmit actions** based on the current beliefs about the tissue state
3. **Acquiring** new data and looping back to the perception step

The :mod:`zea.agent` module provides the building blocks for implementing such perception-action
loops.

.. note::
    The functions currently available implement selection strategies for *focused transmit actions*. Development of action selection functions for more general transmit schemes is currently a work-in-progress.

Action Selection Strategies
===========================

Action selection strategies determine which transmits to fire next, given some belief
about the tissue state.

See the following dropdown for a list of available action selection strategies:

.. dropdown:: **Available strategies**

    - :class:`zea.agent.selection.GreedyEntropy`: Selects lines that maximize entropy reduction.
    - :class:`zea.agent.selection.UniformRandomLines`: Randomly samples scan lines with uniform probability.
    - :class:`zea.agent.selection.EquispacedLines`: Selects equispaced lines that sweep across the image.
    - :class:`zea.agent.selection.CovarianceSamplingLines`: Models line-to-line correlation to select masks with highest entropy.
    - :class:`zea.agent.selection.TaskBasedLines`: Selects lines to maximize information gain with respect to a downstream task.

Basic Usage
===========

.. doctest::

    >>> import zea
    >>> import numpy as np

    >>> agent = zea.agent.selection.GreedyEntropy(
    ...     n_actions=7,
    ...     n_possible_actions=112,
    ...     img_width=112,
    ...     img_height=112,
    ... )

    >>> # (batch, samples, height, width)
    >>> particles = np.random.rand(1, 10, 112, 112)
    >>> lines, mask = agent.sample(particles)

Masks
=====

The :mod:`zea.agent.masks` module provides utilities for converting action representations
(e.g., selected line indices) to image-sized masks that can be applied to observations.

Example Notebooks
=================

We provide example notebooks demonstrating perception-action loops in practice, as companions to recently published papers on the topic:

**Patient-Adaptive Echocardiography**

This tutorial implements a basic perception-action loop using diffusion models for
perception-as-inference and greedy entropy minimization for action selection.

- :doc:`../notebooks/agent/agent_example`

  - Uses :class:`~zea.agent.selection.GreedyEntropy` to select informative scan lines
  - Demonstrates iterative belief refinement with sparse acquisitions
  - Visualizes the reconstruction process over multiple acquisition steps

**Task-Based Transmit Beamforming**

This tutorial implements a task-driven perception-action loop where acquisition decisions
are optimized to gain information about a specific downstream task.

- :doc:`../notebooks/agent/task_based_perception_action_loop`

  - Uses :class:`~zea.agent.selection.TaskBasedLines` for task-aware line selection
  - Computes saliency maps via uncertainty propagation through downstream task models
  - Demonstrates how to integrate domain-specific measurement tasks
"""

from . import masks, selection
