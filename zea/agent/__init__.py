"""Agent subpackage for closing action-perception loop in ultrasound imaging.

The `agent` subpackage provides tools and utilities for agent-based algorithms within the ``zea`` framework, including mask generation and action selection strategies. See :mod:`zea.agent.masks` and :mod:`zea.agent.selection` for key functions implementing intelligent focused transmit selection, such as the :class:`zea.agent.selection.GreedyEntropy` algorithm.

For a practical example, see :doc:`../notebooks/agent/agent_example`.

Example usage
^^^^^^^^^^^^^

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
    >>> lines, mask = agent.sample(particles)  # doctest: +SKIP
"""

from . import masks, selection
