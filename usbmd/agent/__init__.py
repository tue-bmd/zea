"""Agent subpackage for usbmd.

This module provides tools and utilities for agent-based operations within the usbmd framework,
including mask generation and action selection strategies.

Example usage of action selection strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    agent = usbmd.agent.selection.GreedyEntropy(
        n_actions=7,
        n_possible_actions=112,
        img_width=112,
        img_height=112,
        **kwargs,
    )
    particles = np.random.rand(10, 112, 112, 1) # 10 posterior samples
    lines, mask = agent(particles)
"""
