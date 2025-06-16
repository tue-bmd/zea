"""
The `agent` subpackage provides tools and utilities for agent-based algorithms within the `zea` framework,
including mask generation and action selection strategies. The `masks` and `selection` submodules provide
key functions for implementing intelligent focused transmit selection, such as the _Greedy Entropy Minimization_ algorithm.
For a practical example, see [Active Perception for Focused Transmit Selection](http://127.0.0.1:8000/notebooks/agent_example.html).

Example usage of action selection strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    agent = zea.agent.selection.GreedyEntropy(
        n_actions=7,
        n_possible_actions=112,
        img_width=112,
        img_height=112,
        **kwargs,
    )
    particles = np.random.rand(10, 112, 112, 1)  # 10 posterior samples
    lines, mask = agent(particles)
"""
