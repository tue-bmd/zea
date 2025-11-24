.. _contributing:

Contributing
=============

``zea`` is an open-source project and we welcome contributions from the ultrasound community. Whether you want to report a bug, suggest a new feature, or contribute code, your input is valuable. This document outlines how you can contribute to the project.

How to contribute
-----------------
First, take some time to read through this whole guide. Then follow the following steps to contribute:

1. Make an issue
~~~~~~~~~~~~~~~~

Take a look at the current `issues <https://github.com/tue-bmd/zea/issues>`_ on GitHub to see if there are any open issues that you can help with. If your issue is not listed, please create a new issue describing the motivation behind the problem or feature you want to work on. That way we can discuss it and make sure that your contribution is aligned with the project's goals.

.. dropdown:: Example Issue Template

   .. code-block:: markdown

      **Title:** [Bug/Feature/Question]: Brief summary

      **Description:**
      - For problems or bugs
          - What is the problem?
          - Steps to reproduce
          - Expected behavior
          - Actual behavior
      - For feature requests
          - What is the feature and why is it needed?
          - Possible interface or implementation details
      - References (if applicable)
        - Link to relevant documentation, discussions, or related issues.

2. Fork the repository
~~~~~~~~~~~~~~~~~~~~~~

Most contributors will not have write access to the main repository. Instead, you will need to fork the repository to your own GitHub account. You can fork the repository on GitHub by clicking the "Fork" button at the top right of the page. This will create a copy of the repository under your GitHub account. One can also use:

.. code-block:: shell

   gh repo fork tue-bmd/zea --clone

3. Setup environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check out a new branch for your changes.

.. code-block:: shell

   cd zea
   git checkout -b <your_feature_branch_name>

Set up your development environment. We recommend using Docker as described in the :doc:`installation` guide. This ensures consistency and avoids dependency issues. If you prefer to work without Docker, you can set up a local environment using `conda` or `pip`. For example, to set up a conda environment, you can run:

.. code-block:: shell

   conda create -n zea python=3.12
   conda activate zea
   conda install pip
   pip install -e .[dev]
   pre-commit install

Now install the backend(s). If you use the Docker image, the backends are already installed. If you use a local environment, you need to install one of the supported machine learning backends: JAX, PyTorch or TensorFlow. For more information on how to install the backends, see the :ref:`backend installation <backend-installation>`.` guide.

4. Make your changes
~~~~~~~~~~~~~~~~~~~~

In general, we recommend a test-driven development approach:

1. Write a test for the functionality you want to add.

2. Write the code to make the test pass.

You can run tests for your installed environment using `pytest <https://docs.pytest.org/en/stable/>`_:

.. code-block:: shell

   pytest

A few things to keep in mind when making changes:

- Make sure to write backend-agnostic code. This means that your code should work with all supported backends. This can be achieved by using the ``keras.ops`` API (see the `Keras ops documentation <https://keras.io/api/ops/>`_), instead of using backend-specific functions. For example, use ``keras.ops.squeeze`` instead of ``jax.numpy.squeeze`` or ``torch.squeeze``. Also, when converting tensors to numpy arrays, use the ``keras.ops.convert_to_numpy`` function (instead of ``my_tensor.numpy()``) to ensure compatibility with all backends.

- The code is autoformatted using `ruff <https://pypi.org/project/ruff/>`_. You can run the pre-commit hooks to automatically format and check your code using:

.. code-block:: shell

   pre-commit run --all-files

5. Document your changes
~~~~~~~~~~~~~~~~~~~~~~~~

The documentation uses `Sphinx <https://www.sphinx-doc.org/>`_ and generally is written in reStructuredText format. You can find the documentation files in the `docs/source` directory. Docstrings are written in Google style, which you can see examples of in the `example_google_docstrings.py <https://github.com/tue-bmd/zea/blob/main/docs/example_google_docstrings.py>`_ file. If you add new functionality, please make sure to document it in the documentation files. An example of a well structured docstring below:

.. dropdown:: Example docstring

   .. code-block:: python

      def example_function(param1: int, param2: str) -> bool:
         """This is an example function that demonstrates a well-structured docstring.

         Note the spacing and indentation between the sections, which is important
         for the automatic docs generation. Also it is recommended to add an example
         section with doctests, which will automatically test the code snippets in
         the docs generation.

         Args:
            param1 (int): Description of the first parameter.
            param2 (str): Description of the second parameter.

         Returns:
            bool: Description of the return value.

         Raises:
            ValueError: If param1 is negative.

         Example:
            .. doctest::

               >>> example_function(5, "test")
               True
         """
         if param1 < 0:
            raise ValueError("param1 must be non-negative")
         return True

The overall structure of the documentation is manually designed, but the API documentation is auto-generated based on the docstrings in the code. To generate the docs locally you can run:

.. code-block:: shell

   # if you didn't install the dependencies earlier
   pip install -e .[docs]
   cd docs
   make docs-clean && make docs-build
   # you can also serve the docs locally
   make docs-clean && make docs-serve
   # and open them in your browser at http://localhost:8000

See the `README.md <https://github.com/tue-bmd/zea/blob/main/docs/README.md>`_ for more information.


6. Make a pull request
~~~~~~~~~~~~~~~~~~~~~~
Once your changes are ready and all tests pass, push your branch to your forked repository:

.. code-block:: shell

   git add .
   git commit -m "Description of your changes"
   git push origin <your_feature_branch_name>

Then, go to the original repository on GitHub and open a Pull Request (PR) from your branch. In your PR description, clearly explain what changes you made and reference any related issues.

.. dropdown:: Example Pull Request Template

   .. code-block:: markdown

      **Title:** [Bug/Feature]: Brief summary of changes

      **Description:**
      - What changes were made?
      - Why were these changes made?
      - How do these changes address the issue or feature request?
      - Any additional context or references.
      - How to test the changes.

The maintainers will review your PR and may request changes or ask questions. Please respond to feedback and update your PR as needed. Once everything looks good, your changes will be merged!

.. note::

   Anyone can review pull requests, we encourage others to review each other's work, however, only the maintainers can approve a pull request. Pull Requests require at least one approval and all tests passing before being able to merge it.

Thank you for contributing to zea!


Contributing topics
-------------------

Adding notebooks
~~~~~~~~~~~~~~~~

New tutorial or example notebooks are always welcome! Please add them to the `docs/source/notebooks` directory. Make sure to follow the naming conventions and structure of existing notebooks. If you are adding a new tutorial, please also update the `examples.rst` file in the `docs/source` directory to check if your notebook is included.

Adding to ``zea.models``
~~~~~~~~~~~~~~~~~~~~~~~~

Please see the :doc:`models` section for more information on how to add new models to ``zea``.

Adding to ``zea.ops``
~~~~~~~~~~~~~~~~~~~~~

Please see the :doc:`pipeline` section for more information on how to add new ops to ``zea``.
