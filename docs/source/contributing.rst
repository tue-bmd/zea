.. _contributing:

Contributing
=======================

Style conventions
-----------------

Code style (PEP 8)
~~~~~~~~~~~~~~~~~~

To keep our code consistent and ease collaboration, we all follow the PEP 8 style convention. You can read about PEP 8 at https://peps.python.org/pep-0008/. Please take a look and familiarise yourself with these conventions. Some examples of what this style uses are:

- Use 4 spaces per indentation level
- Limit all lines to a maximum of 88 characters
- Imports are always put at the top of the file and are grouped by type.
- Class names should normally use the CapWords convention
- Function and variable names should be lowercase, with words separated by underscores as necessary to improve readability
- (Many more recommendations)

The style of our code will be continuously checked via ``pylint`` when you submit a Pull Request to merge code to branch ``main``. Any inconsistency will be flagged and will block the pull request until it is resolved.

It's recommended to set up your IDE to enforce PEP 8 style. In general, we use VS Code, the black formatter, pylint and isort to achieve this. You can install `black <https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter>`_, `isort <https://marketplace.visualstudio.com/items?itemName=ms-python.isort>`_, `pylint <https://marketplace.visualstudio.com/items?itemName=ms-python.pylint>`_ as extensions to VS Code.

Additionally, it is useful to add the following settings to your VS Code ``settings.json``. This will automatically format your code on save.

.. code-block:: json

   "[python]": {
      "editor.defaultFormatter": "ms-python.black-formatter",
      "editor.formatOnSave": true,
      "editor.codeActionsOnSave": {
         "source.organizeImports": true
      }
   },
   "isort.args": [
      "--profile",
      "black"
   ],

To check if pylint runs successfully, run the following in the root directory:

.. code-block:: bash

   pylint zea

Docstrings (Google)
~~~~~~~~~~~~~~~~~~~

For docstrings we'll rely on the Google style described in ``example_google_docstrings.py``. Your IDE is also able to automatically populate docstrings in our style of choice, so it's a good idea to configure this too (see this `example in PyCharm <https://www.jetbrains.com/help/pycharm/settings-tools-python-integrated-tools.html>`_).

Pull Requests (PR)
------------------

We're following a Git workflow to collaborate at scale and efficiently. If you're not familiar with Git a good starting point would be to read tutorials such as `this one <https://nvie.com/posts/a-successful-git-branching-model/>`_ or these from Atlassian (`tutorial 1 <https://www.atlassian.com/git/tutorials/comparing-workflows#:~:text=A%20Git%20workflow%20is%20a,in%20how%20users%20manage%20changes.>`_, `tutorial 2 <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_).

We'll be using a simplified version of the workflows described in the tutorials above. In its simplest form, our repository will contain three types of branches:

- ``main`` branch: This contains the most stable version of our code. Unit tests should always pass in this branch. We use this branch to release new official versions of the code. We only merge from the develop branch into this branch with each new release.
- ``develop`` branch: This contains the latest version of our code. Unit tests should always pass in this branch and at least one review is mandatory. We use this branch to merge new features into. This is generally where you would branch off from to create a new feature branch, and also where you would PR your feature branch into.
- ``feature`` branches: Feature branches are derived from ``develop`` and are intended for the development of features, so we don't expect them to be tested and reliable at all times. Once the feature is finalised, we'll test core components it brings into the code and merge it into ``develop`` through a Pull Request.

Submitting a Pull Request (PR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The typical PR workflow to make changes to the code will look as follows:

#. Switch to the main branch and pull the latest changes:

   .. code-block:: shell

      git checkout main
      git pull

#. Create a new feature branch for your changes:

   .. code-block:: shell

      git checkout -b <your_feature_branch_name>

   Try to use a descriptive name. A good convention is to use your initials followed by a concise description for what you will implement, for instance:

   .. code-block:: shell

      git checkout -b feature/ts_unet_sr_network

#. Make your changes in your feature branch. Test new code components if necessary.

#. Ensure that all tests pass locally and your code is formatted correctly. You can run the tests and check the code style by running the following commands from the root directory of the repository:

   .. code-block:: shell

      pylint zea
      pytest ./tests

#. Stage the changes to commit

   .. code-block:: shell

      git add <path_to_files_to_stage>

   For instance, from the repository root directory you can add all changes with

   .. code-block:: shell

      git add .

#. Commit your changes using a short but descriptive commit message

   .. code-block:: shell

      git commit -m "<your_commit_message>"

#. (Likely needed) Merge any changes from remote ``develop`` into your branch to incorporate work from others that happened while you were working on your branch. If any conflicts arise, resolve them and repeat steps 3 to 7.

   .. code-block:: shell

      git merge origin/develop

#. Push your branch to the GitHub remote repository:

   .. code-block:: shell

      git push origin <your_feature_branch_name>

   Your PR will now be available on GitHub. A url will show in the console output that can take you directly to it.

#. In GitHub, send a PR to merge your feature branch into the develop branch.

#. Wait for a reviewer to review your PR. After it's accepted, proceed with the merge.

#. After your pull request is merged, make sure that your branch is deleted.

.. note::

   Did you find any issues or inconsistencies following these PR guidelines? Please let a maintainer of the repository know so it's always up to date!

Reviewing a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~

Anyone can review pull requests, we encourage others to review each other's work, however, only the maintainers can approve a pull request. Pull Requests require at least one approval and all tests passing before being able to merge it.

Your First Contribution
~~~~~~~~~~~~~~~~~~~~~~~

Working on your first Pull Request? You can learn how from this *free* series, `How to Contribute to an Open Source Project on GitHub <https://app.egghead.io/playlists/how-to-contribute-to-an-open-source-project-on-github>`_. If you prefer to read through some tutorials, visit https://makeapullrequest.com/ and https://www.firsttimersonly.com/

At this point, you're ready to make your changes! Feel free to ask for help; everyone is a beginner at first :relaxed:

Bumping zea version
---------------------

Bumping and releasing a new version of zea is done by the maintainers of the repository. To bump the version, follow these steps:

First check out the most recent version of the main branch and pull the latest changes:

.. code-block:: shell

   git checkout main
   git pull

Then run the following command to bump the version:

.. code-block:: shell

   ./post-release.sh <new_version> <snellius_username>

The zea ``new_version`` should be a valid version number, e.g. ``v0.1.0``, with the ``v`` prefix and major, minor and patch version numbers. The ``snellius_username`` is the username you use to log in to the Snellius cluster.

Make sure to follow the instructions in the ``post-release.sh`` script. This script will build the new (Docker) environment and version of zea.

Adding models
-------------

Please see the :doc:`models` section for more information on how to add new models to ``zea``.

Adding ops
----------

Please see the :doc:`pipeline` section for more information on how to add new ops to ``zea``.
