Installation
=============

This document describes how to install the zea package and how to use it in a docker container.
Make sure you always use a virtual environment such as ``miniconda`` or ``venv`` to avoid conflicts with other packages!

Backend installation
--------------------

First, install one machine learning backend of choice. Note that zea can run with a numpy backend, but it is not recommended. Also, using the `Docker`_ image will automatically install all compatible backends, so in that case you can skip this step.

- `Install JAX <https://jax.readthedocs.io/en/latest/installation.html>`__
- `Install PyTorch <https://pytorch.org/get-started/locally/>`__
- `Install TensorFlow <https://www.tensorflow.org/install>`__

Editable install
----------------

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (``ultrasound-toolbox``) where the ``pyproject.toml`` file is located and run the following command from terminal:

.. code-block:: bash

   pip install -e .[opencv-python-headless,dev]

This installs the dev dependencies and opencv without the GUI backend. This means it `does not conflict with matplotlib <https://github.com/tue-bmd/ultrasound-toolbox/issues/410>`_.
In case you need the opencv GUI backend, you can install it with ``pip install -e .[opencv-python]``.

Install from github
-------------------

You can also directly install the package from github. This is useful if you want to install a specific release or branch and keep it fixed in your environment.
Note that this is supported from zea v1.2.6 onward.
You can install from Github using either a Github Personal Access Token or an SSH key.

Using a Personal Access Token
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare: `Setup personal access tokens for organisation <https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/setting-a-personal-access-token-policy-for-your-organization#enforcing-an-approval-policy-for-fine-grained-personal-access-tokens>`_

1. `Create a personal access token <https://github.com/settings/personal-access-tokens/new>`__ with the following settings:
   - **Resource owner**: ``tue-bmd``
   - **Repository access**: Only ``ultrasound-toolbox``
   - **Repository permissions**: Contents = *Read-only*

2. Find the release or branch you want to install, e.g. `latest release <https://github.com/tue-bmd/ultrasound-toolbox/releases/latest>`__.

3. Set your GitHub username and token as environment variables in your terminal:

   .. code-block:: bash

      export GITHUB_USER=your-username
      export GITHUB_TOKEN=your-token

4. Install the package using pip:

   .. code-block:: bash

      pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/tue-bmd/ultrasound-toolbox.git@{RELEASE}

   Replace ``{RELEASE}`` with the desired tag or branch, e.g. ``v1.2.7`` or ``main``.

Using an SSH key
~~~~~~~~~~~~~~~~

Alternatively you could use ssh access to the repository and install using::

   pip install git+ssh://git@github.com:tue-bmd/ultrasound-toolbox.git@{RELEASE}

SSH might be a bit harder to setup, but is more convenient in the end.

For this you have to make sure that git is using the correct SSH provider. On windows multiple may exist.
I have set the environment variable ``GIT_SSH=C:\windows\System32\OpenSSH\ssh.exe``

If your ssh key has a passphrase to protect it, you must use an ssh-agent because `pip does not prompt for the passphrase <https://github.com/pypa/pip/issues/7308>`_. Also here, Git for Windows comes with the command ``start-ssh-agent``, which should **NOT** be used if you use OpenSSH from windows. Then you should start it with ``ssh-agent -s``. And add your key with ``ssh-add``.

If you get host key errors, you may need to update your known host for Github, see https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/.

Resources
~~~~~~~~~

- https://docs.readthedocs.io/en/stable/guides/private-python-packages.html
- https://stackoverflow.com/questions/40898981/how-to-discover-where-pip-install-gitssh-is-searching-for-ssh-keys
- https://stackoverflow.com/questions/18683092/how-to-run-ssh-add-on-windows
- https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement
- https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version

Docker
-------


This repository provides multiple Docker images built and tested in our CI pipeline.

Pre-built images
----------------

Public images
~~~~~~~~~~~~~

These images are all built on top of ``Dockerfile.base``:

- `zeahub/all`: This image includes support for all machine learning backends (TensorFlow, PyTorch, and JAX).
- `zeahub/tensorflow`: This image includes support for TensorFlow.
- `zeahub/torch`: This image includes support for PyTorch.
- `zeahub/jax`: This image includes support for JAX.

These images are uploaded to Docker Hub via the CI pipeline and can be used directly in your projects via:

.. code-block:: shell

   docker pull zeahub/all:latest

Private images
~~~~~~~~~~~~~~

- zeahub/private: Built from ``Dockerfile``. This image inherits from `zeahub/all`, copies your repository, performs an editable installation of `zea`, and adds a Message of the Day displaying the `zea` version. This image is also used for development with VSCode, as described below.

The private image is not uploaded to Docker Hub and must be built manually to prevent pushing private code to a public repository. If you use VSCode, you can use the provided ``.devcontainer.json`` file to attach to the private image for development, see `Development in the Container using VSCode`_.

Build
-----

To manually build the base image from its dedicated Dockerfile:

.. code-block:: shell

   docker build -f Dockerfile.base --build-arg BACKEND=numpy . -t zeahub/base:latest

To build the full image with all backends (the default is BACKEND=all):

.. code-block:: shell

   docker build -f Dockerfile.base --build-arg BACKEND=all . -t zeahub/all:latest

To build the private (development) image:

.. code-block:: shell

   docker build . -t zeahub/private:latest

Run
---

Run a container with one of the built images. Ensure you mount your repository at ``/ultrasound-toolbox`` so that changes are reflected inside the container, and use your user and group IDs to avoid permission issues.

.. code-block:: shell

   docker run --name {CONTAINER-NAME} --gpus 'all' \
     -v ~/ultrasound-toolbox:/ultrasound-toolbox \
     -d -it -m 100g --cpus 7 --user "$(id -u):$(id -g)" \
     {IMAGE-NAME}:{IMAGE-TAG}

Which means:

- ``docker run``: create and run a new container from an image.
- ``--name``: name the container.
- ``--gpus``: specify GPU devices to add to the container ('all' to pass all GPUs).
- ``-v`` or ``--volume``: bind mount a volume.
- ``-d`` or ``--detach``: start the container as a background process.
- ``-it``: start an interactive terminal session.
   - ``--interactive``: keep STDIN open.
   - ``--tty``: allocate a pseudo-TTY.
- ``-m`` or ``--memory``: set a memory limit (use g for gigabytes).
- ``--cpus``: specify the number of CPU cores to use.
- ``--user``: run as a specific user.

The container uses ``/bin/bash`` as its entrypoint, allowing you to interactively execute shell commands.

.. important::

   Mount your ``ultrasound-toolbox`` repository to ``/ultrasound-toolbox`` inside the container so that changes are reflected in the `zea` installation inside the container. Additionally, use your user ID and group ID with ``--user "$(id -u):$(id -g)"`` to avoid permission issues when writing to mounted volumes.

.. tip::

   The Docker container sets a random hostname by default. You can set a hostname with the ``--hostname`` flag. This is useful for the ``users.yaml`` file. Alternatively, you can use the hostname wildcard in the ``users.yaml`` file.

Alternative flags:

- ``-w`` or ``--workdir``: set the working directory inside the container.
- ``--rm``: automatically remove the container when it *exits*.
- ``--env-file``: load environment variables from a .env file.

Attach / Start / Stop
---------------------

To attach to the container:

.. code-block:: shell

   docker attach {CONTAINER-NAME}

Start and stop the container with:

.. code-block:: shell

   docker start {CONTAINER-NAME}

.. code-block:: shell

   docker stop {CONTAINER-NAME}

Development in the Container using VSCode
-----------------------------------------

You can use the VSCode Remote Containers extension to attach to the running container for development. A ``.devcontainer.json`` file is provided which specifies the Docker image to use, the volumes to mount, and the extensions to install. To use it, ensure the Remote Containers extension is installed in VSCode, then click the devcontainer icon in the bottom left corner and select "Reopen in Container". To revert to the host environment, click the devcontainer icon again and select "Reopen Locally".

Using git
~~~~~~~~~

Ensure that the ssh-agent is running and your SSH key is added. The local (or remote) ssh-agent is shared with the container upon attaching. More information can be found `here <https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials>`_.

Installing More Packages
~~~~~~~~~~~~~~~~~~~~~~~~

If you need to install additional packages after the image has been built and you are in the container as your user, use ``sudo``:

.. code-block:: shell

   sudo pip install {PACKAGE}
