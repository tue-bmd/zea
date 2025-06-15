Installation
=============

Here we will outline how to install the `zea` package and its dependencies.
Besides installation through `PyPI <https://pypi.org/project/zea>`_, we also provide Docker images for easy development (see the `Docker`_ section below). For instructions on contributing and development installation, see :doc:`contributing`.

.. tab-set::

    .. tab-item:: Conda / Pip

         .. code-block:: shell

               conda create -n zea python=3.12 # 3.10, 3.11, or 3.12
               conda activate zea
               pip install zea

    .. tab-item:: Docker

         .. code-block:: shell

               docker pull zeahub/all:latest
               docker run --gpus 'all' -it zeahub/all:latest


Backend
-------

.. include:: getting-started.rst
   :start-after: .. backend-installation-start
   :end-before: .. backend-installation-end


Docker
-------

This repository provides multiple Docker images built and tested in our CI pipeline.

Pre-built images
~~~~~~~~~~~~~~~~


The following images are available on Docker Hub:

- `zeahub/all`: This image includes support for all machine learning backends (TensorFlow, PyTorch, and JAX).
- `zeahub/tensorflow`: This image includes support for TensorFlow.
- `zeahub/torch`: This image includes support for PyTorch.
- `zeahub/jax`: This image includes support for JAX.

These images are uploaded to Docker Hub via the CI pipeline and can be used directly in your projects via:

.. code-block:: shell

   docker pull zeahub/all:latest


Build
~~~~~

One can build an image for a specific backend using the provided `Dockerfile` and the `BACKEND` build argument.

.. code-block:: shell

   docker build -f Dockerfile --build-arg BACKEND=jax . -t zeahub/jax:latest

To build the full image with all backends, use `BACKEND=all`.

.. code-block:: shell

   docker build -f Dockerfile --build-arg BACKEND=all . -t zeahub/all:latest

Run
~~~

Run a container with one of the built images. Ensure you mount your repository at ``/zea`` so that changes are reflected inside the container, and use your user and group IDs to avoid permission issues.

.. code-block:: shell

   docker run --name {CONTAINER-NAME} --gpus 'all' \
     -v ~/zea:/zea \
     -d -it --user "$(id -u):$(id -g)" \
     {IMAGE-NAME}:{IMAGE-TAG}

.. dropdown:: Docker run command flags explained

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

   Mount your ``zea`` repository to ``/zea`` inside the container so that changes are reflected in the `zea` installation inside the container. Additionally, use your user ID and group ID with ``--user "$(id -u):$(id -g)"`` to avoid permission issues when writing to mounted volumes.

.. tip::

   The Docker container sets a random hostname by default. You can set a hostname with the ``--hostname`` flag. This is useful for the ``users.yaml`` file. Alternatively, you can use the hostname wildcard in the ``users.yaml`` file.

Alternative flags:

- ``-w`` or ``--workdir``: set the working directory inside the container.
- ``--rm``: automatically remove the container when it *exits*.
- ``--env-file``: load environment variables from a .env file.

Attach / Start / Stop
~~~~~~~~~~~~~~~~~~~~~

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
