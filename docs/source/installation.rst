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
               # ! note, you still need to install a backend
               # see below for details

    .. tab-item:: Docker

         .. code-block:: shell

               docker pull zeahub/all:latest
               docker run --gpus 'all' -it zeahub/all:latest


.. _backend-installation:

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

One can build an image for a specific backend using the provided `Dockerfile` and build arguments.

.. code-block:: shell

   docker build --build-arg INSTALL_JAX=gpu . -t zeahub/jax:latest

To build the full image including all backends with GPU support:

.. code-block:: shell

   docker build --build-arg INSTALL_JAX=gpu --build-arg INSTALL_TORCH=gpu --build-arg INSTALL_TF=gpu . -t zeahub/all:latest

Note that these build arguments can be set to either `cpu` or `gpu` depending on your needs.

Run
~~~

Run a container with one of the built images. Ensure you mount your repository at ``/zea`` so that changes are reflected inside the container.
We recommend using `rootless docker <https://docs.docker.com/engine/security/rootless/>`_ to avoid permission issues.

.. code-block:: shell

   docker run --name {CONTAINER-NAME} --gpus 'all' \
     -v ~/zea:/zea \
     -d -it {IMAGE-NAME}:{IMAGE-TAG}

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
   - ``-w`` or ``--workdir``: set the working directory inside the container.
   - ``--rm``: automatically remove the container when it *exits*.
   - ``--env-file``: load environment variables from a .env file.
   - ``--hostname``: set the container hostname (useful for ``users.yaml`` file).

.. important::

   Mount your ``zea`` repository to ``/zea`` inside the container so that changes are reflected in the `zea` installation inside the container.

.. tip::

   The Docker container sets a random hostname by default. You can set a hostname with the ``--hostname`` flag. This is useful for the ``users.yaml`` file. Alternatively, you can use the hostname wildcard in the ``users.yaml`` file.

