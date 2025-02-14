# Docker

This repository provides multiple Docker images built and tested in our CI pipeline.

### Public images Built from [Dockerfile.base](#file:Dockerfile.base-context):
- usbmd/all: This image includes support for all machine learning backends (TensorFlow, PyTorch, and JAX).
- usbmd/tensorflow: This image includes support for TensorFlow.
- usbmd/torch: This image includes support for PyTorch.
- usbmd/jax: This image includes support for JAX.

These images are uploaded to Docker Hub via the CI pipeline and can be used directly in your projects via:

```shell
docker pull usbmd/all:latest
```

### Private images:
- usbmd/private: Built from [Dockerfile](#file:Dockerfile-context). This image inherits from usbmd/all, copies your repository, performs an editable installation of usbmd, and adds a Message of the Day displaying the usbmd version. This image is also used for development with VSCode, as described below.

The private image is not uploaded to Docker Hub and must be built manually to prevent pushing private code to a public repository. If you use VSCode, you can use the provided `.devcontainer.json` file to attach to the private image for development, see [Development in the Container using VSCode](#development-in-the-container-using-vscode).

## Build

To manually build the base image from its dedicated Dockerfile:

```shell
docker build -f Dockerfile.base . -t usbmd/base:latest
```

To build the full image with all backends (the default is BACKEND=all):

```shell
docker build --build-arg BACKEND=all . -t usbmd/all:latest
```

To build the private (development) image:

```shell
docker build . -t usbmd/private:latest
```

## Run

Run a container with one of the built images. Ensure you mount your repository at `/ultrasound-toolbox` so that changes are reflected inside the container, and use your user and group IDs to avoid permission issues.

```shell
docker run --name {CONTAINER-NAME} --gpus 'all' \
  -v ~/ultrasound-toolbox:/ultrasound-toolbox \
  -d -it -m 100g --cpus 7 --user "$(id -u):$(id -g)" \
  {IMAGE-NAME}:{IMAGE-TAG}
```

Which means:
- `docker run`: create and run a new container from an image.
- `--name`: name the container.
- `--gpus`: specify GPU devices to add to the container ('all' to pass all GPUs).
- `-v` or `--volume`: bind mount a volume.
- `-d` or `--detach`: start the container as a background process.
- `-it`: start an interactive terminal session.
   - `--interactive`: keep STDIN open.
   - `--tty`: allocate a pseudo-TTY.
- `-m` or `--memory`: set a memory limit (use g for gigabytes).
- `--cpus`: specify the number of CPU cores to use.
- `--user`: run as a specific user.

The container uses `/bin/bash` as its entrypoint, allowing you to interactively execute shell commands.

> [!IMPORTANT]
> Mount your `ultrasound-toolbox` repository to `/ultrasound-toolbox` inside the container so that changes are reflected in the usbmd installation inside the container. Additionally, use your user ID and group ID with `--user "$(id -u):$(id -g)"` to avoid permission issues when writing to mounted volumes.

> [!TIP]
> The Docker container sets a random hostname by default. You can set a hostname with the `--hostname` flag. This is useful for the `users.yaml` file. Alternatively, you can use the hostname wildcard in the `users.yaml` file.

Alternative flags:
- `-w` or `--workdir`: set the working directory inside the container.
- `--rm`: automatically remove the container when it *exits*.
- `--env-file`: load environment variables from a .env file.

## Attach / Start / Stop

To attach to the container:

```shell
docker attach {CONTAINER-NAME}
```

Start and stop the container with:

```shell
docker start {CONTAINER-NAME}
```

```shell
docker stop {CONTAINER-NAME}
```

## Development in the Container using VSCode

You can use the VSCode Remote Containers extension to attach to the running container for development. A `.devcontainer.json` file is provided which specifies the Docker image to use, the volumes to mount, and the extensions to install. To use it, ensure the Remote Containers extension is installed in VSCode, then click the devcontainer icon in the bottom left corner and select "Reopen in Container". To revert to the host environment, click the devcontainer icon again and select "Reopen Locally".

## Using git

Ensure that the ssh-agent is running and your SSH key is added. The local (or remote) ssh-agent is shared with the container upon attaching. More information can be found [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials).

## Installing More Packages

If you need to install additional packages after the image has been built and you are in the container as your user, use `sudo`:

```shell
sudo pip install {PACKAGE}
```
