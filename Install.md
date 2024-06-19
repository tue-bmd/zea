# Install usbmd

- [Install usbmd](#install-usbmd)
  - [Editable install](#editable-install)
  - [Install from github](#install-from-github)
    - [Using a Personal Access Token](#using-a-personal-access-token)
    - [Using an SSH key](#using-an-ssh-key)
    - [Resources](#resources)
  - [Docker](#docker)
    - [Build](#build)
      - [Base](#base)
      - [Keras 3](#keras-3)
    - [Run](#run)
    - [Attach / start / stop](#attach--start--stop)
    - [Development in the container using vscode](#development-in-the-container-using-vscode)
      - [Using git](#using-git)
      - [Installing more packages](#installing-more-packages)

## Editable install

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`setup.py`](setup.py) file is located and run the following command from terminal:

```bash
python -m pip install -e .
```

## Install from github

You can also directly install the package from github. This is useful if you want to install a specific release or branch and keep it fixed in your environment.
Note that this is supported from usbmd v1.2.6 onward.
You can install from Github using either a Github Personal Access Token or and SSH key.

### Using a Personal Access Token

Prepare: [Setup personal access tokens for organisation](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/setting-a-personal-access-token-policy-for-your-organization#enforcing-an-approval-policy-for-fine-grained-personal-access-tokens)

1. [Create personal access token](https://github.com/settings/personal-access-tokens/new)
    - **Resource owner**: _tue-bmd_
    - **Only select repositories**: _ultrasound-toolbox_
    - **Repository permissions**: Contents = _Read-only_
2. Find the release you want to install, e.g. [the latest](https://github.com/tue-bmd/ultrasound-toolbox/releases/latest)
3. `pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/tue-bmd/ultrasound-toolbox.git@{RELEASE}`
    - e.g. `RELEASE`=v1.2.7
    - e.g. `RELEASE`=main

### Using an SSH key

Alternatively you could use ssh access to the repository and install using:
`pip install git+ssh://git@github.com/tue-bmd/ultrasound-toolbox.git@{RELEASE}`

SSH might be a bit harder to setup, but is more convenient in the end.

For this you have to make sure that git is using the correct SSH provider. On windows multiple may exist.
I have set the environment variable GIT_SSH=C:\windows\System32\OpenSSH\ssh.exe

If your ssh key has a passphrase to protect it, you must use an ssh-agent because [pip does not prompt for the passphrase](https://github.com/pypa/pip/issues/7308). Also here, Git for Windows comes with the command `start-ssh-agent`, which should **NOT** be used if you use OpenSSH from windows. Then you should start it with `ssh-agent -s`. And add your key with `ssh-add`.

If you get host key errors, you may need to update your known host for Github, see https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/.

### Resources

- https://docs.readthedocs.io/en/stable/guides/private-python-packages.html
- https://stackoverflow.com/questions/40898981/how-to-discover-where-pip-install-gitssh-is-searching-for-ssh-keys
- https://stackoverflow.com/questions/18683092/how-to-run-ssh-add-on-windows
- https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement
- https://stackoverflow.com/questions/19548957/can-i-force-pip-to-reinstall-the-current-version

## Docker

### Build

#### Base

You can build a docker image from the dockerfile in this repository.
This will include all the necessary packages and dependencies.

```shell
cd ~/
git clone git@github.com:tue-bmd/ultrasound-toolbox.git
cd ultrasound-toolbox
docker build . -t usbmd/base:latest
```

This will build the image `usbmd/base:latest`.

#### Keras 3

Additionally, the dockerfile also includes the build-arg `KERAS3` which can be set to `true` to install keras 3 with `torch`, `tensorflow` and `jax`.

```shell
docker build --build-arg KERAS3=True . -t usbmd/keras3:latest
```

This will build the image `usbmd/keras3:latest`.

### Run
Here is an example of how to run the docker container with the image you just built. Note that there exist [many flags](https://docs.docker.com/reference/cli/docker/container/run/) you may use.

```shell
docker run --name {CONTAINER-NAME} --gpus 'all' -v ~/{NAS-MOUNT}:/mnt/z/ -v ~/ultrasound-toolbox:/ultrasound-toolbox -d -it -m 100g --cpus 7 --user "$(id -u):$(id -g)" {IMAGE-NAME}:{IMAGE-TAG}
```

Which means:
- `docker run`: create and run a new container from an image
- `--name`: name the container
- `--gpus`: GPU devices to add to the container ('all' to pass all GPUs)
- `-v` = `--volume`: Bind mount a volume
- `-d`=`--detach`: starts a container as a background process that doesn't occupy your terminal window
- `-it`: makes the container start look like a terminal connection session
	- `--interactive`: Keep STDIN open even if not attached
	- `--tty`: Allocate a pseudo-TTY
- `-m` = `--memory`: Memory limit (use g for gigabytes)
- `--cpus`: Number of cores to use
- `--user`: Run as a specific user

> [!IMPORTANT]
> Note that it is important to mount your `ultrasound-toolbox` repository to `/ultrasound-toolbox` inside the container, so that the changes you make are reflected in the usbmd installation inside the container. Additionally, you should use your user id and group id with `--user "$(id -u):$(id -g)"` to avoid permission issues when writing to a mounted volume.

Alternative flags:
- `-w` = `--workdir`: Working directory inside the container
- `--rm`: Automatically remove the container when it *exits*
- `--env-file`: Load environment variables from a .env file

### Attach / start / stop
Now you can attach to the container with:

```shell
docker attach {CONTAINER-NAME}
```

Starting and stopping with `docker start {CONTAINER-NAME}` and `docker stop {CONTAINER-NAME}`.

### Development in the container using vscode

You can use vscode to attach to the running container and develop in it.
You can install your vscode extensions in the container.
The easiest is to keep re-using this container so do not delete it after use.

#### Using git

Make sure that the the ssh-agent is running and your ssh key is added to it. The local (or remote) ssh-agent is shared with the container upon attaching. More information can be found [here](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials).

#### Installing more packages

If you want to install some packages after the image has been build and you are in the container as your user, make sure to use `sudo`.

```shell
sudo pip install {PACKAGE}
```
