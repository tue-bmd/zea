
<!-- This is the readme for the github page (more complete readme for pdocs can be found in usmbd/README.md) -->
# Ultrasound toolbox

The ultrasound toolbox (usbmd) is a collection of ultrasound tools (Python) such as beamforming code, visualization tools and deep learning scripts. Check out the full documentation by opening [index.html](docs/usbmd/index.html) locally in your browser.

The idea of this toolbox is that it is self-sustained, meaning ultrasound researchers can use the tools to create new models / algorithms and after completed, can add them to the toolbox. This repository is being maintained by researchers from the [BM/d lab](https://www.tue.nl/en/research/research-groups/signal-processing-systems/biomedical-diagnostics-lab/) at Eindhoven University of Technology. Currently for [internal](LICENSE) use only.

In case of any questions, feel free to [contact](mailto:t.s.w.stevens@tue.nl).

## Installation

### Editable install

This package can be installed like any open-source python package from PyPI.
Make sure you are in the root folder (`ultrasound-toolbox`) where the [`setup.py`](setup.py) file is located and run the following command from terminal:

```bash
python -m pip install -e .
```

### Install from github

You can also directly install the package from github. This is useful if you want to install a specific release or branch and keep it fixed in your environment.
Note that this is supported from usbmd v1.2.6 onward.
You can install from Github using either a Github Personal Access Token or and SSH key.
#### Using a Personal Access Token

Prepare: [Setup personal access tokens for organisation](https://docs.github.com/en/organizations/managing-programmatic-access-to-your-organization/setting-a-personal-access-token-policy-for-your-organization#enforcing-an-approval-policy-for-fine-grained-personal-access-tokens)

1. [Create personal access token](https://github.com/settings/personal-access-tokens/new)
    - **Resource owner**: _tue-bmd_
    - **Only select repositories**: _ultrasound-toolbox_
    - **Repository permissions**: Contents = _Read-only_
2. Find the release you want to install, e.g. [the latest](https://github.com/tue-bmd/ultrasound-toolbox/releases/latest)
3. `pip install git+https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/tue-bmd/ultrasound-toolbox.git@{RELEASE}`
    - e.g. `RELEASE`=v1.2.6
    - e.g. `RELEASE`=develop

#### Using an SSH key

Alternatively you could use ssh access to the repository and install using:
`pip install git+ssh://git@github.com/tue-bmd/ultrasound-toolbox.git@{RELEASE}`

SSH might be a bit harder to setup, but is more convenient in the end.

For this you have to make sure that git is using the correct SSH provider. On windows multiple may exist.
I have set the environment variable GIT_SSH=C:\windows\System32\OpenSSH\ssh.exe

If your ssh key has a passphrase to protect it, you must use an ssh-agent because [pip does not prompt for the passphrase](https://github.com/pypa/pip/issues/7308). Also here, Git for Windows comes with the command `start-ssh-agent`, which should **NOT** be used if you use OpenSSH from windows. Then you should start it with `ssh-agent -s`. And add your key with `ssh-add`.

If you get host key errors, you may need to update your known host for Github, see https://github.blog/2023-03-23-we-updated-our-rsa-ssh-host-key/.

#### Resources

- https://docs.readthedocs.io/en/stable/guides/private-python-packages.html
- https://stackoverflow.com/questions/40898981/how-to-discover-where-pip-install-gitssh-is-searching-for-ssh-keys
- https://stackoverflow.com/questions/18683092/how-to-run-ssh-add-on-windows
- https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_keymanagement

## Usage

After installation, you can use the package as follows in your own project:

```python
# import usbmd package
import usbmd
# or if you want to use the Tensorflow tools
from usbmd import tensorflow_ultrasound as usmbd_tf
# or if you want to use the Pytorch tools
from usbmd import pytorch_ultrasound as usbmd_torch
```
