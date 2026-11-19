#!/bin/sh
# Entry point for `docker run <img> <cmd>`, which reads neither /etc/bash.bashrc (that is
# interactive-bash only) nor any profile -- so without this, `docker run zeahub/jax python
# train.py` starts with KERAS_BACKEND unset and zea aborts on Keras' 'tensorflow' default.
#
# `docker exec` bypasses ENTRYPOINT in turn, which is why bashrc sources the same script.
. /etc/zea-backend.sh

exec "$@"
