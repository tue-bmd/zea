# Resolve KERAS_BACKEND from the INSTALL_* build args baked into the image.
#
# Sourced, not executed, from both /usr/local/bin/zea-entrypoint and /etc/bash.bashrc:
# no single hook covers every way into the image, so the logic lives here once instead
# of being duplicated at both call sites.
#
# An explicit `docker run -e KERAS_BACKEND=...` always wins.
if [ -z "${KERAS_BACKEND:-}" ]; then
    if [ "${INSTALL_JAX:-false}" != "false" ]; then
        KERAS_BACKEND=jax
    elif [ "${INSTALL_TORCH:-false}" != "false" ]; then
        KERAS_BACKEND=torch
    elif [ "${INSTALL_TF:-false}" != "false" ]; then
        KERAS_BACKEND=tensorflow
    else
        KERAS_BACKEND=numpy
    fi
    export KERAS_BACKEND
fi
