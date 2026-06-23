#!/bin/bash
ZEA_VERSION=$(pip show zea 2>/dev/null | grep Version | awk '{print $2}')
DEV_STATUS="No"
if [ "$DEV" = "true" ]; then
  DEV_STATUS="Yes"
fi
printf '%s\n' "==========================================================="
printf '  ZZZZZ   EEEEE   AAAAA     \e[31mv%s\e[0m\tKERAS_BACKEND: \e[36m%s\e[0m\n' "$ZEA_VERSION" "$KERAS_BACKEND"
printf '     ZZ   EE     AA   AA\t\tDev mode     : \e[35m%s\e[0m\n' "$DEV_STATUS"
printf '    ZZ    EEEE   AAAAAAA\t\tJAX          : \e[33m%s\e[0m\n' "$INSTALL_JAX"
printf '   ZZ     EE     AA   AA\t\tTensorFlow   : \e[33m%s\e[0m\n' "$INSTALL_TF"
printf '  ZZZZZ   EEEEE  AA   AA    \e[31mTU/e 2021\e[0m\tPyTorch      : \e[33m%s\e[0m\n' "$INSTALL_TORCH"
printf '%s\n' "==========================================================="
