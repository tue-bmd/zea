#!/bin/bash
# Message of the Day – shown on every interactive shell in the Docker image.
# Env vars set by /etc/bash.bashrc before this runs:
#   KERAS_BACKEND  INSTALL_JAX  INSTALL_TORCH  INSTALL_TF  DEV

ZEA_VERSION=$(pip show zea 2>/dev/null | awk '/^Version/{print $2}')
ZEA_VERSION=${ZEA_VERSION:-dev}
DEV_STATUS=no; [ "$DEV" = "true" ] && DEV_STATUS=yes

# ── palette ───────────────────────────────────────────────────────────────────
OR=$'\e[38;5;214m'   # brand orange  (#fbb53e)
GY=$'\e[38;5;245m'   # gray (secondary text)
CY=$'\e[36m'         # cyan  (values)
PU=$'\e[38;5;55m'    # purple (flags)
RD=$'\e[38;5;203m'   # red-orange (version)
BD=$'\e[1m'
DM=$'\e[2m'
RS=$'\e[0m'

# ── tiles ─────────────────────────────────────────────────────────────────────
# ▀ = upper-half block: fg fills top 50% of cell, terminal bg fills bottom 50%.
# Combined with the 1-space horizontal gap, tiles appear as distinct landscape
# rectangles separated both horizontally and vertically.
OT=$'\e[38;5;214m▀▀\e[0m'   # brand orange  (#fbb53e)
PT=$'\e[38;5;55m▀▀\e[0m'    # brand purple  (#593c5e)
ET='  '                       # empty slot (no tile)

yn() { [ -n "$1" ] && [ "$1" != "false" ] && printf '%s' "${PU}${1}${RS}" || printf '%s' "${GY}no ${RS}"; }

# ── grid (4 cols × 7 rows, mirroring the transducer-array logo motif) ─────────
# O=orange  P=purple  .=empty
#   . O O .
#   O P O O
#   O O O O
#   P O O P
#   O O P O
#   O O O P
#   . O O .
G='   '

# separator spans exactly "zea  v{version}  TU/e · BM/d"
slen=$(( 3 + 2 + 1 + ${#ZEA_VERSION} + 2 + 11 ))
sep=''; for i in $(seq 1 $slen); do sep="${sep}─"; done

printf '\n'
printf '%s %s %s %s%s%szea%s  %sv%s%s  %sTU/e · BM/d%s\n' \
    "$ET" "$OT" "$OT" "$ET"  "$G"  "$OR$BD" "$RS"  "$RD" "$ZEA_VERSION" "$RS"  "$GY$DM" "$RS"
printf '%s %s %s %s%s%s%s%s\n' \
    "$OT" "$PT" "$OT" "$OT"  "$G"  "$GY" "$sep" "$RS"
printf '%s %s %s %s%sKERAS_BACKEND  %s%s%s\n' \
    "$OT" "$OT" "$OT" "$OT"  "$G"  "$OR" "$KERAS_BACKEND" "$RS"
printf '%s %s %s %s%sdev mode       %s%s%s\n' \
    "$PT" "$OT" "$OT" "$PT"  "$G"  "$OR" "$DEV_STATUS" "$RS"
printf '%s %s %s %s%sjax            %s\n' \
    "$OT" "$OT" "$PT" "$OT"  "$G"  "$(yn "${INSTALL_JAX}")"
printf '%s %s %s %s%storch          %s\n' \
    "$OT" "$OT" "$OT" "$PT"  "$G"  "$(yn "${INSTALL_TORCH}")"
printf '%s %s %s %s%stensorflow     %s\n' \
    "$ET" "$OT" "$OT" "$ET"  "$G"  "$(yn "${INSTALL_TF}")"
printf '\n'
