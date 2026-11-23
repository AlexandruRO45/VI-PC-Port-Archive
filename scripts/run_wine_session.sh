#!/usr/bin/env bash
# Run CI tests on the display that initialized their dedicated Wine prefix.
set -e
if (( $# == 0 )); then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
: "${WINEPREFIX:?Set WINEPREFIX to a dedicated test prefix}"

# Stop this prefix's Wine processes before Xvfb tears down the display.
trap 'wineserver -k || true' EXIT
timeout 5m wineboot --init </dev/null
"$@" </dev/null
