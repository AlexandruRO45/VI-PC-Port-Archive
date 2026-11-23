#!/usr/bin/env bash
# Run CI tests on the display that initialized their dedicated Wine prefix.
# WINE names the Wine to use, as for run_wine_tests.py (default wine; split on spaces, e.g. "umu-run"). Its
# wineserver is WINESERVER, else the one next to that wine (a Proton build's files/bin/), else wineserver on PATH.
set -e
: "${WINEPREFIX:?Set WINEPREFIX to a dedicated test prefix}"

read -ra wine <<<"${WINE:-wine}"
[ "${#wine[@]}" -gt 0 ] || { echo "WINE is blank" >&2; exit 2; }
wineserver=${WINESERVER:-}
if [ -z "$wineserver" ]; then
    found=$(command -v -- "${wine[0]}" || true)
    if [ -n "$found" ] && [ -x "$(dirname -- "$found")/wineserver" ]; then
        wineserver=$(dirname -- "$found")/wineserver
    else
        wineserver=wineserver
    fi
fi

# Stop this prefix's Wine processes before Xvfb tears down the display.
trap '"$wineserver" -k || true' EXIT
timeout 5m "${wine[@]}" wineboot --init </dev/null
"$@" </dev/null
