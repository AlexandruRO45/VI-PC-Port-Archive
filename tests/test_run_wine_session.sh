#!/usr/bin/env bash
# Exercise the wrapper without Wine or a display.
set -eu
wrapper="$(dirname "$0")/../scripts/run_wine_session.sh"
scratch=$(mktemp -d)
trap 'rm -f -- "$scratch/calls" "$scratch/stderr"; rmdir -- "$scratch"' EXIT
export WINEPREFIX="$scratch"
wineboot() { echo init >> "$WINEPREFIX/calls"; }
wineserver() { echo stop >> "$WINEPREFIX/calls"; }
timeout() { shift; "$@"; }
export -f wineboot wineserver timeout

status=0
bash "$wrapper" 2>"$scratch/stderr" || status=$?
[[ $status == 2 && ! -e "$scratch/calls" ]]
grep -qi usage "$scratch/stderr"

status=0
bash "$wrapper" bash -c 'exit "$1"' _ 7 || status=$?
[[ $status == 7 && $(cat "$scratch/calls") == $'init\nstop' ]]
