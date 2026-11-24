#!/usr/bin/env bash
# Start the Windows build with UMU and its Steam Linux Runtime.
set -eu

usage() {
    echo 'Usage: victory-proton.sh TITLE.exe [--execute] [other victory run options]'
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( $# == 0 )); then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ -f $script_dir/victory.exe ]]; then
    default_binary=$script_dir/victory.exe
else
    default_binary=${CARGO_TARGET_DIR:-$script_dir/../target}/x86_64-pc-windows-gnullvm/release/victory.exe
fi
binary=${VICTORY_EXE:-$default_binary}
if [[ ! -f $binary ]]; then
    echo "Missing Windows build: $binary (see Linux / Proton in README.md)" >&2
    exit 1
fi
if [[ ! -f $1 ]]; then
    echo "Missing title executable: $1" >&2
    exit 1
fi
if ! command -v umu-run >/dev/null; then
    echo 'Install umu-launcher to provide umu-run.' >&2
    exit 1
fi
binary=$(realpath -- "$binary")
title=$(realpath -- "$1")
shift

# Hand Proton Windows paths on its standard Z: drive. A Windows executable
# path also lets UMU-Proton attach this CLI to the calling terminal.
export WINEPREFIX=${WINEPREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/victory/proton}
windows_title=${title//\//\\}
windows_binary=${binary//\//\\}
export STEAM_COMPAT_INSTALL_PATH=${STEAM_COMPAT_INSTALL_PATH:-$(dirname -- "$binary")}
exec umu-run "Z:$windows_binary" run "Z:$windows_title" "$@"
