#!/bin/sh
# Runs the retail title through VIctory on Linux, using Proton's Wine so that
# D3D12 goes through vkd3d-proton (host D3D12 is what victory-gpu needs).
#
# Build first (the toolchain pins the target):
#   cargo build --release -p victory --target x86_64-pc-windows-msvc
# with lld-link as the linker and /LIBPATH entries for the xwin SDK, e.g.
#   RUSTFLAGS="-C linker=lld-link -C link-arg=/LIBPATH:$L/msvc ..."
#
# VICTORY_GAME_DIR names the package, PROTON the Proton install, WINE a wine
# binary directly. Everything VIctory writes stays under VICTORY_DATA_DIR and
# VICTORY_CACHE_DIR, outside the repository (docs/asset-policy.md).
set -eu
here=$(dirname "$(readlink -f "$0")")
game=${VICTORY_GAME_DIR:-"$here/../GTA6.Xbox.v1.0.2.1"}
proton_dir=${PROTON:-/usr/share/steam/compatibilitytools.d/proton-cachyos-slr}
exe="$here/target/x86_64-pc-windows-msvc/release/victory.exe"

[ -x "$exe" ] || {
    printf 'no %s: build it with cargo build --release -p victory --target x86_64-pc-windows-msvc\n' "$exe" >&2
    exit 1
}
[ -d "$game" ] || {
    printf 'no game folder %s (VICTORY_GAME_DIR)\n' "$game" >&2
    exit 1
}

export WINEPREFIX=${WINEPREFIX:-"$HOME/.local/share/victory-proton/pfx"}
export WINEDLLOVERRIDES=${WINEDLLOVERRIDES:-"d3d12=n;d3d12core=n;dxgi=n"}
export WINEDEBUG=${WINEDEBUG:--all}
wine=${WINE:-"$proton_dir/files/bin/wine"}

# Convert a Unix path to a Windows path, failing loudly if Wine can't map it.
towin() {
    w=$("$wine" winepath -w "$1" 2>/dev/null | tr -d '\r') || w=
    case $w in
        [A-Za-z]:\\*) printf '%s' "$w" ;;
        *) printf 'winepath did not map %s (got: %s)\n' "$1" "$w" >&2; return 1 ;;
    esac
}

data=${VICTORY_DATA_DIR:-"$(dirname "$here")/victory-data"}
cache=${VICTORY_CACHE_DIR:-"$(dirname "$here")/victory-cache"}
mkdir -p "$data" "$cache"

gamew=$(towin "$game") || exit 1
exew=$(towin "$game/game_scarlett_master.exe") || exit 1

export VICTORY_GAME_DIR="$gamew"
export VICTORY_DATA_DIR=$(towin "$data") || exit 1
export VICTORY_CACHE_DIR=$(towin "$cache") || exit 1

exec "$wine" "$exe" run "$exew" --execute "$@"
