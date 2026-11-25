#!/bin/sh
# Builds a wineserver with the view-tree patch from the Wine source a Proton
# build was made from. The patch is ported to several trees
# (wineserver-view-tree-*.patch.txt); the first that applies is used.
#
# Usage: build-wineserver.sh <git url> <tag or branch> <work dir>
# Examples:
#   proton-cachyos 11.0: https://github.com/CachyOS/wine-cachyos.git cachyos-11.0-20260702-proton-native
#   Valve Proton 10:     https://github.com/ValveSoftware/wine.git proton_10.0
#   Proton Experimental: https://github.com/ValveSoftware/wine.git experimental_11.0 (the 11.0 patch)
#   upstream Wine:       https://gitlab.winehq.org/wine/wine.git master
# GE-Proton 11 builds from Valve's wine plus GE's patches: the
# 11.0 patch applies to it.
#
# The server must speak the same protocol as the Proton it runs with: build
# from the source of that exact Proton release. The result is
# <work dir>/build/server/wineserver.
set -eu
url=$1 ref=$2 work=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$work"
cd "$work"
if [ ! -d src ]; then
    git clone --depth 1 --branch "$ref" "$url" src
fi
cd src
applied=
for p in "$here"/wineserver-view-tree-*.patch.txt; do
    if git apply --check --reverse "$p" 2>/dev/null; then
        applied=$p; break
    fi
    if git apply --check "$p" 2>/dev/null; then
        git apply "$p"; applied=$p; break
    fi
done
[ -n "$applied" ] || { echo "no view-tree patch applies to this tree" >&2; exit 1; }
echo "using $(basename "$applied")"
[ -x configure ] || ./autogen.sh
grep -m1 "define SERVER_PROTOCOL_VERSION" include/wine/server_protocol.h
mkdir -p ../build
cd ../build
# Only the server is built, so nearly every library is left out.
../src/configure --enable-win64 --without-x --without-freetype --without-mingw \
    --without-gstreamer --without-wayland --without-vulkan --without-opengl \
    --without-alsa --without-pulse --without-cups --without-dbus --without-gnutls \
    --without-sane --without-usb --without-v4l2 --without-krb5 --without-netapi \
    --without-pcap --without-sdl --without-udev --without-unwind --without-capi \
    --without-gphoto --without-ffmpeg > configure.log
make -j"$(nproc)" server/wineserver > make.log
echo "built $(pwd)/server/wineserver"
