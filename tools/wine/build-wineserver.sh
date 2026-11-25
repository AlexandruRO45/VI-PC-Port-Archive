#!/bin/sh
# Builds a wineserver with the view-tree patch (wineserver-view-tree.patch.txt)
# from the Wine source a Proton build was made from.
#
# Usage: build-wineserver.sh <git url> <tag or branch> <work dir>
# Example (proton-cachyos 11.0):
#   build-wineserver.sh https://github.com/CachyOS/wine-cachyos.git \
#       cachyos-11.0-20260702-proton-native ~/wineserver-build
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
if ! git apply --check --reverse "$here/wineserver-view-tree.patch.txt" 2>/dev/null; then
    git apply "$here/wineserver-view-tree.patch.txt"
fi
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
