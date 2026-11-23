#!/bin/sh
# Builds the macOS host prototype and its Windows test guest. Needs Homebrew llvm and lld.
set -eu
cd "$(dirname "$0")"
LLVM=${LLVM:-/opt/homebrew/opt/llvm/bin}
LLD=${LLD:-/opt/homebrew/opt/lld/bin}
OUT=${1:-build}
mkdir -p "$OUT"
"$LLVM/llvm-dlltool" -m i386:x86-64 -d guest/kernel32.def -l "$OUT/kernel32.lib"
"$LLVM/clang" --target=x86_64-pc-windows-msvc -O1 -ffreestanding -fms-extensions -fno-stack-protector \
  -c guest/guest.c -o "$OUT/guest.obj"
"$LLD/lld-link" /nologo /entry:mainCRTStartup /subsystem:console /nodefaultlib \
  "$OUT/guest.obj" "$OUT/kernel32.lib" /out:"$OUT/guest.exe"
"$LLD/lld-link" /nologo /entry:mainCRTStartup /subsystem:console /nodefaultlib /fixed \
  "$OUT/guest.obj" "$OUT/kernel32.lib" /out:"$OUT/guest_fixed.exe"
/usr/bin/clang++ -arch x86_64 -std=c++17 -O1 -Wall -Wextra host.cpp -o "$OUT/macos_host"
