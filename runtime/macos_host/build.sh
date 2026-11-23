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
VKH=${VKH:-/opt/homebrew/opt/vulkan-headers/include}
MVK=${MVK:-$HOME/.local/share/moltenvk/MoltenVK/MoltenVK/dynamic/dylib/macOS}  # universal MoltenVK release
/usr/bin/clang++ -arch x86_64 -std=c++17 -O1 -Wall -Wextra -I"$VKH" host.cpp kernel32.cpp kernel32_crt.cpp crt.cpp eh.cpp vulkan.cpp \
  window.mm -framework Cocoa -framework QuartzCore -Wl,-rpath,"$MVK" -o "$OUT/macos_host"

# Leonida's own stand-in game (runtime/test/fixtures/tiny_game.cpp), built as runtime/CMakeLists.txt builds it,
# when an xwin splat is there (docs/testing.md).
XWIN_DIR=${XWIN_DIR:-$HOME/.local/share/xwin-sdk}
[ -d "$XWIN_DIR/VC" ] || exit 0
FIX=../test/fixtures
CL="$LLVM/clang-cl --target=x86_64-pc-windows-msvc /winsysroot $XWIN_DIR /nologo /O2 /MD /utf-8 /EHsc
  /DWIN32_LEAN_AND_MEAN /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS"
$CL /I../../third_party/xtaskqueue/Include /c $FIX/xgameruntime_stub.cpp /Fo"$OUT/xgameruntime_stub.obj"
"$LLD/lld-link" /nologo /dll /def:$FIX/xgameruntime_stub.def /winsysroot:"$XWIN_DIR" \
  "$OUT/xgameruntime_stub.obj" /out:"$OUT/xgameruntime.dll" /implib:"$OUT/xgameruntime.lib"
$CL /c $FIX/tiny_game.cpp /Fo"$OUT/tiny_game.obj"
"$LLD/lld-link" /nologo /winsysroot:"$XWIN_DIR" "$OUT/tiny_game.obj" "$OUT/xgameruntime.lib" /out:"$OUT/tiny_game.exe"
$CL /c guest/eh_probe.cpp /Fo"$OUT/eh_probe.obj"
"$LLD/lld-link" /nologo /winsysroot:"$XWIN_DIR" "$OUT/eh_probe.obj" /out:"$OUT/eh_probe.exe"
printf 'LIBRARY vulkan-1.dll\nEXPORTS\n  vkGetInstanceProcAddr\n' > "$OUT/vulkan-1.def"
"$LLVM/llvm-dlltool" -m i386:x86-64 -d "$OUT/vulkan-1.def" -l "$OUT/vulkan-1.lib"
$CL -I"$VKH" /c guest/vk_clear.c /Fo"$OUT/vk_clear.obj"
"$LLD/lld-link" /nologo /winsysroot:"$XWIN_DIR" "$OUT/vk_clear.obj" "$OUT/vulkan-1.lib" user32.lib /out:"$OUT/vk_clear.exe"
