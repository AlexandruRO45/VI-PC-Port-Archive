# Running on Linux, under Proton

There is no Linux build of `victory`. On Linux the Windows build runs under
Wine, with vkd3d-proton as its Direct3D 12, and the title runs inside it as it
does on Windows. A stock Proton is not enough: two of its parts have to be
replaced, and the first section says which and why. `scripts/play.sh` then
starts the title the way `scripts\play.ps1` does on Windows.

What this page says was last checked on the setup named at its end. Where it
names a version, an older one is known to fail and a newer one should do.

## What a run needs

| Part | Which | Without it |
|---|---|---|
| Wine | A Proton's (11.0 was used), started outside Steam. | |
| `d3d12.dll`, `d3d12core.dll` | vkd3d-proton from 2026-11-25 (`71cae09`) or later. | Its shader converter stops at an assertion in some of the title's pipelines. They are never made, and every frame waits for them: about four times the frame time. |
| `dxgi.dll` | DXVK's, the one the Proton ships. | |
| `d3dcompiler_47.dll` | Microsoft's, 10.0 or later. | Wine's own cannot build the layer's indirect clamp, and most of the world is not drawn (issue #183). |
| `wineserver` | One that keeps a process's mapped views in a tree ([Wine bug 60868](https://bugs.winehq.org/show_bug.cgi?id=60868)). | The stock one walks a list at every map and unmap, and the title maps tens of thousands of views: about six times the frame time, with frames of several seconds. |

The first run's log names what is in use: `vkd3d-proton - build:` is followed by
the commit, `DXVK:` by the version.

## Build

`victory` itself needs nothing but Rust; [cargo-xwin](https://github.com/rust-cross/cargo-xwin)
fetches Microsoft's libraries, so neither Windows nor MinGW is involved.

```sh
rustup toolchain install                 # the version and target pinned in rust-toolchain.toml
cargo install --locked cargo-xwin
cargo xwin build --locked --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
```

### vkd3d-proton

Skip this once your distribution or Proton ships a new enough build. It needs
meson, ninja, a MinGW-w64 GCC and glslang.

```sh
git clone --recursive https://github.com/HansKristian-Work/vkd3d-proton && cd vkd3d-proton
meson setup --cross-file build-win64.txt --buildtype release --strip -Denable_tests=false -Denable_extras=false build.x64
ninja -C build.x64
```

The two files are `build.x64/libs/d3d12/d3d12.dll` and
`build.x64/libs/d3d12core/d3d12core.dll`.

### wineserver

Skip this once your Proton's Wine has the fix. Until then only the server is
rebuilt, from the very source your Proton's Wine was built from: client and
server speak a protocol whose version must match, and Wine refuses to start
when it does not. The patch is
[CachyOS/wine-cachyos#50](https://github.com/CachyOS/wine-cachyos/pull/50).

For proton-cachyos the source commit is the `wine` submodule of the tag named
in the Proton's `version` file:

```sh
P=/usr/share/steam/compatibilitytools.d/proton-cachyos-slr/files   # your Proton's `files` folder
tag=$(cut -d' ' -f2 $P/../version)
commit=$(curl -s "https://api.github.com/repos/CachyOS/proton-cachyos/contents/wine?ref=$tag" | jq -r .sha)

git clone https://github.com/CachyOS/wine-cachyos && cd wine-cachyos
git checkout "$commit"
git fetch origin pull/50/head && git cherry-pick --no-commit FETCH_HEAD
./autogen.sh
mkdir ../wine-build && cd ../wine-build
../wine-cachyos/configure --enable-win64 --disable-tests --without-x --without-wayland --without-freetype \
    --without-alsa --without-pulse --without-gstreamer --without-cups --without-sdl --without-vulkan --without-opengl
make -j"$(nproc)" server/wineserver
```

The result is `server/wineserver`. The `--without` switches only spare the
build libraries the server does not use.

## Set up

`play.sh` keeps everything of its own in one folder, `~/.local/share/victory-linux`
unless `VICTORY_LINUX_HOME` names another: the Wine prefix (`pfx`), the patched
server (`bin/wineserver`), caches, saves and logs. None of it may lie inside
this repository ([asset-policy.md](asset-policy.md)).

```sh
P=/usr/share/steam/compatibilitytools.d/proton-cachyos-slr/files   # your Proton's `files` folder
V=~/.local/share/victory-linux
mkdir -p $V/bin
cp <wine-build>/server/wineserver $V/bin/

WINEPREFIX=$V/pfx LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu $P/bin/wine wineboot -u >/dev/null 2>&1

curl -L -o d3dcompiler_47.dll https://raw.githubusercontent.com/mozilla/fxc2/master/dll/d3dcompiler_47.dll
echo '4432bbd1a390874f3f0a503d45cc48d346abc3a8c0213c289f4b615bf0ee84f3  d3dcompiler_47.dll' | sha256sum -c
cp d3dcompiler_47.dll <vkd3d-proton>/build.x64/libs/d3d12/d3d12.dll <vkd3d-proton>/build.x64/libs/d3d12core/d3d12core.dll \
    $P/lib/wine/dxvk/x86_64-windows/dxgi.dll $V/pfx/drive_c/windows/system32/
```

(The `d3dcompiler_47.dll` above is the file `winetricks d3dcompiler_47` installs.)

The title's executable has to lie in the folder of the package's files: that
folder becomes the title's `G:`. If your dump keeps the executable elsewhere,
make a folder of symbolic links to all of them and start from there. The dump
itself is only read.

## Run

```sh
scripts/play.sh <folder of the package>/<the title's executable>
```

| Argument | |
|---|---|
| `--safe` | Without the newer, faster ways; the first thing to try when something is wrong. |
| `--ray-tracing` | The title's own ray-traced indirect light. |
| `--build release` | `target/.../release` instead of `diag`. |
| `--seconds N` | Ends the run after `N` seconds. |
| `--mem-max 22G` | Runs in a systemd scope that may hold this much RAM and no more. A run that grows past it is ended by the kernel instead of taking the desktop with it. |
| `VAR=value` | One more setting, or another value for one of the script's. |

`PROTON_FILES` names another Proton's `files` folder. The log goes to a file in
`$V/logs`, whose path the script prints first.

The settings are `play.ps1`'s, with one more: `VICTORY_LAND_ECHO=0`. Under Wine,
with that switch on, some characters' heads are drawn without skin.

The first run translates shaders as it meets them. To have all of them
translated ahead, once:

```sh
WINEPREFIX=$V/pfx LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu VICTORY_CACHE_DIR="Z:${V//\//\\}\\cache" \
    $P/bin/wine target/x86_64-pc-windows-msvc/diag/victory.exe shaders 'Z:\<folder of the package>\shaders.rpf' >shaders.log 2>&1
```

## When something is wrong

**A dialog "Assertion failed!" that names `cfg_structurizer.cpp`, and a world at
a few frames a second.** vkd3d-proton is too old. The frame lines of the log
then read `(N being made)` with an `N` that never falls to 0. Do not answer the
dialog: OK ends the run.

**Frames of half a second and more from the first one in the world, though the
card is idle.** The stock wineserver is in use. `play.sh` starts
`$V/bin/wineserver` when it is there; a server left running by another Wine
command on the same prefix is used instead, so end it first
(`WINEPREFIX=$V/pfx $V/bin/wineserver -k`).

**Only menus and flat colour, and `no argument clamp: indirect execution dropped`
in the log.** Wine's own `d3dcompiler_47` is in use (issue #183): the file in
`system32` is not Microsoft's.

**Heads without skin: eyes and teeth only.** `VICTORY_LAND_ECHO` is on, which
it is when the title is started without `play.sh`.

**A shell that never returns after a Wine command.** Its output went through a
pipe (`| tee`, `| tail`). Wine's services outlive the command and hold the pipe
open. Send the output to a file, as `play.sh` does.

**The desktop freezes when memory runs out.** Start with `--mem-max`, a few GB
under what the PC has.

## Memory

The README's figure for memory is what Windows counts as committed. Linux counts
only the pages that were touched, and the same title takes much less there: in
the opening scene the process held 13 GB of RAM on average and 16 GB at most.
With the 7 to 9 GB of the PC's memory that the card uses, about 20 GB were
taken: a PC with 32 GB had 5 GB left at the least. A 16 GB card fills up.

## Last checked

2026-11-26, `81ebb7b`: CachyOS (kernel 7.3.6), Intel i5-13600KF, 32 GB RAM,
Radeon RX 9070 XT 16 GB, Mesa RADV 26.3.1, the Wine of proton-cachyos
11.0-20261015, vkd3d-proton `71cae09`, DXVK 3.1.1, 1080p, the Series X profile.
First-boot screens and the opening scene only; not the open world, not
`--ray-tracing`, no other card.
