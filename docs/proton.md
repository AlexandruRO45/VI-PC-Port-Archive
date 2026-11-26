# On Linux, under Proton

The title runs on Linux as the Windows build of `victory` under Wine, with
vkd3d-proton as its Direct3D 12. Tested on one machine only (2026-11-26):
CachyOS (kernel 7.3.6), Intel i5-13600KF, 32 GB RAM, Radeon RX 9070 XT 16 GB,
Mesa RADV 26.3.1, the Wine of proton-cachyos 11.0-20261015 started outside
Steam, 1080p, the Series X profile.

## What it needs

| Part | Which | Why |
|---|---|---|
| Wine | a Proton's (11.0 here) | |
| `d3d12.dll`, `d3d12core.dll` | vkd3d-proton `71cae09` (2026-11-25) or later | With `30b0eb9` its shader converter stops at an assertion (`cfg_structurizer.cpp`, "Assertion failed!" in a dialog) in ten pipelines, which are then never made: every frame waits for them, 297 ms a frame against 80. |
| `dxgi.dll` | DXVK's, as Proton has it | |
| `d3dcompiler_47.dll` | Microsoft's (10.0.17134, the file `winetricks d3dcompiler_47` installs) | Wine's own does not build the indirect clamp (issue #183); not tried here. |
| `wineserver` | one that keeps a process's views in a tree (Wine bug 60868, CachyOS/wine-cachyos#50) | Proton's own walks a list at every map and unmap: 510 ms a frame against 80. |

## Build

```sh
rustup toolchain install                 # the version and target pinned in rust-toolchain.toml
cargo install --locked cargo-xwin        # fetches Microsoft's libraries; no Windows, no MinGW
cargo xwin build --locked --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
```

vkd3d-proton, until a release has the fix (needs meson, ninja, mingw-w64-gcc and glslang):

```sh
git clone --recursive https://github.com/HansKristian-Work/vkd3d-proton && cd vkd3d-proton
meson setup --cross-file build-win64.txt --buildtype release --strip -Denable_tests=false -Denable_extras=false build.x64
ninja -C build.x64                       # build.x64/libs/d3d12/d3d12.dll, build.x64/libs/d3d12core/d3d12core.dll
```

The wineserver, from the source your Proton's Wine was built from (proton-cachyos
11.0-20261015: CachyOS/wine-cachyos `cd14106e`). Only the server changes; the
rest of that Proton stays as it is.

```sh
git clone https://github.com/CachyOS/wine-cachyos && cd wine-cachyos
git checkout cd14106e
git fetch origin pull/50/head && git cherry-pick -n FETCH_HEAD
./autogen.sh
mkdir ../wine-build && cd ../wine-build
../wine-cachyos/configure --enable-win64 --disable-tests --without-x --without-wayland --without-freetype \
    --without-alsa --without-pulse --without-gstreamer --without-cups --without-sdl --without-vulkan --without-opengl
make -j"$(nproc)" server/wineserver      # server/wineserver
```

## Set up

```sh
P=/usr/share/steam/compatibilitytools.d/proton-cachyos-slr/files   # your Proton's `files`
V=~/.local/share/victory-linux
mkdir -p $V/bin
cp server/wineserver $V/bin/wineserver
WINEPREFIX=$V/pfx LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu $P/bin/wine wineboot -u >/dev/null 2>&1
cp d3d12.dll d3d12core.dll d3dcompiler_47.dll $P/lib/wine/dxvk/x86_64-windows/dxgi.dll $V/pfx/drive_c/windows/system32/
```

Never send Wine's output through a pipe (`| tee`, `| tail`): its services stay
behind, hold the pipe open, and the shell waits for good. Into a file.

The title's executable must lie in the folder of the package's files, which
becomes the title's `G:`. If your dump keeps them apart, a folder of symbolic
links to all of them does.

## Run

```sh
scripts/play.sh <folder of the package>/<the title's exe>
```

It sets what `scripts\play.ps1` sets, `VICTORY_LAND_ECHO=0` besides (see below),
writes the log to `$V/logs`, and keeps caches and saves in `$V/cache` and
`$V/data`. `--safe` and `--ray-tracing` are as on Windows. `--seconds N` ends a
run after that long. `--mem-max 22G` runs it in a systemd scope that may hold so
much memory and no more, so that a run that grows past it is ended instead of
freezing the desktop. `VAR=value` adds a setting.

Every shader translated ahead of the first run (95 s and 963 MB here):

```sh
WINEPREFIX=$V/pfx LD_LIBRARY_PATH=$P/lib/x86_64-linux-gnu VICTORY_CACHE_DIR="Z:${V//\//\\}\\cache" \
    $P/bin/wine target/x86_64-pc-windows-msvc/diag/victory.exe shaders 'Z:\<folder of the package>\shaders.rpf' >shaders.log 2>&1
```

## Measured

Runs of 330 s with the same scripted keys (`VICTORY_KEYS`): the first-boot
screens, then the opening scene, which the engine draws (1,000 to 6,000 draws a
frame). Medians are of the frame lines with 500 draws or more, about 54 a run.
Frames were saved throughout (`VICTORY_CAPTURE`). Two runs of the same settings
came out at 79.0 and 81.1 ms.

| | |
|---|---|
| Menus | 30 fps |
| The opening scene | 72-81 ms a frame (12-14 fps); "preparing" 53-61 ms of it |
| The same with vkd3d-proton `30b0eb9` | 297 ms |
| The same with Proton's own wineserver | 510 ms, a frame in ten over 3 s |
| Memory of the process in RAM | 13 GB on average, 16 GB at most |
| Of that, the PC's memory the card uses (GTT) | 6.7 GB on average, 9.4 GB at most |
| Memory left to the PC | 24-26 GB before the start, 5-6 GB at the least |
| Swap | under 1 GB |
| The card's memory | full: 14.9 of 15.0 GB |

So 32 GB of RAM are enough for this stretch. The README's 40-46 GB are what
Windows counts as committed; Linux counts only the pages that were touched.
Longer play and the open world were not measured.

## Known to be wrong

- With `VICTORY_LAND_ECHO` on (the default of `play.ps1`), a character's head
  is drawn without skin: eyes and teeth only. Seen in three runs of three, in
  none of two with `VICTORY_LAND_ECHO=0`, which costs no frame time here.
  `play.sh` sets it.
- 160-170 draws a frame are left out as stages of the ray-tracing structure
  builder, which is off without `--ray-tracing`. That was not tried here.
- `VICTORY_WINDOW_KB=1024 VICTORY_WINDOW_TAILS=1` saves no memory: 4,249 windows
  of title memory hold 6,778 MB where 1,424 held 6,724 MB, because more of it
  lies in more than one window (3,501 MB against 1,960 MB).
