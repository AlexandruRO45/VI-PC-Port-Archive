# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

The Windows executable can also be launched under Proton on x86-64 Linux. See
[Linux / Proton](#linux--proton) for the build, launcher, and verification scope.

## Windows status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `VICTORY_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk; occlusion, pipeline statistics and timestamp queries run on the host GPU | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; depth bounds, HDR and scaling are missing; console-only query types get stand-in results |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

## Linux / Proton

Install `umu-run` and use an AVX2 CPU with a Vulkan driver that exposes
Shader Model 6.6 through vkd3d-proton.
[UMU installation](https://github.com/Open-Wine-Components/umu-launcher#installation)

To build from source, install Rust with rustup and the x86-64 Linux UCRT
distribution of LLVM-MinGW. Put LLVM-MinGW's `bin` directory first on `PATH`. The loader
relies on native Windows thread-local storage when a title DLL calls back
into Rust during thread startup. The `gnullvm` target provides that layout.
[LLVM-MinGW releases](https://github.com/mstorsjo/llvm-mingw/releases)

```sh
rustup target add x86_64-pc-windows-gnullvm
cargo build --locked --release -p victory --target x86_64-pc-windows-gnullvm
./tests/proton-smoke.sh
./scripts/victory-proton.sh "/path/to/game/title.exe" --execute
```

Supply the title's executable from your game directory outside the checkout.
Omit `--execute` to check mapping and import binding first. The launcher
accepts the remaining `victory run` options and forwards arguments after
`--`. Set `VICTORY_EXE` to use an existing Windows build.

On first launch UMU fetches its Proton build and Steam Linux Runtime.
The default prefix is `${XDG_DATA_HOME:-$HOME/.local/share}/victory/proton`.
Saves and converted data live under the prefix's Windows user profile.
`WINEPREFIX` selects another prefix and `PROTONPATH` selects another Proton
build. Use Windows paths in `victory.toml`.

The smoke test loads a synthetic executable, calls both an Xbox runtime
function and a host function, and checks the title's return value. The
game's menu and world still need verification under Proton. The Windows
status above describes the tests on Windows.

## Getting started on Windows

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

On Windows 10 the in-box D3D12 runtime stops at shader model 6.5, so the host
refuses every translated pipeline. `cargo xtask agility` fetches Microsoft's
D3D12 Agility SDK runtime (a pinned, hash-checked NuGet package) into
`target\<profile>\D3D12\`, where `victory.exe` makes D3D12 load it. Windows 11
needs nothing.

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo xtask agility                      # Windows 10 only: newer D3D12 runtime
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Building and running on Linux

VIctory is a Windows program; on Linux it is cross-compiled for
`x86_64-pc-windows-msvc` with [cargo-xwin](https://github.com/rust-cross/cargo-xwin)
and run under Wine with VKD3D-Proton (Direct3D 12) and DXVK (DXGI). A native
Linux build is not supported. Wine's own D3D12 cannot run the translated
shaders (it rejects the shader model 6.6 query).

Tools: rustup (it reads `rust-toolchain.toml`), cargo-xwin, clang (for
`clang-cl`), lld (`lld-link`), llvm (`llvm-lib`) and Wine. On NixOS or with
Nix, `nix-shell` at the repository root provides them and sets up the
variables below. Elsewhere, install them from your distribution
(`cargo install cargo-xwin` for cargo-xwin) and export the variables yourself.

```sh
nix-shell                                       # or set the variables in shell.nix
rustup show                                     # installs the pinned toolchain and target
cargo xwin build --release --target x86_64-pc-windows-msvc
```

The first build downloads the MSVC CRT and the Windows SDK; using them
means accepting Microsoft's license.

A Wine prefix of its own (`shell.nix` uses `~/.local/share/victory-wine`):

```sh
export WINEPREFIX="$HOME/.local/share/victory-wine"
wineboot -i
# Copy the x64 DLLs of a VKD3D-Proton release (d3d12.dll, d3d12core.dll) and
# a DXVK release (dxgi.dll, d3d11.dll, d3d10core.dll) into
# "$WINEPREFIX/drive_c/windows/system32", then prefer them over Wine's:
export WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3d11,d3d10core=n"
```

Run the title:

```sh
wine target/x86_64-pc-windows-msvc/release/victory.exe run "<game dir>/<the title's exe>" --execute
```

Tests run under Wine too
(`cargo xwin test --workspace --target x86_64-pc-windows-msvc` with
`CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=wine`). Under Wine these fail
for reasons of the environment: `load_probe` and the policy test (they
start `cargo` and `git` from inside Wine), and `stub_pipeline_is_created`
(VKD3D-Proton does not check DXIL hashes); run `cargo xtask policy`
natively. Three GPU tests also fail under VKD3D-Proton and are not yet
explained: `buffer_data_reaches_the_host_texture`, `argument_clamp_builds`
and `a_rejected_blob_gives_an_empty_library`.

Tested on NixOS with Wine 11.21, VKD3D-Proton 3.0.1, DXVK 3.1.1 and an
NVIDIA RTX 5060 Ti: the title starts and renders its first screens.

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
