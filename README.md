# VIctory for Windows

A Windows compatibility project based on [VIctory](https://github.com/Marvisak/VIctory).
It runs an Xbox Series title using files from a copy you supply. The development
and test configuration documented here is the PC listed below.

This repository contains the compatibility code. Game files and assets come
from your own copy. The project is independent of the game's developer and
publisher.

## Development PC

| Component | Configuration |
|-----------|---------------|
| Processor | AMD Ryzen 9 9950X3D (16 cores, 32 threads) |
| Motherboard | MSI MAG X870 TOMAHAWK WIFI (MS-7E51), BIOS 1.A93 |
| Graphics card | NVIDIA GeForce RTX 4090 (24 GB GDDR6X) |
| Graphics driver | NVIDIA 621.40 (WDDM) |
| Memory | 48 GB DDR5 (2 × 24 GB, configured at 8000 MT/s) |
| System storage | Samsung 990 PRO 4 TB NVMe SSD |
| Operating system | Windows 11 Pro 26H2, 64-bit (build 26300.10075) |

## Local test results

The November 24, 2026 test of revision `05a78e3` used the Xbox Series X profile.
It rendered at 2560 × 1440 internally and presented at 3840 × 2160. The command
relay was enabled at level 2. The ray-tracing structure builder and texture
prefetching were disabled.

| Scene | Measurement | FPS | Frame timing |
|-------|-------------|----:|--------------|
| Prologue room, camera still | PresentMon, 268 presents over 35.76 s | 7.47 | 132.2 ms median, 157.2 ms at the 95th percentile |
| Turning the camera in the prologue room | Renderer log, 248 frames over 41.2 s | 6.01 | 166.3 ms average |
| Scene transition | PresentMon, 155 presents over 35.51 s | 4.34 | 152.9 ms median, 521.7 ms at the 95th percentile |

For PresentMon, FPS is the number of intervals between presents divided by the
elapsed time. Frame timing uses the interval between presents. The stationary
capture showed the GPU busy for about 100 ms per frame. The renderer log
reported about 52 ms per frame preparing copies in the same interval.

A separate Series X run on November 25 showed 30 FPS in the early low-draw
samples. Six in-world samples between 00:53:02 and 00:53:29 local time
reported 7.1–7.3 FPS (136.5–141.7 ms per frame).

### Observed rendering problems

The test reached the controllable prologue room and showed the character,
the room geometry, and the control prompt. Camera movement and scene
transitions slowed the frame rate. The captures showed an overexposed window
and sky, stepped shadow edges, missing details near the window, and dark
patches in an exterior shot. See [ray-tracing notes](docs/fork-ray-tracing.md)
for the rendering settings and implementation.

## Credits

- Marvisak and the VIctory contributors built the original compatibility
  layer, including the loader, runtime services, Direct3D 12 backend, and
  shader recompiler. [VIctory](https://github.com/Marvisak/VIctory)
- fxceless and the contributors to the Windows rendering fork developed
  rendering, shader translation, and texture upload improvements used here.
  [VIctory rendering fork](https://github.com/fxceless/victory-fork)
- uiliannn and the VItoriaBr contributors shared fixes for GPU synchronization,
  texture uploads, descriptor allocation, shader translation, and XMA audio.
  [VItoriaBr](https://github.com/uiliannn/VItoriaBr)
- ivall and the XCL contributors provided a reference for title behavior.
  The related changes in this repository are implemented in VIctory.
  [XCL](https://github.com/ivall/xcl)
- TaynanGT maintains this repository and documents testing on the development
  PC. Individual code contributions are credited in the
  [commit history](https://github.com/TaynanGT/victory-windows/commits/local/fxceless-render).

The compatibility code is licensed GPL-3.0-or-later. See [LICENSE](LICENSE).

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
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

Input: XInput pads. While the title's window has focus, the keyboard
supplements the first connected pad (including Enter = A). With none
connected, it drives a stand-in pad. Keyboard directions take priority
while pressed; releasing them restores the controller's stick values:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

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
| `VICTORY_PIPELINE_WAIT_MS=24` | reduce the per-frame wait for background pipeline creation from the default of 500 ms. A draw can be omitted until its pipeline is ready |
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
