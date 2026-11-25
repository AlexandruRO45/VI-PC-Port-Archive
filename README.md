# Vice-Tay

A Windows compatibility project based on [VIctory](https://github.com/Marvisak/VIctory).
It runs an Xbox Series title using files from a copy you supply. The development
and test configuration documented here is the PC listed below.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can (shaders and pipelines too, see Getting started) and on first use
otherwise. See [docs/architecture.md](docs/architecture.md).

This repository contains the compatibility code. Game files and assets come
from your own copy. The project is independent of the game's developer and
publisher.

## Compatibility fixes (2026-11-25)

- Save recovery limits cleanup to the staging names it owns. A staging
  file locked by another process leaves the last committed save readable.
  A write can be retried when the file is released. See
  [VIctory #2716](https://github.com/Marvisak/VIctory/pull/2716).
- Shader fixes improve normalized buffer conversion and declare the required
  feature for writable views in vertex, geometry, and mesh shaders. The
  compute test harness clears its output buffers and textures before dispatch.
- Texture copies between aliases of the same host subresource avoid invalid
  Direct3D 12 commands. Capture logs include alpha ranges and traced barriers.
- The loader releases the source executable buffer as soon as its prepared
  image is available.
- Pixel shader quad reads reach helper lanes at triangle edges. A helper
  lane in a first-lane read uses its own value. The CPU rectangle fallback
  completes the rectangle opposite its right angle in any input order.
- The wave-wide shader path supports independent branches for multiple
  console waves in one group. Descriptor synchronization checks records
  one page at a time under one lock. Diagnostics identify wave64 refusals,
  skipped draws and dispatches, and null buffer views.
- Host constant-buffer views use at most 64 KiB and apply 256-byte
  alignment after the limit. Oversized guest views can still use their
  host-addressable prefix.
- The fork uses VIctory's Clippy lint settings. The optional
  `VICTORY_LIVE` file changes selected performance settings for comparisons
  in one session. See [live switches](docs/live-switches.md).

These changes use the [VIctory main snapshot](https://github.com/Marvisak/VIctory/commit/13f91caee25f2fdf2c062be5f7a27f058e51cbca)
and the [fxceless texture-copy fix](https://github.com/fxceless/victory-fork/commit/a25370661ed1343e2f2d5e79669ec8db4e60dc68)
as references. The CPU rectangle fix follows
[VIctory `8ff7913`](https://github.com/Marvisak/VIctory/commit/8ff7913).
The pixel fixes come from VItoriaBr
[`fe0506a`](https://github.com/uiliannn/VItoriaBr/commit/fe0506a) and
[`2b3c339`](https://github.com/uiliannn/VItoriaBr/commit/2b3c339).
The constant-buffer fix follows
[VItoriaBr `ed6e5af`](https://github.com/uiliannn/VItoriaBr/commit/ed6e5af).
The wave and descriptor work, diagnostics, live switches, and lint
alignment come from [Vice-Tay #11](https://github.com/TaynanGT/Vice-Tay/pull/11),
which includes [#6](https://github.com/TaynanGT/Vice-Tay/pull/6).
The run measurements below identify their own revisions.

The integration is checked with `cargo test --workspace --locked` and
`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings`.
The synthetic shader comparisons across triangle edges use
`VICTORY_CONSOLE_DXC` and a PC `dxc.exe`, as described in the shader tests.
See [Windows CI](https://github.com/TaynanGT/Vice-Tay/actions/workflows/ci.yml)
for the result on each revision.

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

## Upstream status (2026-11-24)

**Not playable yet.** The title boots, runs its menus at its own 30 fps,
loads the world and shows it: slowly, and with rendering defects.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | drawn and shown: streets, buildings, vehicles, characters, the title card and credits; 1-2 s per frame. Surfaces are unlit, a band of distant geometry is missing, characters come out broken in some shots |

> **Warning.** Entering the world on a real GPU has hung it in earlier
> builds, and Windows then resets the display driver (every monitor goes
> black for a moment). The latest runs did not, but it is not ruled out; a
> run can also end on a GPU fault (`crashes.log`). A diagnostics build with
> `VICTORY_D3D12_WARP=1` renders on the software adapter instead (too slow
> to reach the world in minutes).

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | The scene is drawn and presented: mesh pipelines, indirect execution, views the shaders pick from the heap themselves, title memory read by address (ray-tracing structures), depth written from compute, pipelines pre-built or made by worker threads and kept on disk | Lighting (surfaces are unlit) and some missing geometry; frames take 1-2 s; draws of untranslated shaders are dropped; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL, ahead of time (`translate --cache`) or at first use, cached on disk. Menus: all. World: mesh 8,765 of 8,765, vertex/pixel 12,550 of 12,592, compute 2,579 of 2,637 | Merged geometry stages (195 programs) and about 100 others; fused multiply-adds follow the host GPU; some compute programs need a GPU with 64-lane waves ([wave-wide form for 32-lane GPUs](docs/wave64-on-wave32.md)) |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

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
  [commit history](https://github.com/TaynanGT/Vice-Tay/commits/local/fxceless-render).

The compatibility code is licensed GPL-3.0-or-later. See [LICENSE](LICENSE).

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

To install Rust, use rustup (`winget install --id Rustlang.Rustup -e`, or
the installer from https://rustup.rs). The first `cargo` command in this
repository then fetches the pinned toolchain. Building needs the MSVC C++
build tools: Visual Studio or its Build Tools with the "Desktop development
with C++" workload.

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
target\release\victory.exe translate --cache --pipelines <game dir>\shaders.rpf <game dir>\<the title's exe> <game dir>\XboxBvhBuilder.dll <game dir>\amd_fidelityfx_upscaler_xbox.dll
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The `translate --cache` step (minutes; again after a change of the shader
translator) translates every shader the title can ask for into the cache
(about 1.3 GB), so a run does not translate them while it plays; without it
they are translated at first use, and their draws wait. With `--pipelines`
it then makes every pipeline in the title's shader pack (inside
`shaders.rpf`) into the host pipeline library (about 27,000 pipelines, 4 MB;
some 20 minutes on 16 threads, most of it the 2,700 compute ones), so a run
compiles none of those either; the library belongs to the GPU and driver it
was made on. Run again, it skips what is there in seconds; stopped
early, it keeps what it made.

Upscaling (Series S mode): the title upscales with AMD FidelityFX. To run
it with the PC build on your GPU instead of the console build, get the
[FidelityFX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
and point `VICTORY_FIDELITYFX_DIR` (or `[paths].fidelityfx_dir`) at its
`Kits/FidelityFX/signedbin` folder, which holds
`amd_fidelityfx_upscaler_dx12.dll`. Without it, or if that DLL offers no
upscaler for your GPU (the log says so), the title's own DLLs run.

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Set `VICTORY_WINDOW_MAX=1920x1080` before launching to limit the title's
top-level window to that bounding box. Smaller windows keep their size,
larger windows keep their proportions, and the monitor's work area still
limits them. This scales the displayed image; it does not change the
console render resolution or the swap-chain buffers. With the variable
unset, window sizing is unchanged.

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
victory shader-info [paths...] [--verbose] [--disasm] [--effects]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory translate [paths...] --cache [--pipelines] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

### Windows shader-test validator

The shader tests use `dxil.dll` to validate Shader Model 6.6 containers.
An older Windows SDK validator can load successfully but reject them with
`Unknown shader model 'ps_6_6'`. If this happens, use a current x64 validator
from [Microsoft's DirectX Shader Compiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases)
or a current Windows SDK. Set its path for the test session:

```powershell
$env:VICTORY_DXIL = 'C:\tools\dxc\bin\x64\dxil.dll'
cargo test -p victory-shader --lib --locked -- --test-threads=1
```

Without this override, the validator loader selects the newest installed
Windows SDK's `bin/<version>/x64/dxil.dll`, then falls back to the DLL search
path when no SDK copy is found. This is a test/offline-validation dependency;
the translated shaders do not require it at runtime.

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

The on/off switches below (`...=1`) are off when unset or set to `0`.

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_POKE=<rva>:<byte>,...` | writes a title byte twice a second (a hang watchdog's tick, say) |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_BREAKPOINTS=1` | appends first-chance breakpoints to `crashes.log`, including ones a handler may recover from |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_CAPTURE_WORLD=<list>` | with `VICTORY_CAPTURE`, also saves (and with `VICTORY_DUMP_TARGETS` dumps) the frames at these offsets from the first frame that ran 500 draws (the world's first; its number depends on how fast the title loaded) |
| `VICTORY_DUMP_WRITERS=<hash>,...` | with the two above, also the textures a pipeline containing a listed shader hash writes, the first frame it runs, and for its buffers a log line (dwords changed, dwords not zero) |
| `VICTORY_PEEK=<hex address>[:<bytes>],...` | in listed frames, those bytes of title memory logged after each submission. Listed frames also log targets of 16 texels or fewer as values, each table view's view words and whether its host slot is null, the host copy of every buffer of 64 bytes or fewer a draw reads (at submission and after its list), and each 1D or 3D texture a dispatch writes, read back right after it |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_PIPELINE_WAIT_MS=24` | reduce the per-frame wait for background pipeline creation from the default of 500 ms. A draw can be omitted until its pipeline is ready |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands; a device removal names the list it happened in and lists all its commands |
| `VICTORY_TRACE_SHADERS=<hash>,...` | with `VICTORY_D3D12_DRED`, every draw and dispatch of a listed shader logged with what it binds (user data, heap slots, root views and their first dwords; the first 500) |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_PIPELINE_LOG=<file>` | a line per title pipeline created (key, shaders, state), first used, and made or loaded (library name); `translate --cache --pipelines` writes the last kind too |
| `VICTORY_PREBUILD_KEYS=<file>` | `translate --cache --pipelines` makes nothing and lists each pack record's key, shaders and state (or why it is skipped) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

Every build appends unhandled exceptions, panics, and host GPU failures to
`crashes.log` in the title's data folder.

For a startup breakpoint (`0x80000003`), run the diagnostics build with
`VICTORY_CRASH_BREAKPOINTS=1`. Keep the console output along with the new
`crashes.log` entries. A host DLL refusal includes the caller stack in the
console output. The refusal is expected for DLLs outside the host policy,
including `rpcrt4.dll`.

On Series 1.0.2.1, a missing `appdata0.rpf` reproduced the breakpoint at
`+0x2237178` before the first window. Restore the archive from the same
package beside the title executable. `run --execute` checks that it can
read the RPF9 header before starting this build.

To compare hardware and software rendering, repeat with
`VICTORY_D3D12_WARP=1` and check the console for
`host device on the software adapter (WARP)` to confirm the WARP device was
created. Compare the exception's module and offset, stack frames, and
preceding messages across the two runs.

## Performance notes

- [Live switches](docs/live-switches.md): compare two settings in one session (`VICTORY_LIVE`, `VICTORY_METER_MS`).
- [Raw descriptor sync](docs/descriptor-sync.md): what it costs per frame and its switch (`VICTORY_HEAP_SYNC`).
- [64-lane programs on 32-lane GPUs](docs/wave64-on-wave32.md): the wave-wide form (`VICTORY_WAVE_WIDE_ONLY`).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
