# gta6rt

`gta6rt` is the native Windows runtime for the Xbox Series build. It is deliberately small: the game
executable is still missing from the package used here, so this directory contains the machinery that
can be built and tested without it — the PE loader, the measured import shims, and the GPU/Vulkan test
tools.

## Build

The normal path is the repository build script:

```powershell
powershell -File scripts/build_runtime.ps1
```

It uses CMake + Ninja from Visual Studio when available, otherwise `cl.exe`, and writes
`runtime\build\gta6rt.exe`. The direct CMake path works too:

```powershell
cmake -S runtime -B runtime/build
cmake --build runtime/build --config RelWithDebInfo
```

Requirements: Windows x64, Visual Studio 2022 with the C++ desktop workload. The GPU tools need the
Vulkan SDK; the runtime and CPU tests do not. The loader is x64-only and the PE mapper rejects any
other machine.

## Dry run

```powershell
gta6rt --dry-run <image.exe|image.dll> [--game-dir <package>] [--shims <dll>]
gta6rt --init    <image.exe|image.dll> [--game-dir <package>] [--shims <dll>]
gta6rt --run     <image.exe|image.dll> [--game-dir <package>] [--shims <dll>]
```

`--dry-run` maps the image and the bundled modules it imports, applies base relocations, binds every
import through the shim registry, a host DLL, or another bundled module, parses exports, applies
section protections, and prints a per-module report. It does **not** call `DllMain`, TLS callbacks,
or the entry point, so no image code runs. `--init` additionally runs each loaded DLL's TLS callbacks
and `DllMain(DLL_PROCESS_ATTACH)`, dependencies first. `--run` then calls the image's entry point: the
whole start sequence, in one process.

`--shims <dll>` loads a shim DLL before anything is mapped, so its table (for example the test
harness's `kernel32`/`user32`/`d3d12`/`dxgi` entries) is registered. Real targets link their shims into
`gta6rt` itself; the flag is for the PE harness and for experiments.

The run prints which shim tables made it into the link (`shim tables linked: dxgi.dll=yes ...`), because
a table silently dropped by the linker is the difference between a bound import and a stub.

Every import that cannot be bound is also appended to `<game dir>/missing_imports.log` as `dll!name`,
and the file is rewritten on each load, so it always describes the run that just happened. At run time
a call into an unimplemented import prints `[rt] unimplemented: dll!name` once.

The report line is:

```
<name>  base=<addr> size=<image size> entry=+<rva>  shim=N host=N module=N unimplemented=N
```

Shim and unimplemented imports are listed by name. A non-zero `unimplemented` count is the useful
signal: it names the next shim that is missing. The measured result for the 16 DLLs shipped in the
package is recorded in the runtime-imports documentation.

## Self-check without a game

The PE harness exercises the same chain (loader → shim tables → D3D12/DXGI → Vulkan → window) with no
game files at all:

```powershell
cd runtime/build
gta6rt --dry-run leonida_test_harness.exe --game-dir . --shims leonida_shims.dll
gta6rt --run     leonida_test_harness.exe --game-dir . --shims leonida_shims.dll
```

Expected: `0 unimplemented import(s)`, an animated triangle in a window, and `leonida_window.log` with
`wsi=1 shared_device=1 frames=N` (about 220 frames in 4 s here).

How to test with a game you legally own (files from a developer-mode console or your own disc, no DRM
workarounds) is in [`docs/how_to_test_game.md`](../docs/how_to_test_game.md).

## PE loader

`src/pe_loader.cpp` maps x86-64 images at their preferred base, or elsewhere (with relocations) when
that base is taken, copies headers and sections, applies `IMAGE_REL_BASED_DIR64` relocations, binds
imports by name or ordinal, parses the export table, sets section protections, and registers the
image's x64 unwind table (`.pdata`) with `RtlAddFunctionTable`, so exceptions raised in or passing
through a mapped image are dispatched and stack walks can step over its frames. The table is validated
first (entries in bounds, sorted and non-overlapping) and removed when the loader is destroyed. Modules
are deduped by lower-case file name, so an import graph that reaches the same DLL twice maps it once.

Current limits, by design:

- `DllMain`, TLS callbacks and the entry point are left to the caller; `--dry-run` never runs them.
- The TOC of the game's RPF archives is encrypted and is not read here. The loader works on the PE
  files and on any images the caller passes it.

## Shims

`shims/registry.cpp` holds the static tables. The current shims are:

- `xmem.dll` — `XMemAlloc`, `XMemFree`, `XMemGetWorkingSetStatistics`, and the measured three-argument
  `XMemVirtualQuery(process, address, out)`. Allocations are 64-byte aligned; the working-set and
  virtual-query outputs fill the observed 64-byte structures and return `FALSE`/`0` on a null or bad
  address.
- `d3d12_xs.dll` — `D3D12SerializeVersionedRootSignature` and
  `D3D12CreateRootSignatureDeserializer`, with the console root-signature high byte (`0x03000000`)
  cleared on a copy before forwarding to the PC `d3d12.dll`. `D3D12XboxCreateDevice` wraps a device in
  the measured-slot Xbox-ABI proxy: the host PC `ID3D12Device` by default, or the Vulkan-backed device
  (`shims/d3d12_vulkan_device.cpp`) when `LEONIDA_D3D12=vulkan` is set and the runtime was built with the
  Vulkan SDK and Python 3. The Vulkan device covers command submission, descriptor heaps, render-target and
  depth views, committed resources and copies (uploads, readbacks, BC textures); see `docs/d3d12_vulkan.md`
  section 3.2 for what is missing.
- `dxgi.dll` — `CreateDXGIFactory`, `CreateDXGIFactory1` and `CreateDXGIFactory2` (`shims/dxgi.cpp`,
  registered in `shims/dxgi_shims.cpp` so the linker keeps it): one adapter with its primary monitor as an
  `IDXGIOutput` (`GetDesc`, `GetDisplayModeList` from the display's own modes, `FindClosestMatchingMode`,
  `WaitForVBlank`), and swapchains whose back buffers are D3D12 committed render targets presented through the
  shared-device WSI presenter (`vkCmdCopyImage` into the swapchain image + `vkQueuePresentKHR`), with
  `ResizeBuffers` and borderless `SetFullscreenState`. `PresentX` presents through the same path and composes
  several planes with their source/destination rectangles. `docs/d3d12_vulkan.md` section 4.5 has the details
  and what is still a stub.
- `xinput1_4.dll` (`shims/xinput.cpp`) — `XInputGetState`, `XInputSetState`, `XInputGetCapabilities`,
  `XInputEnable`, `XInputGetBatteryInformation`, `XInputGetKeystroke`. A test injects the pad state with
  `XInputSetTestState`; without an injection an older host XInput (`xinput1_3.dll`, `xinput9_1_0.dll`) is
  forwarded to, so a physical pad works. `xinput_test` goes through the table.
- `kernel32.dll` asynchrony (`shims/async_io.cpp`) — `CreateFile2`, `ReadFileScatter`, `WriteFileGather`,
  `CreateIoCompletionPort`, `GetQueuedCompletionStatus`, `PostQueuedCompletionStatus`, forwarded to the host
  through `GetProcAddress` (never a direct call, which would land back in a shim DLL that exports the same
  names). Files and ports stay host handles. `async_io_test` writes two pages with a gather, waits on the port,
  reads them back with a scatter and round-trips a posted packet.
- `xgameruntime.dll` (`shims/xgameruntime.cpp`) — `XGameRuntimeInitialize`/`Uninitialize`, `XSystemGetDeviceType`,
  `XSystemGetConsoleId`, and the XTaskQueue/XAsync surface over `third_party/xtaskqueue` (Microsoft's
  MIT-licensed implementation). `xgameruntime_test` runs work on the thread pool and a completion on the
  "game loop" thread, through the shim table.

The two serializer functions and `D3D12XboxCreateDevice` are the only `d3d12_xs` imports used by the
shipped DLLs (measured; see `docs/design.md` and `docs/d3d12_vulkan.md`). The game executable's own
imports are unknown until it is available.

## Tests

```powershell
ctest --test-dir runtime/build --output-on-failure
```

On Windows the build configures the CPU tests that do not need a GPU or game files, including the
root-signature forwarding test (`root_sig_test`) and the clear-command test. The Vulkan tools are built
when the Vulkan SDK is found:

- `vk_harness` — hand-written RDNA2 shader programs on a real Vulkan device.
- `vk_texture_test`, `vk_compute_test`, `vk_store_test` — texture, LDS/compute, and buffer-store paths.
- `vk_command_replay_test`, `d3d12_com_test`, `d3d12_vulkan_device_test` — the D3D12 command path on
  Vulkan: replay, the COM command objects, and the Vulkan device behind the Xbox-ABI proxy. Set
  `LEONIDA_VK_VALIDATION=1` to run them under the Khronos validation layer.
- `vk_pipeline_check <spirv dir> [cache.bin] [--jobs N] [--start I] [--count N] [--max-bytes N]` -
  creates a shader module and, for compute shaders, a pipeline for every entry in the translator's
  `index.json`; reports `MODULE FAIL` / `PIPELINE FAIL` per shader and a summary. `--max-bytes`
  names and skips modules larger than the limit, which is how the corpus run works around the
  driver's compile time on multi-megabyte ray-tracing shaders. The measured partial pass is in
  [docs/pipeline_check.md](../docs/pipeline_check.md).

The PE-loader and Bink-decode tests that need the retail package are optional: when
`LEONIDA_GAME_DIR` (and, for Bink, `LEONIDA_MOVIE`) is unset they print `SKIP` and exit 0. They read
the files in place and never copy or commit game data.

## Current limits

The game does not boot yet. The missing executable blocks the import-driven work: the real device
entry point, command-list/queue/fence extensions, `PresentX`, and most of the `xgameruntime` surface (an offline
shim covers `XGameRuntimeInitialize`, device/console queries, `XTaskQueue` and `XAsync`; see `xgameruntime_test`). The
D3D12-to-Vulkan design and the measured console ABI are in
[docs/d3d12_vulkan.md](../docs/d3d12_vulkan.md); the shader translator and its GPU checks are described
in [docs/design.md](../docs/design.md), and the differential-validation design is in the shader-validation
work.

No game data, keys, or extracted assets belong in this directory or in the repository.
