# Native Linux runtime

Goal: `victory run <exe> --execute` on Linux as an ELF program, with no
Wine and no Windows DLL. The title's x86-64 code still runs unchanged in
our process; on Linux we also supply what the Windows loader, kernel and
system DLLs supply on Windows. The Windows build keeps working unchanged.

## What changes, and where

| On Windows | On Linux | Where |
|------------|----------|-------|
| `LoadLibraryExW` maps the prepared image | our PE mapper: `mmap`, relocations, protections | `victory-loader/src/linux/image.rs` |
| the TEB, static TLS (`gs:[0x58]`), last error | a TEB at each title thread's GS base (`wrgsbase`, else `arch_prctl(ARCH_SET_GS)`) | `victory-loader/src/linux/teb.rs` |
| `CreateThread` | guest threads: TEB, stack with guard page, TLS callbacks | `victory-loader/src/linux/thread.rs` |
| SEH: the kernel and `ntdll` dispatch faults through `.pdata` | signals to `EXCEPTION_RECORD` + `CONTEXT`, our own dispatcher and unwinder | `victory-loader/src/linux/seh.rs`, `signals.rs` |
| `VirtualAlloc` and friends | the region table over `mmap` (64 KiB granularity, `memfd` for physical pages) | `victory-loader/src/linux/vm.rs` |
| host pass-through (`HostPolicy`) | our implementation of each allowed DLL | `victory-hle/src/linux/<family>_<slice>.rs` |
| host D3D12 and DXGI | host D3D12 on Vulkan (see GPU below) | `victory-gpu` |
| Win32 window, XInput | a Linux window and gamepads | `victory-gpu` (window), `victory-hle` (input) |

`victory-loader::linux` is the platform (memory, threads, exceptions);
`victory-hle::linux` is the Win32 API surface built on it. `victory-hle`
depends on `victory-loader`, never the other way round.

## Rules

- **Calling convention.** Every function title code calls, and every
  title function we call, is `extern "win64"`. `extern "system"` is only
  for callbacks handed to a Windows host API; on Linux it is SysV.
- **Import resolution is unchanged.** `HostPolicy` still decides which
  DLLs pass through; on Linux, `host::imp::resolve` looks the import up in
  `victory-hle::linux` (`lookup`, `lookup_ordinal`). The modules in
  `victory-hle::modules` still win over it, and DLLs the title ships
  (`vcruntime140`, `msvcp140`, ...) are still mapped and run, not
  replaced.
- **Adding a function.** In `victory-hle/src/linux/<family>_<slice>.rs`
  (family: the real DLL name, lower case, without `.dll`): an
  `extern "win64"` function named like the export, listed with
  `hle_export!` in the file's `module()`, and that `module()` added to
  `linux::modules()`. A module's `dlls` lists the DLL and the API sets
  that forward to it (`kernel32.dll`, `kernelbase.dll`,
  `api-ms-win-core-synch-*`). An export by ordinal is named `#<n>`.
- **Guest types.** `DWORD`/`LONG`/`BOOL` are 32 bits, wide text is `u16`,
  `HANDLE` is pointer-sized. Never `c_long` or `wchar_t` (64 and 32 bits
  on Linux). Structures the title reads keep their Windows layout, with
  their size and field offsets checked by tests.
- **Errors.** Report through `victory_loader::linux::teb::set_last_error`
  with Windows error codes; NTSTATUS where the API returns one.
- **Paths.** The `victory-hle` file module rewrites console paths to host
  paths before the Linux layer sees them. The Linux layer takes a host
  path as Windows code spells it: `\` and `/` both separate, no drive
  letter means the current directory.
- **Shared state lives in one place.** One handle table (files, events,
  threads, all waitable through the same `WaitForMultipleObjects`), one
  region table, one TEB layout.
- **No Wine, no Windows DLL, no system DLL forwarding.** Where a Linux
  library already does the job (Vulkan, a D3D12-on-Vulkan library, an
  audio API), call it through its C API.
- **Unsafe code** keeps the repository's `// SAFETY:` comments.
- **Tests run without game data** (`cargo test --workspace` on Linux).
  Guest code under test is a PE we build: `tests/fixtures/hle-probe`
  (`x86_64-pc-windows-msvc`, linked with `lld-link` against an xwin SDK).
  Tests that need the game take `VICTORY_GAME_DIR` and skip when it is
  unset; they assert properties, never snapshots
  ([asset policy](asset-policy.md)).

## GPU

`victory-gpu` translates D3D12.X to host D3D12 and console shaders to
DXIL. On Linux the host D3D12 is a D3D12 implementation on Vulkan built as
a native Linux library and loaded at run time, so the D3D12 code and the
shader translator stay shared. Its COM methods and ours must agree on the
calling convention; check before you bind.

## Status

| Part | State |
|------|-------|
| Workspace builds and tests on Linux | done |
| Title-facing ABI is `win64` | done |
| PE mapper, TEB, guest threads, TLS, entry | done |
| SEH, signals, unwinding | done: dispatcher, unwinder, signals, `Rtl*`/`RaiseException`, crash report (`linux/seh.rs`) |
| kernel32: handles, waits, sync objects and locks, threads, TLS/FLS, thread pool, time, system, process | done (`victory-hle/src/linux/handles.rs`, `kernel32_*.rs`; SuspendThread uses `SIGRTMIN+6`) |
| Region table (`vm.rs`), kernel32 memory, heap, files at host level, text, locale | done: `kernel32_{memory,heap,file,io,path,text,locale}.rs`; overlapped I/O completes inside the call |
| UCRT | every import the title and its DLLs take (`ucrt_*.rs`); C locale only |
| GPU, window, input | not started |
| Audio, network (offline), crypto, COM and the other host DLLs | done: every import of these DLLs resolves. Sound through PulseAudio/PipeWire (`libpulse-simple`, loaded at run time): the game's Spatial Audio stream (`mmdevapi.rs`) and Bink's XAudio2 (`xaudio2.rs`); loopback-only Winsock, WinHTTP that cannot connect, Media Foundation without sources, an empty registry |
| xmem, virtual drives (`files`), persistent local storage | done: xmem on the region table (reservations inside the console's address limits, the physical pool, `alias` for the GPU's page mappings, `MEM_DECOMMIT` to unmap; no placeholders); `G:`/`T:`/storage drives rewritten to Linux folders, names found whatever their case; the title's own `msvcp140` reaches them through its imports |
| Other Windows-only `victory-hle` modules (lifecycle, dstorage, ...) | not started |
