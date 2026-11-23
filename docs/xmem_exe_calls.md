# XMem: how the retail executable calls it (RT-5)

`game_scarlett_master.exe` imports 12 functions from `xmem.dll` (`docs/runtime_imports.md`, "The retail executable").
Seven have no shim yet. This page records, from the file alone, how the executable calls them: how many call sites each
has and the constant arguments those sites pass. Nothing was run: `tools/import_call_sites.py` finds the call sites, and
the instructions that load the argument registers before each call were read by hand. No code or data of the executable
is quoted, only the constants a call passes and the counts. Measured at `main` `1cc6e3c1`, SHA-256
`e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993`.

```powershell
python -I tools/import_call_sites.py game_scarlett_master.exe --dll xmem.dll --dll d3d12_xs.dll --dll xg_xs.dll
```

The tool finds each import's thunk (`jmp [rip+disp32]` through its address-table slot) and the `call`/`jmp rel32`
instructions that reach it. Its counts cover direct calls only, and a stray `E8`/`E9` byte could add one, so they are
bounds, not a proof that every site is reached.

| Import | Call sites | Shim at `main` `1cc6e3c1` |
|---|---:|---|
| `XMemVirtualAlloc` | 23 | yes, refuses every one of these 23 (below) |
| `XMemMapPhysicalPages` | 12 | no |
| `XMemAllocatePhysicalPages` | 4 | no |
| `XMemFreePhysicalPages` | 3 | no |
| `XMemMapPhysicalPagesScatter` | 2 | no |
| `XMemAlloc` | 3 | yes |
| `XMemFree` | 3 (2 calls, 1 tail jump) | yes |
| `XMemGetWorkingSetStatistics` | 3 | yes |
| `XMemTransferMemory` | 1 | yes |
| `XMemIsToolsMemAvailable` | 1 | no |
| `XMemAllocDefault`, `XMemFreeDefault` | 1 tail jump each, no call | no |

The executable also exports one symbol, `__XMemAllocationHooks__`.

## `XMemVirtualAlloc`: every site passes an attribute word

The 23 sites pass these constants (`allocation_type`, `attributes`, `protect`; the shim's parameter order is
`address, size, allocation_type, attributes, protect`):

| `allocation_type` | `attributes` | `protect` | Sites |
|---|---|---|---:|
| `0x20403000` | `0x181` | `0x4` | 3 |
| `0x20402000` | `0x21` | `0x4` | 3 |
| `0x20003000` | `0x183` | `0x22004` | 3 |
| `0x20402000` | `0x3` | `0x801` | 2 |
| `0x20402000` | `0x23` | `0x2004` | 2 |
| `0x20402000` | `0x23` | `0x2404` | 1 |
| `0x20003000` | `0x283` | `0x2004` | 1 |
| `0x20003000` | `0x283` | `0x2404` | 1 |
| `0x20401000` | `0x283` | `0x42404` | 1 |
| `0x20406000` | `0x27` | `0x1004`, `0x1404`, `0x8404`, `0x2404`, `0x2004` | 5 (one each) |
| `0x42000` | `0x7` | `0x1` | 1 |

Every `attributes` value is nonzero, and `runtime/shims/xmem.cpp` returns `ERROR_NOT_SUPPORTED` for a nonzero value
("until their host mapping is known"), so none of the 23 calls succeeds there. Past that check the shim would also
refuse the calls with `allocation_type` `0x20003000` (5 sites: the 2 MiB page mode) and those with bits it does not
know: `0x4000` in `0x20406000` (5 sites) and `0x40000` in `0x42000` (1 site).

The `protect` values are not Win32 protections alone. The Windows SDK's `winnt.h` defines the graphics page protections
the high bits match: `PAGE_GRAPHICS_NOACCESS` `0x800`, `_READONLY` `0x1000`, `_READWRITE` `0x2000`, `_EXECUTE_READ`
`0x8000`, `_COHERENT` `0x20000`, `_NOCACHE` `0x40000`, beside `PAGE_WRITECOMBINE` `0x400`. Read that way, `0x2004` is
`PAGE_READWRITE` plus `PAGE_GRAPHICS_READWRITE`, and `0x801` is `PAGE_NOACCESS` plus `PAGE_GRAPHICS_NOACCESS`. That is
an inference from the constants matching the SDK's names, not something the executable states. What the `attributes`
bits mean (`0x1`, `0x2`, `0x4`, `0x20`, `0x80`, `0x100`, `0x200` appear) is **unmeasured**.

## The physical-page family

Four imports share one shape, and it is the Win32 AWE family's (`AllocateUserPhysicalPages`, `MapUserPhysicalPages`,
`FreeUserPhysicalPages`, `MapUserPhysicalPagesScatter`) with the process handle replaced or dropped. These signatures
are **inferred** from the registers loaded before each call; no documentation of these functions is public:

| Function | Inferred signature | Evidence |
|---|---|---|
| `XMemAllocatePhysicalPages` | `BOOL (ULONG_PTR flags, ULONG_PTR* pages, ULONG_PTR* pageArray)` | 4 sites. `pages` points to a stack slot holding the wanted count; `pageArray` is a buffer of `count * 8` bytes. One site compares the count read back with the wanted one, so a partial allocation is possible. `flags` is `0x380`, `0x280` (2 sites) and `0x180` |
| `XMemMapPhysicalPages` | `BOOL (void* virtualAddress, ULONG_PTR pages, const ULONG_PTR* pageArray)` | 12 sites. Four pass a null `pageArray` (an unmap); one maps one page through a one-element array; the result is tested at the sites that read it |
| `XMemFreePhysicalPages` | `? (ULONG_PTR pages, ULONG_PTR* pageArray)` | 3 sites. The count is passed by value (one site frees the sum of four counts it had just unmapped), the array by pointer. Only two of the three sites were read to the argument, so this one is the least certain |
| `XMemMapPhysicalPagesScatter` | `BOOL (void** virtualAddresses, ULONG_PTR pages, const ULONG_PTR* pageArray)` | 2 sites. Shaped like the Win32 call; the result was not followed |

The page unit looks like **64 KiB**: the code that unmaps shifts byte sizes right by 16 before it passes page counts,
and the reservations that precede the map calls use the 64 KiB page mode (`0x20402000`, above). That is inferred from
those two facts. The meaning of `flags`, and whether the page numbers in `pageArray` carry anything beyond an index,
are **unmeasured**.

The sequence in the first function that calls them is: `XMemAllocatePhysicalPages`, then `XMemVirtualAlloc` of a range
with `0x20402000` and attributes `0x23`, then `XMemMapPhysicalPages` of those pages at that range. So the physical-page calls depend on the
attribute check above: until `XMemVirtualAlloc` accepts these attribute words, the map calls are never reached from
there.

The host's own AWE functions need `SeLockMemoryPrivilege`, which an ordinary user does not hold, so a shim would
emulate the pool (a pagefile-backed section of 64 KiB pages, with views placed at the reserved range) instead of
forwarding. That is a design note, not a measurement.

## The other imports

Observed constants, where the window before the call loads one:

| Import | Sites | Constants seen |
|---|---:|---|
| `XMemAlloc` | 3 | attributes `0xf0814000` (2 sites); one size `0x60000`; one size `0x80` |
| `XMemFree` | 3 | attributes `0x3` and `0xf0814000`; one attribute not a constant |
| `XMemTransferMemory` | 1 | direction `1`, size `0x1000000` (16 MiB) |
| `XMemGetWorkingSetStatistics` | 3 | kind `0x2000000a` (2 sites), `0x20000001` |
| `XMemIsToolsMemAvailable` | 1 | arguments not read |

`runtime/shims/xmem.cpp` says one call site "passes `0x5e50000`" as the attribute word. The two `XMemAlloc` sites here
whose attribute is a constant pass `0xf0814000`, not that value, so it comes from another image.

## The boot's call order (measured)

One `gta6rt --run --trace` on the retail executable, `main` `9f53c502e`, own 1.0.2.1 copy, Windows 11 26200, RTX
3070 Ti (driver 32.0.15.9649), `LEONIDA_D3D12=vulkan`, MSVC `RelWithDebInfo`. Before the first allocation the title
calls `d3d12_xs.dll!D3D12ConfigureVirtualMemoryX` once, unimplemented at `main`. The trace keeps only the first call
for each distinct attribute word, so this is the first five distinct attribute words in first-seen order, not the 23
sites:

| # | `attributes` | Size (bytes) | Page mode / type | Outcome |
|---:|---|---|---|---|
| 1 | `0x21` | 25,769,803,776 (24 GiB) | 64 KiB page mode (`0x20402000`) | succeeds |
| 2 | `0x3` | 25,769,803,776 (24 GiB) | 64 KiB page mode (`0x20402000`) | succeeds |
| 3 | `0x183` | 1,228,931,072 (about 1.14 GiB) | 2 MiB page mode (`0x20003000`) | succeeds |
| 4 | `0x283` | 2,305,818,624 (about 2.15 GiB) | 2 MiB page mode (`0x20003000`) | succeeds |
| 5 | `0x181` | 20,971,520 (20 MiB) | 64 KiB committed (`0x20403000`), other caller | succeeds |

The shim now serves all five: #6908 records the words, #6919 adds the 2 MiB page mode, placeholders, physical pages
and the default allocators. The sizes are the bytes the executable requests, not the rounded values the host reserves.
After the five, plain `main` faults on a bad TLS read (module TLS index 0); with the open pull request that fixes that
load, the title instead asks `xgameruntime.dll!XErrorReport` through `GetProcAddress`, which `main` does not serve, and
hits its own `int3`. Both stops are open pull requests, not facts about `main`.

## What this does not show

The order is measured for the first five distinct attribute words, in the section above; what follows them, and whether
the boot reaches every one of the 23 sites, is not. A call site is code that can call, and the trace keeps one event per
distinct attribute word, so a later allocation that reuses a word is missing from this order as well. The next run
(`--trace`, `xmem_attributes` events) extends it.
