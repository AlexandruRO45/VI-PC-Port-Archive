# Call shapes of the 23 unimplemented Xbox-only imports

**Status: measured 2026-11-23 from the RT-5 executable, static analysis only, nothing executed.**
Nothing here quotes the executable: no addresses, offsets, IID values or bytes; only export names,
argument counts, argument roles, observed immediates, and call-site counts. This follows the same rule as
[retail_boot_chain.md](retail_boot_chain.md).

This page complements [runtime_imports.md](runtime_imports.md), which lists *which* imports are unimplemented,
and [audio_hal_exe_calls.md](audio_hal_exe_calls.md), which records call-site counts. What it adds is the
arity and in/out role of each argument, so a stand-in can be declared with the right signature before the
title reaches it.

## Reachability

All 23 unimplemented imports have live code cross-references. None is dead code, so each one needs a
stand-in eventually. There are 37 call sites in total, and every import is reached through only one to four
thin wrapper functions, so the surface to implement is 23 signatures rather than five subsystems.

Note that none of these appears in [retail_boot_chain.md](retail_boot_chain.md) as of stop 13. They are
required, but not yet reached: the audio set in particular sits behind an initialiser that has no direct
caller (it is invoked through a pointer), so it is not on the path measured so far.

## `AcpHal.dll` (9)

| Export | Args | Shape |
|---|---:|---|
| `AcpHalCreate` | 1 | out: pointer-sized object handle written to a global. Returns a status tested `< 0` |
| `AcpHalAllocateShapeContexts` | 1 | out: 16-byte structure written to a global. Status tested `< 0` |
| `AcpHalReleaseShapeContexts` | 0 | teardown counterpart of the above |
| `ApuCreateHeap` | 2 | in: a byte count computed as a megabyte value shifted left 20; in: a second fixed byte count. Status tested `< 0` |
| `ApuAlloc` | 5 | in/out: allocation handle; a zero; in: size; in: a fixed alignment immediate of 2048; in: caller flags |
| `ApuFree` | 1 | in: the handle from `ApuAlloc` |
| `ApuIsVirtualAddressValid` | 2 | in: a CPU address; in: a size class. Observed size classes are 64, 128 and 2048. Returns a boolean used as `unsigned __int8` |
| `ApuMapVirtualAddress` | 1 | in: a CPU address. Returns a **32-bit** value stored as `_DWORD`, so the result is an APU-space address, not a pointer |
| `ApuMapApuAddress` | 4 | in: the 32-bit value previously produced by `ApuMapVirtualAddress`, plus three further arguments. The inverse direction of the mapping |

The pairing of `ApuMapVirtualAddress` (returns 32-bit) with `ApuMapApuAddress` (consumes it) is the load-bearing
detail: the APU address space is 32-bit and distinct from the CPU one, so a stand-in cannot return a truncated
host pointer and must hand back a handle it can map back.

## `xapu.dll` (5)

| Export | Args | Shape |
|---|---:|---|
| `XApuConnect` | 3 | in: a 16-byte structure; in: a second 16-byte structure; out: a handle written to a global. Status tested `>= 0` |
| `XApuEnqueueCommand` | 3 | in: the connect handle; in: a command buffer; in: a zero. Four call sites, all with the same shape |
| `XApuGetResultCount` | 2 | in: the connect handle; in: a zero. Returns a count |
| `XApuDequeueResult` | 2 | in: the connect handle; out: a result slot |
| `XApuDisconnect` | 0 | teardown |

The handle `XApuConnect` writes is consumed by the other four, so `XApuConnect` must succeed and return a
distinguishable non-null handle before any of them is called.

## `chad.dll` (5)

| Export | Args | Shape |
|---|---:|---|
| `XDspConnectWithMaximumStreamLimit` | **6** | in: a pointer from the caller's object; in: a 32-bit field from it; in: a caller-supplied stream limit; in: a fixed immediate of 128; in: a fixed immediate of 1; out: a slot in the caller's object. Success tested `== 0` |
| `XDspActivate` | 4 | four caller arguments, with the third and fourth passed in swapped order relative to the wrapper's own parameters. Status tested as non-zero for success |
| `XDspSubmitCommand` | 2 | the wrapper's two arguments, passed in reverse order. Success tested `== 0` |
| `XDspDeactivate` | 0 | teardown |
| `XDspDisconnect` | 0 | teardown |

`XDspConnectWithMaximumStreamLimit` taking six arguments is worth recording: the name suggests a stream limit
and little else, and a two or three argument guess would corrupt the stack on a `__fastcall` stand-in that
writes through its out parameter.

Both `XDspActivate` and `XDspSubmitCommand` pass arguments in an order that differs from their wrapper's
parameter order, so the wrapper signature is not a guide to the export signature.

## `xg_xs.dll` (2)

| Export | Args | Shape |
|---|---:|---|
| `XGCreateTextureComputer` | 2 | in: a descriptor; out: a computer object. Four call sites, all two arguments. A factory returning a status cast to `unsigned int` |
| `XGSuggestSwizzleMode` | 3 | in: a descriptor; in: a zero at every call site; out: a slot inside the caller's object. Two call sites |

Both are plain factories rather than COM interfaces: the out parameter is written directly and no interface id
is passed, so a stand-in needs an object whose own vtable the title then calls.

## `XCRDAPI.dll` (1)

| Export | Args | Shape |
|---|---:|---|
| `DStorageGetFactory` | 2 | in: an interface id held in read-only data; out: a global, which is loaded afterwards and tested for null |

This matches the public `DStorageGetFactory(REFIID, void**)` shape already recorded in
[audio_hal_exe_calls.md](audio_hal_exe_calls.md). The interface id the title requests is a single fixed
constant, so one stand-in interface covers it. Boot stop 5 in
[retail_boot_chain.md](retail_boot_chain.md) is this import returning null.

## `d3d12_xs.dll` (1)

| Export | Args | Shape |
|---|---:|---|
| `D3D12ConfigureVirtualMemoryX` | 1 | in: a descriptor assembled on the caller's stack. Status tested `< 0`, and the caller treats failure as fatal |

## What this does not establish

Argument counts and roles come from the call sites, not from any declaration, so an argument the title never
varies could be a default rather than a required one, and a trailing argument the title never passes would not
appear here at all. No return value is known beyond how the title tests it. Nothing on this page has been run:
these are shapes a stand-in must match, not behaviour it must reproduce.
