# Memory tiers: one console memory on a PC with two

The console has one memory that CPU and GPU both address. A discrete PC card
has its own memory behind PCIe, and the host cannot make the two one. What
this project does about it today, and where Direct3D's newer heap types fit.

## What is in place

| Mechanism | What it does | Switch |
|---|---|---|
| Host copies (`shadow.rs`) | each title buffer or 4 MiB window of GPU memory gets a system-memory copy the GPU reads; the title's writes are found by write watching and copied before a submission, shader writes are copied back after | `VICTORY_SWEEP`, window settings |
| Card buffers (`shadow::card_for`) | big copies (16 MiB and up, 512 MiB in all, while the card has room) get a default-heap buffer on the card, with the system-memory copy as staging; changed ranges move by copy commands | `VICTORY_CARD_BUFFERS`, `_MIN_MB`, `_MB`, `VICTORY_CARD_WINDOWS_MB` |
| In place (`inplace.rs`) | a heap opened on the title's own memory, so the GPU reads and writes the title's bytes with no copy (the GPU still reads them over PCIe) | `VICTORY_INPLACE`, `VICTORY_INPLACE_MB` |
| Memory budget | `Host::video_memory` reads the system's budget and use of the card's memory; card copies stop at four fifths of it | |

## GPU upload heaps

Direct3D 12 (Agility SDK 1.613 and later) has `D3D12_HEAP_TYPE_GPU_UPLOAD`:
the CPU maps a buffer and its writes are forwarded over PCIe into the card's
memory. No staging copy, no copy command, and the GPU reads it at full speed.

From Microsoft's specification and the vendors' notes:

- `D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported` says whether the
  device has them. Creating one when it says no fails.
- On a card with its own memory they exist only if all of the card's memory is
  CPU-visible: Resizable BAR on in the firmware setup, and a driver that
  exposes it. Parts with one memory (integrated GPUs) always have them.
- The CPU's pages are write-combined, like an upload heap's. Sequential
  writes are fast; CPU reads are very slow, slower than from an upload heap.
- Retail use needed Windows 11 build 26080 or later when it was announced.

### Why they are not used by the copies yet

The host copies are unordered-access buffers, because shaders may write them,
and the copy-back finds what shaders wrote by comparing bytes on the CPU
(`Shadow::writing`). Both are in doubt here:

1. Whether a GPU upload buffer may have `ALLOW_UNORDERED_ACCESS` is not
   settled by anything the probe's author could confirm; the probe tries it.
2. Reading write-combined memory on the CPU for the comparison is slow.

So the first use would be narrower than the copies as a whole: copies that no
shader writes (vertex, index and constant data), filled with no staging copy
and no copy command. That needs the copies to know ahead that no shader
writes them, which today they do not.

### The probe

`VICTORY_GPU_UPLOAD_PROBE=1` logs, once, a few seconds into the run:

- the adapter's architecture (one memory or two);
- `GPUUploadHeapSupported`;
- whether a GPU upload buffer is made at all, with which initial state, and
  whether one with `ALLOW_UNORDERED_ACCESS` is;
- the card's memory use before and after;
- the CPU's write rate into a GPU upload buffer against an upload buffer,
  and its read rate from the GPU upload buffer.

Lines start with `gpu upload probe:` in the log. It is one pass on one
machine: run it a few times, with Resizable BAR on and off, before reading
anything into the numbers. It was written without a Windows machine to run
it on, so the first thing to check is that it builds and that a run logs.

### What would follow

If the probe says a GPU upload heap is there, unordered access is refused or
the CPU read is slow, and the card path's copy commands show up in the
frame time (`VICTORY_GPU_TIMING`), the narrow use above is worth trying:
a read-only card side filled by the CPU directly, behind its own switch,
with the card path as the fallback. If the heap is not there (Resizable BAR
off, an older driver), nothing changes.
