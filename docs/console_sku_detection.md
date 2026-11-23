# How the title decides which console it is running on

**Status: measured 2026-11-23 from the RT-5 executable, static analysis only, nothing executed.**
Nothing here quotes the executable: no addresses, offsets, IID values or bytes; only export names, public SDK
types, vtable slot numbers, call shapes, immediates and counts, the same rule as
[retail_boot_chain.md](retail_boot_chain.md). Registry paths, value names and string literals are quoted because
they are what a host has to match.

The title identifies its hardware from a **single registry DWORD**, and the answer changes a hard assert, two
startup virtual-memory reservations, a set of memory budgets, several render target sizes and which asset
variant it asks for. Nothing in the tree currently reads or sets it, so this is a host-environment knob that is
being left at its default by accident rather than by choice.

## The switch

One function, called once per process and cached in a thread-safe static, does this:

```
RegOpenKeyW(HKEY_LOCAL_MACHINE,
            L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM")
RegQueryValueExW(key, L"ConsoleMode", ..., 4 bytes)
```

and maps the DWORD to a small generation id:

| `ConsoleMode` | id |
|---|---:|
| the key is absent, the value is absent, or either call fails | **1** |
| `0x10`, `0x20`, `0x21`, `0x30`, `0x31` | 2 |
| `0x40` | 3 |
| `0x50` | 4 |
| `0x58` | 5 |
| `0x60` | **6** |
| `0x64` | 7 |
| `0x68` | 8 |
| any other value | **0** |

Both failure paths return 1, so a machine with no such value gets id 1 rather than id 0.

## Consumer 1: a hard assert on the id

The title's virtual-memory configuration function, the one that calls
`D3D12ConfigureVirtualMemoryX` ([d3d12_xs_surface.md](d3d12_xs_surface.md) has its shape), begins by asserting
that the id is one of **1, 6, 7 or 8**, and `__debugbreak()`s on anything else.

| State | id | Result |
|---|---:|---|
| No `ConsoleMode` value, which is every ordinary Windows machine | 1 | **passes** |
| `0x60`, `0x64`, `0x68` | 6, 7, 8 | **passes** |
| `0x10`, `0x20`, `0x21`, `0x30`, `0x31`, `0x40`, `0x50`, `0x58` | 2 to 5 | **fatal assert** |
| Anything else | 0 | **fatal assert** |

**So do not invent a value for this.** The natural absent state is one the title explicitly accepts, and eight
of the recognised values kill it before the device exists. If a host ever does want to set it, the only safe
choices are `0x60`, `0x64` and `0x68`.

That function is also where the two large startup reservations happen, and the id feeds their size:

| | id 6 | every other accepted id, including 1 |
|---|---|---|
| Each of the two `XMemVirtualAlloc` reserves | 16 GiB | **24 GiB** |
| The `D3D12ConfigureVirtualMemoryX` count at `+4` of its 32-byte block | 2 | **3** |

Both reserves use the `0x20402000` form with no `MEM_COMMIT`, with attribute and protect words `0x21`/`0x4` and
`0x3`/`0x801`, which are the two forms [xmem_exe_calls.md](xmem_exe_calls.md) tabulates and
[retail_boot_chain.md](retail_boot_chain.md) stop 1 names. So **stop 1 is reserving two 24 GiB ranges**, not two
small ones, and a host that caps the reservation will fail here rather than later.

## Consumer 2: the Series S predicate

A second wrapper reduces the id to one boolean, `id == 6`, also cached. **63 distinct functions consult it
across 71 call sites.**

What it means is not a guess. One of those call sites selects an asset variant suffix:

```
if (predicate) suffix = ":xbss";
else           suffix = ":xbsx";
```

`xbsx` and `xbss` are the Xbox Series X and Series S platform tags. So:

- **`ConsoleMode` absent, which is the current state of every machine in the tree, means the title runs as a
  Series X.** That is almost certainly what the project wants.
- **`ConsoleMode` set to `0x60` makes it a Series S**, in every one of those 71 decisions at once, including
  which asset variants it asks the archives for.

Ids 7 and 8 pass the assert but do not flip this predicate, so they are Series X variants as far as the title's
per-SKU behaviour is concerned. Whatever distinguishes them is read from the raw id by two other functions, not
from this predicate.

### What the predicate changes

A sample of the 71 sites, to show the range rather than to enumerate it:

| Site | Series X (predicate false) | Series S (predicate true) |
|---|---|---|
| Asset variant suffix | `":xbsx"` | `":xbss"` |
| A standalone budget getter | 64 MiB | 40 MiB |
| The ten-entry budget table | see below | see below |
| The back-buffer arena budget | 64 MiB | about 82 MiB |
| A limit | 455 | 255 |
| Two render target dimensions | as computed | **right-shifted by one**, so half size |
| Two further dimensions | as computed | clamped to 1024 |
| A quality or tier value | 6 | 7, or 6, depending on a second flag |
| A resource flag word | `0` | `8519734` |

The two shift sites are the clearest illustration of why this matters: the predicate is read **inline in the
expression that computes a render target's width and height**, so turning it on does not just change a budget,
it changes the size of surfaces the Vulkan path has to allocate.

### The budget table

One function writes ten budget globals from the predicate, in a single block, early enough that these are the
numbers the rest of the title plans against. All of them are exact multiples of 1 MiB:

| Budget | Series X | Series S |
|---|---:|---:|
| 1 | **6,236 MiB** | **3,977 MiB** |
| 2 | 1,855 MiB | 1,029 MiB |
| 3 | 246 MiB | 96 MiB |
| 4 | 200 MiB | 100 MiB |
| 5 | 64 MiB | 16 MiB |
| 6 | 100 MiB | 100 MiB |
| 7 | 80 MiB | 80 MiB |
| 8 | 8 MiB | 8 MiB |
| 9 | 1 MiB | 1 MiB |
| 10 | 0 | 0 |

Budget 5 is written to two separate globals, so nine distinct values cover ten slots. The Series X column sums to
about 8.65 GiB, which is a plausible whole-title figure for a console with 16 GB of shared memory, and budget 1
on its own is the number a host's memory planning has to accommodate.

These are the title's own budgets, not allocations, so they are not a stop in themselves. They are recorded here
because they say what the title believes it has, and because they are the clearest confirmation that the
predicate selects a whole hardware configuration rather than one feature.

### How the suffix is used

The asset variant suffix is not appended to a path. It is passed, with a name, to a combining function, and the
result is looked up in one of the title's in-memory name tables. A miss returns false and the caller treats that
as "this variant is not present". So per-SKU resources are resolved by name at runtime, and the suffix is part of
the key rather than part of a filename. [resource_names.md](resource_names.md) covers the separate, unrelated
`name.#dr` style type suffixes carried in RPF9 stream data.

## Consequences for the host

1. **Leave the value absent.** It is an accepted state and it selects Series X, the larger configuration.
2. **If a crash appears in virtual-memory setup on a machine that works elsewhere, check this value.** Some
   Windows installations on OEM hardware do have an `OEM` key. A `ConsoleMode` DWORD of `0x40` or similar
   left there by an OEM would make the title assert before the device is created, with no diagnostic, and the
   assert would look like an unrelated allocation failure. Reading the value and logging the id would be a
   cheap addition to the runtime's startup log.
3. **Stop 1's reserves are 24 GiB each.** Two of them, reserve-only, no commit. Worth stating explicitly next to
   stop 1, because "two 64 KiB reserve forms" describes the flags rather than the sizes.
4. **Do not use this to make the title cheaper.** Setting `0x60` to get the Series S budgets also switches the
   asset variant, so the archives would be asked for streams the host may not have extracted, and 71 other
   decisions change at the same time.

## What this does not establish

What distinguishes ids 7 and 8 from 1 was not traced; two functions read the raw id rather than the predicate and
only one of them was read. What each of the ten budgets governs is not identified, only their values and that
they are written together. The 71 call sites were sampled, not enumerated: the table above is 9 of them. The asset
suffix is read as a platform tag from the literals themselves, and how it is combined with an archive entry name
was not followed. Nothing here has been run.
