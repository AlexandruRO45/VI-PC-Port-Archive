# How the title uses its descriptor heaps

**Status: measured 2026-11-23 from the RT-5 executable, static analysis only, nothing executed.**
Nothing here quotes the executable: no addresses, offsets, IID values or bytes; only export names, public SDK
types, vtable slot numbers, call shapes, immediates and counts, the same rule as
[retail_boot_chain.md](retail_boot_chain.md). Runtime structure field offsets are given where a shim has to
match them.

This is the other half of [descriptor_heap_allocation_hook.md](descriptor_heap_allocation_hook.md), which covers
where the first heap's memory comes from. This page covers what the title then does with it, because the answer
constrains any descriptor model a Vulkan-backed device can use.

## The one-sentence version

The title sub-allocates descriptors itself, by pointer arithmetic on the slot 15 increment inside the heap's CPU
allocation, keeps its free list in that same memory, and calls the device only to fill in each descriptor's
contents. **Nothing on the path traced here tells the device that a descriptor slot has been taken or released**:
the reservation, the free list and every pop and push happen in the title's own memory.

That is a statement about the four heaps that get a free list, not about every heap. #9368 observes at runtime
that the title stages samplers with `CopyDescriptorsSimple`, and the two findings fit together rather than
conflicting: the shader-visible `SAMPLER` heap is the one heap of the five that gets no free list, only a plain
bump reservation, so it is the one heap whose contents arrive through a copy call instead of being written in
place.

## The heap block

The descriptor heap wrapper fills a 56-byte block, and that block is the only handle the rest of the title uses.
Its fields, by offset:

| Offset | Field |
|---|---|
| 0 | a pointer to the block itself, so a sub-allocator can reach the increment through one indirection |
| 8 | the bump cursor, starting at the CPU base |
| 16 | the end pointer |
| 32 | the slot 9 CPU handle |
| 40 | the slot 9 handle plus the heap's byte size |
| 48 | the slot 15 increment for the heap's type |
| 56 | for a shader-visible heap, slot 10 minus slot 9. For a heap that is not shader visible, the poison constant `0xBADADD0000000000` |

Reservations out of a heap are a compare-and-swap bump allocator over fields 8 and 16: it takes
`increment * count` bytes, optionally aligned up to a byte alignment the caller passes, and returns null when the
cursor would pass the end. So **a descriptor range in this title is a CPU address and a stride, not an index**.

## The free list lives inside the descriptor memory

Right after the five heaps are created, the title reserves descriptors out of four of them and **writes a singly
linked free list into that descriptor memory**: at the start of each descriptor slot it stores the index of the
next slot as a `UINT`, striding by the slot 15 increment, and stores `0xFFFFFFFF` in the last. Four separate
functions do it, with identical bodies:

| Heap | Descriptors put on the free list |
|---|---|
| The big type 0 `CBV_SRV_UAV` heap, the one that asserts | the configured descriptor count that also sized sub-range 1 of the stop 10 placeholder |
| The second `SAMPLER` heap, the one that is not shader visible | 128 |
| The `RTV` heap | 4096 |
| The `DSV` heap | 2048 |

The shader-visible `SAMPLER` heap is the exception: it gets a plain 128-descriptor bump reservation instead, with
no free list.

**`RTV` and `DSV` heaps are included.** That is worth stating on its own, because a render target view on the PC
is an opaque handle into a driver-owned heap, and here the title writes a `UINT` over the first four bytes of
every one of 4096 `RTV` slots and 2048 `DSV` slots before any view is created in them, then reads those bytes
back. Any descriptor model that cannot survive the title scribbling on unallocated `RTV` and `DSV` slots will not
work.

Allocating a descriptor is a lock-free pop that **reads the next index back out of the descriptor memory** and
compare-and-swaps the head, with an ABA counter in the head's upper 32 bits.

Three consequences for a shim.

- The heap's CPU allocation has to be **plain readable and writable CPU memory the title can scribble on**. It
  cannot hold shim metadata, a descriptor-table header, or anything else: the first four bytes of every
  descriptor slot are overwritten with a free-list index before any descriptor is ever created there, and are
  read back later.
- The **slot 15 increment is the stride the title uses for everything**, so whatever a shim reports has to be
  the stride at which it is willing to have descriptors written, and has to stay the same for the life of the
  run. [descriptor_heap_allocation_hook.md](descriptor_heap_allocation_hook.md) notes that the same value also
  sizes the 185 MiB placeholder, which is read before the heap exists.
- For these four heaps there is **no API event when a descriptor is allocated or freed**. A shim that wants to know which slots are
  live has to infer it from the create-view calls below, or shadow the free list by reading the same memory.

## Views are created through the device, with PC-shaped descriptors

The contents of each descriptor do go through the device. Two thin wrappers, each of which creates the view and
then calls a per-descriptor bookkeeping function:

| Wrapper | Device slot | PC method | Call shape |
|---|---|---|---|
| View writer 1 | 18 | `CreateShaderResourceView` | `(resource, desc, dest_handle)` |
| View writer 2 | 19 | `CreateUnorderedAccessView` | `(resource, null counter, desc, dest_handle)` |

Both match the PC argument order exactly, which is worth noting next to
[retail_boot_chain.md](retail_boot_chain.md) §3: not every device slot on the console diverges, and these two do
not.

### The null descriptor set

Bring-up's first use of the heap is a set of null views, each into its own popped descriptor. The descriptors are
built by the title on its stack as PC structures, so they can be read directly:

- **Five null SRVs**, all with `Format = DXGI_FORMAT_R32_UINT` and
  `Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING (0x1688)`, and `ViewDimension` in turn
  `TEXTURE1D`, `TEXTURE2D`, `TEXTURECUBE`, `TEXTURE3D`, `BUFFER`. The resource pointer is null in all five.
- **Five null UAVs**, same format, `ViewDimension` in turn `TEXTURE1D`, `TEXTURE2D`, `TEXTURE2DARRAY`,
  `TEXTURE3D`, `BUFFER`. Resource and counter resource both null.
- **One null CBV**, see below.
- Six further descriptors are popped and kept without a view being created at bring-up time.

A shim therefore has to accept a **null resource pointer** on slots 18 and 19 and produce a descriptor that reads
as "nothing bound" of the right dimension. On the PC that is legal and well defined, so this is a shape note
rather than a problem, but a proxy that asserts on a null resource will stop here.

### The null CBV has a raw fallback gated on a device field

The constant buffer view is the exception. The title tests **a byte at offset 24 of the device object itself**,
not through the vtable:

- **non-zero**: call device slot 17 (`CreateConstantBufferView` on the PC) for the popped descriptor.
- **zero**: skip the device entirely and **write two 64-bit hardware descriptor words straight into the
  descriptor memory**.

That second path cannot be intercepted by a proxy at all, and the descriptor it writes is a console hardware
encoding with no PC meaning. So a device proxy wants that byte to be **non-zero**, and a proxy whose object
happens to have a zero there will silently get a hand-written hardware descriptor in its heap instead of a
`CreateConstantBufferView` call.

This is the first measured case of the title reading a raw field out of the device object rather than calling
through it. It is worth checking the proxy's object layout for anything else the title might read at a small
offset.

## What bring-up does after the five descriptor heaps

All of the above and everything below happens in the same device bring-up function, in this order, with no
return to the caller in between. Listing it in order because each item is a candidate for the next stop once
stop 13 clears:

1. The descriptor free list is built over the big type 0 heap, as above.
2. The same free list is built over the non-shader-visible `SAMPLER`, `RTV` and `DSV` heaps, 128, 4096 and 2048
   descriptors.
3. Heap slot 9 is called on a second heap block and the handle is stored in two globals.
4. A 128-descriptor reservation is made out of the shader-visible sampler heap, and a slot 9 difference against a
   third heap block is stored.
5. The null SRV set, the null UAV set and the null CBV are created, as above.
6. Three index values are computed as `(descriptor address minus a base) / a stride`, so the title also converts
   descriptor addresses back into indexes. The base and stride are globals, not the heap block's fields.
7. A **submission ring** is built over **sub-range 3** of the placeholder: `N` nodes of 88 bytes each, linked
   circularly, where node `i` owns the block at `sub-range 3 base + i * block size`. `N` and the block size are
   the configured count and block size that sized sub-range 3, 32 and 256 KiB by default. That is what
   sub-range 3 is for.
8. One 8-byte object is allocated **directly** from the 64 KiB pool that the fence interceptor also serves, so
   that pool is a general small-object pool and not fence-only.
9. **Two command allocator pools are created, `N` allocators each**, reusing the ring's `N`: one pool of
   `D3D12_COMMAND_LIST_TYPE_DIRECT` and one of `COPY`, each through device slot 9
   (`CreateCommandAllocator(type, riid, out)`), each allocator stored in a 48-byte pool entry. With the default
   `N` that is **64 `CreateCommandAllocator` calls**, and they pair one to one with the 32 ring nodes and their
   256 KiB blocks.

   **A failure HRESULT from any one of those 64 calls aborts bring-up with no assert and no message.** The
   failure propagates as a false return all the way out, the function frees what it built and returns, and the
   process exits cleanly. This is the one place in bring-up where a shim problem produces no diagnostic at all,
   so it is worth logging slot 9 failures explicitly rather than only non-zero exit codes.
10. Four 128-byte pointer arrays are allocated, so the batch that follows creates at most 16 of each.
11. **The arena batch runs**, which needs the allocation hook chain's third interceptor kind. See
    [descriptor_heap_allocation_hook.md](descriptor_heap_allocation_hook.md): the first matching hook call
    substitutes a 64 MiB or 82 MiB budget for the requested size, and the batch ends in an assert that the arena
    was consumed to within 2 MiB, which also fires if the hook never ran. This is the next hard stop after the
    descriptor heaps, and unlike stop 13 it has no minimal workaround: the chain has to work.
12. `XboxBvhBuilder` is constructed on the device. [runtime_imports.md](runtime_imports.md) records that DLL
    loading with zero unimplemented imports, so this is the real console DLL calling back into the device proxy,
    with `xdxr.md`'s structures.
13. Two further initialisers, then bring-up returns.

## What this does not establish

The two initialisers in step 13 are not identified. The base and stride used for the address-to-index conversion
in step 6 are globals written elsewhere and were not traced. The 88-byte ring node's fields past the block
pointer are not decoded, only that three of them are initialised to `-1` and two to zero. The six descriptors popped without a view are not followed to their use. The hardware descriptor words in
the CBV fallback are read as two 64-bit immediates, with no claim about what their fields mean. Nothing on this
page has been run.
