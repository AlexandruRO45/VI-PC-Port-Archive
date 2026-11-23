# Xbox GPU page mapping on Vulkan

`runtime/src/xbox_paging.h` models the Xbox page-mapping calls on Vulkan sparse binding. It is header-only and not
wired into the device proxy yet: the vtable slots of `RegisterPagePoolX`, `CopyPageMappingsX`,
`CopyPageMappingsBatchX` and `ClearPageMappingsX` are not measured, so nothing calls it. `runtime/test/
vk_xbox_paging_test.cpp` checks it on a real device.

## The model

- **Physical pages** are 64 KiB, the console's page size. Their GPU memory comes from 64 MiB device-local
  `VkDeviceMemory` chunks (`PagePool`), so the OS pages at a useful granularity under pressure. On NVIDIA only
  device-local memory can back a sparse buffer page; host-visible and imported memory are refused.
- **The arena** (`PagedRegion`) is one sparse, device-addressable `VkBuffer` with a large virtual size. A console VA is
  an offset into it, and its device address never changes: a remap changes only what backs a page. Unbound pages read
  0 where the device reports `residencyNonResidentStrict`.
- **The calls** (`PageMapper`): `RegisterPagePoolX` registers an ordered list of physical pages. `CopyPageMappingsX`
  and `CopyPageMappingsBatchX` map arena pages to pool pages, and `ClearPageMappingsX` unbinds them. A call is
  validated in full before anything is bound, and becomes one `vkQueueBindSparse`. Runs that are contiguous both in
  the arena and in one chunk merge into one `VkSparseMemoryBind`, and changes that leave a page as it is bind nothing.
- **Ordering**: `VulkanQueue::bind_sparse_ordered` submits the bind on the queue's timeline as `bind_tiles` does for
  D3D12 tiled resources. It waits for everything the queue submitted before it, and the next submission waits for
  it, which is the console's GPU-timeline order for these calls. While the queue holds work behind a `Wait()` whose
  `Signal()` is not submitted yet, the bind is held behind it too and submitted in queue order when it is released;
  its ticket reaches the pool then, and until it does `trim()` keeps the chunks it references. A bind dropped with
  the fence it waits on is reported, so its chunks can go.
- **Lifetimes**: a region's pages come from one `PagePool`, which outlives it, and one `PageMapper` changes it.
  A pool's binds all go through one queue, since tickets of different timelines do not compare: the pool takes the
  timeline of its first bind (`BindQueue::timeline`) and refuses another before anything is bound. Destroying a region
  gives back the mappings it still holds. A released page is not handed out again while an arena still maps it, so
  a stale mapping never shows a new owner's data. A call refuses pages that are free when it runs; a page id has no
  generation, though, so a title that maps from a pool after releasing its pages and allocating new ones maps the
  new owner's pages. That is the title using memory it freed, as it would be on the console.
- Only `D3D12XBOX_PAGE_MAPPING_RANGE_TYPE` 0 (incrementing page indices) is documented publicly (the
  Xbox-GDK-Samples AdvancedESRAM README). Other range types are refused until a call site shows them.

## Measured cost

GTX 1660 SUPER, driver 610.88, `vk_xbox_paging_test` (median of 15; CPU time from the call to the bind's timeline
signal):

| pages | contiguous, 1 bind | scattered, one bind per page |
|---:|---:|---:|
| 1 | 0.28 ms | 0.29 ms |
| 16 | 0.34 ms | 0.50 ms |
| 256 | 0.55 ms | 2.4 ms |
| 1024 | 1.6 ms | 7.9 ms |

A call costs about 0.25 ms plus 1.5–2 µs per contiguous page or 8–10 µs per scattered one, so a title's remaps
should reach the driver as few calls per frame. A remap between two dispatches added 0.67 ms to the gap between them
(64 scattered pages; 0.012 ms without one), and returned on the CPU in 0.1 ms while the first dispatch was still
running: the bind waits on the GPU, not on the CPU.

## Physical pages on the CPU

On the console a physical page is one piece of memory that the CPU and GPU both address. Here the GPU side must be
device-local, so `runtime/src/xbox_physical.h` gives each physical page two halves and keeps them coherent:

- **CPU half** (`PhysicalPageStore`): a 64 KiB slot of one pagefile-backed section, committed when the page is
  allocated. Views of it are placed at the title's VAs over placeholder reservations (`VirtualAlloc2` with
  `MEM_RESERVE_PLACEHOLDER`, then `MapViewOfFile3` with `MEM_REPLACE_PLACEHOLDER`), so one page can be mapped at
  several VAs, as the console allows. A slot is split out of its reservation's placeholder only when it is first
  mapped; a slot whose neighbours were split already is a placeholder of its own and is not split again.
- **GPU half** (`PhysicalGpu`): a `PagePool` page, allocated when the page is first uploaded or bound.
- **CPU to GPU**: clean pages are read-only in every view. The first write faults, a vectored exception handler marks
  the page dirty and makes that view writable, and the write resumes. `take_dirty()` lists the dirty pages and
  write-protects them again; `record_upload` copies them into their GPU halves ahead of the title's commands on the
  same queue, after earlier GPU work on it. An upload that fails leaves its pages dirty. A new GPU half is marked
  dirty, so its first upload gives it the CPU half's contents.
- **GPU to CPU**: only on request (readbacks, query results), with `record_readback` then `finish_readback`. A page
  the CPU wrote since its last upload is not read back (the readback would overwrite those writes), and a readback
  is dropped if the CPU writes one of its pages before it finishes.

`runtime/test/vk_xbox_physical_test.cpp` (Windows only) checks a page mapped at two VAs, CPU writes read back through
the arena after one upload, a GPU fill read through a CPU view, and the refusals above. Measured on the GTX 1660
SUPER (median of 8): the first write to a clean page costs 3.7 µs (fault and unprotect), `take_dirty` of 256 pages
0.25 ms, and an upload of 256 dirty pages (16 MiB: copy to staging, GPU copy, wait) 8.7 ms, about 1.9 GB/s (7.8 to
9.3 ms over the runs).

The staging buffer is used from offset 0 by every record call, so a command buffer holds one upload or readback,
and it completes before the next one is recorded. A ring per frame in flight lifts that once the device proxy drives
it.

## Not done

- Held batches for D3D12 tiled resources: `update_tiles` and `copy_tile_mappings` still bind at once while the queue
  holds work, so a tile remap can run ahead of work submitted before it. `bind_sparse_ordered` holds its binds;
  `bind_tiles` can do the same.
- Wiring into `d3d12_xs_device` once the slots are measured from a real run.
- Wiring `xbox_physical.h` to `XMemAllocatePhysicalPages` and `XMemMapPhysicalPages[Scatter]` once a call site shows
  their arguments.
- Freeing physical pages: `PhysicalPageStore` commits pages but has no free yet, so its capacity only fills.
- Pages both sides write: uploads and readbacks copy whole 64 KiB pages, so a page the CPU and the GPU both write
  between two syncs loses one side's writes (a GPU-written resource sharing a page with a CPU-written object is
  overwritten by the next upload). A readback of a CPU-written page is refused rather than lose the CPU's writes;
  the other direction needs the GPU-written ranges tracked, or uploads of only the CPU-written bytes. Needed before
  wiring.
- Writes by the kernel: dirty tracking sees user-mode writes through the fault handler, so `ReadFile`, overlapped
  I/O or `recv` into a clean page fails with `ERROR_NOACCESS` instead of marking it dirty. The I/O shims would mark
  the target pages dirty and writable first. Whether `MEM_WRITE_WATCH` can apply to section views is not checked.
- The `XALLOC_ATTRIBUTES` layout that tells `XMemAlloc` to hand out physical pages is not public.
