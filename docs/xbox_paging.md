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

## Not done

- Held batches for D3D12 tiled resources: `update_tiles` and `copy_tile_mappings` still bind at once while the queue
  holds work, so a tile remap can run ahead of work submitted before it. `bind_sparse_ordered` holds its binds;
  `bind_tiles` can do the same.
- Wiring into `d3d12_xs_device` once the slots are measured from a real run.
- The CPU side of physical pages: on the console the CPU and GPU see the same page; here CPU writes need a host copy
  uploaded before the GPU reads the page.
- The `XALLOC_ATTRIBUTES` layout that tells `XMemAlloc` to hand out physical pages is not public.
