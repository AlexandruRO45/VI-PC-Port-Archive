# Raw descriptor sync

Descriptors normally reach the host through the descriptor API: the title
calls `CreateShaderResourceView`, `CopyDescriptors` and so on, and the layer
makes the matching host descriptor (`HostHeap::mirror`). A title can also copy
the runtime's 32-byte descriptor records straight into a shader-visible heap
with a memory copy. Those copies bypass the API and leave the host slot null
or stale.

The raw descriptor sync (`descriptor::sync_heaps`, on by default) closes that
gap. Each placed shader-visible heap is write-watched. Before every host
submission, the pages written since the last look are read through the kernel
(a concurrent unmap cannot fault the read), and each record that changed is
mirrored as if the API had written it.

## Cost

In the first room the title rewrites a ring of descriptors every frame. With
about 30 submissions a frame:

| What | Per frame |
|---|---|
| Heap pages read (64 KiB each) | 110 to 140 |
| Records that changed and were mirrored | 35,000 to 41,000 |
| Time on the submitting thread | 30 to 50 ms |

The submitting thread bounds the frame rate (the GPU waits for it), so this
time shows up in the frame time. The heap pass that later makes the buffer
views among those records costs another 50 to 80 ms a frame on buffer slots.

## Checking a page under one lock

Most records of a written page are unchanged. The sync used to call
`mirror_copied` for each of a page's 2,048 records, which locked the host
heap's `made` and `copied` tables for every record. `mirror_copied_run` now
compares the whole page with the kept records under one lock and mirrors only
those that changed, so an unchanged record costs a comparison.

Measured in the first room, 25 s per run after the room is steady:

| Base | Build | FPS | ms/frame | Sync ms/frame |
|---|---|---|---|---|
| `2146f54` | without | 6.5 | 155 | 38 |
| `2146f54` | with | 7.1 | 141 | 29 |
| `77e1ac7` | without (2 runs) | 6.6, 7.2 | 152, 139 | 38, 36 |
| `77e1ac7` | with (2 runs) | 7.0, 7.0 | 143, 144 | 30, 32 |

The sync is 5 to 9 ms a frame cheaper in every pair. On `77e1ac7` the frame
rate difference is within the run-to-run spread of the control.

## Switch

| Switch | Effect |
|---|---|
| `VICTORY_HEAP_SYNC=0` | no raw descriptor sync (heaps are not tracked either) |

With the sync off, the same scene ran at 7.6 FPS (132 ms) against 6.5 FPS
(155 ms) on `2146f54`. The sync was written for missing walls and roofs
(fxceless/victory-fork#305) and has not yet been shown to fix them, so whether
it should stay on by default is open.

The frame log has one line for it every five seconds:

```
raw descriptor copies per frame: 118.8 heap pages read, 40876.7 records mirrored, 50.89 ms
```

## Tests

In `crates/victory-gpu/src/descriptor.rs`:
`records_copied_as_bytes_are_made_at_the_next_sync` (a record copied as bytes
is made at the next sync, then replaced, then removed by zeroing it) and
`raw_heap_sync_stops_before_dynamic_host_slots` (slots past a capped host heap
are left alone). Both pass with the page compared under one lock.
