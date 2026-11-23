# Boot stop 13: the descriptor heap starts at the stop 10 placeholder's base

**Status: measured 2026-11-23 from the RT-5 executable, static analysis only, nothing executed.**
Nothing here quotes the executable: no addresses, offsets, IID values or bytes; only export names, public SDK
types, vtable slot numbers, call shapes, immediates and counts, the same rule as
[retail_boot_chain.md](retail_boot_chain.md).

## The answer

[retail_boot_chain.md](retail_boot_chain.md) stop 13 records that right after the first `CreateDescriptorHeap`
the title calls the heap's slot 9 and asserts the handle equals a value it already holds, and offers the
hypothesis that "a console descriptor heap starts at an address the title chose or can predict".

It is neither chosen nor predicted. **The value asserted against is the pointer the stop 10 placeholder reserve
returned**: the result of the single `XMemVirtualAlloc(nullptr, about 185 MiB, 0x42000, 0x7, PAGE_NOACCESS)` call,
which [xmem_exe_calls.md](xmem_exe_calls.md) lists as the one `0x42000` site. The title stores that pointer in a
global and passes it, unchanged, as the desired CPU descriptor base for its first descriptor heap. The heap's byte
size is the sum of the first three of the five sub-ranges the title then carves out of that placeholder at
stop 11.

So the first descriptor heap is the first three sub-ranges of the placeholder, viewed as descriptor storage. A
`CreateDescriptorHeap` that allocates anywhere else cannot satisfy the assert, and no choice of address helps
unless it is that one.

## The wrapper and its five call sites

One function creates every descriptor heap in the title. Its parameters are
`(out_block, heap_type, desired_cpu_base, descriptor_count, shader_visible)`, and it has two callers covering
five heaps:

| Call site | `heap_type` | `desired_cpu_base` | `descriptor_count` | `shader_visible` |
|---|---|---|---|---|
| The asserting one | 0 (`CBV_SRV_UAV`) | the placeholder base | byte size / slot 15 increment for type 0 | 1 |
| Second caller, call 1 | 1 (`SAMPLER`) | 0 | 128 | 1 |
| Second caller, call 2 | 1 (`SAMPLER`) | 0 | 128 | 0 |
| Second caller, call 3 | 2 (`RTV`) | 0 | 4096 | 0 |
| Second caller, call 4 | 3 (`DSV`) | 0 | 2048 | 0 |

**Only the first supplies a base, and the assert is guarded on that argument being non-zero.** The four that
follow pass zero, accept whatever base comes back, and tile it out across their own descriptor arrays by
repeatedly adding the slot 15 increment. They are indifferent to where the heap lands, so clearing stop 13 should
let all five through, not just the first.

The five are created back to back, the asserting one first, so the trace at stop 13 holding exactly one
descriptor heap is consistent.

## Where the base and the size come from

The stop 10 and stop 11 allocations are one function, and it is called immediately before the descriptor heaps,
inside the same device bring-up function. It takes five sizes, reserves their sum as one placeholder, splits it
with four `VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)` calls and replaces each piece with the
`0x20406000` form [xmem_exe_calls.md](xmem_exe_calls.md) already tabulates. Along the way it records, in globals:

- the **placeholder base**, which is what the descriptor heap asserts against,
- the **combined size of sub-ranges 1 and 2**,
- the **size of sub-range 3**, separately.

The byte size handed to the descriptor heap wrapper is the third of those plus the second, that is sub-ranges
1 + 2 + 3. The descriptor count is that byte size divided by the slot 15 increment for type 0.

The five sub-range sizes themselves:

| Sub-range | Size | `protect` at its `0x20406000` replace |
|---|---|---|
| 1 | slot 15 increment for type 0, times a configured descriptor count, rounded up to 64 KiB | `0x1004` |
| 2 | from a configured 16-byte pair, truncated to 64 KiB | `0x1404` |
| 3 | a configured count times a 64 KiB-rounded configured block size, 32 times 256 KiB by default, so 8 MiB | `0x8404` |
| 4 | the other half of that configured pair, truncated to 64 KiB | `0x2404` |
| 5 | a fifth configured size | `0x2004` |

That ordering is the order of the `protect` values in
[xmem_exe_calls.md](xmem_exe_calls.md)'s `0x20406000` row, so the row's five values map one to one onto the five
sub-ranges in address order.

### The slot 15 increment sizes the placeholder

Sub-range 1's size is computed by calling device slot 15 for type 0 **before** the placeholder is reserved. So
the value a shim returns from `GetDescriptorHandleIncrementSize` decides:

- the size of sub-range 1, and therefore the size of the whole placeholder,
- the descriptor count passed to `CreateDescriptorHeap`,
- the byte size the title expects the heap to occupy.

Two consequences. The reported increment has to be stable: the wrapper calls slot 15 again when it creates the
heap, and multiplies by the count, so a different answer the second time changes the size the title expects.
And the increment should be chosen deliberately rather than forwarded from the host device, because it scales
the 185 MiB reservation. #9368 makes the layer's answer per-type for exactly that kind of reason and keeps type 0
at 32, so the reservation's size is unaffected by it; the point here is only that type 0's answer is the one that
has a second, less obvious effect.

### A fifth descriptor heap type

Immediately before the descriptor heaps, the title calls device slot 15 five times, for types 0, 1, 2, 3 **and
4**. The PC enum stops at 3, with `D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES` equal to 4. A shim that validates the
type argument against the PC range will reject the fifth call. Only the type 0 result is kept; the other four
are called for their side effect or discarded.

## Clearing it

Two routes, and they are not equivalent.

**Minimal.** Make the heap's slot 9 handle equal the `desired_cpu_base` the title passed, when it passed one.
The whole heap is CPU-side descriptor storage in memory the title already owns and has already mapped physical
pages into, so handing its own pointer back is consistent rather than a fiction. Nothing else in the wrapper
depends on the handle except the end pointer it derives from it (base plus byte size) and, for a shader-visible
heap, the slot 10 minus slot 9 difference. For a heap that is not shader visible the wrapper stores the poison
constant `0xBADADD0000000000` in that field instead, so the GPU-side difference is only read for the two
shader-visible heaps.

**Faithful.** Honour the allocation hook chain below, which is the mechanism the console uses to get that same
answer. The title installs an interceptor around the slot 14 call precisely so that the runtime asks it where to
put the heap.

The minimal route clears the stop with one conditional. The faithful route is what the other two interceptor
kinds need as well, so it has to be built eventually.

## The hook chain: `__XMemAllocationHooks__`

The executable exports exactly one symbol, a data symbol, as
[xmem_exe_calls.md](xmem_exe_calls.md) notes. It is a table of three pointers followed by two null slots:

| Entry | What it is |
|---|---|
| 0 | a thunk to the **allocation** chain walker |
| 1 | a thunk to the **release** chain walker |
| 2 | read-only data: six records of `{32-bit identifier, sequential ordinal 0 to 5, zero}`, terminated by a record of `{zero, -1, -1}` |
| 3, 4 | null |

Which of the two walkers is allocation and which is release is now measured rather than inferred from position:
entry 1's push helper is the one the title uses to wrap an `IUnknown::Release` call, and the interceptor it
installs there returns a non-zero constant on a match and performs no free, so entry 1 is the release side.

Both walkers have the same shape:

```
hook_entry(size, class_word)
    token = current thread's scope token       // a lazily initialised thread-local slot
    lock(shared_lock)
    node = chain_head[ (class_word >> 16) & 0xFF ]
    collected = []
    while node:
        if node.token == token or node.token == -1:
            collected.append({ node.callback, node.context, node.token })
        node = node.next
    unlock(shared_lock)
    link each collected entry's next to the following one, last one's next = 0
    return collected[0].callback(size, class_word, collected[0].context, collected[1])
```

Six facts in that shape matter for a shim.

- **The chain head is a 256-entry array indexed by byte 2 of the class word**, that is `(class_word >> 16) & 0xFF`.
  There is one such array per walker, each 256 pointers long, and the two arrays sit next to each other in
  writable data with the exported table immediately after the second.
- **The title only ever registers at index 128.** Its push helper writes to one fixed slot, `0x80`, not to a slot
  chosen from the class word. So a class word whose byte 2 is not `0x80` reaches a chain the title has nothing in,
  and the interceptor never fires. The host decides byte 2, so for the descriptor heap allocation it has to be
  `0x80`.
- **Nothing in the executable initialises either array, and the walker dereferences the indexed slot before
  testing it for null.** Both arrays are referenced only by their own walker. So the host has to pre-populate
  every index it will ever pass with at least one node, whose token is `-1` so every thread sees it, and whose
  callback performs the real allocation. An empty slot is a null dereference inside the title's own export.
- **A node is 4 pointers**: callback, context, scope token, next. The push and pop helpers take nodes from a
  free list and splice them under one lock shared by all four helpers and both walkers.
- **Nodes are thread-scoped.** A node is collected only if its token equals the calling thread's token or is
  `-1`. The token comes from a thread-local slot initialised on first use. So an interceptor installed on one
  thread is invisible to allocations made on another, and a shim that services a descriptor heap allocation from
  a worker thread will not see the interceptor the title installed on the calling thread.
- **The collect buffer holds ten nodes.** The walker copies matching nodes into a fixed stack array with room for
  ten, and does not bound the loop. An eleventh matching node in one chain overruns the walker's frame. Ten is
  far more than the title uses, but a host that registers a default per index should register one, not a stack.

The callback signature is `(size, class_word, context, next_node)`. A callback that declines chains by calling
`next_node.callback(size, class_word, next_node.context, next_node.next)`. The last collected entry's next is set
to null, so the final callback in the chain, which is the host's own default, must not chain further.

### The class word

Every interceptor tests a different field of the same 33-bit word, so the word is a packed descriptor and not an
enum:

| Bits | Mask | Meaning |
|---|---|---|
| 0 to 13 | `0x3FFF` | allocation kind. `4` is a descriptor heap |
| 14 to 15 | `0xC000` | a second kind field. `2` selects the arena sub-allocator below |
| 16 to 23 | `0xFF0000` | the chain head index. The title registers only at `0x80` |
| 24 to 28 | `0x1F000000` | **log2 of the required alignment** |
| 29 to 32 | `0x1E0000000` | a category. `10` selects the fence pool below |

The alignment field is the one a host cannot guess: one of the interceptors computes its alignment as
`1 << ((class_word >> 24) & 0x1F)` and aligns the returned pointer up to it, so the host has to encode the
allocation's alignment there as a log2 value rather than aligning the result itself.

## The three interceptor kinds, and what each does

All three are installed by the same device bring-up function, or by a function it calls, so all three are on the
boot path and all three are reached before the first frame.

| Kind | Test on the class word | Installed around | What it returns |
|---|---|---|---|
| Descriptor heap | low 14 bits `== 4`, and the requested size equals the byte size the wrapper computed | device slot 14 | the base the caller supplied |
| Fence pool | bits 29 to 32 `== 10` | device slot 36 (`CreateFence`) | a block from a 64 KiB title-owned pool, aligned to `1 << (bits 24 to 28)` |
| Arena | bits 14 to 15 `== 2` | a batch of resource creations later in bring-up | a 64 KiB-granular block bump-allocated out of one large arena |

Each is pushed immediately before the call it guards and popped immediately after, so the window in which it is
visible is one call wide.

### Descriptor heap

Its test is the pair: the kind field equals `4` **and** the requested size equals the byte size the wrapper
computed as slot 15's increment times the descriptor count. On a match it returns the caller's chosen base; on
anything else it chains. **The size has to match exactly.** A shim that rounds the allocation up, adds a header,
or batches several heaps into one allocation misses the test, falls through, and gets an unrelated address back,
which then fails the assert in a way that looks like the hook was never honoured at all.

### Fence pool

Installed by the function that wraps a freshly created command queue, which creates one fence per queue, so it
fires five times during bring-up, once per queue. On a match it serves the request from a 64 KiB block the title
obtained with a single `XMemVirtualAlloc(nullptr, 0x10000, 0x20401000, 0x283, 0x42404)`, which is the one
`0x20401000` site in [xmem_exe_calls.md](xmem_exe_calls.md)'s table. The suballocator is a lock-free
compare-and-swap bump pointer, it aligns up using the class word's alignment field, and it asserts if the
aligned block would pass the block's 64 KiB end. So that `0x20401000` reservation has to succeed and has to be
at least 64 KiB of readable, writable, CPU-visible memory, or five fences in a row fail inside the title.

### Arena

Installed around a batch of resource creations, with a two-field stack context of `{cursor, budget}` where the
budget is 64 MiB, or about 82 MiB when a configuration check passes. Its behaviour has one feature a host must
know about:

- On the **first** matching call it chains, **substituting the budget for the requested size**. So the host's
  default handler sees a request for the whole 64 MiB or 82 MiB arena with that class word, not the small size
  the title asked for. The returned pointer becomes both the arena base and the initial cursor.
- Every call, matching or not, records the class word in a global.
- Each matching request is rounded **up to 64 KiB**, the cursor is advanced by that, and the old cursor is
  returned. Overrunning the budget is a fatal assert.
- After the batch, the installer asserts that the arena was consumed to within 2 MiB of the budget. That assert
  also fires if the hook never ran, because the cursor is still zero.

So an unimplemented chain shows up here as an assert about the arena rather than as a failed allocation.

## The release chain

Entry 1's chain is used once on the boot path: the queue-wrapping function, if fence creation fails, installs an
interceptor around the fence's `Release` call that matches the same category `10` as the fence pool's allocation
interceptor and returns without freeing anything. That is correct for a pointer into the title's own 64 KiB
pool, and it tells a shim what the release side is for: allocations the title satisfied itself must not be
returned to the host allocator, and the title communicates that by installing a matching release interceptor
rather than by tagging the pointer.

Nothing on the boot path installs a release interceptor for the descriptor heap or the arena, so the host should
not expect one.

## What this does not establish

Sub-range 3's size is identified as the configured count times the configured block size by the pairing in the
teardown path, not by reading the argument at the call; the other four sizes are read directly. The six records
in the table's third entry are a shape, not a meaning: what the 32-bit identifiers select is unknown. The fifth
descriptor heap type is observed being queried, not used. The arena's batch creates a configured number of
resources through a slot that is not identified here. Nothing on this page has been run: this is the shape the
calls have to take, not a verified fix. The obvious next measurement is a `--run` with the minimal slot 9 change
in place, to see whether all five descriptor heaps go through and what the next stop is.
