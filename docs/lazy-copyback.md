# Lazy copy-back: merge only what the CPU touches

Status: design, step 0 (measurement) done. Nothing below is built yet.

## Why

On an RTX 4070 / Ryzen 5 7500F in the first gameplay room, the relay's
copy-back costs 44-55 ms of a 112-123 ms frame, and the title's fences are
set only once it is done, so it adds to the frame almost one for one:

- about 1,250 pieces of 64 KiB a frame are fetched from card copies (78 MB),
  96 % of them hold bytes shaders wrote, 34 MB of bytes that changed;
- merging them on several threads gained nothing (the cost is the kernel's
  writes into title memory and the fetches, not the compare).

`VICTORY_BACK_READ_PROBE=8` (step 0) marked one page in eight of what the
copy-back wrote with `PAGE_GUARD` and counted the first touch:

| per frame | |
|---|---|
| pages marked | 650-1,230 |
| first touched by a CPU read | **0.0** |
| first touched by a CPU write | 2.5-3 |
| first touched by our own access (mostly the next merge) | 820-1,220 |

The title's CPU code does not read back what its shaders write to these
buffers. Almost all of the copy-back is bytes only the card uses.

The same run went to about 1 frame a second: changing a page's protection
(`VirtualProtect`) on the title's big view memory is expensive, and the
probe did it some 2,000 times a frame. **A design that protects and
unprotects pages every frame cannot win.**

## Design: protect once

A 64 KiB page of title memory that shaders write and the CPU does not
touch becomes **card-only**:

- its title views (every address that shows the page:
  `crate::inplace` knows them) are made `PAGE_NOACCESS` once;
- from then on, a flight that writes it costs nothing on the CPU: the
  piece is not fetched, not merged, and the flight's fences are not held
  for it; the copy in the card's memory is the page's truth;
- the page stays card-only, with no further protection change, for as
  long as nothing on the CPU side touches it.

A page leaves card-only (**materialize**) when anything touches it:

- the title's CPU code (an access violation on a `PAGE_NOACCESS` page,
  taken by the vectored handler);
- a kernel call on the title's behalf (file writes, DirectStorage) that
  fails on it;
- the layer's own reads of title memory (`host::read`, `copy_mapped`,
  `read_into`, fingerprints) and writes (`write_mapped`).

Materialize: fetch the page from the card copy (a copy list on the copy
queue and a wait), merge it as the copy-back does now
(`Shadow::card_merge_relay`, through our own view so that no title view's
protection must change), give the title views their protection back. The
page is then ordinary until a flight's compare flags it again, when it
may become card-only again (at most once per N frames per page, so that a
page the CPU does touch every frame does not pay a protection change
every frame).

## Hazards

- **Deadlock.** Materialize waits for the card. A fault on a layer thread
  that holds the submission lock, a queue's lock or the guards' lock would
  wait for itself. Layer threads must never touch a card-only page in
  process memory; every layer read of title memory goes through the
  kernel wrappers, which materialize before they read, outside those
  locks. Faults taken on a layer thread are counted and must be zero.
- **Aliases.** One physical page is shown at several title addresses;
  all of them must be protected and given back together.
- **The guards.** `crate::watch` protects view memory read-only to see
  the title's writes. A card-only page's protection belongs to this
  module until it is materialized; the guards must leave it alone (and
  must not count it as written).
- **Kernel calls.** A kernel call that fails on a card-only page on the
  title's behalf must be retried after materializing; the places that
  call the kernel with title addresses (HLE file and storage modules)
  must be found and wrapped.
- **Refreshes of other copies.** A texture or another buffer copy over
  the same memory reads title memory to send it to the card: through the
  wrappers, so it materializes first.

## Steps, each tested in the game before the next

1. **Registry and materialize, no protection.** Card-only pages are only
   *noted*: flagged pieces still fetched and merged as now, and a counter
   says how many would have been skipped, per frame, and how many layer
   reads touched a noted page. Proves the bookkeeping, changes nothing.
2. **Skip the merge, keep the fetch.** Card-only pages are protected once
   and not merged; the fetched bytes are kept until the next flight's, so
   materialize needs no wait for the card. Fences no longer wait for those
   merges. Measures the gain with the least risk.
3. **Skip the fetch too.** Materialize fetches the page itself.
4. **On by default**, after runs past the first rooms, a save, a load.

`VICTORY_LAZY_BACK=1` turns it on at every step until step 4.
