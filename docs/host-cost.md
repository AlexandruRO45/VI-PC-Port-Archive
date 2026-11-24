# What a world frame costs on the host

Measured on BartekPC (Ryzen 9 7900X, RX 6700 XT), `--profile diag`, GTA VI Xbox
1.0.2.1. Every number here comes from the frame log the host already writes,
`grep "ms/frame"` on a run's stderr.

**Read this first: a world frame is not GPU-bound.** In every logged world
frame the host GPU executes for 10-35% of the wall time. The rest is this
crate moving title memory around on the CPU. The hypotheses this lane was
sent to price (translated loops spinning to 2^20, per-frame pipeline
creation, descriptor or barrier churn) all show up as *GPU* time, and GPU
time is the small share. They are worth killing on those grounds before
anyone spends a day on them.

## The table

World windows only (frames with 100 or more draws, so the frame log is past
the menus and the account picker). Windows and their source logs:

| run log | ms/frame | preparing | copies | textures | on the GPU | copying back | draws/frame | dropped |
|---|---|---|---|---|---|---|---|---|
| `victory-hardware2.stdout.log` (9 windows) | 1233 | 577 | 307 ms / 107 MB | 26 ms / 105 MB | 123 | 350 ms / 4.9 MB | 494 | 1000 |
| `victory-hardware.stdout.log` (13 windows) | 2750 | 1106 | 507 ms / 316 MB | 25 ms / 92 MB | 974 | 492 ms / 10.6 MB | 1884 | 1709 |
| `victory-render-debug.stdout.log` (6 windows) | 4757 | 2411 | 1614 ms / 379 MB | 124 ms / 165 MB | 1595 | 696 ms / 13.8 MB | 3250 | 846 |

Averaged over the windows of each run. All three logs are in
`D:\victory-scratch`, from runs on `stable` before the first-world-frame
crash (`docs/../AGENTS.md`, "The crash that ends every run") made world
frames unobtainable. `victory-after5.stdout.log` is deliberately absent: its
windows carry 12 draws/frame, which is the menus or the account picker, not
the world, and this lane was warned that such a window measures nothing about
rendering.

Per frame, from the same windows (`copies per frame:` lines):

- 19 to 27 submissions/frame, 3400 refreshes/frame asking for 4778 MB of
  refreshed ranges, of which 107 MB is copied;
- 983 write-watch askings taking 57 ms, checking 6126 MB, of which 6051 MB
  was unchanged (98.7% of the checked bytes cost time and produced nothing);
- 274,000 page faults/frame.

## What the shares mean

Of a 1233 ms world frame:

- **46.8% preparing title memory**: copies of stale title pages into the host
  copies buffers need (307 ms), texture detiling (26 ms), and the rest in
  the submission's other deferred steps.
- **28.4% copying back what shaders wrote**: 350 ms to find that 4.9 MB of
  the GPU-written ranges actually changed.
- **10.0% on the GPU**: 123 ms for 494 draws.
- the remainder is the title's own work and the frame clock.

So the two title-memory paths (`shadow.rs`, `watch.rs`, `texture.rs`) are 75%
of a world frame. Everything the world draws fits in the other 10%.

The second run logs the same shape at a different scale (2750 ms/frame, of
which 1106 preparing and 492 copying back, against 974 on the GPU), and the
third — a build with the debug layer and GPU-based validation on — keeps the
ratio. The shape is not an artefact of one build.

## Copy rates, and why they are what they are

- Title memory is copied with `ReadProcessMemory`, not `memcpy`, so a page
  the title has unmapped reads as zeros instead of faulting
  (`host::copy_mapped`). `VirtualQuery` was tried instead and made copies run
  at 100-200 MB/s, because it walks page tables to the end of the region.
- The measured end-to-end rate of the refresh path is 107 MB in 307 ms
  (350 MB/s), but only 10.75 ms/frame of that is inside `ReadProcessMemory`
  (`moves` in the log). The rest is the write-watch asking (57 ms), the
  per-64 KiB page loop in `Shadow::copy_stale`, and locking: 983 askings and
  3400 `update` calls per frame.
- Copy-back compares the buffers' host copies against the bytes they held
  before the submission, 64 KiB at a time, and copies back only what
  differs. That comparison is ~2.3 GB/frame of `memcmp` (26 submissions of
  the world's declared ranges, its 174 MB heap among them) to find 4.9 MB.

## What to do next, in order

1. **Declare the GPU-written ranges once per frame, not once per submission**
   (`Shadow::writing`/`copy_back`). The world declares the same big heap
   dozens of times per frame. The frame log now splits that cost in two
   (`{:.1} MB kept before shaders write in {:.1} ms, {:.1} MB compared after
   them` on the `copies per frame` line): keeping the before-bytes is a
   memcpy per declared range per submission, and it goes away entirely once
   the copy kept across submissions is updated in `copy_back` instead of
   discarded. The **compare survives** — it is the price of not knowing which
   bytes a shader wrote, and declaring a range once per frame does not tell
   us that; only an incremental compare could, and that needs the GPU's own
   write ranges.
2. **Ship the frame log's per-submission breakdown** (this branch): it names
   the costliest submissions with their pipelines, draws, dispatches and
   command counts, and on a device removal it dumps the timings of the list
   that died.
3. Only then look at GPU time. 123 ms for 494 draws is 0.25 ms a draw, which
   a translated vertex/pixel pair can plausibly cost; the loop-limit
   experiment (`VICTORY_SHADER_LOOP_LIMIT`) prices that directly, and the
   per-command stamps from `VICTORY_GPU_TIMING` name the draw.

## The harness

One command, bounded, with the crew's key schedule:

```powershell
powershell -NoProfile -File D:\victory-scratch\runs\world.ps1 `
  -Name world-timing -Seconds 240 -Set 'VICTORY_GPU_TIMING=1'
```

It refuses to start while a title process is alive or another agent's lock is
fresh, writes `D:\victory-session.lock` with its own token, and removes only
that lock. Logs land in `D:\victory-scratch\runs\<Name>.err.log`; the key
schedule is `D:\victory-scratch\runs\keys.txt` (pad A every 4 s from 5 s to
300 s).

Then, for that run:

```powershell
Select-String 'ms/frame|submissions per frame|submission \d+ ms|slow host list|slow list command' `
  D:\victory-scratch\runs\world-timing.err.log
```

`VICTORY_GPU_TIMING=1` alone is enough for the per-command names (it used to
need `VICTORY_D3D12_DRED=1` as well, which also turns every draw into a debug
marker; the two are now independent).

To price the copy machinery itself, run the same command with
`VICTORY_COPY_MODE=none` in `-Set` (copies and refreshes skipped, timing
only) and compare `ms/frame`: the difference is the whole copy share.

Naming commands is not free: with `VICTORY_GPU_TIMING` on, every draw,
dispatch and transition builds its own description and takes the note lock, so
a frame measured with timing on carries that CPU cost. Compare `ms/frame`
with timing on in *both* runs (one of them with `VICTORY_GPU_TIMING=1
VICTORY_SKIP_PIPELINES=<hash>`), never against a run with timing off.

## What is still blocked

No run today survives the first world frame (`AGENTS.md`, "The crash that
ends every run"), and skipping the compute that removes the device does not
get past it: `VICTORY_SKIP_PIPELINES=e30970f1727063fd` still ends in a
removal at the same submission. So there is no *current* world-frame number
in this document, and none is claimed: the table is the last set of world
frames that survived, and it is stable across three builds. The instrument
below is built so that the first surviving world frame names its own cost
within one run.

## Two tests that were measuring the machine, not the code

Both are in this crate and both red the suite at random, so a green
`cargo test --workspace` was luck. Fixed here; the gate is
`cargo test --workspace -- --test-threads=1`.

- `frame::tests::waits_end_on_ticks_and_count_frames` asserted that four
  consecutive flip waits land on four consecutive ticks. The contract the
  clock documents is weaker and is the right one: ticks are absolute, so a
  late sleep never lets two waits for one event share a tick. A thread
  descheduled for several intervals is not a bug in the frame clock. The test
  now asserts flips are strictly increasing and no faster than the clock that
  made them (measured 4 failures in 20 runs before, DGR-6).
- The snapshot tests asserted on the process-global snapshot arena
  (`shadow.rs`). Every test that submits a list with declared GPU-written
  ranges adds parts to that arena from its own thread — `draw.rs`,
  `indirect.rs` and `list.rs` all do — so a snapshot test's part count was
  partly other tests' parts (measured 1 failure in 20 runs). Each copy now
  reports its own parts (`Shadow::kept`), the tests assert on the copies they
  made, and the release-on-drop check counts parts given up (monotone, so a
  concurrent copy can only add to the evidence, never withdraw it).

Both were found by DGR-6, not by me; the numbers above are its measurements.

## Prepared: declare a shader's write ranges once per frame

Not landed, deliberately: it lives in the copy-back path that has already
corrupted title memory once (`title+0x4e9b54a`, a fence list read as -1), and
no run can contradict it today. Written out here so it lands in one piece when
one can. Verify the counters above first.

**The invariant.** For every byte of a declared range, the bytes kept before a
submission equal the copy as it stood at the start of *that* submission. Then
`copy_back` sees only what the GPU changed, and the per-submission memcpy of
the same ranges disappears.

**Three changes in `shadow.rs`.**

1. `Shadow::writing` keeps its parts across submissions: `copy_back` stops
   calling `forget` when `pending` empties. Parts still go on `Drop` and when
   the runtime remaps title memory (`watch::title_memory_remapped`).
2. `Shadow::copy`, the refresh from title memory, writes the bytes it copies
   into the parts covering them as well. Without this, a title write between
   two submissions is read as the GPU's and copied back.
3. `copy_back` writes the bytes it leaves in the copy (`*h = *t`) into the
   covering parts too, so the kept copy follows the copy instead of going stale.

**What it costs.** One extra memcpy of the refreshed bytes: 107 MB/frame in
the `hardware2` world window, about 15 ms. What it saves is the whole snapshot
half, which `copies per frame` now reports as `MB kept before shaders write in
N ms`. The compare half survives — see the note above.

**Tests it needs, in `shadow/snapshot_tests.rs`** (they already have two
copies, `Shadow::kept`, and a host device):

- two submissions over the same range where the GPU changed nothing: the
  second copies back 0 bytes;
- the same, with one byte written by the shader: exactly that byte goes back;
- a title write between the two submissions and no GPU write: 0 bytes go back,
  so a title write is never attributed to the GPU.

**The A/B when a world frame survives.**
`powershell -File D:\victory-scratch\runs\world.ps1 -Name ranges -Seconds 120 -Set 'VICTORY_GPU_TIMING=1'`,
then read `MB kept before shaders write in N ms` on the `copies per frame`
line: it should fall to near zero after the first frame, and `ms/frame`
should fall by that amount.
