# The frame profiler: what a frame's budget went to

`VICTORY_FRAME_PROFILE=<file>` (diagnostics builds only) measures the host
code the runtime runs during each frame, attributes every measurement to the
frame it belongs to and to the thread that ran it, and writes a report that
ranks functions by the share of the frame budget they take.

This is not the sampler. `VICTORY_PROFILE` (`crates/victory-gpu/src/profile.rs`)
stops the threads and walks their stacks: it finds the functions that take
the time, statistically, title code included, but knows nothing about
frames. This one is exact and frame-attributed for the code we can open and
close a scope in, and it says how much of the frame *nothing* accounts for —
which is where the title's own code is.

## The model

- **A frame is the stretch between two presents.** The boundary is the top
  of `show()` in `crates/victory-gpu/src/queue.rs`: the frame that ends is
  closed, the next begins. The wall time between boundaries is the frame
  budget, and every share in the report divides by it.
- **A probe is a scope.** `crate::frameprof::enter(&SITE)` opens it, the
  guard it returns closes it. Each site is declared once with a stable name,
  a tag and a kind (`cpu`, `gpu`, `memory`, `sync`, `idle`).
- **A scope counts in the frame whose boundary it crosses** (`close_frame`
  reads deltas of monotonically increasing counters, so nothing is reset
  under a running thread). A copy that starts before a present and ends
  after it is one frame later than the frame it started in. At most one
  such scope per thread per frame, so the error is one scope's self time on
  a boundary frame.
- **`self` is the time with nested scopes taken out; `inclusive` is the
  whole scope including what it calls.** The ranking is by `self`, because
  otherwise a function that calls the hot one twice would be counted twice
  in the frame budget. The inclusive table is the second view.
- **The frame has one presenter thread**: the thread that ran `show()`. The
  report measures what that thread's scopes account for and calls the rest
  *outside any scope*: the title's own code, the system, and host code
  without a probe.
- **Waiting on the host GPU** is a scope of its own around the fence wait
  (`host::submit.wait`): with synchronous submission, the time the submitting
  thread spends in `WaitForSingleObject` is the host GPU's execution of that
  submission, plus a few microseconds of signal. With
  `VICTORY_ASYNC_SUBMIT=1` (the default) that wait happens at the next
  submission's drain, which is why `host::submit.drain` is a separate
  `gpu`-kind row. The per-list GPU timestamps of `VICTORY_GPU_TIMING` (real
  D3D12 timestamp queries on the host) remain the finer instrument for the
  GPU itself; the title-visible timestamps are the CPU's clock (see
  `crates/victory-gpu/src/queue.rs`, `gpu_clock`).
- **Waiting for a lock** is a `sync`-kind scope (`host::submit.lock`,
  `descriptor::lock`): the time a submission waits for the submission lock
  and the descriptor heaps' locks. Together with the thread column this is
  the evidence for threading and scheduling.

## Switches

| Variable | Effect |
|----------|--------|
| `VICTORY_FRAME_PROFILE=<file>` | write the report there (created, parents too) |
| `VICTORY_FRAME_PROFILE_TAGS=host,copy` | only probes whose tag or name starts with one of these (default: all); `-` leaves every probe out |
| `VICTORY_FRAME_PROFILE_SECONDS=<n>` | rewrite the report every `n` seconds (default 20) |
| `VICTORY_FRAME_PROFILE_FRAMES=<n>` | stop after `n` frames, write the last report, stop paying for probes |
| `VICTORY_FRAME_PROFILE_COMPARE=<file>` | add a before/after table against an earlier report |

Compiled out entirely without the `diagnostics` feature: `lib.rs` declares
`mod frameprof` under `#[cfg(feature = "diagnostics")]` and every call site
is a `#[cfg(feature = "diagnostics")]` statement, so a release build has no
code and no branch for it.

With the feature in but the variable unset, each probe costs one `OnceLock`
load and a branch: about 1 ns per probe on the machine this was written on
(optimised test build). A probe that is open costs two clock reads, two
thread-local accesses and three relaxed atomic adds: about 60 ns per
open/close pair. At 9,000 draws a frame the per-draw probe is therefore
about 0.5 ms of a frame, which is 3% of a 150 ms frame and the reason it is
in its own tag (`draw`) and worth turning off when reading anything else.

## Reading the report

```
VIctory frame profile: 120 frames in 20.00 s (6.0 frames/s), 12 sites
frame time: 165.3 ms mean, 164.0 ms median, 201.2 ms max, 120 of 120 frames over 16.7 ms
-- self time, ranked by the share of the frame budget it takes
  share  self ms/f  incl ms/f  calls/f  kind    thread  site
   31.2%      51.5       51.5       5.0  cpu     title-main  host::submit
   ...
-- the worst frames
  frame 1219 (title 4402): 201.2 ms
     38.1%      76.6 ms  host::submit (title-main)
-- what the frames are made of (self time)
cpu 52.3%  gpu 30.1%  memory 12.0%  sync 4.2%  idle 0.8%  outside any scope 12.6%
-- threads
  title-main  ...
-- compared with <file> (self time per frame)
  before     after    change   site
```

- `ms/f` is per frame over the run, so it is comparable between runs of
  different lengths and to the frame time at the top. A thread's share can
  pass 100% when threads run in parallel: it is that thread's busy time
  against the frame, not a part of the frame.
- `calls/f` is what separates "expensive function" from "cheap function
  called far too often".
- `-- the worst frames` gives, for the worst frames by wall time, their own
  breakdown: this is the section for a spike, where the average hides the
  frame that stutters.
- `-- what the frames are made of` with `outside any scope` is the honest
  remainder: at 5 to 7 frames per second in the world, a large `outside` is
  the title's own code, which no probe can see; `VICTORY_PROFILE` (the
  sampler) is where to look then.
- The comparison table needs the same site names in both reports, so keep
  the names stable when moving a probe.

## The probes

| Site | Kind | What it wraps |
|------|------|---------------|
| `host::submit` | cpu | `Host::execute_inner` |
| `host::submit.lock` | sync | taking the submission lock |
| `host::submit.prewarm` | cpu | `Host::prewarm` |
| `host::submit.drain` | gpu | waiting for the previous async submission |
| `host::submit.prepare` | cpu | batch steps, deferred copies, bindless plan, card engine |
| `host::submit.execute` | cpu | `ExecuteCommandLists` |
| `host::submit.wait` | gpu | signalling and waiting for the submission's fence |
| `host::finish.copy_back` | memory | the card's and the shadow's copy-back |
| `host::finish.readbacks` | memory | title read-backs |
| `host::finish.writes` | cpu | bottom-of-pipe writes and timestamp resolves |
| `host::copy.parallel` | memory | `in_parallel` (the parallel copy-back) |
| `host::present` | idle | `Host::present` (swap chain, meter) |
| `shadow::update` | memory | `Shadow::update_lazily` |
| `shadow::copy_back` | memory | `Shadow::copy_back` |
| `list::prelude` | cpu | the prelude list |
| `list::run_deferred` | cpu | a list's submission-time steps |
| `descriptor::plan` | cpu | `HostHeap::bindless_plan` |
| `descriptor::record` | cpu | `prepare_bindless_record` |
| `descriptor::lock` | sync | the descriptor heaps' lock |
| `frame::wait` | idle | the title's frame pacing wait |
| `draw::record` | cpu | one draw, indexed or not (`draw` tag) |
| `queue::present` | cpu | `show()` (sweep, capture, present) |

A new probe is a static and one statement at the top of the function:

```rust
pub(crate) static FOO: Site = Site::new("module::foo", "module", Kind::Cpu);
// ...
#[cfg(feature = "diagnostics")]
let _prof = crate::frameprof::enter(&crate::frameprof::FOO);
```

The value must be bound to a name (`_prof`), not `_`: it closes the scope when
it is dropped, and `let _ = ...` drops it at once. `MOST_SITES` is 64 sites
and `MOST_THREADS` 16 threads; past those, probes are left out.

## What it cannot see

- Title code. Every frame of this title spends most of its time in code
  mapped from the user's game: it has no probes. That time is `outside any
  scope`; the sampler (`VICTORY_PROFILE`) and `VICTORY_STACKS` cover it.
- GPU execution as the GPU sees it. `VICTORY_GPU_TIMING` (host timestamp
  queries per list) is the instrument for that; the GPU-kind rows here are
  the submitting thread's blocked time.
- Memory bandwidth as a counter. `memory`-kind rows are copies, and their
  evidence is the bytes in the five-second frame-rate line (`PREPARING`,
  `COPY_BACK`) and the PIX capture, not a hardware counter.
- Anything before the first present after the profiler starts: that first
  frame is dropped, because it may hold minutes of loading.

## Verifying a run

1. Run the world or the menus for at least a minute with
   `VICTORY_FRAME_PROFILE=prof.txt VICTORY_FRAME_PROFILE_SECONDS=10`.
2. Compare the report's `frame time` and `frames/s` with the window caption
   (the `Meter`, an independent counter over the same presents): means
   within 2%, and the same order of magnitude of spikes. They read the same
   clock, so a disagreement means attribution is wrong, not that time runs
   at two speeds.
3. Run twice, once with `VICTORY_FRAME_PROFILE_TAGS=-` (every probe left
   out, the frames still measured) and once with the tags you care about,
   and compare the frame time: the difference is the probes' cost on that
   scene.
4. Sanity-check attribution: a function that runs once per frame must show
   `calls/f` 1.0 on the thread that ran it; a copy-back that runs twice a
   submission must show twice the submissions per frame.
5. For a before/after of a fix: keep both files and run the second with
   `VICTORY_FRAME_PROFILE_COMPARE=<the first>`.

The 11 unit tests of the module (`crates/victory-gpu/src/frameprof.rs`) need
no game, no GPU and no Windows: they cover the frame attribution, the self
time, the tag selection, the threads, the ranking, the comparison, the wall
accounting and the overhead ceiling. Run them with
`cargo test -p victory-gpu --features diagnostics`.

## Design notes

- Counters are monotonic per (site, thread) and the boundary reads deltas:
  no reset races the recording threads, and a probe that closes late lands
  in the frame it closed in.
- No allocation and no lock on the recording path: two thread-local lookups,
  one clock read, three relaxed adds. The boundary takes the locks (once per
  present, not once per probe).
- A report write can take milliseconds. It happens on the presenting thread
  after the frame was closed, and the next frame starts after the write, so
  the write is in no frame.
- `site.id` is a process-wide atomic: a `Site` belongs to one profiler. The
  tests build their own sites so they never share one.
