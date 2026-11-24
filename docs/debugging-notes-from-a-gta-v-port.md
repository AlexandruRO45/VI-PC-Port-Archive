# Debugging notes from a GTA V port

Lessons from bringing GTA V (RAGE, D3D11 path) up natively on an Android
phone (ARM64, D3D11 over DXVK and Vulkan on an Adreno 740). That port hit
the stages VIctory is at now: the title loads the world, then the scene is
dark, frames are slow and the GPU can hang. What found the causes there is
written down here as things to try. None of it has been checked against
GTA VI.

Nothing from either game is in this note: no code, names from it or data.

## A dark scene after loading

**Find the pass where the image goes empty before guessing.** In the port
the world was black and characters were flat grey shapes. Lighting was the
obvious suspect and it was wrong: a dump of every bound render target and
depth buffer before each change of targets (141 passes in one frame) showed
the G-buffer and lighting were correct, and the frame went black in the
copy that feeds post-processing. Min, max and mean per channel for each
dump is enough to spot the pass at a glance. `VICTORY_DUMP_TARGETS` already
saves targets; dumping at every target change, not only at the end of a
frame, is what pinned it.

**The title expects something the console does for it.** The black world
came from a step the title only takes on some platforms: the platform being
ported to took the branch that skips a back-buffer copy, and the
post-process read a stale copy. The general form for VIctory: anything the
console GPU or runtime does implicitly (compression metadata on render
targets, fast clears, decompression or resolve on transition, memory that
CPU and GPU see without copies) has to be done by the host layer, or a
later pass reads an empty or cleared target. The README lists
render-target metadata planes as missing; a target whose contents only
exist in that metadata would read as its clear colour.

**Auto-exposure turns one bad value into a dark frame.** If the luminance
reduction produces 0 or NaN, the whole scene goes dark while the G-buffer
is fine. It works both ways: in the port, fixing the exposure showed the
scene was already black before it, so exposure was ruled out in one run. In
VIctory, a dropped compute dispatch in the exposure chain would look exactly
like a dark scene. Forcing a fixed exposure, or finding the adaptation
buffer and checking its value, separates the two.

**Dropped or wrong passes.** In an earlier PC build, shaders whose passes did
not match what the code indexed gave "black world, HUD visible". Since VIctory
drops draws whose shaders are not translated yet, it helps to log which
passes lose draws or dispatches in a dark frame. Drawing those in a debug
colour instead of skipping them shows what is missing on screen.

**A constant that arrives as 0 makes a shader sample texel (0,0).** A skin
pass in the port received its screen-size constant as 0, so every pixel read
the top-left texel: characters came out one flat colour, exactly that
pixel's. When a surface is a single flat colour, compare it with pixel (0,0)
of the pass's inputs, then check the constant buffer for zeros (wrong
offset, size or root-parameter mapping).

## GPU hangs

**Separate the hang from the rendering first.** The "flat grey shapes" seen
at one point in the port were a half-drawn frame left on screen by a GPU
hang, not a shading bug. WARP, or full synchronisation, shows what the
frame looks like when nothing hangs.

**Experiments that narrowed it down:**

* Waiting for the GPU to finish before each present: no hangs (7 clean runs
  in a row). As a test, it shows the fault depends on how much CPU and GPU
  work overlap, not on one shader.
* One submission per draw: never hung in over 10 minutes, at 3 fps.
* Ending the render pass at every shader change: moved the hang and once
  showed corrupted texture blocks (overwritten BC data). That pointed to
  memory reused while the GPU still read it.
* Early fences: the title recycles memory when its fences say the GPU is
  done; a fence signalled early lets it overwrite data in flight. Tested by
  marking fences complete only when the whole command list had finished,
  and logging any fence the driver reported earlier. That gave 0 cases, so
  the hypothesis was dropped. In VIctory the title manages console fences
  and memory itself, so the same check on the mapping of title fences to
  host fences is worth one run.
* The cause there turned out to be the phone vendor's driver with that
  command pattern, fixed by switching to Mesa's Turnip driver. On a desktop
  AMD GPU the answer may differ, but the steps above apply.

**A hang at a fixed point is a gift.** It happened about 44 s in, at the same
cinematic-to-gameplay switch every time. Recording what was drawn and what
memory had just been freed or reused right before that point narrows it
much faster than random runs.

## Loads that stall

Each stall in the port was one of these:

* data the title waits on without reporting it (audio banks missing);
* data still encrypted, or in a form the loader did not expect;
* budgets and pool sizes from the platform's config (a streaming budget of
  0 MB, a loaded-object pool too small for the data set): either a silent
  wait or an abort that names the pool;
* a function with no return value on the platform's branch, returning
  garbage (a thread count used as an array index);
* streaming reads that never complete, intermittently. A dump of the loader's
  queue when its head stops moving shows which request is stuck.

## Tooling that paid off

* Switches that need no rebuild, like VIctory's environment variables.
* A crash handler that names the function and walks the frame chain even
  when the program counter is invalid.
* All threads' stacks on demand for hangs with no crash
  (`VICTORY_STACKS`).
* Short, bounded logging: one line per frame was enough to cost frame rate.
* The same frame-rate measurement every run, several samples per run.

## What does not carry over

That port recompiled the title for another CPU; VIctory runs its x86-64 code
unchanged. These problems from the port do not apply:

* memory ordering on ARM (empty barriers, counters seen before data);
* compiler differences (signed `char`, null checks after `new`, references
  assumed non-null);
* the data formats and encryption of GTA V's PC release.
