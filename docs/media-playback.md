# The boot movie's rate, measured rather than read off the caption

The presentation rate in the window caption is a one-line summary of the last second
and it cannot tell these apart:

* the title asked for 15 frames and got 15 presented,
* the title asked for 30 and our layer presented 15,
* the title asked for 30 and presented 30 copies of 15 distinct images,
* the title asked for 30 and got 30, with one image per frame.

Only the last one is "the movie is 30 fps". The instruments below separate all four
cheaply; every number in this file came from them.

## Instruments

| What | How | What it answers |
|------|-----|-----------------|
| Presented frames and the title's own frame tokens | `cargo build --profile diag -p victory --features diagnostics`, then `VICTORY_LOG=victory_gpu::frame=debug`, which logs `present token=<n>` once per presented frame | A token that jumps means the title counted frames we never presented; a token that advances one per present, with presents 33 ms apart, means the title itself is pacing slower |
| The images themselves | `VICTORY_CAPTURE=<dir>` with `VICTORY_CAPTURE_FRAMES=<a,b,c,…>` saving **consecutive** frames, then diffing neighbouring pairs | Identical neighbours on alternate pairs is a movie presenting at half its content rate; a difference on every pair is not |
| The movie's audio | `VICTORY_AUDIO_CAPTURE=<file.wav>` (process loopback, 48 kHz stereo) | Cross-correlating two captures over the same stretch of movie: `corr 1.0000` and a mean absolute sample difference under 1 LSB is bit-identical audio |
| Cost per frame | the `NN fps, MM ms/frame (… copies …, textures …; … on the GPU; …)` line at info level, every 5 s | Whether a rate is a pacing problem or work: 60 ms *on the GPU* is a shader or a draw, 30 ms of *preparing* is host-side copying |

Two traps, both measured here:

* `VICTORY_LOG=…,victory_hle=debug` suppresses presentation entirely — one run presented
  **0 frames in 54 s** against 1,477 on the same binary with the default filter. Never
  measure a rate with a subsystem-wide debug filter on.
* A run that overlaps another title session measures contention. `victory.exe run` in
  the process list is what counts; `victory.exe translate …` is offline and holds no GPU.

## Where the movie is

On this build the boot sequence presents, per run and in frame numbers:

| Frames | Screen |
|--------|--------|
| 1–160 | black |
| 160–250 | flat dark-blue title card, drifting |
| 256–560 | the movie: photographic, a different image on **every** present |
| 560+ | menus and loading |

The movie's audio starts at **9.25 s** and runs about 6 s of loud material, which is
what makes it easy to align two captures of different runs.

## Numbers, 2026-11-24

| Head | Profile | Presented | Content | Audio |
|------|---------|-----------|---------|-------|
| `5728350` | diag | 30.0 fps, tokens one per present | one image per present | reference |
| `e3dac6e` | diag | 27–30 fps, tokens one per present | one image per present | `corr 1.0000`, 0.3 LSB |
| `991cff7` | diag | 24–35 fps (mean ≈ 29) | one image per present | `corr 1.0000`, 0.3 LSB |
| `991cff7` | release | 30.0 fps | — | — |
| `187d156` | diag, cold shader cache (1,811 first binds) | 19–38 fps (mean ≈ 29) | — | — |
| `b77a938` | diag | 30 fps with dips to 8–20 while lane B's `translate` ran alongside | — | `corr 1.0000` |

Full-load check: with 18 busy processes saturating all 24 logical CPUs, the movie still
presented at 30–31 fps, so its rate is not CPU starvation.

## The other mode in the record

Runs earlier in the day show the movie at **11–15 fps with 56–66 ms on the GPU** per
frame at 3–6 draws and 12.7 MB of texture a frame (`victory-run.stderr.log` 14:42,
`victory-audio-debug.stdout.log` 15:48, `world-capture2.log` 16:32), while the runs from
17:05 onward show the same window at **0–2 ms on the GPU**. Those runs make one image
per present in both modes, so the halving there is a *per-frame GPU cost*, not a pacing
or presentation defect. `VICTORY_GPU_TIMING=1` on a diagnostics build names the slow
list and its slowest commands for the frame, which is the instrument to reach for if
that mode comes back.

## The pacing contract this lane holds

`victory-gpu/src/frame.rs`: a frame's origin poll (which always times out in this title
and still takes the frame's token) and its blocking flip wait must both end on that
frame's tick. Two ticks a frame presents exactly half the rate the title asked for while
the token keeps counting one frame per flip — the one signature a caption, a token log
and an audio capture all agree is "fine". `a_frames_origin_and_flip_share_a_tick` in that
file asserts it: 8 frames must not take 16 intervals.

One latent way to break it, worth knowing: the per-kind "last tick" table is indexed
`LAST[kind.min(2)]`, so every event kind from 3 upward shares the flip's slot. Two waits
of different kinds at or above 2 in one frame would then take two ticks. This title asks
for kinds 0 and 2 only (measured over six runs), so nothing is halved today.
