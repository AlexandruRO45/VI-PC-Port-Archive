# Console 64-lane compute programs on a 32-lane GPU

The console runs many compute programs in 64-lane waves. NVIDIA GPUs run
32-lane waves only, and the driver refuses a pipeline whose DXIL declares
`WaveSize(64)` (`E_INVALIDARG`). The DXIL itself is valid (DXV 1.9 accepts
it), so the refusal is the host's, not the translator's.

Most such programs translate exactly to 32-lane waves anyway. This note is
about the ones whose cross-lane code spans the whole console wave, and about
the wave-wide form that runs them as two host waves.

## Programs seen in the first room

| Hash | Program | Thread group | Before | Now (wave-wide) |
|---|---|---|---|---|
| `6f225b641c5b0708` | water transmission volumetrics | 64 | refused | runs |
| `5dc3b25986622949` | water transmission volumetrics | 64 | refused | runs |
| `369cbe906ea72764` | GPU particle visibility | 64 | refused | runs |
| `2cc4c197ee29e291` | ray-tracing tile classification | 64 | refused | runs |
| `926c46bab3b86bd6` | (not identified) | 64 | refused | runs |
| `02384a60fda6f2d0` | effects triangle clipping, relocation pass | 128 | refused | runs |
| `6b82bc7d22016cb5` | effects triangle clipping, triangle collection | 1024 | refused | runs (see LDS below) |
| `f133e35efed2a48a` | cascade shadows of alpha-tested casters | 128 | not translated | runs |
| `2da4beb386509b77` | alpha transmission prefilter | 256 | refused | still refused |

A refused or untranslated pipeline drops every draw or dispatch that uses it,
so the effect is simply missing from the frame.

## How the translator places a console wave

A console wave of 64 lanes becomes two host waves of 32. Code that only
reads its own lane, or exchanges values within 32-lane halves (DPP rows,
`ds_swizzle`, `v_permlane16`), is exact as it is. Code that reads across the
halves (`v_readlane`, `v_mbcnt`, lane masks read as numbers, `ds_bpermute`)
needs the two host waves to exchange: each writes its half to LDS, the
group waits, each reads both.

That exchange needs both host waves at the same place. Inside a per-lane
branch, one host wave may have no lane taking it while the other does, so the
exchange cannot happen there. The wave-wide form takes such branches as the
console takes them: by the whole console wave at once (taken if some lane, or
every lane, of the 64 wants it). Both host waves then go the same way and can
meet inside.

## Groups of several console waves

With one console wave per thread group, the two host waves are the whole
group, and they meet at group barriers. With several console waves in a group
(128, 256 or 1024 threads), each console wave takes the wave-wide branches on
its own, so a group barrier inside one would wait for console waves that
never get there.

There the two host waves of each console wave meet only each other
(`lower.rs`, `pair_wait`):

- Each console wave has a counter dword in LDS, zeroed in the entry block
  before a group barrier (the entry block is run by every thread).
- At a meeting, each host wave's first lane adds 1 to its console wave's
  counter. The wave that found an even count `2k` arrived first and spins until
  the count reaches `2k + 2`. The counter only grows, so a wave that already
  went on to the next meeting cannot confuse one still spinning.
- LDS fences before and after the wait order the halves' LDS accesses as a
  barrier would.
- After 2^18 reads the wave goes on regardless: a missing partner gives wrong
  values, not a hung GPU.

The wait is a loop, so it splits the current block. `Module::split_before`
moves the code emitted so far into a new block and places the wait between
it and the current one. Code that callers add to the current block later
runs after the wait, which keeps every caller that holds the current block
correct.

A lane mask made in an earlier block and exchanged where it is used is
exchanged at the use, where every lane of the console wave must run.

### LDS budget

The pair meetings use one LDS dword per console wave, besides the exchange
dwords (two per console wave) and the wave counter. `6b82bc7d` already uses all
32 KiB, as a bitset over triangles read from its start. For programs named in
`VICTORY_WAVE_WIDE_ONLY` whose LDS would exceed 32 KiB, these dwords go over
the program's last ones. This is a shortcut: it assumes the program does not
reach its last bytes.

## Switches

| Switch | Effect |
|---|---|
| `VICTORY_WAVE_WIDE_ONLY=<hash>,...` | the wave-wide form for the listed programs only |
| `VICTORY_WAVE_WIDE=1` | the wave-wide form for every program (lost the GPU in earlier runs; diagnostic only) |
| `VICTORY_WAVE_WHY=1` | the translator's reasons for each wave64 program's form, on stderr |

The list that covers the first room:

```
VICTORY_WAVE_WIDE_ONLY=6f225b641c5b0708,5dc3b25986622949,369cbe906ea72764,2cc4c197ee29e291,926c46bab3b86bd6,02384a60fda6f2d0,6b82bc7d22016cb5,f133e35efed2a48a
```

## Results

Offline (`victory translate --validate --pipelines` with
`VICTORY_HOST_WAVE=32 VICTORY_WAVE_WIDE=1`, RTX 4090): 8 of the 9 programs
above translate, validate and create pipelines.

Live, first room, same binary, only the list differs (4 programs against the
8 above), 25 s per run after the room is steady:

| Run | List | FPS | ms/frame | Refused pipelines |
|---|---|---|---|---|
| control 1 | 4 | 7.2 | 138 | 3 |
| control 2 | 4 | 6.9 | 145 | 3 |
| change 1 | 8 | 6.6 | 151 | 1 |
| change 2 | 8 | 6.3 | 159 | 1 |

About 8% slower, because three more programs now run instead of being
dropped. No device loss in any run.

## Known gaps

- `2da4beb386509b77` builds a sampler inside a region the console skips when
  no lane runs it. Taken wave-wide, that region is skipped on the host too, and
  the sampler is then undefined after it. It stays refused.
- The LDS overlap above is a shortcut for `6b82bc7d`.
- The spin limit trades a hang for wrong values if a partner wave never
  arrives.

## Tests

`crates/victory-shader/src/emit/wave64_tests.rs`:
`two_console_waves_take_a_wave_wide_branch_apart` runs a 128-thread group in
which only the first console wave takes the branch. It runs on the GPU when
one with 32-lane waves is present.
