# What still stops compute shaders

Measured on branch `cap/compute-permute` (rebased onto `fork/main` `61ca6fe`) from its
own worktree, over lane B's container corpus:

```
cargo build --release -p victory
target\release\victory.exe translate D:\victory-scratch\lane-b\containers --list
```

25,889 containers, 26,084 programs. The summary prints emitted-of-total for the three
graphics classes, and `--list` prints one line per refused program, counted here by
stage:

| Emitted | | Refused programs | |
|--------:|--:|--------:|--:|
| vertex and pixel | 13,022 of 14,031 | pixel sequences | 1,209 |
| compute | 2,457 of 2,611 | mesh sequences | 384 |
| mesh | 8,381 of 8,713 | compute sequences | 225 |
| | | geometry sequences | 195 |
| | | vertex sequences | 16 |
| | | **total** | **2,029** |

The two columns count different things — a program can refuse one pass and be emitted
by a later one — so they are not meant to add up to one denominator.

A refusal is an attempt, not an absence. A program refused at one emit pass can be
reached again by a later one, and a pipeline refused at creation may come into
existence on a later attempt — so this column measures failed attempts, not programs
that never run. Only a hash whose pipeline never appears at all is work owed.

The mesh column is about **emission**, not creation: lane B's wave-size fix took mesh
pipeline refusals in the title from 545 to zero, so a mesh refusal measured here is a
program that does not become DXIL, not a PSO the host cannot make.

## What the opening cutscene needs

Measured from one session log (`D:\victory-scratch\cutscene.log`, 1,024 frames
presented, world packs opened at 19:37:02, draws rising to 2,031 by frame 1024):

- the session requested **442** shader-stage programs once the world packs opened —
  11 vertex, 208 pixel, 220 mesh, 3 compute;
- **all 442 emit on this branch**, and the 3 compute also pass `dxil.dll` validation
  (`victory.exe translate D:\victory-scratch\cutscene-set --list --validate` →
  `3 emitted of 3 compute programs; 3 validated`);
- the only pipelines that fail in that session are **505 compute PSOs**, every one
  `CreateComputePipelineState` returning `E_INVALIDARG`, on the first attempt and again
  on the retry.

So no program the cutscene requested refuses on anything in this document. The three
compute programs it binds every frame (`052a2316a03b9ba1`, `a133bf0a24aff410`,
`4c0ac499ad613809` — containers 22438, 22439, 22441) translate and validate; they are
dropped at creation, which is the host's gap, not the emitter's.

## The families, in corpus order

| Programs | Refusal | Stages |
|---------:|---------|--------|
| 372 | `emit: a descriptor's high address read as data` | 219 Ps, 95 Ms, 58 Cs |
| 290 | `emit: a descriptor's base address read as data` | 276 Ps, 11 Ms, 3 Cs |
| 242 | `emit: a lane mask read as data` | 110 Ps, 109 Ms, 23 Cs |
| 207 | `I/O: an interpolation with computed barycentrics` | 207 Ps |
| 195 | `resources: an address in an SGPR that is no root parameter` | 195 Gs |
| 168 | `emit: a packed input half no element holds` | 168 Ps |
| 131 | `emit: a descriptor's format read as data` | 89 Ms, 23 Cs, 15 Ps, 4 Vs |
| 74 | `emit: a cross-lane operation outside compute` | 74 Ps |
| 64 | `resources: an address not from a root parameter` | 64 Cs |
| 49 | `emit: a descriptor read only as data` | 31 Ps, 15 Ms, 3 Vs |
| 41 | `emit: a descriptor chosen by control flow` | 37 Ps, 3 Ms, 1 Cs |
| 38 | `resources: a descriptor read from the first lane of computed VGPRs` | 34 Ms, 4 Vs |
| 28 | `emit: the first half of an interpolation` | 28 Ps |
| 22 | `emit: a descriptor's address or format read as data` | 14 Cs, 8 Ps |
| 17 | `emit: a constant buffer's descriptor read as data` | 7 Cs, 6 Ms, 4 Ps |
| 15 | `resources: a waterfall loop whose compares do not agree` | 12 Ms, 3 Vs |
| 13 | `emit: a constant lane mask` | 13 Ps |
| 12 | `emit: a lane mask holding lanes elsewhere read as data` | 12 Cs |
| 8 | `resources: a descriptor partly joined by control flow` | 6 Ms, 2 Vs |

Examples: `01115` at `0x55d0`, `01858` at `0x5d40`, `00595` at `0xd98`, `10550` at
`0x3790`, `21358` at `0x44`, `11190` at `0x5b4`, `01822` at `0x5bb4`.

## A descriptor dword with no host value, read as a number

832 programs — the five descriptor-dword rows above, 372 + 290 + 131 + 22 + 17 — read a
descriptor's own dwords and test them as numbers:

```
s_load_dwordx4 s[4:7], s[2:3], null      ; the descriptor itself
s_and_b32    s0, s4, 0x100               ; a bit of its address
s_cmp_eq_u32 s0, 0
s_cbranch_scc1
s_and_b32    s4, s19, 0xff0000           ; the format field
```

`descriptor_data` (`emit/lower.rs`) rebuilds an image's dimensions and a buffer's
record count from the host view, which is exact, and deliberately leaves the address
and format dwords without a value: a host resource has no GPU address, and the format
dword is the title's, not the view's. A guessed bit here silently changes a path in
~660 programs, so these refuse instead.

The honest fix is a raw view over the title's own descriptor heap, keyed by the heap
index the handle carries, so the shader reads the dwords the title wrote rather than a
value invented for it. `victory-gpu` keeps no byte mirror of a title heap, but a title
heap is guest memory the shadow already copies, so a read view is a binding over a
title address resolved at bind time. Contract in progress with the graphics thread; the
hidden parameter goes last in the synthesised signature so no program that does not read
a descriptor as data changes.

These are two replicated shapes, not 832 different programs: `01115`/`01119`/`01123`
refuse at the same offset with the same instruction sequence, and `01858`/`01862`/`01866`
likewise.

## The pointer whose dwords come from a descriptor

64 compute programs stop on `resources: an address not from a root parameter`, and 38 in
the two dump sets used to stop on the same thing. The doc used to call this "a pointer
read from a buffer's contents"; that was wrong, and lane B's measurement corrected it —
no program in either dump set reads a pointer out of a buffer.

What the programs actually do (`compute-190fbf0ebe040169`, container 21809, at `0xcc`):

```
s_load_dwordx16 s[8:23], s[2:3], 0x18     ; dwords out of the root descriptor table
v_add_co_u32   v5, s0, s10, v0           ; a per-lane offset onto them
v_add_co_ci_u32 v6, s0, s11, 0, s0
global_load_dwordx2 v[5:6], v[5:6], null
```

So the address is assembled from a descriptor's dwords, not from a buffer's contents:
the same missing capability as the section above, one step further along. `Root::Pointer`
in `emit/host.rs` and main's `reg.checked_sub(256)` branch in `victory-gpu/src/draw.rs`
bind the one-hop form where the pointer sits in the root constants; what these need is
the dword to come from the descriptor the program loaded.

Examples: `compute-190fbf0ebe040169` at `0xcc`, `compute-6d7fe7745ab9c6d8` at `0x834`,
`compute-f22a76fcb967ee59` at `0x7e8`, `compute-17829ac71aab4e8d` at `0x88c`.

## Descriptors read from per-lane computed VGPRs

38 mesh and 4 vertex programs stop on `a descriptor read from the first lane of computed
VGPRs`, and 8 mesh on `a descriptor partly joined by control flow`. The `waterfall` proof
refuses them, so the descriptor differs between lanes in a way the tracer cannot
attribute to a mask, a `v_cndmask_b32` or a control-flow join.

Examples: `mesh-89a622617919c92a` at `0x5fe8`, `mesh-fe660f2d6d57de76` at `0x5d90`.

## Single instructions

| Programs | Refusal | Instruction |
|---------:|---------|-------------|
| 2 | `resources: an address from a phi` | an address whose halves a phi joins from arms that are not one sum |
| 1 | `resources: a descriptor neither loaded nor built` | `compute-2da4beb386509b77` at `0x4d0` |
| 1 | `resources: an address from a root parameter that is no address` | `compute-f6e47efa50226beb` at `0x13ec` |
| 1 | `resources: an address in an SGPR that is no root parameter` | `compute-162216b9cfd91aaf` at `0x8c` |
| 1 | `emit: a descriptor chosen by control flow` | `compute-39f0e9aeb156a61c` at `0x288dc` |
| 1 + 1 | `emit: a lane-mask join of an untracked mask` | `compute-a53a7977fe762268` at `0x10a78` |

## Worth knowing before you start

- `victory.exe translate --list` prints the offset of the instruction that *failed*,
  which for an emit error is the one that needed a missing value, not the one that
  produced it. `victory.exe shader-info <file> --disasm` and the offsets around it are
  the way in.
- Refusals that split by field — `base address`, `high address`, `format` — were added
  on this branch precisely so the queue reads as four gaps instead of one.
- 64-lane waves: a program that permutes with `ds_bpermute_b32` or `v_readlane_b32`
  across the 32-lane boundary gets the host's 32-lane answer, because `cross_lane` has
  no emulation for it. `console_lanes` relabelling in `emit/lower.rs` is the machinery
  that would have to carry it.
- A refusal line in a title log is a failed attempt. In the cutscene session 505 compute
  PSOs were refused and the session still presented 1,024 frames; read the refusal
  against the number of hashes whose pipeline never appears at all.
