# What still stops compute shaders

Measured with the two dump sets a diagnostics run leaves behind, on branch
`cap/compute-descriptors`, from its own worktree:

```
cargo build --release -p victory
target\release\victory.exe translate <game dir>\VIctory-data\failed-shaders --list
target\release\victory.exe translate <game dir>\VIctory-data\failed-shaders-after-session --list
```

| Set | Containers | Compute emitted, main | Compute emitted, this branch |
|-----|-----------:|----------------------:|-----------------------------:|
| `failed-shaders` | 58 | 13 of 24 | 19 of 28 |
| `failed-shaders-after-session` | 54 | 0 of 15 | 6 of 19 |

The denominator grows because more programs get past resource tracking and
reach the emitter. Every refusal below is measured, not predicted.

## The one that matters most: a pointer the host cannot bind

15 programs in the first set and 23 in the second still stop on
`resources: an address not from a root parameter`. Nearly all of them are
the same shape, and it is not a translation gap — it is a missing host
capability:

```
v_readfirstlane_b32 s4, v1        ; or a plain per-lane value
s_mul_i32 s4, s4, 28
s_addk_i32 s4, 0x0050
s_add_u32 s4, s4, s8              ; s8:s9 = a pointer the program read from memory
s_addc_u32 s5, 0, s9
v_mov_b32 v1, s4
v_mov_b32 v2, s5
global_load_dword v1, v[1:2], null
```

The pointer's *value* lives in GPU memory — it was read by a scalar or global
load from a buffer the host does bind — so the host cannot turn it into a root
view at bind time today. `resource::Tracer::pointer_location` refuses it as
`a pointer to a pointer`, which is the honest answer until the host can follow
the hop.

What the host would need: `victory-gpu` already keeps a shadow copy of every
title buffer and already refreshes the ranges a submission reads before it
runs, so the qword at `(the location the program loaded from)` can be read at
bind time, resolved with `shadow::find`, and bound as a root SRV. That is the
same two steps as `Root::Pointer` in `victory-shader/src/emit/host.rs`, one
level deeper, and it needs a wider root-view register encoding than
`((dword + 1) << 8) | slot` because the location is `(param, offset)` of
another parameter, not necessarily the constants. Main already binds the
one-hop form (`victory-gpu/src/draw.rs`, the `reg.checked_sub(256)` branch), so
the register encoding is the only thing in the way.

Examples: `compute-190fbf0ebe040169` at `0xcc`, `compute-6d7fe7745ab9c6d8` at
`0x834`, `compute-f22a76fcb967ee59` at `0x7e8`, `compute-17829ac71aab4e8d` at
`0x88c`.

## A descriptor dword with no host value, read as a number

| Programs | Refusal | What the program does |
|---------:|---------|-----------------------|
| 4 + 5 | `a descriptor's high address read as data` | arithmetic on the descriptor's base address bits |
| 2 | `a descriptor's address or format read as data` | one of the two |
| 1 + 2 | `a descriptor's format read as data` | compares or masks the format field |
| 1 + 2 | `a constant buffer's descriptor read as data` | sizes a row from a CBV descriptor |

`descriptor_data` (`emit/lower.rs`) rebuilds an image's dimensions and a
buffer's record count from the host view, which is exact, and deliberately
leaves the address and format dwords without a value: a host resource has no
GPU address, and the format dword is the title's, not the view's. A program
that reads those has to keep refusing. The only route would be to have the
host pass the address it actually bound, which changes the root view contract
for every shader, and to carry a format table through the cache.

Examples: `compute-1c02f8131c508c75` at `0x7c90`, `compute-3fa1fda35402c93f` at
`0x26dc`, `compute-585ecfc8622dffca` at `0x163c`.

## Descriptors read from per-lane computed VGPRs

4 mesh programs and 1 compute stop on
`a descriptor read from the first lane of computed VGPRs`, and 1 mesh on
`a descriptor partly joined by control flow`. The `waterfall` proof refuses
them, so the descriptor differs between lanes in a way the tracer cannot
attribute to a mask, a `v_cndmask_b32` or a control-flow join. These are in
the resource layer even though they are mesh programs.

Examples: `mesh-89a622617919c92a` at `0x5fe8`, `mesh-fe660f2d6d57de76` at
`0x5d90`.

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

- `victory.exe translate --list` prints the offset of the instruction that
  *failed*, which for an emit error is the one that needed a missing value, not
  the one that produced it. `victory.exe shader-info <file> --disasm` and the
  offsets around it are the way in.
- Refusals that split by field — `base address`, `high address`, `format` —
  were added on this branch precisely so the queue reads as four gaps instead
  of one.
- 64-lane waves: nothing in either dump needs a host wave wider than 32 any
  more, but a program that permutes a `ds_bpermute_b32` or a `v_readlane_b32`
  across the 32-lane boundary gets the host's 32-lane answer, because
  `cross_lane` has no emulation for it. `console_lanes` relabelling in
  `emit/lower.rs` is the machinery that would have to carry it.
